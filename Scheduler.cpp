//
//  Scheduler.cpp
//  CloudSim
//
//  Created by ELMOOTAZBELLAH ELNOZAHY on 10/20/24.
//

#include <cstdint>
#include "Scheduler.hpp"
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <climits>
#include <deque>
#include <iostream>
#include <string>
#include <cmath>

using namespace std;

typedef enum { CUSTOM_GREEDY, PMAPPER, EECO, PABFD } AlgorithmType;
static AlgorithmType CURRENT_ALGO = PMAPPER;  // Change this to switch algorithms

// ============================================================================
// 1. CUSTOM_GREEDY IMPLEMENTATION
// ============================================================================

static map<MachineId_t, vector<VMId_t>> greedy_m2v;
static set<MachineId_t> greedy_waking;
static map<CPUType_t, deque<TaskId_t>> greedy_pending_by_cpu;
static set<VMId_t> greedy_migrating;
static map<TaskId_t, VMId_t> greedy_task_to_vm;
static unsigned greedy_rr_idx = 0;
// Set to true whenever a task completes or a machine wakes to S0, so
// PeriodicCheck only scans pending tasks when capacity actually changed.
static bool greedy_retry_pending = false;

// Local VM property cache.
static map<VMId_t, VMType_t>    greedy_vm_type;
static map<VMId_t, CPUType_t>   greedy_vm_cpu;
static map<VMId_t, MachineId_t> greedy_vm_machine;
static map<VMId_t, unsigned>    greedy_vm_tasks;  // count of active tasks on VM

// Local machine property cache — avoids calling Machine_GetInfo in the hot path.
struct GreedyMachineCache {
    MachineState_t s_state;
    CPUType_t      cpu;
    unsigned       memory_size;
    unsigned       memory_used;
    unsigned       active_tasks;
    unsigned       active_vms;
    bool           gpus;
};
static map<MachineId_t, GreedyMachineCache> greedy_mc;

static bool Greedy_VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true;
}

static Priority_t Greedy_SLAPrio(SLAType_t sla) {
    if (sla == SLA0 || sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2)                return MID_PRIORITY;
    return LOW_PRIORITY;
}

static VMId_t Greedy_EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct, vector<VMId_t>& all_vms) {
    for (VMId_t vm : greedy_m2v[mid]) {
        if (greedy_migrating.count(vm)) continue;
        if (greedy_vm_type[vm] == vt && greedy_vm_cpu[vm] == ct) return vm;
    }
    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    greedy_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
    greedy_mc[mid].active_vms++;
    all_vms.push_back(vm);
    greedy_m2v[mid].push_back(vm);
    greedy_vm_type[vm]    = vt;
    greedy_vm_cpu[vm]     = ct;
    greedy_vm_machine[vm] = mid;
    greedy_vm_tasks[vm]   = 0;
    return vm;
}

static MachineId_t Greedy_BestFit(TaskId_t task, const vector<MachineId_t>& mlist) {
    CPUType_t rc  = RequiredCPUType(task);
    VMType_t  rv  = RequiredVMType(task);
    unsigned  rm  = GetTaskMemory(task);
    bool      rg  = IsTaskGPUCapable(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict   = (sla == SLA0 || sla == SLA1);

    if (!Greedy_VMCPUOk(rv, rc)) return MachineId_t(UINT_MAX);

    unsigned n = mlist.size();
    MachineId_t best = MachineId_t(UINT_MAX);
    int best_score = INT_MAX;

    for (unsigned i = 0; i < n; i++) {
        MachineId_t mid = mlist[(greedy_rr_idx + i) % n];
        const GreedyMachineCache& mc = greedy_mc[mid];
        if (mc.s_state != S0 || mc.cpu != rc) continue;

        bool has_vm = false;
        for (VMId_t vm : greedy_m2v[mid]) {
            if (greedy_migrating.count(vm)) continue;
            if (greedy_vm_type[vm] == rv && greedy_vm_cpu[vm] == rc) { has_vm = true; break; }
        }
        unsigned extra = has_vm ? 0 : VM_MEMORY_OVERHEAD;
        if (mc.memory_size < mc.memory_used + rm + extra) continue;

        int load  = (int)mc.active_tasks;
        int score = strict ? load : -load;
        if (rg && !mc.gpus) score += 1000;

        if (score < best_score) { best = mid; best_score = score; }
    }
    if (best != MachineId_t(UINT_MAX)) greedy_rr_idx = (greedy_rr_idx + 1) % n;
    return best;
}

static void Greedy_TrySleep(MachineId_t mid, vector<VMId_t>& all_vms) {
    if (greedy_waking.count(mid)) return;
    GreedyMachineCache& mc = greedy_mc[mid];
    if (mc.s_state != S0 || mc.active_tasks > 0) return;

    vector<VMId_t>& vlist = greedy_m2v[mid];
    vector<VMId_t> keep;
    for (VMId_t vm : vlist) {
        if (greedy_migrating.count(vm)) { keep.push_back(vm); continue; }
        if (greedy_vm_tasks[vm] == 0) {
            VM_Shutdown(vm);
            mc.memory_used -= VM_MEMORY_OVERHEAD;
            mc.active_vms--;
            greedy_vm_type.erase(vm);
            greedy_vm_cpu.erase(vm);
            greedy_vm_machine.erase(vm);
            greedy_vm_tasks.erase(vm);
            all_vms.erase(remove(all_vms.begin(), all_vms.end(), vm), all_vms.end());
        } else {
            keep.push_back(vm);
        }
    }
    vlist = keep;
    if (mc.active_vms == 0) {
        // Use S3 (warm standby) instead of S5 so machines wake much faster
        // when the next burst arrives. Energy cost is small relative to SLA gains.
        Machine_SetState(mid, S3);
        mc.s_state = S3;
    }
}

// ============================================================================
// 2. PMAPPER IMPLEMENTATION
// ============================================================================
//
// pMapper: Power-aware VM placement using Best-Fit Decreasing (BFD).
//
// New task placement:
//   Among active (S0) machines, pick the most-utilized one that can still fit
//   the task. This consolidates load onto fewer hosts, leaving lightly-used
//   machines idle so they can be powered down.
//
// Periodic consolidation:
//   Machines with utilization < PMAPPER_U_LOW are candidates.
//   Their VMs are migrated to the most-utilized machine that still has
//   headroom (< PMAPPER_U_HIGH). Emptied machines are put to sleep (S5).
// ============================================================================

static const double PMAPPER_U_LOW  = 0.20;
static const double PMAPPER_U_HIGH = 0.85;
// Only run consolidation every N periodic checks to avoid O(n²) every tick.
static const unsigned PMAPPER_CONSOLIDATE_INTERVAL = 200;

static map<MachineId_t, vector<VMId_t>> pm_m2v;
static map<TaskId_t, VMId_t>            pm_task_to_vm;
static map<CPUType_t, deque<TaskId_t>>  pm_pending_by_cpu;
static set<MachineId_t>                 pm_transitioning;
static set<VMId_t>                      pm_migrating;
static bool                             pm_retry_pending  = false;
static unsigned                         pm_check_counter  = 0;

// Local machine cache — avoids copying Machine_GetInfo vectors in hot paths.
struct PMachineCache {
    MachineState_t s_state;
    CPUType_t      cpu;
    unsigned       num_cpus;
    unsigned       memory_size;
    unsigned       memory_used;
    unsigned       active_tasks;
    unsigned       active_vms;
    bool           gpus;
};
static map<MachineId_t, PMachineCache> pm_mc;

// Local VM cache.
struct PMVMCache {
    VMType_t    vm_type;
    CPUType_t   cpu;
    MachineId_t machine_id;
    unsigned    task_count;
    unsigned    memory_used;  // VM_MEMORY_OVERHEAD + sum of task memory
};
static map<VMId_t, PMVMCache> pm_vc;

static Priority_t PMapper_SLAPrio(SLAType_t sla) {
    if (sla == SLA0 || sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2)                return MID_PRIORITY;
    return LOW_PRIORITY;
}

static bool PMapper_VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true;
}

static VMId_t PMapper_EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct,
                                vector<VMId_t>& all_vms) {
    for (VMId_t vm : pm_m2v[mid]) {
        if (pm_migrating.count(vm)) continue;
        if (pm_vc[vm].vm_type == vt && pm_vc[vm].cpu == ct) return vm;
    }
    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    pm_m2v[mid].push_back(vm);
    pm_vc[vm] = { vt, ct, mid, 0, VM_MEMORY_OVERHEAD };
    pm_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
    pm_mc[mid].active_vms++;
    return vm;
}

// Cache-based host check — no Machine_GetInfo / VM_GetInfo calls.
static bool PMapper_CanHost(MachineId_t mid, CPUType_t rc, VMType_t rv, unsigned rm) {
    const PMachineCache& mc = pm_mc[mid];
    if (mc.s_state != S0 || mc.cpu != rc) return false;
    if (!PMapper_VMCPUOk(rv, rc)) return false;
    bool has_vm = false;
    for (VMId_t vm : pm_m2v[mid]) {
        if (pm_migrating.count(vm)) continue;
        if (pm_vc[vm].vm_type == rv && pm_vc[vm].cpu == rc) { has_vm = true; break; }
    }
    unsigned need = rm + (has_vm ? 0 : VM_MEMORY_OVERHEAD);
    return mc.memory_size >= mc.memory_used + need;
}

// Placement: SLA0/SLA1 → least-loaded (minimize queue wait for tight deadlines).
//             SLA2/SLA3 → most-loaded BFD (consolidation for energy efficiency).
// GPU-capable tasks get a large penalty for non-GPU machines.
static MachineId_t PMapper_BestFit(TaskId_t task, const vector<MachineId_t>& mlist) {
    CPUType_t rc  = RequiredCPUType(task);
    VMType_t  rv  = RequiredVMType(task);
    unsigned  rm  = GetTaskMemory(task);
    bool      rg  = IsTaskGPUCapable(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict   = (sla == SLA0 || sla == SLA1);

    MachineId_t best      = MachineId_t(UINT_MAX);
    int         best_score = INT_MAX;

    for (MachineId_t mid : mlist) {
        if (!PMapper_CanHost(mid, rc, rv, rm)) continue;
        const PMachineCache& mc = pm_mc[mid];
        int load  = (int)mc.active_tasks;
        int score = strict ? load : -load;
        if (rg && !mc.gpus) score += 1000;
        if (score < best_score) { best_score = score; best = mid; }
    }
    return best;
}

void PMapper_Init(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    for (MachineId_t mid : machines) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        Machine_SetState(mid, S0);
        pm_mc[mid] = { S0, mi.cpu, mi.num_cpus, mi.memory_size,
                       mi.memory_used, mi.active_tasks, mi.active_vms, mi.gpus };
        VMType_t vt = (mi.cpu == POWER) ? AIX : LINUX;
        VMId_t vm = VM_Create(vt, mi.cpu);
        VM_Attach(vm, mid);
        vms.push_back(vm);
        pm_m2v[mid].push_back(vm);
        pm_vc[vm] = { vt, mi.cpu, mid, 0, VM_MEMORY_OVERHEAD };
        pm_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
        pm_mc[mid].active_vms++;
        for (unsigned c = 0; c < mi.num_cpus; c++)
            Machine_SetCorePerformance(mid, c, P0);
    }
}

void PMapper_NewTask(Time_t now, TaskId_t task,
                     vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    MachineId_t mid = PMapper_BestFit(task, machines);
    if (mid == MachineId_t(UINT_MAX)) {
        CPUType_t rc = RequiredCPUType(task);
        SLAType_t sla = RequiredSLA(task);
        bool strict = (sla == SLA0 || sla == SLA1);
        // Wake ALL sleeping machines for strict SLA bursts, one for non-strict.
        for (MachineId_t m : machines) {
            if (pm_transitioning.count(m)) continue;
            if (pm_mc[m].cpu == rc && pm_mc[m].s_state != S0) {
                Machine_SetState(m, S0);
                pm_transitioning.insert(m);
                if (!strict) break;
            }
        }
        pm_pending_by_cpu[rc].push_back(task);
    } else {
        VMId_t vm = PMapper_EnsureVM(mid, RequiredVMType(task), RequiredCPUType(task), vms);
        VM_AddTask(vm, task, PMapper_SLAPrio(RequiredSLA(task)));
        pm_task_to_vm[task] = vm;
        unsigned tm = GetTaskMemory(task);
        pm_vc[vm].task_count++;
        pm_vc[vm].memory_used += tm;
        pm_mc[mid].active_tasks++;
        pm_mc[mid].memory_used += tm;
    }
}

void PMapper_PeriodicCheck(Time_t now,
                            vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    bool has_pending = false;
    for (auto& [_, bkt] : pm_pending_by_cpu) {
        if (!bkt.empty()) { has_pending = true; break; }
    }
    pm_check_counter++;
    bool do_consolidate = (pm_check_counter % PMAPPER_CONSOLIDATE_INTERVAL == 0);

    if (!pm_retry_pending && !has_pending && !do_consolidate) return;

    // --- 1. Consolidation (rate-limited) -------------------------------------
    if (do_consolidate) {
        vector<pair<double, MachineId_t>> running_by_util;
        for (MachineId_t mid : machines) {
            if (pm_transitioning.count(mid)) continue;
            const PMachineCache& mc = pm_mc[mid];
            if (mc.s_state != S0) continue;
            double util = mc.num_cpus > 0 ? (double)mc.active_tasks / mc.num_cpus : 0.0;
            running_by_util.push_back({util, mid});
        }
        sort(running_by_util.begin(), running_by_util.end());

        for (auto& [util, src] : running_by_util) {
            if (util >= PMAPPER_U_LOW) break;
            if (pm_transitioning.count(src)) continue;
            const PMachineCache& smc = pm_mc[src];

            bool all_migrated = true;
            for (VMId_t vm : pm_m2v[src]) {
                if (pm_migrating.count(vm)) continue;
                if (pm_vc[vm].task_count == 0) continue;  // idle VM

                unsigned vm_mem = pm_vc[vm].memory_used;

                MachineId_t dst      = MachineId_t(UINT_MAX);
                double      dst_util = -1.0;
                for (MachineId_t cand : machines) {
                    if (cand == src || pm_transitioning.count(cand)) continue;
                    const PMachineCache& cmc = pm_mc[cand];
                    if (cmc.s_state != S0 || cmc.cpu != smc.cpu) continue;
                    if (cmc.memory_size < cmc.memory_used + vm_mem) continue;
                    double cu = cmc.num_cpus > 0
                                ? (double)cmc.active_tasks / cmc.num_cpus : 0.0;
                    if (cu >= PMAPPER_U_HIGH) continue;
                    if (cu > dst_util) { dst_util = cu; dst = cand; }
                }
                if (dst == MachineId_t(UINT_MAX)) { all_migrated = false; continue; }

                // Update caches for the migration
                pm_mc[dst].memory_used  += vm_mem;
                pm_mc[dst].active_tasks += pm_vc[vm].task_count;
                pm_mc[src].memory_used  -= vm_mem;
                pm_mc[src].active_tasks -= pm_vc[vm].task_count;
                pm_vc[vm].machine_id = dst;

                VM_Migrate(vm, dst);
                pm_migrating.insert(vm);
                pm_m2v[dst].push_back(vm);
                pm_m2v[src].erase(remove(pm_m2v[src].begin(), pm_m2v[src].end(), vm),
                                   pm_m2v[src].end());
            }

            if (!all_migrated) continue;

            vector<VMId_t> keep;
            for (VMId_t vm : pm_m2v[src]) {
                if (pm_migrating.count(vm)) { keep.push_back(vm); continue; }
                pm_mc[src].memory_used -= pm_vc[vm].memory_used;
                pm_mc[src].active_vms--;
                pm_vc.erase(vm);
                VM_Shutdown(vm);
                vms.erase(remove(vms.begin(), vms.end(), vm), vms.end());
            }
            pm_m2v[src] = keep;
            if (pm_mc[src].active_vms == 0) {
                // S3 wakes faster than S5 — reduces SLA violations on next burst.
                Machine_SetState(src, S3);
                pm_mc[src].s_state = S3;
                pm_transitioning.insert(src);
            }
        }
    }

    // --- 2. Retry pending tasks (SLA priority order) -------------------------
    if (pm_retry_pending || has_pending) {
        pm_retry_pending = false;
        for (auto& [cpu_type, bucket] : pm_pending_by_cpu) {
            if (bucket.empty()) continue;
            stable_sort(bucket.begin(), bucket.end(), [](TaskId_t a, TaskId_t b) {
                SLAType_t sa = RequiredSLA(a), sb = RequiredSLA(b);
                int pa = (sa==SLA0)?0:(sa==SLA1)?1:(sa==SLA2)?2:3;
                int pb = (sb==SLA0)?0:(sb==SLA1)?1:(sb==SLA2)?2:3;
                return pa < pb;
            });
            while (!bucket.empty()) {
                TaskId_t t = bucket.front();
                MachineId_t mid = PMapper_BestFit(t, machines);
                if (mid != MachineId_t(UINT_MAX)) {
                    bucket.pop_front();
                    VMId_t vm = PMapper_EnsureVM(mid, RequiredVMType(t), cpu_type, vms);
                    VM_AddTask(vm, t, PMapper_SLAPrio(RequiredSLA(t)));
                    pm_task_to_vm[t] = vm;
                    unsigned tm = GetTaskMemory(t);
                    pm_vc[vm].task_count++;
                    pm_vc[vm].memory_used += tm;
                    pm_mc[mid].active_tasks++;
                    pm_mc[mid].memory_used += tm;
                } else {
                    // Wake a machine for this CPU type and stop for now.
                    for (MachineId_t m : machines) {
                        if (pm_transitioning.count(m)) continue;
                        if (pm_mc[m].cpu == cpu_type && pm_mc[m].s_state != S0) {
                            Machine_SetState(m, S0);
                            pm_transitioning.insert(m);
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
}

void PMapper_TaskComplete(Time_t now, TaskId_t task,
                           vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    auto it = pm_task_to_vm.find(task);
    if (it == pm_task_to_vm.end()) return;
    VMId_t vm = it->second;
    MachineId_t mid = pm_vc[vm].machine_id;
    unsigned tm = GetTaskMemory(task);
    pm_vc[vm].task_count--;
    pm_vc[vm].memory_used -= tm;
    pm_mc[mid].active_tasks--;
    pm_mc[mid].memory_used -= tm;
    pm_task_to_vm.erase(it);
    pm_retry_pending = true;
}

void PMapper_MigrationComplete(VMId_t vm,
                                vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    pm_migrating.erase(vm);
    pm_retry_pending = true;
}

void PMapper_Shutdown(vector<VMId_t>& vms) {
    for (VMId_t vm : vms) {
        if (pm_vc.count(vm) && pm_vc[vm].task_count == 0) VM_Shutdown(vm);
    }
}



// ============================================================================
// 3. EECO IMPLEMENTATION
// ============================================================================
//
// Three-tier model:
//   Running     (S0)  — hosts actively serving tasks
//   Intermediate (S3) — standby hosts, fast to wake (~seconds)
//   Switched off (S5) — deep sleep, slow to wake
//
// Tier sizes are recomputed on every PeriodicCheck using EECO formulae:
//
//   U_cur   = current cluster utilization  (active_tasks / capacity)
//   N_run   = ceil(U_cur / U_target)       running hosts needed
//   N_inter = ceil(alpha * N_run)           intermediate buffer (alpha ∈ [0.1, 0.3])
//   N_off   = Total - N_run - N_inter
//
// No VM migration. Tasks are placed on any S0 machine with capacity.
// When workload rises, intermediate hosts wake (S3→S0) first.
// When workload falls, excess running hosts demote to S3, then S5.
// ============================================================================

// ---- Tunables ---------------------------------------------------------------
static const double EECO_U_TARGET = 0.70;  // target utilization for running tier
static const double EECO_ALPHA    = 0.20;  // intermediate tier = 20% of running tier
// -----------------------------------------------------------------------------

static map<MachineId_t, vector<VMId_t>> eeco_m2v;
static map<TaskId_t, VMId_t>            eeco_task_to_vm;
static vector<TaskId_t>                 eeco_pending;
static set<MachineId_t>                 eeco_transitioning;  // Track machines currently changing state

// Helper: map SLA → priority (same logic as Greedy)
static Priority_t EECO_SLAPrio(SLAType_t sla) {
    if (sla == SLA0 || sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2)                return MID_PRIORITY;
    return LOW_PRIORITY;
}

// Helper: find or create a VM of the right type on a machine
static VMId_t EECO_EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct,
                             vector<VMId_t>& all_vms) {
    for (VMId_t vm : eeco_m2v[mid]) {
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.vm_type == vt && vi.cpu == ct) return vm;
    }
    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    eeco_m2v[mid].push_back(vm);
    return vm;
}

// Helper: can this machine accept this task right now?
static bool EECO_CanHost(MachineId_t mid, TaskId_t task) {
    MachineInfo_t mi = Machine_GetInfo(mid);
    if (mi.s_state != S0)          return false;
    if (mi.cpu != RequiredCPUType(task)) return false;

    VMType_t vt = RequiredVMType(task);
    CPUType_t ct = RequiredCPUType(task);
    if (vt == AIX && ct != POWER)  return false;

    // Check memory: need task memory + overhead if no matching VM yet
    bool has_vm = false;
    for (VMId_t vm : eeco_m2v[mid]) {
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.vm_type == vt && vi.cpu == ct) { has_vm = true; break; }
    }
    unsigned need = GetTaskMemory(task) + (has_vm ? 0 : VM_MEMORY_OVERHEAD);
    return mi.memory_size >= mi.memory_used + need;
}

// Helper: pick the best running machine for a task (first-fit among S0 hosts)
static MachineId_t EECO_FindHost(TaskId_t task, const vector<MachineId_t>& mlist) {
    MachineId_t best = MachineId_t(UINT_MAX);
    unsigned    best_load = UINT_MAX;
    for (MachineId_t mid : mlist) {
        if (!EECO_CanHost(mid, task)) continue;
        unsigned load = Machine_GetInfo(mid).active_tasks;
        if (load < best_load) { best = mid; best_load = load; }
    }
    return best;
}

// Core EECO tier-rebalancing logic — called from PeriodicCheck
static void EECO_Rebalance(const vector<MachineId_t>& mlist, vector<VMId_t>& all_vms) {
    unsigned total = (unsigned)mlist.size();
    if (total == 0) return;

    // ---- Tally tiers first --------------------------------------------------
    vector<MachineId_t> running, intermediate, off_hosts;
    for (MachineId_t mid : mlist) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        if      (mi.s_state == S0) running.push_back(mid);
        else if (mi.s_state == S3) intermediate.push_back(mid);
        else                       off_hosts.push_back(mid);
    }

    // ---- Measure utilization ------------------------------------------------
    unsigned cap_sum  = 0;
    unsigned task_sum = 0;
    for (MachineId_t mid : running) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        cap_sum  += mi.num_cpus;
        task_sum += mi.active_tasks;
    }
    double U_cur = (cap_sum > 0) ? (double)task_sum / (double)cap_sum : 0.0;

    // ---- Compute target tier sizes ------------------------------------------
    // How many running hosts are needed to serve current load at U_TARGET?
    // Derived from: U_cur * N_run_cur / U_TARGET = N_run_target
    unsigned N_run_cur = (unsigned)running.size();
    unsigned N_run_target;

    if (U_cur == 0.0) {
        N_run_target = 1;  // keep at least one warm
    } else {
        N_run_target = (unsigned)ceil((U_cur / EECO_U_TARGET) * (double)N_run_cur);
    }

    // Hard clamp: can't exceed total, must be at least 1
    N_run_target = max(1u, min(N_run_target, total));

    // Intermediate buffer gets what's left after running, capped by ALPHA
    unsigned headroom       = total - N_run_target;
    unsigned N_inter_target = min((unsigned)ceil(EECO_ALPHA * (double)N_run_target), headroom);

    // ---- Step 1: promote intermediate → running if under-supplied -----------
    if (N_run_cur < N_run_target) {
        unsigned need = N_run_target - N_run_cur;
        for (unsigned i = 0; i < need && i < intermediate.size(); i++) {
            MachineId_t mid = intermediate[i];
            if (!eeco_transitioning.count(mid)) {
                Machine_SetState(mid, S0);
                MachineInfo_t mi = Machine_GetInfo(mid);
                for (unsigned c = 0; c < mi.num_cpus; c++)
                    Machine_SetCorePerformance(mid, c, P0);
                eeco_transitioning.insert(mid);
            }
        }
        // If intermediate ran dry, pull from off_hosts too
        if (need > intermediate.size()) {
            unsigned still_need = need - (unsigned)intermediate.size();
            for (unsigned i = 0; i < still_need && i < off_hosts.size(); i++) {
                MachineId_t mid = off_hosts[i];
                if (!eeco_transitioning.count(mid)) {
                    Machine_SetState(mid, S0);
                    MachineInfo_t mi = Machine_GetInfo(mid);
                    for (unsigned c = 0; c < mi.num_cpus; c++)
                        Machine_SetCorePerformance(mid, c, P0);
                    eeco_transitioning.insert(mid);
                }
            }
        }
    }

    // ---- Step 2: promote off → intermediate if buffer is thin ---------------
    unsigned inter_cur = (unsigned)intermediate.size();
    if (inter_cur < N_inter_target) {
        unsigned need = N_inter_target - inter_cur;
        for (unsigned i = 0; i < need && i < off_hosts.size(); i++) {
            MachineId_t mid = off_hosts[i];
            if (!eeco_transitioning.count(mid)) {
                Machine_SetState(mid, S3);
                eeco_transitioning.insert(mid);
            }
        }
    }

    // ---- Step 3: demote surplus running → intermediate (ONLY if idle) -------
    if (N_run_cur > N_run_target) {
        // Sort so we try to demote least-loaded first
        vector<MachineId_t> candidates = running;
        sort(candidates.begin(), candidates.end(), [](MachineId_t a, MachineId_t b){
            return Machine_GetInfo(a).active_tasks < Machine_GetInfo(b).active_tasks;
        });
        unsigned demoted = 0;
        unsigned surplus = N_run_cur - N_run_target;
        for (MachineId_t mid : candidates) {
            if (demoted >= surplus) break;
            if (eeco_transitioning.count(mid)) continue;  // Skip if transitioning
            MachineInfo_t mi = Machine_GetInfo(mid);
            if (mi.active_tasks > 0) continue;  // never evict a busy host
            for (VMId_t vm : eeco_m2v[mid]) {
                if (VM_GetInfo(vm).active_tasks.empty()) {
                    VM_Shutdown(vm);
                    all_vms.erase(remove(all_vms.begin(), all_vms.end(), vm), all_vms.end());
                }
            }
            eeco_m2v[mid].clear();
            Machine_SetState(mid, S3);
            eeco_transitioning.insert(mid);
            demoted++;
        }
    }

    // ---- Step 4: demote surplus intermediate → off --------------------------
    vector<MachineId_t> inter_now;
    for (MachineId_t mid : mlist)
        if (Machine_GetInfo(mid).s_state == S3) inter_now.push_back(mid);

    if ((unsigned)inter_now.size() > N_inter_target) {
        unsigned surplus = (unsigned)inter_now.size() - N_inter_target;
        for (unsigned i = 0; i < surplus && i < inter_now.size(); i++) {
            MachineId_t mid = inter_now[i];
            if (!eeco_transitioning.count(mid)) {
                Machine_SetState(mid, S5);
                eeco_transitioning.insert(mid);
            }
        }
    }
}

// Init funciton to go into switch logic
void EECO_Init(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    unsigned total = (unsigned)machines.size();
    unsigned run_n   = max(1u, total);
    unsigned inter_n = 0;

    for (unsigned i = 0; i < total; i++) {
        MachineId_t mid = machines[i];
        MachineInfo_t mi = Machine_GetInfo(mid);

        if (i < run_n) {
            // Running tier — create a starter VM and set cores to P0
            Machine_SetState(mid, S0);
            VMId_t vm = VM_Create((mi.cpu == POWER) ? AIX : LINUX, mi.cpu);
            VM_Attach(vm, mid);
            vms.push_back(vm);
            eeco_m2v[mid].push_back(vm);
            for (unsigned c = 0; c < mi.num_cpus; c++)
                Machine_SetCorePerformance(mid, c, P0);
        } else if (i < run_n + inter_n) {
            Machine_SetState(mid, S3);   // intermediate (standby)
        } else {
            Machine_SetState(mid, S5);   // switched off
        }
    }
}

void EECO_NewTask(Time_t now, TaskId_t task,
                  vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    MachineId_t mid = EECO_FindHost(task, machines);
    if (mid == MachineId_t(UINT_MAX)) {
        // No running host available — queue and let PeriodicCheck promote one
        eeco_pending.push_back(task);
    } else {
        VMId_t vm = EECO_EnsureVM(mid, RequiredVMType(task),
                                   RequiredCPUType(task), vms);
        VM_AddTask(vm, task, EECO_SLAPrio(RequiredSLA(task)));
        eeco_task_to_vm[task] = vm;
    }
}

void EECO_PeriodicCheck(Time_t now,
                         vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    EECO_Rebalance(machines, vms);

    vector<TaskId_t> still_pending;
    for (TaskId_t t : eeco_pending) {
        MachineId_t mid = EECO_FindHost(t, machines);
        if (mid != MachineId_t(UINT_MAX)) {
            VMId_t vm = EECO_EnsureVM(mid, RequiredVMType(t), RequiredCPUType(t), vms);
            VM_AddTask(vm, t, EECO_SLAPrio(RequiredSLA(t)));
            eeco_task_to_vm[t] = vm;
        } else {
            // Force-promote a matching host if none is running for this CPU type
            CPUType_t rc = RequiredCPUType(t);
            for (MachineId_t m : machines) {
                MachineInfo_t mi = Machine_GetInfo(m);
                // Skip if already transitioning or already in target state
                if (eeco_transitioning.count(m)) continue;
                if (mi.cpu == rc && mi.s_state != S0) {
                    Machine_SetState(m, S0);
                    for (unsigned c = 0; c < mi.num_cpus; c++)
                        Machine_SetCorePerformance(m, c, P0);
                    eeco_transitioning.insert(m);  // Mark as transitioning
                    break;  // one at a time; retry next tick
                }
            }
            still_pending.push_back(t);
        }
    }
    eeco_pending = still_pending;
}

void EECO_TaskComplete(Time_t now, TaskId_t task,
                        vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    eeco_task_to_vm.erase(task);
    // Rebalance triggers demotion of now-idle hosts on the next periodic check
}

void EECO_Shutdown(vector<VMId_t>& vms) {
    for (VMId_t vm : vms) {
        if (VM_GetInfo(vm).active_tasks.empty()) VM_Shutdown(vm);
    }
}

// ============================================================================
// 3. MANDATORY SCHEDULER METHODS (The "Switch" logic)
// ============================================================================

static Scheduler theScheduler;

void Scheduler::Init() {
    unsigned total = Machine_GetTotal();
    for (unsigned i = 0; i < total; i++) machines.push_back(MachineId_t(i));

    switch(CURRENT_ALGO) {
        case CUSTOM_GREEDY:
            for (MachineId_t mid : machines) {
                MachineInfo_t mi = Machine_GetInfo(mid);
                greedy_mc[mid] = { mi.s_state, mi.cpu, mi.memory_size,
                                   mi.memory_used, mi.active_tasks,
                                   mi.active_vms, mi.gpus };
                VMType_t vt = (mi.cpu == POWER) ? AIX : LINUX;
                VMId_t vm = VM_Create(vt, mi.cpu);
                VM_Attach(vm, mid);
                greedy_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
                greedy_mc[mid].active_vms++;
                vms.push_back(vm);
                greedy_m2v[mid].push_back(vm);
                greedy_vm_type[vm]    = vt;
                greedy_vm_cpu[vm]     = mi.cpu;
                greedy_vm_machine[vm] = mid;
                greedy_vm_tasks[vm]   = 0;
                for (unsigned c = 0; c < mi.num_cpus; c++) Machine_SetCorePerformance(mid, c, P0);
            }
            break;
        case PMAPPER:
            PMapper_Init(machines, vms);
            break;
        case EECO:
            EECO_Init(machines, vms);
            break;
        case PABFD: /* Call PABFD_Init here */ break;
        default: break;
    }
}

void Scheduler::NewTask(Time_t now, TaskId_t task_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        MachineId_t mid = Greedy_BestFit(task_id, machines);
        if (mid == MachineId_t(UINT_MAX)) {
            CPUType_t rc = RequiredCPUType(task_id);
            SLAType_t sla = RequiredSLA(task_id);
            bool strict = (sla == SLA0 || sla == SLA1);
            // For strict SLA tasks wake ALL sleeping machines of the right type
            // immediately — a burst of SLA0/SLA1 tasks can't afford to wake one
            // machine at a time. For non-strict, still limit to one.
            for (MachineId_t m : machines) {
                if (greedy_waking.count(m)) continue;
                if (greedy_mc[m].cpu == rc && greedy_mc[m].s_state != S0) {
                    Machine_SetState(m, S0);
                    greedy_waking.insert(m);
                    if (!strict) break;
                }
            }
            greedy_pending_by_cpu[rc].push_back(task_id);
        } else {
            VMId_t vm = Greedy_EnsureVM(mid, RequiredVMType(task_id), RequiredCPUType(task_id), vms);
            VM_AddTask(vm, task_id, Greedy_SLAPrio(RequiredSLA(task_id)));
            greedy_vm_tasks[vm]++;
            greedy_mc[mid].memory_used  += GetTaskMemory(task_id);
            greedy_mc[mid].active_tasks++;
            greedy_task_to_vm[task_id] = vm;
        }
    }
    else if (CURRENT_ALGO == PMAPPER) {
        PMapper_NewTask(now, task_id, machines, vms);
    }
    else if(CURRENT_ALGO == EECO){
        EECO_NewTask(now, task_id, machines, vms);
    }
}

void Scheduler::PeriodicCheck(Time_t now) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        bool has_pending = false;
        for (auto& [_, bkt] : greedy_pending_by_cpu) {
            if (!bkt.empty()) { has_pending = true; break; }
        }
        if (!greedy_retry_pending && !has_pending) return;
        greedy_retry_pending = false;

        for (auto& [cpu_type, bucket] : greedy_pending_by_cpu) {
            if (bucket.empty()) continue;
            // Prioritize strict-SLA tasks (SLA0/SLA1) to front of bucket.
            stable_sort(bucket.begin(), bucket.end(), [](TaskId_t a, TaskId_t b) {
                SLAType_t sa = RequiredSLA(a), sb = RequiredSLA(b);
                int pa = (sa == SLA0) ? 0 : (sa == SLA1) ? 1 : (sa == SLA2) ? 2 : 3;
                int pb = (sb == SLA0) ? 0 : (sb == SLA1) ? 1 : (sb == SLA2) ? 2 : 3;
                return pa < pb;
            });
            while (!bucket.empty()) {
                MachineId_t mid = Greedy_BestFit(bucket.front(), machines);
                if (mid != MachineId_t(UINT_MAX)) {
                    TaskId_t t = bucket.front();
                    bucket.pop_front();
                    VMId_t vm = Greedy_EnsureVM(mid, RequiredVMType(t), cpu_type, vms);
                    VM_AddTask(vm, t, Greedy_SLAPrio(RequiredSLA(t)));
                    greedy_vm_tasks[vm]++;
                    greedy_mc[mid].memory_used  += GetTaskMemory(t);
                    greedy_mc[mid].active_tasks++;
                    greedy_task_to_vm[t] = vm;
                } else {
                    break;
                }
            }
        }
    }
    else if (CURRENT_ALGO == PMAPPER) {
        PMapper_PeriodicCheck(now, machines, vms);
    }
    else if (CURRENT_ALGO == EECO){
        EECO_PeriodicCheck(now, machines, vms);
    }
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        auto it = greedy_task_to_vm.find(task_id);
        if (it != greedy_task_to_vm.end()) {
            VMId_t vm = it->second;
            MachineId_t mid = greedy_vm_machine[vm];
            greedy_vm_tasks[vm]--;
            greedy_mc[mid].memory_used  -= GetTaskMemory(task_id);
            greedy_mc[mid].active_tasks--;
            greedy_task_to_vm.erase(it);
            Greedy_TrySleep(mid, vms);
            greedy_retry_pending = true;
        }
    }
    else if (CURRENT_ALGO == PMAPPER) {
        PMapper_TaskComplete(now, task_id, machines, vms);
    }
    else if (CURRENT_ALGO == EECO){
        EECO_TaskComplete(now, task_id, machines, vms);
    }
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) greedy_migrating.erase(vm_id);
    else if (CURRENT_ALGO == PMAPPER)  PMapper_MigrationComplete(vm_id, machines, vms);
}

void Scheduler::Shutdown(Time_t time) {
    if (CURRENT_ALGO == PMAPPER) { PMapper_Shutdown(vms); return; }
    for (VMId_t vm : vms) {
        if (CURRENT_ALGO == CUSTOM_GREEDY) {
            MachineId_t mid = greedy_vm_machine.count(vm) ? greedy_vm_machine[vm]
                                                           : MachineId_t(UINT_MAX);
            if (mid != MachineId_t(UINT_MAX) && greedy_mc[mid].s_state != S0) continue;
        }
        if (VM_GetInfo(vm).active_tasks.empty()) VM_Shutdown(vm);
    }
}

// ============================================================================
// 4. PUBLIC INTERFACE (Wrappers for theScheduler)
// ============================================================================

void InitScheduler() { theScheduler.Init(); }
void HandleNewTask(Time_t time, TaskId_t task_id) { theScheduler.NewTask(time, task_id); }
void HandleTaskCompletion(Time_t time, TaskId_t task_id) { theScheduler.TaskComplete(time, task_id); }
void MemoryWarning(Time_t time, MachineId_t machine_id) { }
void MigrationDone(Time_t time, VMId_t vm_id) { theScheduler.MigrationComplete(time, vm_id); }
void SchedulerCheck(Time_t time) { theScheduler.PeriodicCheck(time); }
void SLAWarning(Time_t time, TaskId_t task_id) {
    SetTaskPriority(task_id, HIGH_PRIORITY);
    CPUType_t rc = RequiredCPUType(task_id);
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        unsigned total = Machine_GetTotal();
        for (unsigned i = 0; i < total; i++) {
            MachineId_t m = MachineId_t(i);
            if (greedy_waking.count(m)) continue;
            if (greedy_mc[m].cpu == rc && greedy_mc[m].s_state != S0) {
                Machine_SetState(m, S0);
                greedy_waking.insert(m);
            }
        }
        greedy_retry_pending = true;
    } else if (CURRENT_ALGO == PMAPPER) {
        for (auto& [mid, mc] : pm_mc) {
            if (pm_transitioning.count(mid)) continue;
            if (mc.cpu == rc && mc.s_state != S0) {
                Machine_SetState(mid, S0);
                pm_transitioning.insert(mid);
            }
        }
        pm_retry_pending = true;
    }
}

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        greedy_waking.erase(machine_id);
        MachineInfo_t mi = Machine_GetInfo(machine_id);
        greedy_mc[machine_id].s_state = mi.s_state;
        if (mi.s_state == S0) {
            for (unsigned c = 0; c < mi.num_cpus; c++) Machine_SetCorePerformance(machine_id, c, P0);
            greedy_retry_pending = true;
        }
    }
    else if (CURRENT_ALGO == PMAPPER) {
        pm_transitioning.erase(machine_id);
        MachineInfo_t mi = Machine_GetInfo(machine_id);
        pm_mc[machine_id].s_state = mi.s_state;
        if (mi.s_state == S0) {
            for (unsigned c = 0; c < mi.num_cpus; c++)
                Machine_SetCorePerformance(machine_id, c, P0);
            pm_retry_pending = true;
        }
    }
    else if (CURRENT_ALGO == EECO) {
        eeco_transitioning.erase(machine_id);  // Remove from transitioning set
        MachineInfo_t mi = Machine_GetInfo(machine_id);
        if (mi.s_state == S0) {
            for (unsigned c = 0; c < mi.num_cpus; c++)
                Machine_SetCorePerformance(machine_id, c, P0);
        }
    }
}

void SimulationComplete(Time_t time) {
    // This function is called before the simulation terminates Add whatever you feel like.
    cout << "SLA violation report" << endl;
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;     // SLA3 do not have SLA violation issues
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    cout << "Simulation run finished in " << double(time)/1000000 << " seconds" << endl;
    SimOutput("SimulationComplete(): Simulation finished at time " + to_string(time), 4);

    theScheduler.Shutdown(time);
}
