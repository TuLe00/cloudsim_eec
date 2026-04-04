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

// ============================================================================
// PABFD SCHEDULER
// ============================================================================

// ============================================================================
// 4. PABFD IMPLEMENTATION
// ============================================================================
//
// Power-Aware Best Fit Decreasing (PABFD) — Beloglazov & Buyya (2010/2012)
//
// Core idea:
//   Model host power as P(u) = P_idle + (P_max - P_idle) * u  (linear).
//   When placing a task, choose the active (S0) host whose power draw
//   increases the LEAST after the task lands, while staying below a
//   utilization ceiling (PABFD_U_MAX).  This is "Best Fit" in the power
//   dimension instead of the memory dimension.
//
//   The "Decreasing" part: pending tasks are sorted by memory demand
//   (largest first) before placement, so large jobs don't get stranded
//   after small ones have consumed the last free slots.
//
// Power model (normalised, machine-class agnostic):
//   P(u) = PABFD_P_IDLE + (1.0 - PABFD_P_IDLE) * u
//   ΔP   = P(u_after) - P(u_before)
//
// Machine management:
//   • All machines start S0.
//   • Machines emptied during consolidation transition to S3 (fast-wake
//     standby) — not S5 — so the next burst can be served quickly.
//   • PeriodicCheck consolidates: machines below PABFD_U_LOW have their
//     VMs migrated to the best-fit (min ΔP) host with headroom below
//     PABFD_U_HIGH, then the empty machine goes to S3.
//   • SLA-strict tasks (SLA0/SLA1) bypass the ΔP criterion and pick
//     the least-loaded host to minimise queue wait.
//   • GPU-capable tasks prefer GPU-equipped machines (large ΔP penalty
//     on non-GPU hosts keeps them as a last resort, not a hard block).
//   • When no S0 host can accept a task it is queued per CPU type;
//     a sleeping machine of the right type is woken immediately.
//     SLA0/SLA1 bursts wake ALL matching machines at once.
// ============================================================================
 
// ---- Tunables ---------------------------------------------------------------
static const double   PABFD_P_IDLE  = 0.60;   // idle power fraction of peak
static const double   PABFD_U_MAX   = 0.90;   // hard utilisation ceiling
static const double   PABFD_U_LOW   = 0.10;   // consolidation source threshold (only near-empty machines)
static const double   PABFD_U_HIGH  = 0.80;   // consolidation destination ceiling
static const double   PABFD_GPU_PENALTY = 500.0; // ΔP penalty for mismatched GPU
static const unsigned PABFD_CONSOLIDATE_INTERVAL = 300; // periodic-check ticks
static const unsigned PABFD_MIN_ACTIVE = 4;   // min S0 machines to keep per CPU type
 
// ---- State ------------------------------------------------------------------
struct PABFDMachineCache {
    MachineState_t s_state;
    CPUType_t      cpu;
    unsigned       num_cpus;
    unsigned       memory_size;
    unsigned       memory_used;
    unsigned       active_tasks;
    unsigned       active_vms;
    bool           gpus;
    unsigned       mips_p0;   // MIPS at P0 — used to prefer faster machines
};
static map<MachineId_t, PABFDMachineCache> pabfd_mc;
static unsigned pabfd_ref_mips = 1;  // peak MIPS across all machines (set at Init)
 
struct PABFDVMCache {
    VMType_t    vm_type;
    CPUType_t   cpu;
    MachineId_t machine_id;
    unsigned    task_count;
    unsigned    memory_used;  // VM_MEMORY_OVERHEAD + sum of hosted task memory
};
static map<VMId_t, PABFDVMCache> pabfd_vc;
 
static map<MachineId_t, vector<VMId_t>> pabfd_m2v;
static map<TaskId_t, VMId_t>            pabfd_task_to_vm;
static map<CPUType_t, deque<TaskId_t>>  pabfd_pending;   // queued per CPU type
static set<MachineId_t>                 pabfd_transitioning;
static set<VMId_t>                      pabfd_migrating;
static bool                             pabfd_retry_pending = false;
static unsigned                         pabfd_check_counter = 0;
 
// ---- Power model ------------------------------------------------------------
 
// Normalised instantaneous power for a host at utilisation u ∈ [0,1].
static double PABFD_Power(double u) {
    return PABFD_P_IDLE + (1.0 - PABFD_P_IDLE) * u;
}
 
// Power increase on a machine if we add `extra_tasks` tasks.
// Returns a large sentinel when the result would exceed PABFD_U_MAX.
static double PABFD_DeltaPower(const PABFDMachineCache& mc, unsigned extra_tasks) {
    if (mc.num_cpus == 0) return 1e18;
    double u_before = (double)mc.active_tasks / mc.num_cpus;
    double u_after  = (double)(mc.active_tasks + extra_tasks) / mc.num_cpus;
    if (u_after > PABFD_U_MAX) return 1e18;  // over-ceiling: reject
    return PABFD_Power(u_after) - PABFD_Power(u_before);
}
 
// ---- Helpers ----------------------------------------------------------------
 
static bool PABFD_VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true;  // LINUX runs on any CPU
}
 
static Priority_t PABFD_SLAPrio(SLAType_t sla) {
    if (sla == SLA0 || sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2)                return MID_PRIORITY;
    return LOW_PRIORITY;
}
 
// Find or create a VM of the required type/CPU on machine `mid`.
static VMId_t PABFD_EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct,
                               vector<VMId_t>& all_vms) {
    for (VMId_t vm : pabfd_m2v[mid]) {
        if (pabfd_migrating.count(vm)) continue;
        if (pabfd_vc[vm].vm_type == vt && pabfd_vc[vm].cpu == ct) return vm;
    }
    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    pabfd_m2v[mid].push_back(vm);
    pabfd_vc[vm] = { vt, ct, mid, 0, VM_MEMORY_OVERHEAD };
    pabfd_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
    pabfd_mc[mid].active_vms++;
    return vm;
}
 
// True iff machine `mid` can host a task with requirements (rc,rv,rm).
static bool PABFD_CanHost(MachineId_t mid, CPUType_t rc, VMType_t rv, unsigned rm) {
    const PABFDMachineCache& mc = pabfd_mc[mid];
    if (mc.s_state != S0 || mc.cpu != rc) return false;
    if (!PABFD_VMCPUOk(rv, rc))           return false;
 
    bool has_vm = false;
    for (VMId_t vm : pabfd_m2v[mid]) {
        if (pabfd_migrating.count(vm)) continue;
        if (pabfd_vc[vm].vm_type == rv && pabfd_vc[vm].cpu == rc) {
            has_vm = true;
            break;
        }
    }
    unsigned need = rm + (has_vm ? 0u : (unsigned)VM_MEMORY_OVERHEAD);
    return mc.memory_size >= mc.memory_used + need;
}
 
// ---- Core placement: minimum power-increase (PABFD criterion) ---------------
//
// For SLA0/SLA1 (strict deadline): switch criterion to minimum active_tasks
// (least-loaded) so the task starts executing as soon as possible.
//
// GPU penalty: tasks that are GPU-capable get PABFD_GPU_PENALTY added to ΔP
// when placed on a non-GPU machine, making GPU-equipped hosts strongly
// preferred but non-GPU hosts still usable as overflow.
static MachineId_t PABFD_BestFit(TaskId_t task, const vector<MachineId_t>& mlist) {
    CPUType_t rc  = RequiredCPUType(task);
    VMType_t  rv  = RequiredVMType(task);
    unsigned  rm  = GetTaskMemory(task);
    bool      rg  = IsTaskGPUCapable(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict   = (sla == SLA0 || sla == SLA1);
 
    if (!PABFD_VMCPUOk(rv, rc)) return MachineId_t(UINT_MAX);
 
    MachineId_t best    = MachineId_t(UINT_MAX);
    double      best_dp = 1e18;
    int         best_load = INT_MAX;  // used in strict mode
 
    for (MachineId_t mid : mlist) {
        if (!PABFD_CanHost(mid, rc, rv, rm)) continue;
        const PABFDMachineCache& mc = pabfd_mc[mid];
 
        if (strict) {
            // Strict SLA: minimise queue wait → pick least-loaded host.
            // Skip machines whose MIPS is below 50% of the cluster peak:
            // tasks run >2× slower there and will reliably miss their deadlines.
            if (pabfd_ref_mips > 0 && mc.mips_p0 * 2 < pabfd_ref_mips) continue;
            // Normalise by core count so larger/faster machines are preferred
            // when raw task counts are equal (handles heterogeneous clusters).
            int load = mc.num_cpus > 0
                       ? (int)mc.active_tasks * 1024 / (int)mc.num_cpus
                       : INT_MAX;
            if (rg && !mc.gpus) load += 10000;
            if (load < best_load) { best_load = load; best = mid; }
        } else {
            // Non-strict: minimise power increase (PABFD criterion).
            double dp = PABFD_DeltaPower(mc, 1);
            if (rg && !mc.gpus) dp += PABFD_GPU_PENALTY;
            if (dp < best_dp) { best_dp = dp; best = mid; }
        }
    }
    return best;
}
 
// Place a task that has already been matched to a host.
static void PABFD_PlaceTask(TaskId_t task, MachineId_t mid, vector<VMId_t>& vms) {
    VMId_t vm = PABFD_EnsureVM(mid, RequiredVMType(task),
                                 RequiredCPUType(task), vms);
    VM_AddTask(vm, task, PABFD_SLAPrio(RequiredSLA(task)));
    pabfd_task_to_vm[task]  = vm;
    unsigned tm = GetTaskMemory(task);
    pabfd_vc[vm].task_count++;
    pabfd_vc[vm].memory_used += tm;
    pabfd_mc[mid].active_tasks++;
    pabfd_mc[mid].memory_used += tm;
}
 
// Wake sleeping machines of the requested CPU type.
// For strict SLA wake ALL matching machines; for non-strict wake one.
static void PABFD_WakeMachines(const vector<MachineId_t>& machines,
                                CPUType_t rc, bool strict) {
    for (MachineId_t m : machines) {
        if (pabfd_transitioning.count(m)) continue;
        if (pabfd_mc[m].cpu == rc && pabfd_mc[m].s_state != S0) {
            Machine_SetState(m, S0);
            pabfd_transitioning.insert(m);
            if (!strict) return;   // one wake-up is enough for non-strict
        }
    }
}
 
// ---- PABFD consolidation ----------------------------------------------------
//
// Source hosts: utilisation < PABFD_U_LOW.
// Destination hosts: most-loaded host with headroom (util < PABFD_U_HIGH)
// whose power increase is minimised — classic BFD packing.
static void PABFD_Consolidate(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    // Skip consolidation entirely if any SLA0 or SLA1 task is queued.
    // Migrating VMs during a spike only delays placement and increases violations.
    for (auto& [cpu_type, bucket] : pabfd_pending) {
        for (TaskId_t t : bucket) {
            SLAType_t s = RequiredSLA(t);
            if (s == SLA0 || s == SLA1) return;
        }
    }

    // Count active (S0) machines per CPU type so we can enforce PABFD_MIN_ACTIVE.
    map<CPUType_t, unsigned> active_count;
    for (MachineId_t mid : machines) {
        if (!pabfd_transitioning.count(mid) && pabfd_mc[mid].s_state == S0)
            active_count[pabfd_mc[mid].cpu]++;
    }

    // Build list of S0 machines sorted by utilisation ascending.
    // For equal utilisation, sort by mips_p0 ascending so slow machines are
    // consolidated first, preserving fast machines as the active hot pool.
    vector<pair<double, MachineId_t>> by_util;
    for (MachineId_t mid : machines) {
        if (pabfd_transitioning.count(mid)) continue;
        const PABFDMachineCache& mc = pabfd_mc[mid];
        if (mc.s_state != S0) continue;
        double u = mc.num_cpus > 0
                   ? (double)mc.active_tasks / mc.num_cpus : 0.0;
        by_util.push_back({u, mid});
    }
    sort(by_util.begin(), by_util.end(), [&](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        // Tie: consolidate slow machines first, keeping fast ones active.
        return pabfd_mc[a.second].mips_p0 < pabfd_mc[b.second].mips_p0;
    });

    for (auto& [util, src] : by_util) {
        if (util >= PABFD_U_LOW) break;   // remaining hosts are not under-loaded
        if (pabfd_transitioning.count(src)) continue;

        const PABFDMachineCache& smc = pabfd_mc[src];
        // Never drop below the minimum active machine pool for this CPU type.
        if (active_count[smc.cpu] <= PABFD_MIN_ACTIVE) continue;
        bool all_placed = true;
 
        // Try to migrate every active VM off the source.
        for (VMId_t vm : pabfd_m2v[src]) {
            if (pabfd_migrating.count(vm)) continue;
            if (pabfd_vc[vm].task_count == 0) continue;  // idle VM, skip
 
            unsigned vm_mem = pabfd_vc[vm].memory_used;
 
            // Find destination: minimum ΔP, utilisation < PABFD_U_HIGH.
            MachineId_t dst    = MachineId_t(UINT_MAX);
            double      dst_dp = 1e18;
            for (MachineId_t cand : machines) {
                if (cand == src || pabfd_transitioning.count(cand)) continue;
                const PABFDMachineCache& cmc = pabfd_mc[cand];
                if (cmc.s_state != S0 || cmc.cpu != smc.cpu) continue;
                if (cmc.memory_size < cmc.memory_used + vm_mem) continue;
                if (cmc.num_cpus == 0) continue;
                double cu = (double)cmc.active_tasks / cmc.num_cpus;
                if (cu >= PABFD_U_HIGH) continue;
                double dp = PABFD_DeltaPower(cmc, pabfd_vc[vm].task_count);
                if (dp < dst_dp) { dst_dp = dp; dst = cand; }
            }
 
            if (dst == MachineId_t(UINT_MAX)) { all_placed = false; continue; }
 
            // Update caches, initiate migration.
            unsigned tc = pabfd_vc[vm].task_count;
            pabfd_mc[dst].memory_used  += vm_mem;
            pabfd_mc[dst].active_tasks += tc;
            pabfd_mc[src].memory_used  -= vm_mem;
            pabfd_mc[src].active_tasks -= tc;
            pabfd_vc[vm].machine_id     = dst;
 
            VM_Migrate(vm, dst);
            pabfd_migrating.insert(vm);
            pabfd_m2v[dst].push_back(vm);
            pabfd_m2v[src].erase(
                remove(pabfd_m2v[src].begin(), pabfd_m2v[src].end(), vm),
                pabfd_m2v[src].end());
        }
 
        if (!all_placed) continue;  // can't fully empty this host right now
 
        // Shut down idle VMs that remain on src, then send host to S3.
        vector<VMId_t> keep;
        for (VMId_t vm : pabfd_m2v[src]) {
            if (pabfd_migrating.count(vm)) { keep.push_back(vm); continue; }
            pabfd_mc[src].memory_used -= pabfd_vc[vm].memory_used;
            pabfd_mc[src].active_vms--;
            pabfd_vc.erase(vm);
            VM_Shutdown(vm);
            vms.erase(remove(vms.begin(), vms.end(), vm), vms.end());
        }
        pabfd_m2v[src] = keep;
 
        if (pabfd_mc[src].active_vms == 0) {
            // S3 (warm standby) wakes faster than S5 — better for SLA recovery.
            Machine_SetState(src, S3);
            pabfd_mc[src].s_state = S3;
            pabfd_transitioning.insert(src);
            // Update the active count so subsequent iterations respect the new state.
            active_count[smc.cpu]--;
        }
    }
}
 
// ---- PABFD pending retry (BFD order: largest memory first) ------------------
static void PABFD_DrainPending(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    for (auto& [cpu_type, bucket] : pabfd_pending) {
        if (bucket.empty()) continue;
 
        // Sort by SLA urgency first (SLA0 > SLA1 > SLA2 > SLA3), then by
        // earliest deadline within the same tier (EDF).  This ensures that
        // the most critical tasks are placed first rather than large-memory
        // SLA2/SLA3 tasks jumping the queue.
        stable_sort(bucket.begin(), bucket.end(), [](TaskId_t a, TaskId_t b) {
            SLAType_t sa = RequiredSLA(a), sb = RequiredSLA(b);
            int pa = (sa==SLA0)?0:(sa==SLA1)?1:(sa==SLA2)?2:3;
            int pb = (sb==SLA0)?0:(sb==SLA1)?1:(sb==SLA2)?2:3;
            if (pa != pb) return pa < pb;
            // Within the same SLA tier, prefer tasks closest to their deadline.
            TaskInfo_t ia = GetTaskInfo(a), ib = GetTaskInfo(b);
            return ia.target_completion < ib.target_completion;
        });
 
        deque<TaskId_t> retry;
        while (!bucket.empty()) {
            TaskId_t t = bucket.front();
            bucket.pop_front();
            MachineId_t mid = PABFD_BestFit(t, machines);
            if (mid != MachineId_t(UINT_MAX)) {
                PABFD_PlaceTask(t, mid, vms);
            } else {
                // Wake a machine for this CPU type and re-queue the task.
                SLAType_t sla = RequiredSLA(t);
                PABFD_WakeMachines(machines, cpu_type,
                                   sla == SLA0 || sla == SLA1);
                retry.push_back(t);
                // Push remaining tasks straight back without re-sorting.
                while (!bucket.empty()) {
                    retry.push_back(bucket.front());
                    bucket.pop_front();
                }
                break;
            }
        }
        bucket = retry;
    }
}
 
// ---- Public PABFD entry points ----------------------------------------------
 
void PABFD_Init(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    for (MachineId_t mid : machines) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        // Bring every machine to S0 at startup so we can immediately absorb
        // the initial task burst without queuing delays.
        Machine_SetState(mid, S0);
        unsigned mips = mi.performance.empty() ? 1u : mi.performance[0];
        if (mips > pabfd_ref_mips) pabfd_ref_mips = mips;
        pabfd_mc[mid] = {
            S0, mi.cpu, mi.num_cpus,
            mi.memory_size, mi.memory_used,
            mi.active_tasks, mi.active_vms,
            mi.gpus, mips
        };
        // Pre-create one starter VM so the first task placement is fast.
        VMType_t vt = (mi.cpu == POWER) ? AIX : LINUX;
        VMId_t vm = VM_Create(vt, mi.cpu);
        VM_Attach(vm, mid);
        vms.push_back(vm);
        pabfd_m2v[mid].push_back(vm);
        pabfd_vc[vm] = { vt, mi.cpu, mid, 0, VM_MEMORY_OVERHEAD };
        pabfd_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
        pabfd_mc[mid].active_vms++;
        // All cores at peak performance (P0) so task execution is fast.
        for (unsigned c = 0; c < mi.num_cpus; c++)
            Machine_SetCorePerformance(mid, c, P0);
    }
}
 
void PABFD_NewTask(Time_t now, TaskId_t task,
                   vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    MachineId_t mid = PABFD_BestFit(task, machines);
    if (mid == MachineId_t(UINT_MAX)) {
        // No suitable running host — queue and wake a sleeping one.
        CPUType_t rc  = RequiredCPUType(task);
        SLAType_t sla = RequiredSLA(task);
        PABFD_WakeMachines(machines, rc, sla == SLA0 || sla == SLA1);
        pabfd_pending[rc].push_back(task);
    } else {
        PABFD_PlaceTask(task, mid, vms);
    }
}
 
void PABFD_PeriodicCheck(Time_t now,
                          vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    bool has_pending = false;
    bool has_strict_pending = false;
    for (auto& [_, bkt] : pabfd_pending) {
        if (bkt.empty()) continue;
        has_pending = true;
        for (TaskId_t t : bkt) {
            SLAType_t s = RequiredSLA(t);
            if (s == SLA0 || s == SLA1) { has_strict_pending = true; break; }
        }
        if (has_strict_pending) break;
    }

    pabfd_check_counter++;
    // Don't consolidate when strict-SLA tasks are waiting — PABFD_Consolidate
    // has its own guard too, but skipping the call here avoids wasted work.
    bool do_consolidate = !has_strict_pending &&
        (pabfd_check_counter % PABFD_CONSOLIDATE_INTERVAL == 0);

    if (!pabfd_retry_pending && !has_pending && !do_consolidate) return;

    // Consolidation first: free up under-used machines before we retry.
    if (do_consolidate)
        PABFD_Consolidate(machines, vms);

    // Retry pending tasks (SLA-priority + EDF order).
    if (pabfd_retry_pending || has_pending) {
        pabfd_retry_pending = false;
        PABFD_DrainPending(machines, vms);
    }
}
 
void PABFD_TaskComplete(Time_t now, TaskId_t task,
                         vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    auto it = pabfd_task_to_vm.find(task);
    if (it == pabfd_task_to_vm.end()) return;
 
    VMId_t      vm  = it->second;
    MachineId_t mid = pabfd_vc[vm].machine_id;
    unsigned    tm  = GetTaskMemory(task);
 
    pabfd_vc[vm].task_count--;
    pabfd_vc[vm].memory_used  -= tm;
    pabfd_mc[mid].active_tasks--;
    pabfd_mc[mid].memory_used  -= tm;
    pabfd_task_to_vm.erase(it);
 
    pabfd_retry_pending = true;
}
 
void PABFD_MigrationComplete(VMId_t vm,
                              vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    pabfd_migrating.erase(vm);
    // Update the destination machine's VM list state (already updated in cache).
    pabfd_retry_pending = true;
}
 
void PABFD_Shutdown(vector<VMId_t>& vms) {
    for (VMId_t vm : vms) {
        if (pabfd_vc.count(vm) && pabfd_vc[vm].task_count == 0)
            VM_Shutdown(vm);
    }
}

// ============================================================================
// 2. SCHEDULER INTERFACE
// ============================================================================

static Scheduler theScheduler;

void Scheduler::Init() {
    unsigned total = Machine_GetTotal();
    for (unsigned i = 0; i < total; i++) machines.push_back(MachineId_t(i));
    PABFD_Init(machines, vms);
}

void Scheduler::NewTask(Time_t now, TaskId_t task_id) {
    PABFD_NewTask(now, task_id, machines, vms);
}

void Scheduler::PeriodicCheck(Time_t now) {
    PABFD_PeriodicCheck(now, machines, vms);
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    PABFD_TaskComplete(now, task_id, machines, vms);
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    PABFD_MigrationComplete(vm_id, machines, vms);
}

void Scheduler::Shutdown(Time_t time) {
    PABFD_Shutdown(vms);
}

// ============================================================================
// 3. PUBLIC INTERFACE (Wrappers for theScheduler)
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
    // Wake ALL sleeping machines of the matching CPU type immediately.
    for (auto& [mid, mc] : pabfd_mc) {
        if (pabfd_transitioning.count(mid)) continue;
        if (mc.cpu == rc && mc.s_state != S0) {
            Machine_SetState(mid, S0);
            pabfd_transitioning.insert(mid);
        }
    }
    pabfd_retry_pending = true;
}

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    pabfd_transitioning.erase(machine_id);
    MachineInfo_t mi = Machine_GetInfo(machine_id);
    pabfd_mc[machine_id].s_state = mi.s_state;
    if (mi.s_state == S0) {
        for (unsigned c = 0; c < mi.num_cpus; c++)
            Machine_SetCorePerformance(machine_id, c, P0);
        pabfd_retry_pending = true;
    }
}

void SimulationComplete(Time_t time) {
    // This function is called before the simulation terminates Add whatever you feel like.
    cout << "SLA violation report" << endl;
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    cout << "Simulation run finished in " << double(time)/1000000 << " seconds" << endl;
    SimOutput("SimulationComplete(): Simulation finished at time " + to_string(time), 4);

    theScheduler.Shutdown(time);
}
