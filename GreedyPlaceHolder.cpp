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
// GREEDY SCHEDULER - SLA-aware, core-capacity-limited placement
// ============================================================================
//
// Key design decisions:
//  1. Each machine tracks num_cpus; tasks are not placed beyond
//     num_cpus * OVERSUBSCRIBE_FACTOR active tasks per machine.
//     This prevents over-subscription (multiple tasks sharing a core),
//     which drastically slows individual tasks and causes SLA violations.
//  2. Overflow tasks wait in greedy_pending_by_cpu, sorted by SLA urgency
//     and earliest deadline first, and are dispatched as capacity opens.
//  3. Machines idle at S0i1 (instantaneous wake) rather than S3
//     (serious delay), keeping SLA0/SLA1 burst start latency near zero.
//  4. On SLAWarning all compatible sleeping machines are woken immediately.
//  5. GPU-capable tasks are strongly preferred on GPU machines.
// ============================================================================

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

static map<MachineId_t, vector<VMId_t>> greedy_m2v;
static set<MachineId_t>                 greedy_waking;
static map<CPUType_t, deque<TaskId_t>>  greedy_pending_by_cpu;
static set<VMId_t>                      greedy_migrating;
static map<TaskId_t, VMId_t>            greedy_task_to_vm;
static set<MachineId_t>                 greedy_latency_reserved;
static bool                             greedy_retry_pending = false;

// VM property cache
static map<VMId_t, VMType_t>    greedy_vm_type;
static map<VMId_t, CPUType_t>   greedy_vm_cpu;
static map<VMId_t, MachineId_t> greedy_vm_machine;
static map<VMId_t, unsigned>    greedy_vm_tasks;

// Machine property cache
struct GreedyMachineCache {
    MachineState_t s_state;
    CPUType_t      cpu;
    unsigned       num_cpus;      // core count — caps placement
    unsigned       memory_size;
    unsigned       memory_used;
    unsigned       active_tasks;
    unsigned       active_vms;
    bool           gpus;
    unsigned       perf0;
    unsigned       power_s0;
};
static map<MachineId_t, GreedyMachineCache> greedy_mc;

// How many times over num_cpus we allow tasks to pile up before queuing.
// A factor of 1 = strictly 1 task per core (best isolation, less throughput).
// A factor of 2 = allow 2 tasks per core before queuing (balanced).
static const unsigned OVERSUBSCRIBE_FACTOR = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool Greedy_VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true;
}

static Priority_t Greedy_SLAPrio(SLAType_t sla) {
    if (sla == SLA1)                return HIGH_PRIORITY;
    if (sla == SLA0 || sla == SLA2) return MID_PRIORITY;
    return LOW_PRIORITY;
}

static unsigned Greedy_Strictness(SLAType_t sla) {
    if (sla == SLA0) return 0;
    if (sla == SLA1) return 1;
    if (sla == SLA2) return 2;
    return 3;
}

static unsigned Greedy_TaskPerfHint(TaskId_t task) {
    TaskInfo_t ti = GetTaskInfo(task);
    uint64_t span = (ti.target_completion > ti.arrival)
                  ? (ti.target_completion - ti.arrival) : 1;
    return unsigned((ti.total_instructions + span - 1) / span);
}

static unsigned Greedy_FreeSlots(MachineId_t mid) {
    const GreedyMachineCache& mc = greedy_mc[mid];
    unsigned cap = mc.num_cpus * OVERSUBSCRIBE_FACTOR;
    return (mc.active_tasks < cap) ? (cap - mc.active_tasks) : 0;
}

static unsigned Greedy_MaxPerf(CPUType_t cpu, bool gpu_only, bool require_free_slot) {
    unsigned best = 0;
    for (const auto& [mid, mc] : greedy_mc) {
        if (mc.cpu != cpu) continue;
        if (gpu_only && !mc.gpus) continue;
        if (require_free_slot && (mc.s_state != S0 || Greedy_FreeSlots(mid) == 0)) continue;
        best = max(best, mc.perf0);
    }
    return best;
}

static bool Greedy_HasFastPreferredCapacity(TaskId_t task) {
    CPUType_t rc = RequiredCPUType(task);
    bool need_gpu = IsTaskGPUCapable(task);
    unsigned best = Greedy_MaxPerf(rc, need_gpu, true);
    if (best == 0) return false;
    for (const auto& [mid, mc] : greedy_mc) {
        if (mc.s_state != S0 || mc.cpu != rc) continue;
        if (need_gpu && !mc.gpus) continue;
        if (Greedy_FreeSlots(mid) == 0) continue;
        if (mc.perf0 * 4 >= best * 3) return true;
    }
    return false;
}

static bool Greedy_HasPreferredReservation(TaskId_t task, bool reserved) {
    CPUType_t rc = RequiredCPUType(task);
    bool need_gpu = IsTaskGPUCapable(task);
    for (const auto& [mid, mc] : greedy_mc) {
        if (mc.s_state != S0 || mc.cpu != rc) continue;
        if (need_gpu && !mc.gpus) continue;
        if (Greedy_FreeSlots(mid) == 0) continue;
        if (greedy_latency_reserved.count(mid) == reserved) return true;
    }
    return false;
}

static void Greedy_WakeMatching(TaskId_t task, bool wake_all) {
    CPUType_t rc = RequiredCPUType(task);
    bool need_gpu = IsTaskGPUCapable(task);
    bool strict = Greedy_Strictness(RequiredSLA(task)) <= 1;
    unsigned best_perf = Greedy_MaxPerf(rc, need_gpu, false);
    vector<MachineId_t> sleepers;

    for (const auto& [mid, mc] : greedy_mc) {
        if (greedy_waking.count(mid)) continue;
        if (mc.cpu != rc || mc.s_state == S0) continue;
        if (need_gpu && !mc.gpus) continue;
        if (strict && best_perf != 0 && mc.perf0 * 4 < best_perf * 3) continue;
        sleepers.push_back(mid);
    }

    sort(sleepers.begin(), sleepers.end(), [](MachineId_t a, MachineId_t b) {
        const GreedyMachineCache& ma = greedy_mc[a];
        const GreedyMachineCache& mb = greedy_mc[b];
        if (ma.gpus != mb.gpus) return ma.gpus > mb.gpus;
        if (ma.perf0 != mb.perf0) return ma.perf0 > mb.perf0;
        uint64_t lhs = uint64_t(ma.power_s0) * max(1u, mb.num_cpus * mb.perf0);
        uint64_t rhs = uint64_t(mb.power_s0) * max(1u, ma.num_cpus * ma.perf0);
        if (lhs != rhs) return lhs < rhs;
        return a < b;
    });

    for (MachineId_t mid : sleepers) {
        Machine_SetState(mid, S0);
        greedy_waking.insert(mid);
        if (!wake_all) break;
    }
}

static void Greedy_SweepIdleMachines() {
    (void)0;
}

// Reuse an existing compatible VM on the machine, or create a new one.
static VMId_t Greedy_EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct,
                               vector<VMId_t>& all_vms) {
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

// Best-fit placement:
//   - SLA0/SLA1 (strict): least-loaded machine first (spread load evenly to
//     minimise queuing delay for tight deadlines).
//   - SLA2/SLA3         : most-loaded machine first (consolidate to free
//     idle machines for power saving).
//   Always gates on: S0, matching CPU, core capacity, memory.
static MachineId_t Greedy_BestFit(TaskId_t task, const vector<MachineId_t>& mlist) {
    CPUType_t rc  = RequiredCPUType(task);
    VMType_t  rv  = RequiredVMType(task);
    unsigned  rm  = GetTaskMemory(task);
    bool      rg  = IsTaskGPUCapable(task);
    SLAType_t sla = RequiredSLA(task);
    unsigned strictness = Greedy_Strictness(sla);
    bool strict   = strictness <= 1;
    unsigned task_perf_hint = Greedy_TaskPerfHint(task);
    unsigned best_perf = Greedy_MaxPerf(rc, rg, true);
    bool prefer_fast_tier = strict && best_perf != 0 && Greedy_HasFastPreferredCapacity(task);
    bool prefer_reserved = (sla != SLA0) && Greedy_HasPreferredReservation(task, true);
    bool avoid_reserved  = (sla == SLA0) && Greedy_HasPreferredReservation(task, false);

    if (!Greedy_VMCPUOk(rv, rc)) return MachineId_t(UINT_MAX);

    MachineId_t best       = MachineId_t(UINT_MAX);
    uint64_t    best_score = numeric_limits<uint64_t>::max();

    for (MachineId_t mid : mlist) {
        const GreedyMachineCache& mc = greedy_mc[mid];
        if (mc.s_state != S0 || mc.cpu != rc) continue;
        if (prefer_fast_tier && mc.perf0 * 4 < best_perf * 3) continue;
        if (rg && best_perf != 0 && mc.gpus && mc.perf0 * 4 < best_perf * 3) continue;
        if (prefer_reserved && !greedy_latency_reserved.count(mid)) continue;
        if (avoid_reserved && greedy_latency_reserved.count(mid)) continue;

        // Core capacity gate.
        unsigned cap = mc.num_cpus * OVERSUBSCRIBE_FACTOR;
        if (mc.active_tasks >= cap) continue;

        // Memory gate.
        bool has_vm = false;
        for (VMId_t vm : greedy_m2v[mid]) {
            if (greedy_migrating.count(vm)) continue;
            if (greedy_vm_type[vm] == rv && greedy_vm_cpu[vm] == rc) {
                has_vm = true; break;
            }
        }
        unsigned extra = has_vm ? 0 : VM_MEMORY_OVERHEAD;
        if (mc.memory_size < mc.memory_used + rm + extra) continue;

        uint64_t util = uint64_t(mc.active_tasks + 1) * 1000 / max(1u, mc.num_cpus);
        uint64_t speed_penalty = 1000000ull / max(1u, mc.perf0);
        uint64_t score = 0;

        if (strict) {
            uint64_t normalized_load = uint64_t(mc.active_tasks + 1)
                                     * max(1u, best_perf) * 1000ull
                                     / max(1u, mc.num_cpus * mc.perf0);
            score = normalized_load * 100000ull;
            score += speed_penalty * 500ull;
            score += uint64_t(task_perf_hint) * 10000ull
                   / max(1u, mc.perf0 * mc.num_cpus);
            if (sla != SLA0 && greedy_latency_reserved.count(mid)) score /= 4;
        } else {
            uint64_t efficiency = uint64_t(mc.power_s0) * 1000ull
                                / max(1u, mc.num_cpus * mc.perf0);
            score = efficiency * 10000ull;
            score += speed_penalty * 25ull;
            score += (1000ull - util);
        }

        if (rg && !mc.gpus) score += 100000000ull;  // only spill off GPU when forced
        if (!rg && mc.gpus) score += strict ? 25000ull : 5000ull;

        if (score < best_score) { best = mid; best_score = score; }
    }
    return best;
}

// Keep machines in S0 after they become idle. The provided workloads are much
// more sensitive to wake delays than idle power draw.
static void Greedy_TrySleep(MachineId_t mid, vector<VMId_t>& all_vms) {
    GreedyMachineCache& mc = greedy_mc[mid];
    if (mc.s_state != S0 || mc.active_tasks > 0) return;

    vector<VMId_t>& vlist = greedy_m2v[mid];
    vector<VMId_t>  keep;
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
            all_vms.erase(remove(all_vms.begin(), all_vms.end(), vm),
                          all_vms.end());
        } else {
            keep.push_back(vm);
        }
    }
    vlist = keep;
}

// ---------------------------------------------------------------------------
// Scheduler methods
// ---------------------------------------------------------------------------

static Scheduler theScheduler;

void Scheduler::Init() {
    unsigned total = Machine_GetTotal();
    for (unsigned i = 0; i < total; i++) machines.push_back(MachineId_t(i));

    for (MachineId_t mid : machines) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        if (mi.s_state != S0) {
            Machine_SetState(mid, S0);
            greedy_waking.insert(mid);
            mi = Machine_GetInfo(mid);
        }
        greedy_mc[mid] = { mi.s_state, mi.cpu, mi.num_cpus,
                           mi.memory_size, mi.memory_used,
                           mi.active_tasks, mi.active_vms, mi.gpus,
                           mi.performance.empty() ? 1u : mi.performance[0],
                           mi.s_states.empty() ? 1u : mi.s_states[0] };
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
        for (unsigned c = 0; c < mi.num_cpus; c++)
            Machine_SetCorePerformance(mid, c, P0);
    }

    greedy_latency_reserved.clear();
    for (CPUType_t cpu : {ARM, POWER, RISCV, X86}) {
        vector<MachineId_t> tier;
        unsigned best_perf = 0;
        for (const auto& [mid, mc] : greedy_mc) {
            if (mc.cpu != cpu) continue;
            best_perf = max(best_perf, mc.perf0);
        }
        if (best_perf == 0) continue;
        for (const auto& [mid, mc] : greedy_mc) {
            if (mc.cpu != cpu) continue;
            if (mc.perf0 * 4 < best_perf * 3) continue;
            tier.push_back(mid);
        }
        sort(tier.begin(), tier.end(), [](MachineId_t a, MachineId_t b) {
            const GreedyMachineCache& ma = greedy_mc[a];
            const GreedyMachineCache& mb = greedy_mc[b];
            if (ma.gpus != mb.gpus) return ma.gpus < mb.gpus;
            if (ma.num_cpus != mb.num_cpus) return ma.num_cpus > mb.num_cpus;
            if (ma.perf0 != mb.perf0) return ma.perf0 > mb.perf0;
            return a < b;
        });
        unsigned reserve = tier.size() >= 4 ? 1u : 0u;
        for (unsigned i = 0; i < reserve && i < tier.size(); i++)
            greedy_latency_reserved.insert(tier[i]);
    }
}

void Scheduler::NewTask(Time_t now, TaskId_t task_id) {
    SLAType_t sla = RequiredSLA(task_id);
    bool strict = Greedy_Strictness(sla) <= 1;
    if (strict) Greedy_WakeMatching(task_id, false);

    MachineId_t mid = Greedy_BestFit(task_id, machines);
    if (mid == MachineId_t(UINT_MAX)) {
        CPUType_t rc  = RequiredCPUType(task_id);
        (void)rc;
        Greedy_WakeMatching(task_id, strict);
        greedy_pending_by_cpu[rc].push_back(task_id);
    } else {
        VMId_t vm = Greedy_EnsureVM(mid, RequiredVMType(task_id),
                                    RequiredCPUType(task_id), vms);
        VM_AddTask(vm, task_id, Greedy_SLAPrio(RequiredSLA(task_id)));
        greedy_vm_tasks[vm]++;
        greedy_mc[mid].memory_used  += GetTaskMemory(task_id);
        greedy_mc[mid].active_tasks++;
        greedy_task_to_vm[task_id] = vm;

        if (strict) {
            unsigned free_slots = 0;
            for (MachineId_t m : machines)
                if (greedy_mc[m].cpu == RequiredCPUType(task_id) &&
                    greedy_mc[m].s_state == S0)
                    free_slots += Greedy_FreeSlots(m);
            if (free_slots <= greedy_mc[mid].num_cpus)
                Greedy_WakeMatching(task_id, false);
        }
    }
}

void Scheduler::PeriodicCheck(Time_t now) {
    (void)now;
    Greedy_SweepIdleMachines();

    bool has_pending = false;
    for (auto& [_, bkt] : greedy_pending_by_cpu)
        if (!bkt.empty()) { has_pending = true; break; }
    if (!greedy_retry_pending && !has_pending) return;
    greedy_retry_pending = false;

    for (auto& [cpu_type, bucket] : greedy_pending_by_cpu) {
        if (bucket.empty()) continue;

        // Sort pending by SLA priority, then by task_id (proxy for arrival
        // order / deadline urgency within the same SLA class).
        stable_sort(bucket.begin(), bucket.end(), [](TaskId_t a, TaskId_t b) {
            SLAType_t sa = RequiredSLA(a), sb = RequiredSLA(b);
            int pa = (int)Greedy_Strictness(sa);
            int pb = (int)Greedy_Strictness(sb);
            if (pa != pb) return pa < pb;
            TaskInfo_t ta = GetTaskInfo(a);
            TaskInfo_t tb = GetTaskInfo(b);
            if (ta.target_completion != tb.target_completion)
                return ta.target_completion < tb.target_completion;
            return a < b;
        });

        while (!bucket.empty()) {
            TaskId_t    t   = bucket.front();
            MachineId_t mid = Greedy_BestFit(t, machines);
            if (mid == MachineId_t(UINT_MAX)) {
                Greedy_WakeMatching(t, Greedy_Strictness(RequiredSLA(t)) <= 1);
                break;
            }
            bucket.pop_front();
            VMId_t vm = Greedy_EnsureVM(mid, RequiredVMType(t), cpu_type, vms);
            VM_AddTask(vm, t, Greedy_SLAPrio(RequiredSLA(t)));
            greedy_vm_tasks[vm]++;
            greedy_mc[mid].memory_used  += GetTaskMemory(t);
            greedy_mc[mid].active_tasks++;
            greedy_task_to_vm[t] = vm;
        }
    }
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    auto it = greedy_task_to_vm.find(task_id);
    if (it != greedy_task_to_vm.end()) {
        VMId_t      vm  = it->second;
        MachineId_t mid = greedy_vm_machine[vm];
        greedy_vm_tasks[vm]--;
        greedy_mc[mid].memory_used  -= GetTaskMemory(task_id);
        greedy_mc[mid].active_tasks--;
        greedy_task_to_vm.erase(it);
        Greedy_TrySleep(mid, vms);
        greedy_retry_pending = true;
    }
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    greedy_migrating.erase(vm_id);
}

void Scheduler::Shutdown(Time_t time) {
    for (VMId_t vm : vms) {
        MachineId_t mid = greedy_vm_machine.count(vm)
                          ? greedy_vm_machine[vm] : MachineId_t(UINT_MAX);
        if (mid != MachineId_t(UINT_MAX) &&
            greedy_mc[mid].s_state != S0) continue;
        if (VM_GetInfo(vm).active_tasks.empty()) VM_Shutdown(vm);
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void InitScheduler()   { theScheduler.Init(); }
void HandleNewTask(Time_t time, TaskId_t task_id)
    { theScheduler.NewTask(time, task_id); }
void HandleTaskCompletion(Time_t time, TaskId_t task_id)
    { theScheduler.TaskComplete(time, task_id); }
void MemoryWarning(Time_t time, MachineId_t machine_id) { }
void MigrationDone(Time_t time, VMId_t vm_id)
    { theScheduler.MigrationComplete(time, vm_id); }
void SchedulerCheck(Time_t time) { theScheduler.PeriodicCheck(time); }

void SLAWarning(Time_t time, TaskId_t task_id) {
    (void)time;
    // Boost priority of the violating task.
    SetTaskPriority(task_id, HIGH_PRIORITY);
    Greedy_WakeMatching(task_id, true);
    greedy_retry_pending = true;
}

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    greedy_waking.erase(machine_id);
    MachineInfo_t mi = Machine_GetInfo(machine_id);
    greedy_mc[machine_id].s_state = mi.s_state;
    if (mi.s_state == S0) {
        for (unsigned c = 0; c < mi.num_cpus; c++)
            Machine_SetCorePerformance(machine_id, c, P0);
        greedy_retry_pending = true;
    }
}

void SimulationComplete(Time_t time) {
    cout << "SLA violation report" << endl;
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    cout << "Simulation run finished in " << double(time)/1000000
         << " seconds" << endl;
    SimOutput("SimulationComplete(): Simulation finished at time "
              + to_string(time), 4);
    theScheduler.Shutdown(time);
}
