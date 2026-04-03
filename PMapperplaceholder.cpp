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

static map<MachineId_t, vector<VMId_t>> sched_m2v;
static map<TaskId_t, VMId_t>            sched_task_to_vm;
static map<CPUType_t, deque<TaskId_t>>  sched_pending_by_cpu;
static set<MachineId_t>                 sched_waking;
static set<VMId_t>                      sched_migrating;
static map<VMId_t, VMType_t>            sched_vm_type;
static map<VMId_t, CPUType_t>           sched_vm_cpu;
static map<VMId_t, MachineId_t>         sched_vm_machine;
static map<VMId_t, unsigned>            sched_vm_tasks;
static map<CPUType_t, unsigned>         sched_best_perf;
static map<CPUType_t, unsigned>         sched_best_non_gpu_perf;
static bool                             sched_retry_pending = false;

struct MachineCache {
    MachineState_t s_state;
    CPUType_t      cpu;
    unsigned       num_cpus;
    unsigned       perf_p0;
    unsigned       memory_size;
    unsigned       memory_used;
    unsigned       active_tasks;
    unsigned       active_vms;
    bool           gpus;
};
static map<MachineId_t, MachineCache> sched_mc;

static bool VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true;
}

static bool IsStrict(SLAType_t sla) {
    return sla == SLA0 || sla == SLA1;
}

static int SLARank(SLAType_t sla) {
    if (sla == SLA0) return 0;
    if (sla == SLA1) return 1;
    if (sla == SLA2) return 2;
    return 3;
}

static Priority_t SLAPrio(SLAType_t sla) {
    if (sla == SLA0) return HIGH_PRIORITY;
    if (sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2) return LOW_PRIORITY;
    return LOW_PRIORITY;
}

static unsigned PreferredBaseline(TaskId_t task) {
    CPUType_t cpu = RequiredCPUType(task);
    if (!IsTaskGPUCapable(task)) {
        auto it = sched_best_non_gpu_perf.find(cpu);
        if (it != sched_best_non_gpu_perf.end() && it->second > 0) return it->second;
    }
    auto it = sched_best_perf.find(cpu);
    return (it == sched_best_perf.end()) ? 0u : it->second;
}

static bool HasMatchingVM(MachineId_t mid, VMType_t vt, CPUType_t ct) {
    for (VMId_t vm : sched_m2v[mid]) {
        if (sched_migrating.count(vm)) continue;
        if (sched_vm_type[vm] == vt && sched_vm_cpu[vm] == ct) return true;
    }
    return false;
}

static bool FastEnough(TaskId_t task, const MachineCache& mc) {
    if (!IsStrict(RequiredSLA(task))) return true;
    unsigned baseline = PreferredBaseline(task);
    if (baseline == 0) return true;
    unsigned threshold_pct = (RequiredSLA(task) == SLA0) ? 80u : 40u;
    return mc.perf_p0 * 100 >= baseline * threshold_pct;
}

static bool CanHost(TaskId_t task, MachineId_t mid, bool require_fast) {
    const MachineCache& mc = sched_mc[mid];
    const CPUType_t rc = RequiredCPUType(task);
    const VMType_t  rv = RequiredVMType(task);

    if (mc.s_state != S0 || mc.cpu != rc) return false;
    if (!VMCPUOk(rv, rc)) return false;
    if (IsTaskGPUCapable(task) && !mc.gpus) return false;
    if (require_fast && !FastEnough(task, mc)) return false;

    unsigned extra = HasMatchingVM(mid, rv, rc) ? 0 : VM_MEMORY_OVERHEAD;
    if (IsStrict(RequiredSLA(task)) && HasMatchingVM(mid, rv, rc) && mc.active_vms < mc.num_cpus) {
        bool has_idle_vm = false;
        for (VMId_t vm : sched_m2v[mid]) {
            if (sched_migrating.count(vm)) continue;
            if (sched_vm_type[vm] == rv && sched_vm_cpu[vm] == rc && sched_vm_tasks[vm] == 0) {
                has_idle_vm = true;
                break;
            }
        }
        if (!has_idle_vm) extra = VM_MEMORY_OVERHEAD;
    }
    return mc.memory_size >= mc.memory_used + GetTaskMemory(task) + extra;
}

static double PlacementScore(TaskId_t task, const MachineCache& mc) {
    unsigned baseline = PreferredBaseline(task);
    if (baseline == 0) baseline = max(1u, mc.perf_p0);

    double effective_slots = (double)mc.num_cpus * (double)mc.perf_p0 / (double)baseline;
    if (effective_slots < 1.0) effective_slots = 1.0;
    double load = (double)mc.active_tasks / effective_slots;

    const bool strict = IsStrict(RequiredSLA(task));
    const bool gpu_required = IsTaskGPUCapable(task);

    if (strict) {
        TaskInfo_t ti = GetTaskInfo(task);
        double service_time = (mc.perf_p0 > 0) ? (double)ti.remaining_instructions / (double)mc.perf_p0 : 1e100;
        double queue_factor = 1.0 + ((double)mc.active_tasks / max(1u, mc.num_cpus));
        double predicted_finish = (double)Now() + service_time * queue_factor;
        double lateness = max(0.0, predicted_finish - (double)ti.target_completion);

        if (RequiredSLA(task) == SLA1) {
            double desired_perf = max(1.0, (double)baseline * 0.40);
            double score = lateness * 1000.0 + predicted_finish / 1000.0;
            score += load * 100000.0;
            score += fabs((double)mc.perf_p0 - desired_perf) * 100.0;
            if (!gpu_required && mc.gpus) score += 25000.0;
            return score;
        }

        double score = lateness * 1000.0 + predicted_finish / 1000.0;
        score += load * 100000.0;
        score += (double)(baseline - min(baseline, mc.perf_p0)) * 100.0;
        if (!gpu_required && mc.gpus) score += 25000.0;
        return score;
    }

    double score = 0.0;
    if (!gpu_required && mc.gpus) score += 50000.0;
    score += (double)mc.perf_p0 * 10.0;
    score -= load * 10000.0;
    return score;
}

static MachineId_t FindPlacement(TaskId_t task, const vector<MachineId_t>& machines) {
    const bool strict = IsStrict(RequiredSLA(task));

    for (int pass = 0; pass < (strict ? 2 : 1); pass++) {
        bool require_fast = (pass == 0 && strict);
        MachineId_t best = MachineId_t(UINT_MAX);
        double best_score = 1e100;

        for (MachineId_t mid : machines) {
            if (!CanHost(task, mid, require_fast)) continue;
            double score = PlacementScore(task, sched_mc[mid]);
            if (score < best_score) {
                best_score = score;
                best = mid;
            }
        }

        if (best != MachineId_t(UINT_MAX)) return best;
    }

    return MachineId_t(UINT_MAX);
}

static VMId_t EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct, vector<VMId_t>& all_vms, TaskId_t task) {
    VMId_t best = VMId_t(UINT_MAX);
    unsigned best_load = UINT_MAX;
    for (VMId_t vm : sched_m2v[mid]) {
        if (sched_migrating.count(vm)) continue;
        if (sched_vm_type[vm] == vt && sched_vm_cpu[vm] == ct && sched_vm_tasks[vm] < best_load) {
            best = vm;
            best_load = sched_vm_tasks[vm];
        }
    }

    if (best != VMId_t(UINT_MAX)) {
        const MachineCache& mc = sched_mc[mid];
        bool strict = IsStrict(RequiredSLA(task));
        bool can_split =
            strict &&
            best_load > 0 &&
            mc.active_vms < mc.num_cpus &&
            mc.memory_size >= mc.memory_used + VM_MEMORY_OVERHEAD;
        if (!can_split) return best;
    }

    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    sched_m2v[mid].push_back(vm);
    sched_vm_type[vm] = vt;
    sched_vm_cpu[vm] = ct;
    sched_vm_machine[vm] = mid;
    sched_vm_tasks[vm] = 0;
    sched_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
    sched_mc[mid].active_vms++;
    return vm;
}

static void AddTaskToMachine(TaskId_t task, MachineId_t mid, vector<VMId_t>& all_vms) {
    VMId_t vm = EnsureVM(mid, RequiredVMType(task), RequiredCPUType(task), all_vms, task);
    VM_AddTask(vm, task, SLAPrio(RequiredSLA(task)));
    sched_vm_tasks[vm]++;
    sched_mc[mid].memory_used += GetTaskMemory(task);
    sched_mc[mid].active_tasks++;
    sched_task_to_vm[task] = vm;
}

static void WakeEligibleMachines(TaskId_t task, const vector<MachineId_t>& machines) {
    CPUType_t rc = RequiredCPUType(task);
    bool gpu_required = IsTaskGPUCapable(task);
    for (MachineId_t mid : machines) {
        if (sched_waking.count(mid)) continue;
        MachineCache& mc = sched_mc[mid];
        if (mc.cpu != rc || mc.s_state == S0) continue;
        if (gpu_required && !mc.gpus) continue;
        Machine_SetState(mid, S0);
        sched_waking.insert(mid);
    }
}

static Scheduler theScheduler;

void Scheduler::Init() {
    sched_m2v.clear();
    sched_task_to_vm.clear();
    sched_pending_by_cpu.clear();
    sched_waking.clear();
    sched_migrating.clear();
    sched_vm_type.clear();
    sched_vm_cpu.clear();
    sched_vm_machine.clear();
    sched_vm_tasks.clear();
    sched_best_perf.clear();
    sched_best_non_gpu_perf.clear();
    sched_mc.clear();
    sched_retry_pending = false;
    vms.clear();
    machines.clear();

    unsigned total = Machine_GetTotal();
    for (unsigned i = 0; i < total; i++) machines.push_back(MachineId_t(i));

    for (MachineId_t mid : machines) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        unsigned perf_p0 = mi.performance.empty() ? 0 : mi.performance[0];
        sched_best_perf[mi.cpu] = max(sched_best_perf[mi.cpu], perf_p0);
        if (!mi.gpus) sched_best_non_gpu_perf[mi.cpu] = max(sched_best_non_gpu_perf[mi.cpu], perf_p0);
    }

    for (MachineId_t mid : machines) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        unsigned perf_p0 = mi.performance.empty() ? 0 : mi.performance[0];

        Machine_SetState(mid, S0);
        sched_mc[mid] = { S0, mi.cpu, mi.num_cpus, perf_p0, mi.memory_size,
                          mi.memory_used, mi.active_tasks, mi.active_vms, mi.gpus };

        VMType_t vt = (mi.cpu == POWER) ? AIX : LINUX;
        VMId_t vm = VM_Create(vt, mi.cpu);
        VM_Attach(vm, mid);
        vms.push_back(vm);
        sched_m2v[mid].push_back(vm);
        sched_vm_type[vm] = vt;
        sched_vm_cpu[vm] = mi.cpu;
        sched_vm_machine[vm] = mid;
        sched_vm_tasks[vm] = 0;
        sched_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
        sched_mc[mid].active_vms++;

        for (unsigned c = 0; c < mi.num_cpus; c++) Machine_SetCorePerformance(mid, c, P0);
    }
}

void Scheduler::NewTask(Time_t now, TaskId_t task_id) {
    MachineId_t mid = FindPlacement(task_id, machines);
    if (mid == MachineId_t(UINT_MAX)) {
        WakeEligibleMachines(task_id, machines);
        sched_pending_by_cpu[RequiredCPUType(task_id)].push_back(task_id);
        sched_retry_pending = true;
        return;
    }

    AddTaskToMachine(task_id, mid, vms);
}

void Scheduler::PeriodicCheck(Time_t now) {
    bool has_pending = false;
    for (auto& [_, bucket] : sched_pending_by_cpu) {
        if (!bucket.empty()) {
            has_pending = true;
            break;
        }
    }

    if (!sched_retry_pending && !has_pending) return;
    sched_retry_pending = false;

    for (auto& [cpu_type, bucket] : sched_pending_by_cpu) {
        if (bucket.empty()) continue;

        stable_sort(bucket.begin(), bucket.end(), [](TaskId_t a, TaskId_t b) {
            int ra = SLARank(RequiredSLA(a));
            int rb = SLARank(RequiredSLA(b));
            if (ra != rb) return ra < rb;
            if (IsTaskGPUCapable(a) != IsTaskGPUCapable(b)) return IsTaskGPUCapable(a) > IsTaskGPUCapable(b);
            return GetTaskMemory(a) > GetTaskMemory(b);
        });

        while (!bucket.empty()) {
            TaskId_t task = bucket.front();
            MachineId_t mid = FindPlacement(task, machines);
            if (mid == MachineId_t(UINT_MAX)) {
                WakeEligibleMachines(task, machines);
                break;
            }
            bucket.pop_front();
            AddTaskToMachine(task, mid, vms);
        }
    }
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    auto it = sched_task_to_vm.find(task_id);
    if (it == sched_task_to_vm.end()) return;

    VMId_t vm = it->second;
    MachineId_t mid = sched_vm_machine[vm];
    if (sched_vm_tasks[vm] > 0) sched_vm_tasks[vm]--;
    sched_mc[mid].memory_used -= GetTaskMemory(task_id);
    if (sched_mc[mid].active_tasks > 0) sched_mc[mid].active_tasks--;
    sched_task_to_vm.erase(it);
    sched_retry_pending = true;
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    sched_migrating.erase(vm_id);
}

void Scheduler::Shutdown(Time_t time) {
    for (VMId_t vm : vms) {
        if (VM_GetInfo(vm).active_tasks.empty()) VM_Shutdown(vm);
    }
}

void InitScheduler() { theScheduler.Init(); }
void HandleNewTask(Time_t time, TaskId_t task_id) { theScheduler.NewTask(time, task_id); }
void HandleTaskCompletion(Time_t time, TaskId_t task_id) { theScheduler.TaskComplete(time, task_id); }
void MemoryWarning(Time_t time, MachineId_t machine_id) { }
void MigrationDone(Time_t time, VMId_t vm_id) { theScheduler.MigrationComplete(time, vm_id); }
void SchedulerCheck(Time_t time) { theScheduler.PeriodicCheck(time); }

void SLAWarning(Time_t time, TaskId_t task_id) {
    SetTaskPriority(task_id, HIGH_PRIORITY);
    vector<MachineId_t> machines;
    unsigned total = Machine_GetTotal();
    machines.reserve(total);
    for (unsigned i = 0; i < total; i++) machines.push_back(MachineId_t(i));
    WakeEligibleMachines(task_id, machines);
    sched_retry_pending = true;
}

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    sched_waking.erase(machine_id);
    MachineInfo_t mi = Machine_GetInfo(machine_id);
    sched_mc[machine_id].s_state = mi.s_state;
    if (mi.s_state == S0) {
        for (unsigned c = 0; c < mi.num_cpus; c++) Machine_SetCorePerformance(machine_id, c, P0);
        sched_retry_pending = true;
    }
}

void SimulationComplete(Time_t time) {
    cout << "SLA violation report" << endl;
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    cout << "Simulation run finished in " << double(time)/1000000 << " seconds" << endl;
    SimOutput("SimulationComplete(): Simulation finished at time " + to_string(time), 4);

    theScheduler.Shutdown(time);
}
