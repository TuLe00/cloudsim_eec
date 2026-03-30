//
//  Scheduler.cpp
//  CloudSim
//

#include <cstdint>
#include "Scheduler.hpp"
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <climits>
#include <iostream>
#include <string>

using namespace std;

typedef enum { CUSTOM_GREEDY, PMAPPER, EECO, PABFD } AlgorithmType;
static AlgorithmType CURRENT_ALGO = CUSTOM_GREEDY;

// ============================================================================
// 1. CUSTOM_GREEDY IMPLEMENTATION
// ============================================================================

static map<MachineId_t, vector<VMId_t>> greedy_m2v;
static set<MachineId_t> greedy_waking;
static vector<TaskId_t> greedy_pending;
static set<VMId_t> greedy_migrating;
static map<TaskId_t, VMId_t> greedy_task_to_vm;
static unsigned greedy_rr_idx = 0;

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
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.vm_type == vt && vi.cpu == ct) return vm;
    }
    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    greedy_m2v[mid].push_back(vm);
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
        MachineInfo_t mi = Machine_GetInfo(mid);
        if (mi.s_state != S0 || mi.cpu != rc) continue;

        bool has_vm = false;
        for (VMId_t vm : greedy_m2v[mid]) {
            if (greedy_migrating.count(vm)) continue;
            VMInfo_t vi = VM_GetInfo(vm);
            if (vi.vm_type == rv && vi.cpu == rc) { has_vm = true; break; }
        }
        unsigned extra = has_vm ? 0 : VM_MEMORY_OVERHEAD;
        if (mi.memory_size < mi.memory_used + rm + extra) continue;

        int load  = (int)mi.active_tasks;
        int score = strict ? load : -load;
        if (rg && !mi.gpus) score += 1000;

        if (score < best_score) { best = mid; best_score = score; }
    }
    if (best != MachineId_t(UINT_MAX)) greedy_rr_idx = (greedy_rr_idx + 1) % n;
    return best;
}

static void Greedy_TrySleep(MachineId_t mid, vector<VMId_t>& all_vms) {
    if (greedy_waking.count(mid)) return;
    MachineInfo_t mi = Machine_GetInfo(mid);
    if (mi.s_state != S0 || mi.active_tasks > 0) return;

    vector<VMId_t>& vlist = greedy_m2v[mid];
    vector<VMId_t> keep;
    for (VMId_t vm : vlist) {
        if (greedy_migrating.count(vm)) { keep.push_back(vm); continue; }
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.active_tasks.empty()) {
            VM_Shutdown(vm);
            all_vms.erase(remove(all_vms.begin(), all_vms.end(), vm), all_vms.end());
        } else {
            keep.push_back(vm);
        }
    }
    vlist = keep;
    if (Machine_GetInfo(mid).active_vms == 0) Machine_SetState(mid, S5);
}

// ============================================================================
// 2. PLACEHOLDERS FOR OTHER ALGORITHMS (PMAPPER, EECO, PABFD)
// ============================================================================

// Add your research-based PABFD logic here later...

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
                VMId_t vm = VM_Create((mi.cpu == POWER) ? AIX : LINUX, mi.cpu);
                VM_Attach(vm, mid);
                vms.push_back(vm);
                greedy_m2v[mid].push_back(vm);
                for (unsigned c = 0; c < mi.num_cpus; c++) Machine_SetCorePerformance(mid, c, P0);
            }
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
            for (MachineId_t m : machines) {
                if (greedy_waking.count(m)) continue;
                if (Machine_GetInfo(m).cpu == rc && Machine_GetInfo(m).s_state != S0) {
                    Machine_SetState(m, S0);
                    greedy_waking.insert(m);
                    break;
                }
            }
            greedy_pending.push_back(task_id);
        } else {
            VMId_t vm = Greedy_EnsureVM(mid, RequiredVMType(task_id), RequiredCPUType(task_id), vms);
            VM_AddTask(vm, task_id, Greedy_SLAPrio(RequiredSLA(task_id)));
            greedy_task_to_vm[task_id] = vm;
        }
    }
}

void Scheduler::PeriodicCheck(Time_t now) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        vector<TaskId_t> rem;
        for (TaskId_t t : greedy_pending) {
            MachineId_t mid = Greedy_BestFit(t, machines);
            if (mid != MachineId_t(UINT_MAX)) {
                VMId_t vm = Greedy_EnsureVM(mid, RequiredVMType(t), RequiredCPUType(t), vms);
                VM_AddTask(vm, t, Greedy_SLAPrio(RequiredSLA(t)));
                greedy_task_to_vm[t] = vm;
            } else rem.push_back(t);
        }
        greedy_pending = rem;
        for (MachineId_t mid : machines) Greedy_TrySleep(mid, vms);
    }
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        auto it = greedy_task_to_vm.find(task_id);
        if (it != greedy_task_to_vm.end()) {
            MachineId_t mid = VM_GetInfo(it->second).machine_id;
            greedy_task_to_vm.erase(it);
            Greedy_TrySleep(mid, vms);
        }
    }
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) greedy_migrating.erase(vm_id);
}

void Scheduler::Shutdown(Time_t time) {
    for (VMId_t vm : vms) {
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
void SLAWarning(Time_t time, TaskId_t task_id) { SetTaskPriority(task_id, HIGH_PRIORITY); }

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    if (CURRENT_ALGO == CUSTOM_GREEDY) {
        greedy_waking.erase(machine_id);
        MachineInfo_t mi = Machine_GetInfo(machine_id);
        for (unsigned c = 0; c < mi.num_cpus; c++) Machine_SetCorePerformance(machine_id, c, P0);
    }
}

void SimulationComplete(Time_t time) {
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;
    cout << "SLA3: " << GetSLAReport(SLA3) << "%" << endl;
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    theScheduler.Shutdown(time);
}