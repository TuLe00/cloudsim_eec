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
#include <algorithm>
#include <climits>

// ---------- Static bookkeeping ----------

static map<MachineId_t, vector<VMId_t>> m2v;       // machine → VMs on it
static set<MachineId_t> waking;                     // machines currently powering on
static vector<TaskId_t> pending;                    // tasks waiting for a free CPU slot
static set<VMId_t> migrating;                       // VMs mid-migration
static map<TaskId_t, VMId_t> task_to_vm;           // task → its current VM
static unsigned rr_idx = 0;                         // round-robin index for tie-breaking

// ---------- Utilities ----------

static bool VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true; // LINUX, LINUX_RT valid on all CPUs
}

static Priority_t SLAPrio(SLAType_t sla) {
    if (sla == SLA0 || sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2)                return MID_PRIORITY;
    return LOW_PRIORITY;
}

// Return (or create) a VM on mid with matching (vm_type, cpu_type).
static VMId_t EnsureVM(MachineId_t mid, VMType_t vt, CPUType_t ct, vector<VMId_t>& all_vms) {
    for (VMId_t vm : m2v[mid]) {
        if (migrating.count(vm)) continue;
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.vm_type == vt && vi.cpu == ct) return vm;
    }
    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    m2v[mid].push_back(vm);
    return vm;
}

// Find best machine for a task.
//
//  SLA0/SLA1 (strict):
//    - Spread load: pick the machine with the fewest active tasks so no single
//      machine becomes a bottleneck.
//    - Round-robin tie-breaking: rr_idx rotates the starting point so equal-
//      load machines are selected in sequence across consecutive calls.
//
//  SLA2/SLA3 (relaxed):
//    - Consolidate: pick the most loaded machine that still fits (best-fit),
//      leaving other machines idle so they can be slept to save energy.
//
//  GPU tasks:
//    - Strongly prefer GPU-equipped machines regardless of SLA class.
//
static MachineId_t BestFit(TaskId_t task, const vector<MachineId_t>& mlist) {
    CPUType_t rc  = RequiredCPUType(task);
    VMType_t  rv  = RequiredVMType(task);
    unsigned  rm  = GetTaskMemory(task);
    bool      rg  = IsTaskGPUCapable(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict   = (sla == SLA0 || sla == SLA1);

    if (!VMCPUOk(rv, rc)) return MachineId_t(UINT_MAX);

    unsigned n = mlist.size();
    MachineId_t best = MachineId_t(UINT_MAX);
    int best_score = INT_MAX;

    for (unsigned i = 0; i < n; i++) {
        MachineId_t mid = mlist[(rr_idx + i) % n];
        MachineInfo_t mi = Machine_GetInfo(mid);
        if (mi.s_state != S0 || mi.cpu != rc) continue;

        // Memory check (conservatively account for new-VM overhead if needed)
        bool has_vm = false;
        for (VMId_t vm : m2v[mid]) {
            if (migrating.count(vm)) continue;
            VMInfo_t vi = VM_GetInfo(vm);
            if (vi.vm_type == rv && vi.cpu == rc) { has_vm = true; break; }
        }
        unsigned extra = has_vm ? 0 : VM_MEMORY_OVERHEAD;
        if (mi.memory_size < mi.memory_used + rm + extra) continue;

        // Score: strict → least loaded (spread); relaxed → most loaded (consolidate)
        int load  = (int)mi.active_tasks;
        int score = strict ? load : -load;
        if (rg && !mi.gpus) score += 1000; // heavily prefer GPU for GPU tasks

        if (score < best_score) { best = mid; best_score = score; }
    }

    if (best != MachineId_t(UINT_MAX)) rr_idx = (rr_idx + 1) % n;
    return best;
}

// Drain the pending queue, placing tasks onto newly-available machines.
static void PlacePending(const vector<MachineId_t>& mlist, vector<VMId_t>& all_vms) {
    vector<TaskId_t> rem;
    for (TaskId_t t : pending) {
        if (IsTaskCompleted(t)) continue;
        MachineId_t mid = BestFit(t, mlist);
        if (mid != MachineId_t(UINT_MAX)) {
            VMId_t vm = EnsureVM(mid, RequiredVMType(t), RequiredCPUType(t), all_vms);
            VM_AddTask(vm, t, SLAPrio(RequiredSLA(t)));
            task_to_vm[t] = vm;
        } else {
            rem.push_back(t);
        }
    }
    pending = rem;
}

// Place a single incoming task, waking a sleeping machine if necessary.
static void DoPlace(TaskId_t task, const vector<MachineId_t>& mlist, vector<VMId_t>& all_vms) {
    MachineId_t mid = BestFit(task, mlist);
    if (mid == MachineId_t(UINT_MAX)) {
        // Wake the first sleeping machine with the right CPU type
        CPUType_t rc = RequiredCPUType(task);
        for (MachineId_t m : mlist) {
            if (waking.count(m)) continue;
            MachineInfo_t mi = Machine_GetInfo(m);
            if (mi.cpu == rc && mi.s_state != S0) {
                Machine_SetState(m, S0);
                waking.insert(m);
                SimOutput("DoPlace(): Waking machine " + to_string(m), 3);
                break;
            }
        }
        pending.push_back(task);
        return;
    }
    VMId_t vm = EnsureVM(mid, RequiredVMType(task), RequiredCPUType(task), all_vms);
    VM_AddTask(vm, task, SLAPrio(RequiredSLA(task)));
    task_to_vm[task] = vm;
}

// Shut down idle VMs and sleep the machine if it becomes empty.
static void TrySleep(MachineId_t mid, vector<VMId_t>& all_vms) {
    if (waking.count(mid)) return;
    MachineInfo_t mi = Machine_GetInfo(mid);
    if (mi.s_state != S0 || mi.active_tasks > 0) return;

    vector<VMId_t>& vlist = m2v[mid];
    vector<VMId_t> keep;
    for (VMId_t vm : vlist) {
        if (migrating.count(vm)) { keep.push_back(vm); continue; }
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.active_tasks.empty()) {
            VM_Shutdown(vm);
            all_vms.erase(remove(all_vms.begin(), all_vms.end(), vm), all_vms.end());
        } else {
            keep.push_back(vm);
        }
    }
    vlist = keep;

    mi = Machine_GetInfo(mid);
    if (mi.active_vms == 0) {
        Machine_SetState(mid, S5);
        SimOutput("TrySleep(): Machine " + to_string(mid) + " sleeping", 3);
    }
}

// ---------- Scheduler methods ----------

void Scheduler::Init() {
    SimOutput("Scheduler::Init(): Total machines = " + to_string(Machine_GetTotal()), 1);
    unsigned total = Machine_GetTotal();
    for (unsigned i = 0; i < total; i++) {
        MachineId_t mid = MachineId_t(i);
        machines.push_back(mid);
        MachineInfo_t mi = Machine_GetInfo(mid);
        VMType_t vt = (mi.cpu == POWER) ? AIX : LINUX;
        VMId_t vm = VM_Create(vt, mi.cpu);
        VM_Attach(vm, mid);
        vms.push_back(vm);
        m2v[mid].push_back(vm);
        // Start every core at full speed; DVFS will scale back when idle
        for (unsigned c = 0; c < mi.num_cpus; c++)
            Machine_SetCorePerformance(mid, c, P0);
    }
    SimOutput("Scheduler::Init(): Created " + to_string(vms.size()) +
              " VMs on " + to_string(total) + " machines", 1);
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    migrating.erase(vm_id);
    SimOutput("Scheduler::MigrationComplete(): VM " + to_string(vm_id) +
              " done at " + to_string(time), 4);
}

void Scheduler::NewTask(Time_t now, TaskId_t task_id) {
    DoPlace(task_id, machines, vms);
}

void Scheduler::PeriodicCheck(Time_t now) {
    // Try to schedule any tasks waiting for a free CPU slot
    PlacePending(machines, vms);

    // Power management: sleep idle machines, tune P-states on busy ones
    for (MachineId_t mid : machines) {
        if (waking.count(mid)) continue;
        MachineInfo_t mi = Machine_GetInfo(mid);
        if (mi.s_state != S0) continue;
        if (mi.active_tasks == 0)
            TrySleep(mid, vms);
        // No P-state changes while tasks are running (unsafe on C0 cores)
    }
}

void Scheduler::Shutdown(Time_t time) {
    for (VMId_t vm : vms) {
        if (migrating.count(vm)) continue;
        VMInfo_t vi = VM_GetInfo(vm);
        if (vi.active_tasks.empty())
            VM_Shutdown(vm);
    }
    SimOutput("SimulationComplete(): Finished!", 4);
    SimOutput("SimulationComplete(): Time is " + to_string(time), 4);
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    SimOutput("Scheduler::TaskComplete(): Task " + to_string(task_id) +
              " at " + to_string(now), 4);

    // Identify which machine the completed task was on so we can
    // (a) immediately fill its freed CPU slot with a pending task, and
    // (b) opportunistically sleep it if it becomes idle.
    MachineId_t freed_machine = MachineId_t(UINT_MAX);
    auto it = task_to_vm.find(task_id);
    if (it != task_to_vm.end()) {
        VMInfo_t vi = VM_GetInfo(it->second);
        freed_machine = vi.machine_id;
        task_to_vm.erase(it);
    }

    // Sleep the freed machine if it is now empty
    if (freed_machine != MachineId_t(UINT_MAX))
        TrySleep(freed_machine, vms);
}

// ---------- Public interface ----------

static Scheduler Scheduler;

void InitScheduler() {
    SimOutput("InitScheduler(): Initializing scheduler", 4);
    Scheduler.Init();
}

void HandleNewTask(Time_t time, TaskId_t task_id) {
    SimOutput("HandleNewTask(): Task " + to_string(task_id) + " at " + to_string(time), 4);
    Scheduler.NewTask(time, task_id);
}

void HandleTaskCompletion(Time_t time, TaskId_t task_id) {
    SimOutput("HandleTaskCompletion(): Task " + to_string(task_id) +
              " at " + to_string(time), 4);
    Scheduler.TaskComplete(time, task_id);
}

void MemoryWarning(Time_t time, MachineId_t machine_id) {
    SimOutput("MemoryWarning(): Machine " + to_string(machine_id) +
              " at " + to_string(time), 0);
}

void MigrationDone(Time_t time, VMId_t vm_id) {
    SimOutput("MigrationDone(): VM " + to_string(vm_id) + " at " + to_string(time), 4);
    Scheduler.MigrationComplete(time, vm_id);
}

void SchedulerCheck(Time_t time) {
    SimOutput("SchedulerCheck(): at " + to_string(time), 4);
    Scheduler.PeriodicCheck(time);
}

void SimulationComplete(Time_t time) {
    cout << "SLA violation report" << endl;
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;     // SLA3 has no violation issues
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    cout << "Simulation run finished in " << double(time)/1000000 << " seconds" << endl;
    SimOutput("SimulationComplete(): Simulation finished at time " + to_string(time), 4);
    Scheduler.Shutdown(time);
}

void SLAWarning(Time_t time, TaskId_t task_id) {
    // Immediately boost priority; the task may already be at HIGH but this
    // handles edge cases (e.g., a task whose priority was downgraded).
    SetTaskPriority(task_id, HIGH_PRIORITY);
    SimOutput("SLAWarning(): Task " + to_string(task_id) +
              " boosted at " + to_string(time), 2);
}

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    SimOutput("StateChangeComplete(): Machine " + to_string(machine_id) +
              " ready at " + to_string(time), 4);
    // Machine has finished waking up; restore full performance before tasks run.
    waking.erase(machine_id);
    MachineInfo_t mi = Machine_GetInfo(machine_id);
    for (unsigned c = 0; c < mi.num_cpus; c++)
        Machine_SetCorePerformance(machine_id, c, P0);
}