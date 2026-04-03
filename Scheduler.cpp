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
// EECO IMPLEMENTATION
// ============================================================================
//
// Three-tier model:
//   Running      (S0) — hosts actively serving tasks
//   Intermediate (S3) — standby hosts, fast to wake
//   Switched off (S5) — deep sleep, slow to wake
// ============================================================================

static const double EECO_U_TARGET      = 0.70;
static const double EECO_ALPHA         = 0.25;
static const double EECO_MIN_RUN_FRAC  = 0.25;
static const unsigned EECO_MAX_DEMOTE  = 2;
static const Time_t EECO_STRICT_COOLDOWN = 5000000;

struct EECOMachineCache {
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

struct EECOVMCache {
    VMType_t    vm_type;
    CPUType_t   cpu;
    MachineId_t machine_id;
    unsigned    task_count;
    unsigned    memory_used;
};

static map<MachineId_t, EECOMachineCache> eeco_mc;
static map<VMId_t, EECOVMCache> eeco_vc;
static map<MachineId_t, vector<VMId_t>> eeco_m2v;
static map<TaskId_t, VMId_t> eeco_task_to_vm;
static map<CPUType_t, deque<TaskId_t>> eeco_pending_by_cpu;
static set<MachineId_t> eeco_transitioning;
static bool eeco_retry_pending = false;
static unsigned eeco_strict_active = 0;
static Time_t eeco_strict_until = 0;

static Priority_t EECO_SLAPrio(SLAType_t sla) {
    if (sla == SLA0 || sla == SLA1) return HIGH_PRIORITY;
    if (sla == SLA2) return MID_PRIORITY;
    return LOW_PRIORITY;
}

static bool EECO_VMCPUOk(VMType_t vm, CPUType_t cpu) {
    if (vm == AIX) return cpu == POWER;
    if (vm == WIN) return cpu == ARM || cpu == X86;
    return true;
}

static VMId_t EECO_EnsureVM(MachineId_t mid, TaskId_t task, vector<VMId_t>& all_vms) {
    VMType_t vt = RequiredVMType(task);
    CPUType_t ct = RequiredCPUType(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict = (sla == SLA0 || sla == SLA1);
    VMId_t best_vm = VMId_t(UINT_MAX);
    unsigned best_load = UINT_MAX;
    unsigned matching_vms = 0;

    for (VMId_t vm : eeco_m2v[mid]) {
        if (eeco_vc[vm].vm_type != vt || eeco_vc[vm].cpu != ct) continue;
        matching_vms++;
        if (eeco_vc[vm].task_count < best_load) {
            best_vm = vm;
            best_load = eeco_vc[vm].task_count;
        }
    }

    bool can_add_vm =
        matching_vms < eeco_mc[mid].num_cpus &&
        eeco_mc[mid].memory_size >= eeco_mc[mid].memory_used + VM_MEMORY_OVERHEAD;

    if (best_vm != VMId_t(UINT_MAX) && (!strict || best_load == 0 || !can_add_vm)) {
        return best_vm;
    }

    VMId_t vm = VM_Create(vt, ct);
    VM_Attach(vm, mid);
    all_vms.push_back(vm);
    eeco_m2v[mid].push_back(vm);
    eeco_vc[vm] = {vt, ct, mid, 0, VM_MEMORY_OVERHEAD};
    eeco_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
    eeco_mc[mid].active_vms++;
    return vm;
}

static bool EECO_CanHost(MachineId_t mid, CPUType_t rc, VMType_t rv, unsigned rm) {
    const EECOMachineCache& mc = eeco_mc[mid];
    if (mc.s_state != S0 || mc.cpu != rc) return false;
    if (!EECO_VMCPUOk(rv, rc)) return false;

    bool has_vm = false;
    for (VMId_t vm : eeco_m2v[mid]) {
        if (eeco_vc[vm].vm_type == rv && eeco_vc[vm].cpu == rc) {
            has_vm = true;
            break;
        }
    }

    unsigned need = rm + (has_vm ? 0u : (unsigned)VM_MEMORY_OVERHEAD);
    return mc.memory_size >= mc.memory_used + need;
}

static int EECO_HostTierForTask(TaskId_t task, const EECOMachineCache& mc) {
    bool strict = RequiredSLA(task) == SLA0 || RequiredSLA(task) == SLA1;
    bool gpu_task = IsTaskGPUCapable(task);
    bool fast = mc.perf_p0 >= 2500;

    if (strict) {
        if (gpu_task) return fast ? 0 : 1;
        if (fast && !mc.gpus) return 0;
        if (fast && mc.gpus) return 1;
        return 2;
    }

    if (gpu_task) return mc.gpus ? 0 : 1;
    if (!mc.gpus && !fast) return 0;
    if (!mc.gpus) return 1;
    return 2;
}

static MachineId_t EECO_FindHost(Time_t now, TaskId_t task, const vector<MachineId_t>& mlist) {
    CPUType_t rc = RequiredCPUType(task);
    VMType_t rv = RequiredVMType(task);
    unsigned rm = GetTaskMemory(task);
    bool gpu_task = IsTaskGPUCapable(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict = (sla == SLA0 || sla == SLA1);
    TaskInfo_t ti = GetTaskInfo(task);

    MachineId_t best = MachineId_t(UINT_MAX);
    double best_score = strict ? 1e300 : -1e300;
    bool best_meets_deadline = false;
    int best_tier = INT_MAX;

    for (MachineId_t mid : mlist) {
        if (!EECO_CanHost(mid, rc, rv, rm)) continue;

        const EECOMachineCache& mc = eeco_mc[mid];
        int tier = EECO_HostTierForTask(task, mc);
        double perf = max(1u, mc.perf_p0);
        double service_time = (double)ti.remaining_instructions / perf;
        double waves = max(1.0, ((double)mc.active_tasks + 1.0) / max(1u, mc.num_cpus));
        double projected_finish = (double)now + service_time * waves;
        bool meets_deadline = projected_finish <= (double)ti.target_completion;

        if (strict) {
            double util = (double)(mc.active_tasks + 1) / max(1u, mc.num_cpus);
            double deadline_penalty = meets_deadline ? 0.0
                : (projected_finish - (double)ti.target_completion);
            double score = deadline_penalty * 1000.0
                + projected_finish
                + util * 1e6
                - perf * 10.0;
            if (gpu_task && mc.gpus) score -= 2e5;
            if (!gpu_task && mc.gpus) score += 2e5;

            if (best == MachineId_t(UINT_MAX)) {
                best = mid;
                best_score = score;
                best_meets_deadline = meets_deadline;
                best_tier = tier;
                continue;
            }
            if (meets_deadline != best_meets_deadline) {
                if (meets_deadline) {
                    best = mid;
                    best_score = score;
                    best_meets_deadline = true;
                    best_tier = tier;
                }
                continue;
            }
            if (tier != best_tier) {
                if (tier < best_tier) {
                    best = mid;
                    best_score = score;
                    best_meets_deadline = meets_deadline;
                    best_tier = tier;
                }
                continue;
            }
            if (score < best_score) {
                best = mid;
                best_score = score;
                best_meets_deadline = meets_deadline;
                best_tier = tier;
            }
        } else {
            double util = (double)mc.active_tasks / max(1u, mc.num_cpus);
            double score = util * 1e6 - perf * 100.0;
            if (best == MachineId_t(UINT_MAX) || tier < best_tier) {
                best = mid;
                best_score = score;
                best_tier = tier;
                continue;
            }
            if (tier > best_tier) continue;
            if (score > best_score) {
                best = mid;
                best_score = score;
                best_tier = tier;
            }
        }
    }

    return best;
}

static void EECO_WakeMachines(const vector<MachineId_t>& machines,
                              CPUType_t rc,
                              bool strict,
                              bool needs_gpu,
                              bool avoid_gpu) {
    auto wake_pass = [&](bool require_gpu, bool forbid_gpu) -> bool {
        bool woke_any = false;
        for (MachineId_t mid : machines) {
            if (eeco_transitioning.count(mid)) continue;
            const EECOMachineCache& mc = eeco_mc[mid];
            if (mc.cpu != rc || mc.s_state == S0) continue;
            if (require_gpu && !mc.gpus) continue;
            if (forbid_gpu && mc.gpus) continue;
            Machine_SetState(mid, S0);
            eeco_transitioning.insert(mid);
            woke_any = true;
            if (!strict) return true;
        }
        return woke_any;
    };

    if (needs_gpu) {
        bool woke_gpu = wake_pass(true, false);
        if (!woke_gpu) wake_pass(false, false);
        return;
    }

    if (avoid_gpu) {
        bool woke_non_gpu = wake_pass(false, true);
        if (!woke_non_gpu) wake_pass(false, false);
        return;
    }

    wake_pass(false, false);
}

static void EECO_PlaceTask(TaskId_t task, MachineId_t mid, vector<VMId_t>& vms) {
    VMId_t vm = EECO_EnsureVM(mid, task, vms);
    SLAType_t sla = RequiredSLA(task);
    VM_AddTask(vm, task, EECO_SLAPrio(sla));
    eeco_task_to_vm[task] = vm;

    unsigned tm = GetTaskMemory(task);
    eeco_vc[vm].task_count++;
    eeco_vc[vm].memory_used += tm;
    eeco_mc[mid].active_tasks++;
    eeco_mc[mid].memory_used += tm;

    if (sla == SLA0 || sla == SLA1) eeco_strict_active++;
}

static void EECO_Rebalance(Time_t now, const vector<MachineId_t>& mlist, vector<VMId_t>& all_vms) {
    unsigned total = (unsigned)mlist.size();
    if (total == 0) return;

    bool has_strict_pending = false;
    for (auto& [_, bucket] : eeco_pending_by_cpu) {
        for (TaskId_t task : bucket) {
            SLAType_t s = RequiredSLA(task);
            if (s == SLA0 || s == SLA1) {
                has_strict_pending = true;
                break;
            }
        }
        if (has_strict_pending) break;
    }

    bool has_pending = false;
    for (auto& [_, bucket] : eeco_pending_by_cpu) {
        if (!bucket.empty()) {
            has_pending = true;
            break;
        }
    }

    vector<MachineId_t> running;
    vector<MachineId_t> intermediate;
    vector<MachineId_t> off_hosts;
    for (MachineId_t mid : mlist) {
        MachineState_t s = eeco_mc[mid].s_state;
        if (s == S0) running.push_back(mid);
        else if (s == S3) intermediate.push_back(mid);
        else off_hosts.push_back(mid);
    }

    unsigned N_run_cur = (unsigned)running.size();

    if (eeco_strict_active > 0 || has_strict_pending || now < eeco_strict_until) {
        for (MachineId_t mid : intermediate) {
            if (!eeco_transitioning.count(mid)) {
                Machine_SetState(mid, S0);
                eeco_transitioning.insert(mid);
            }
        }
        for (MachineId_t mid : off_hosts) {
            if (!eeco_transitioning.count(mid)) {
                Machine_SetState(mid, S0);
                eeco_transitioning.insert(mid);
            }
        }
        return;
    }

    unsigned cap_sum = 0;
    unsigned task_sum = 0;
    for (MachineId_t mid : running) {
        cap_sum += eeco_mc[mid].num_cpus;
        task_sum += eeco_mc[mid].active_tasks;
    }
    double U_cur = (cap_sum > 0) ? (double)task_sum / (double)cap_sum : 0.0;

    unsigned N_run_floor = max(1u, (unsigned)ceil(total * EECO_MIN_RUN_FRAC));
    unsigned N_run_target;
    if (U_cur == 0.0) {
        N_run_target = N_run_floor;
    } else {
        N_run_target = (unsigned)ceil((U_cur / EECO_U_TARGET) * (double)N_run_cur);
        N_run_target = max(N_run_floor, N_run_target);
    }

    if (has_pending) N_run_target = max(N_run_target, N_run_cur);
    N_run_target = max(1u, min(N_run_target, total));

    unsigned headroom = total - N_run_target;
    unsigned N_inter_target = min((unsigned)ceil(EECO_ALPHA * (double)N_run_target), headroom);

    if (N_run_cur < N_run_target) {
        unsigned need = N_run_target - N_run_cur;
        for (unsigned i = 0; i < need && i < intermediate.size(); i++) {
            MachineId_t mid = intermediate[i];
            if (!eeco_transitioning.count(mid)) {
                Machine_SetState(mid, S0);
                eeco_transitioning.insert(mid);
            }
        }
        if (need > intermediate.size()) {
            unsigned still_need = need - (unsigned)intermediate.size();
            for (unsigned i = 0; i < still_need && i < off_hosts.size(); i++) {
                MachineId_t mid = off_hosts[i];
                if (!eeco_transitioning.count(mid)) {
                    Machine_SetState(mid, S0);
                    eeco_transitioning.insert(mid);
                }
            }
        }
    }

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

    if (!has_pending && N_run_cur > N_run_target) {
        vector<pair<unsigned, MachineId_t>> candidates;
        for (MachineId_t mid : running) {
            candidates.push_back({eeco_mc[mid].active_tasks, mid});
        }
        sort(candidates.begin(), candidates.end());

        unsigned demoted = 0;
        unsigned surplus = min(N_run_cur - N_run_target, EECO_MAX_DEMOTE);
        for (auto& [tasks, mid] : candidates) {
            if (demoted >= surplus) break;
            if (eeco_transitioning.count(mid) || tasks > 0) continue;

            vector<VMId_t> keep;
            for (VMId_t vm : eeco_m2v[mid]) {
                if (eeco_vc[vm].task_count == 0) {
                    eeco_mc[mid].memory_used -= eeco_vc[vm].memory_used;
                    eeco_mc[mid].active_vms--;
                    eeco_vc.erase(vm);
                    VM_Shutdown(vm);
                    all_vms.erase(remove(all_vms.begin(), all_vms.end(), vm), all_vms.end());
                } else {
                    keep.push_back(vm);
                }
            }
            eeco_m2v[mid] = keep;

            if (eeco_mc[mid].active_vms == 0) {
                Machine_SetState(mid, S3);
                eeco_mc[mid].s_state = S3;
                eeco_transitioning.insert(mid);
                demoted++;
            }
        }
    }

    if (!has_pending) {
        vector<MachineId_t> inter_now;
        for (MachineId_t mid : mlist) {
            if (eeco_mc[mid].s_state == S3 && !eeco_transitioning.count(mid)) {
                inter_now.push_back(mid);
            }
        }

        if ((unsigned)inter_now.size() > N_inter_target) {
            unsigned surplus = (unsigned)inter_now.size() - N_inter_target;
            surplus = min(surplus, EECO_MAX_DEMOTE);
            for (unsigned i = 0; i < surplus && i < inter_now.size(); i++) {
                MachineId_t mid = inter_now[i];
                if (!eeco_transitioning.count(mid)) {
                    Machine_SetState(mid, S5);
                    eeco_mc[mid].s_state = S5;
                    eeco_transitioning.insert(mid);
                }
            }
        }
    }
}

static void EECO_DrainPending(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    for (auto& [cpu_type, bucket] : eeco_pending_by_cpu) {
        if (bucket.empty()) continue;

        stable_sort(bucket.begin(), bucket.end(), [](TaskId_t a, TaskId_t b) {
            SLAType_t sa = RequiredSLA(a);
            SLAType_t sb = RequiredSLA(b);
            int pa = (sa == SLA0) ? 0 : (sa == SLA1) ? 1 : (sa == SLA2) ? 2 : 3;
            int pb = (sb == SLA0) ? 0 : (sb == SLA1) ? 1 : (sb == SLA2) ? 2 : 3;
            return pa < pb;
        });

        deque<TaskId_t> retry;
        while (!bucket.empty()) {
            TaskId_t task = bucket.front();
            bucket.pop_front();
            MachineId_t mid = EECO_FindHost(Now(), task, machines);
            if (mid != MachineId_t(UINT_MAX)) {
                EECO_PlaceTask(task, mid, vms);
            } else {
                SLAType_t sla = RequiredSLA(task);
                bool gpu_task = IsTaskGPUCapable(task);
                EECO_WakeMachines(machines, cpu_type, sla == SLA0 || sla == SLA1, gpu_task, !gpu_task);
                retry.push_back(task);
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

void EECO_Init(vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    eeco_mc.clear();
    eeco_vc.clear();
    eeco_m2v.clear();
    eeco_task_to_vm.clear();
    eeco_pending_by_cpu.clear();
    eeco_transitioning.clear();
    eeco_retry_pending = false;
    eeco_strict_active = 0;
    eeco_strict_until = 0;

    for (MachineId_t mid : machines) {
        MachineInfo_t mi = Machine_GetInfo(mid);
        Machine_SetState(mid, S0);
        eeco_mc[mid] = {
            S0,
            mi.cpu,
            mi.num_cpus,
            mi.performance.empty() ? 1u : mi.performance[0],
            mi.memory_size,
            mi.memory_used,
            mi.active_tasks,
            mi.active_vms,
            mi.gpus
        };

        VMType_t vt = (mi.cpu == POWER) ? AIX : LINUX;
        VMId_t vm = VM_Create(vt, mi.cpu);
        VM_Attach(vm, mid);
        vms.push_back(vm);
        eeco_m2v[mid].push_back(vm);
        eeco_vc[vm] = {vt, mi.cpu, mid, 0, VM_MEMORY_OVERHEAD};
        eeco_mc[mid].memory_used += VM_MEMORY_OVERHEAD;
        eeco_mc[mid].active_vms++;

        for (unsigned c = 0; c < mi.num_cpus; c++) {
            Machine_SetCorePerformance(mid, c, P0);
        }
    }
}

void EECO_NewTask(Time_t now, TaskId_t task, vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    CPUType_t rc = RequiredCPUType(task);
    SLAType_t sla = RequiredSLA(task);
    bool strict = (sla == SLA0 || sla == SLA1);
    bool gpu_task = IsTaskGPUCapable(task);

    if (strict) {
        EECO_WakeMachines(machines, rc, true, gpu_task, !gpu_task);
    }

    MachineId_t mid = EECO_FindHost(now, task, machines);
    if (mid == MachineId_t(UINT_MAX)) {
        if (!strict) {
            EECO_WakeMachines(machines, rc, false, gpu_task, !gpu_task);
        }
        eeco_pending_by_cpu[rc].push_back(task);
    } else {
        EECO_PlaceTask(task, mid, vms);
    }
}

void EECO_PeriodicCheck(Time_t now, vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    bool has_pending = false;
    for (auto& [_, bucket] : eeco_pending_by_cpu) {
        if (!bucket.empty()) {
            has_pending = true;
            break;
        }
    }

    EECO_Rebalance(now, machines, vms);

    if (eeco_retry_pending || has_pending) {
        eeco_retry_pending = false;
        EECO_DrainPending(machines, vms);
    }
}

void EECO_TaskComplete(Time_t now, TaskId_t task, vector<MachineId_t>& machines, vector<VMId_t>& vms) {
    (void)machines;
    (void)vms;
    auto it = eeco_task_to_vm.find(task);
    if (it == eeco_task_to_vm.end()) return;

    VMId_t vm = it->second;
    MachineId_t mid = eeco_vc[vm].machine_id;
    unsigned tm = GetTaskMemory(task);
    eeco_vc[vm].task_count--;
    eeco_vc[vm].memory_used -= tm;
    eeco_mc[mid].active_tasks--;
    eeco_mc[mid].memory_used -= tm;
    eeco_task_to_vm.erase(it);

    SLAType_t sla = RequiredSLA(task);
    if (sla == SLA0 || sla == SLA1) {
        if (eeco_strict_active > 0) eeco_strict_active--;
        eeco_strict_until = max(eeco_strict_until, now + EECO_STRICT_COOLDOWN);
    }

    eeco_retry_pending = true;
}

void EECO_Shutdown(vector<VMId_t>& vms) {
    for (VMId_t vm : vms) {
        if (eeco_vc.count(vm) && eeco_vc[vm].task_count == 0) {
            VM_Shutdown(vm);
        }
    }
}

// ============================================================================
// REQUIRED SCHEDULER METHODS
// ============================================================================

static Scheduler theScheduler;

void Scheduler::Init() {
    machines.clear();
    vms.clear();
    unsigned total = Machine_GetTotal();
    for (unsigned i = 0; i < total; i++) {
        machines.push_back(MachineId_t(i));
    }
    EECO_Init(machines, vms);
}

void Scheduler::MigrationComplete(Time_t time, VMId_t vm_id) {
    (void)time;
    (void)vm_id;
}

void Scheduler::NewTask(Time_t now, TaskId_t task_id) {
    EECO_NewTask(now, task_id, machines, vms);
}

void Scheduler::PeriodicCheck(Time_t now) {
    EECO_PeriodicCheck(now, machines, vms);
}

void Scheduler::Shutdown(Time_t now) {
    (void)now;
    EECO_Shutdown(vms);
}

void Scheduler::TaskComplete(Time_t now, TaskId_t task_id) {
    EECO_TaskComplete(now, task_id, machines, vms);
}

// ============================================================================
// PUBLIC INTERFACE
// ============================================================================

void InitScheduler() { theScheduler.Init(); }
void HandleNewTask(Time_t time, TaskId_t task_id) { theScheduler.NewTask(time, task_id); }
void HandleTaskCompletion(Time_t time, TaskId_t task_id) { theScheduler.TaskComplete(time, task_id); }
void MemoryWarning(Time_t time, MachineId_t machine_id) {
    (void)time;
    (void)machine_id;
}
void MigrationDone(Time_t time, VMId_t vm_id) { theScheduler.MigrationComplete(time, vm_id); }
void SchedulerCheck(Time_t time) { theScheduler.PeriodicCheck(time); }

void SLAWarning(Time_t time, TaskId_t task_id) {
    (void)time;
    SetTaskPriority(task_id, HIGH_PRIORITY);
    CPUType_t rc = RequiredCPUType(task_id);
    bool task_gpu = IsTaskGPUCapable(task_id);

    auto wake_sla_pass = [&](bool require_gpu, bool forbid_gpu) -> bool {
        bool woke_any = false;
        for (auto& [mid, mc] : eeco_mc) {
            if (eeco_transitioning.count(mid)) continue;
            if (mc.cpu != rc || mc.s_state == S0) continue;
            if (require_gpu && !mc.gpus) continue;
            if (forbid_gpu && mc.gpus) continue;
            Machine_SetState(mid, S0);
            eeco_transitioning.insert(mid);
            woke_any = true;
        }
        return woke_any;
    };

    if (task_gpu) {
        if (!wake_sla_pass(true, false)) wake_sla_pass(false, false);
    } else {
        if (!wake_sla_pass(false, true)) wake_sla_pass(false, false);
    }

    eeco_retry_pending = true;
}

void StateChangeComplete(Time_t time, MachineId_t machine_id) {
    (void)time;
    eeco_transitioning.erase(machine_id);
    MachineInfo_t mi = Machine_GetInfo(machine_id);
    eeco_mc[machine_id].s_state = mi.s_state;
    if (mi.s_state == S0) {
        for (unsigned c = 0; c < mi.num_cpus; c++) {
            Machine_SetCorePerformance(machine_id, c, P0);
        }
        eeco_retry_pending = true;
    }
}

void SimulationComplete(Time_t time) {
    cout << "SLA violation report" << endl;
    cout << "SLA0: " << GetSLAReport(SLA0) << "%" << endl;
    cout << "SLA1: " << GetSLAReport(SLA1) << "%" << endl;
    cout << "SLA2: " << GetSLAReport(SLA2) << "%" << endl;
    cout << "Total Energy " << Machine_GetClusterEnergy() << "KW-Hour" << endl;
    cout << "Simulation run finished in " << double(time) / 1000000 << " seconds" << endl;
    SimOutput("SimulationComplete(): Simulation finished at time " + to_string(time), 4);
    theScheduler.Shutdown(time);
}
