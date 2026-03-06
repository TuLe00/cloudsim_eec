machine class:
{
        Number of machines: 16
        CPU type: X86
        Number of cores: 4
        Memory: 64
        S-States: [250,200,150,100,50,10,0]
        P-States: [20,15,10,5]
        C-States: [15,10,5,0]
        MIPS: [500,400,300,200]
        GPUs: yes
}
# Task class 1: Early burst of long-running CPU-bound web tasks.
# Tasks arrive very frequently and run for a long time, creating heavy load early in the simulation.
# How it punishes a naive scheduler:
# A simple scheduler that assigns tasks sequentially to the first machines will quickly overload them,
# causing long queues and SLA violations because the tasks cannot finish before new ones arrive.
task class:
{
        Start time: 0
        End time: 50000
        Inter arrival: 500
        Expected runtime: 1000000
        Memory: 8
        VM type: LINUX
        GPU enabled: no
        SLA type: SLA0
        CPU type: X86
        Task type: WEB
        Seed: 123
}

# Task class 2: Overlapping GPU-capable tasks that arrive during the burst period.
# These tasks are shorter but can benefit from GPUs if scheduled correctly.
# How it punishes a naive scheduler:
# A naive scheduler that ignores GPU capability will place these tasks anywhere,
# wasting the GPU acceleration and increasing runtime. This reduces throughput and
# increases contention with the already-running long tasks.
task class:
{
        Start time: 20000
        End time: 100000
        Inter arrival: 1000
        Expected runtime: 100000
        Memory: 8
        VM type: LINUX
        GPU enabled: yes
        SLA type: SLA0
        CPU type: X86
        Task type: WEB
        Seed: 456
}


# Task class 3: Late-arriving background workload with lower intensity and weaker SLA.
# Tasks arrive less frequently and run for short durations.
# How it punishes a naive scheduler:
# A naive scheduler may keep all machines active after the earlier burst, wasting energy.
# Smarter schedulers would consolidate these light workloads onto fewer machines and
# place idle machines into sleep states to save energy.
task class:
{
        Start time: 150000
        End time: 300000
        Inter arrival: 5000
        Expected runtime: 20000
        Memory: 4
        VM type: LINUX
        GPU enabled: no
        SLA type: SLA1
        CPU type: X86
        Task type: WEB
        Seed: 789
}