"""
mlfq_simulation.py
--------------------
**SUMMARY**:

This module implements a discrete-event simulation of a Multi-Level Feedback Queue (MLFQ)

The system models a multi-core CPU where arriving jobs (processes) are scheduled
according to three priority levels (Q1, Q2, Q3). High priority jobs in Q1 and
Q2 use round-robin scheduling with configurable time quanta, while the lowest
priority queue Q3 uses a simple first-come, first-served (FCFS) discipline. The
design follows the MLFQ rules and data structures detailed below.

• **Priority and Preemption:** Jobs in higher priority queues always run before those in
  lower priority queues. When a new high-priority job arrives and all cores are busy,
  the scheduler preempts the lowest priority running job.

• **Demotion and Yielding:** If a job exhausts its time quantum in Q1 or Q2 without
  finishing, it is demoted to a lower queue. If a job voluntarily yields (e.g. for I/O)
  before its quantum expires, it retains its current priority level.

• **Priority Boost:** To prevent starvation, a periodic booster moves all jobs
  waiting in Q2 and Q3 back to Q1 after a configurable period S.

• **Data Structures:** Each queue is represented by a `ProcessQueue` with methods
  to add or remove jobs from the head or tail. Jobs are
  represented by the `Process` class, which tracks static attributes (PID,
  arrival time, total CPU time) and dynamic state (current level, remaining
  CPU time, remaining quantum, etc.).

• **Scheduler and Cores:** A central `CPUScheduler` coordinates all scheduling
  activities. It maintains three queues, manages a collection of `Core`
  instances (one per CPU), and provides event handlers for arrival,
  completion, quantum expiration and boosting. Each core
  runs as a SimPy process that executes jobs and handles preemption.

• **Metrics Collection:** A `StatisticsCollector` records waiting times and
  turnaround times for each job, allowing computation of average metrics at
  the end of the simulation.

The simulation uses SimPy to model time and concurrency. Jobs arrive
according to a Poisson process with rate λ, and their service demands
(CPU burst times) follow an exponential distribution with mean 1/µ. The number of CPU cores
is configurable.  Running the `simulate()` function will produce average
waiting and turnaround times alongside the number of jobs processed.

Usage example:

```
results = simulate(num_cpus=4,
                   lam=2.0, mu=1.0,
                   q1=0.05, q2=0.1,
                   s_period=1.0,
                   simulation_time=100.0,
                   seed=123)
print(results)
```

This example runs a 10-second simulation with four cores, arrival rate λ=2
jobs/sec, service rate µ=1 job/sec, Q1 time quantum of 0.05s, Q2 quantum of
0.1s, and priority boost period of 1 second. The function returns a
dictionary containing the average waiting time, average turnaround time and
total number of completed jobs.

The above instance can be modified to include I/O operations by setting the
`io_probability` and `io_rate` parameters in the `simulate()` function. This
will model jobs that may yield the CPU for I/O waits, adding realism to the
simulation.

We need to evaluate the performance of this MLFQ scheduler with many different
parameter combinations to understand its behavior under various workloads.

"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import simpy


@dataclass
class Process:
    """Represents a job in the MLFQ system.

    Attributes
    ----------
    pid: Unique process identifier.
    arrival_time: Time when the job arrives into the system.
    total_cpu_time: Total CPU time required by the job.

    current_level: Current priority level (1=Q1, 2=Q2, 3=Q3).
    cpu_left: Remaining CPU time to finish the job.
    quantum_left: Remaining time quantum in the current queue.
    time_entered_current_queue: Timestamp when the job entered its current queue.
    start_time: Timestamp of the job's first CPU execution (for response time).
    finish_time: Timestamp when the job completes (for turnaround time).
    """

    pid: int
    arrival_time: float
    total_cpu_time: float
    current_level: int = 1
    cpu_left: float = field(init=False)
    quantum_left: float = field(init=False)
    time_entered_current_queue: float = field(default=0.0)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    last_run_start: Optional[float] = None

    def __post_init__(self):
        # Initialize remaining CPU time and quantum based on queue level
        self.cpu_left = self.total_cpu_time
        self.quantum_left = float("inf")  # will be set by scheduler on scheduling


class ProcessQueue:
    """A simple queue structure for holding processes of the same priority."""

    def __init__(self):
        self.queue: List[Process] = []

    def add_to_end(self, p: Process) -> None:
        """Add a process to the end of the queue (used for normal enqueuing)."""
        self.queue.append(p)

    def add_to_head(self, p: Process) -> None:
        """Add a process to the head of the queue (used after preemption)."""
        self.queue.insert(0, p)

    def remove_from_head(self) -> Optional[Process]:
        """Remove and return the process at the head of the queue, if any."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def is_empty(self) -> bool:
        return not self.queue

    def __len__(self) -> int:
        return len(self.queue)


class StatisticsCollector:
    """Collects and computes performance metrics for the MLFQ simulation."""

    def __init__(self):
        self.wait_times: List[float] = []
        self.turnaround_times: List[float] = []
        self.boost_events: int = 0  # count of priority boosts

    def record_wait_time(self, t: float) -> None:
        self.wait_times.append(t)

    def record_turnaround_time(self, t: float) -> None:
        self.turnaround_times.append(t)

    def calculate_averages(self) -> Dict[str, float]:
        num = len(self.wait_times)
        avg_wait = sum(self.wait_times) / num if num > 0 else 0.0
        avg_turn = sum(self.turnaround_times) / num if num > 0 else 0.0
        return {
            "average_wait_time": avg_wait,
            "average_turnaround_time": avg_turn,
            "num_jobs": num,
            "boost_events": self.boost_events,
        }


class Core:
    """Represents a single CPU core that executes jobs.

    Each core runs a loop, requesting the next process from the scheduler when idle
    and executing it until completion, quantum expiry or preemption. Preemption
    is modelled using SimPy interrupts.
    """

    def __init__(self, env: simpy.Environment, scheduler: "CPUScheduler", core_id: int):
        self.env = env
        self.scheduler = scheduler
        self.core_id = core_id
        self.current_process: Optional[Process] = None
        # Start the run loop as a SimPy process
        self.process = env.process(self.run_loop())

    def run_loop(self):
        while True:
            # Request next process from scheduler
            p, quantum = self.scheduler.request_next_process(self)
            if p is None:
                # No processes available; wait a bit before polling again
                yield self.env.timeout(0.001)
                continue
            self.current_process = p
            # Set the remaining quantum for this run
            p.quantum_left = quantum
            # Record start time if first execution
            if p.start_time is None:
                p.start_time = self.env.now
                wait_time = p.start_time - p.arrival_time
                self.scheduler.stats.record_wait_time(wait_time)
            # Determine actual run time: minimum of remaining CPU and quantum
            run_time = (
                p.cpu_left if p.current_level == 3 else min(p.cpu_left, p.quantum_left)
            )
            run_time = max(0.0, run_time)
            p.last_run_start = self.env.now
            # Run and allow for preemption
            try:
                yield self.env.timeout(run_time)
            except simpy.Interrupt as interrupt:
                # Preempted by scheduler; record remaining CPU and quantum consumed
                slice_start = (
                    p.last_run_start if p.last_run_start is not None else self.env.now
                )
                consumed = max(0.0, self.env.now - slice_start)
                p.cpu_left = max(0.0, p.cpu_left - consumed)
                p.quantum_left = max(0.0, p.quantum_left - consumed)
                # Place back at head of its current queue
                self.scheduler.on_preempt(p)
                self.current_process = None
                continue
            # Normal completion of run segment
            p.cpu_left = max(0.0, p.cpu_left - run_time)
            p.quantum_left = max(0.0, p.quantum_left - run_time)
            # Determine outcome: finished, I/O yield, quantum expired or voluntary yield
            if p.cpu_left <= 0:
                # Process finished
                p.finish_time = self.env.now
                turn_time = p.finish_time - p.arrival_time
                self.scheduler.stats.record_turnaround_time(turn_time)
                self.scheduler.on_process_complete(p)
            else:
                # Check for I/O event
                if (
                    self.scheduler.io_probability > 0
                    and random.random() < self.scheduler.io_probability
                ):
                    # Start I/O; process leaves queues and will return when complete
                    self.scheduler.start_io(p)
                else:
                    # CPU still left; decide demotion or yield
                    if p.current_level < 3 and p.quantum_left <= 0:
                        # Quantum expired; demote
                        self.scheduler.on_quantum_expired(p)
                    else:
                        # Voluntary CPU yield; keep level
                        self.scheduler.on_process_yield(p)
            self.current_process = None

    def preempt(self) -> None:
        """Interrupt the currently running process (if any)."""
        if self.current_process is not None:
            self.process.interrupt("preempt")


class CPUScheduler:
    """Central scheduler managing MLFQ queues and cores."""

    def __init__(
        self,
        env: simpy.Environment,
        num_cpus: int,
        q1: float,
        q2: float,
        s_period: float,
        io_probability: float = 0.0,
        io_rate: float = 1.0,
    ):
        """Create a scheduler with optional I/O parameters.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment for the simulation.
        num_cpus : int
            Number of CPU cores.
        q1 : float
            Time quantum for Q1.
        q2 : float
            Time quantum for Q2.
        s_period : float
            Priority boost period S.
        io_probability : float, optional
            Probability that a process initiates an I/O operation after a CPU burst.
            When set to zero, the system behaves like a pure CPU scheduler with no I/O.
        io_rate : float, optional
            Rate parameter (1/mean) of the exponential distribution governing I/O wait times.
        """
        self.env = env
        self.num_cpus = num_cpus
        self.q1 = q1
        self.q2 = q2
        self.s_period = s_period
        self.io_probability = io_probability
        self.io_rate = io_rate
        # Initialize queues for Q1, Q2, Q3
        self.queues: List[ProcessQueue] = [ProcessQueue() for _ in range(3)]
        # All processes in the system (for boosting)
        self.all_processes: List[Process] = []
        # Statistics collector
        self.stats = StatisticsCollector()
        # Create cores
        self.cores: List[Core] = [Core(env, self, i) for i in range(num_cpus)]
        # Launch periodic priority booster
        env.process(self.priority_booster())

    # --- Event handlers ---
    def add_process(self, process: Process) -> None:
        """Add a newly arrived process to Q1 and track it."""
        process.current_level = 1
        self.queues[0].add_to_end(process)
        process.time_entered_current_queue = self.env.now
        self.all_processes.append(process)

    def on_preempt(self, process: Process) -> None:
        """Handle a preempted process: enqueue at head of its current queue."""
        idx = process.current_level - 1
        self.queues[idx].add_to_head(process)

    def on_process_yield(self, process: Process) -> None:
        """Handle a process that yields voluntarily. It keeps its level and goes to tail."""
        idx = process.current_level - 1
        self.queues[idx].add_to_end(process)

    def on_quantum_expired(self, process: Process) -> None:
        """Demote a process whose quantum expired and enqueue at tail of next queue."""
        if process.current_level < 3:
            process.current_level += 1
        idx = process.current_level - 1
        # Reset quantum for new level on next scheduling
        process.quantum_left = float("inf")
        self.queues[idx].add_to_end(process)

    def on_process_complete(self, process: Process) -> None:
        """Remove a finished process from tracking."""
        if process in self.all_processes:
            self.all_processes.remove(process)

    # --- I/O handling ---
    def start_io(self, process: Process) -> None:
        """Start an I/O operation for a process.

        The process leaves the ready queues and, after waiting for a random
        I/O duration drawn from an exponential distribution, returns to
        the tail of its current priority queue. While blocked on I/O, the
        process does not consume CPU time and is not counted as waiting
        for the CPU.

        Parameters
        ----------
        process : Process
            The process that initiates an I/O operation.
        """
        # Launch an asynchronous SimPy process to handle I/O completion
        io_time = random.expovariate(self.io_rate) if self.io_rate > 0 else 0.0
        self.env.process(self._io_handler(process, io_time))

    def _io_handler(self, process: Process, io_time: float):
        """SimPy process modeling the I/O wait for a process."""
        # Process is now blocked; wait for I/O to finish
        yield self.env.timeout(io_time)
        # Upon completion, return to its ready queue at the same level
        self.on_io_complete(process)

    def on_io_complete(self, process: Process) -> None:
        """Reinsert a process that has completed I/O back into the ready queue."""
        # Reset its quantum; will be set properly when scheduled
        process.quantum_left = float("inf")
        idx = process.current_level - 1
        process.time_entered_current_queue = self.env.now
        self.queues[idx].add_to_end(process)

    # --- Scheduling helpers ---
    def priority_booster(self):
        """Periodically boost all waiting processes back to Q1."""
        while True:
            yield self.env.timeout(self.s_period)
            # Move all jobs in Q2 and Q3 back to Q1
            self.stats.boost_events += 1
            for level in [1, 2]:  # indices for Q2, Q3
                queue = self.queues[level]
                while not queue.is_empty():
                    p = queue.remove_from_head()
                    if p is None:
                        break
                    p.current_level = 1
                    self.queues[0].add_to_end(p)

    def request_next_process(self, core: Core):
        """Return the highest priority process available for execution and its quantum.

        This method is invoked by cores when they finish executing a process. It scans
        Q1, Q2, then Q3 for the first available process. If none are available,
        returns (None, 0).
        """
        # Preemption check is performed in on_process_arrival when a new high‑priority job arrives
        for level, queue in enumerate(self.queues, start=1):
            if not queue.is_empty():
                p = queue.remove_from_head()
                # Determine quantum for this level
                if level == 1:
                    quantum = self.q1
                elif level == 2:
                    quantum = self.q2
                else:
                    quantum = float("inf")  # FCFS, no quantum limit
                p.quantum_left = quantum
                return p, quantum
        # No process ready
        return None, 0.0

    def preempt_if_needed(self, new_process: Process) -> None:
        """If a new job arrives in Q1 and all cores are busy, preempt the lowest priority job."""
        # Only preempt for Q1 processes
        if new_process.current_level > 1:
            return
        # Check if any core is free
        idle = [core for core in self.cores if core.current_process is None]
        if idle:
            return  # at least one core free
        # Find lowest priority running process; among those, choose the one that entered its queue last (FIFO order reversed)
        victim_core: Optional[Core] = None
        victim_level = -1
        for core in self.cores:
            p = core.current_process
            if p is None:
                continue
            if p.current_level > victim_level:
                victim_level = p.current_level
                victim_core = core
        if victim_core is not None and victim_core.current_process is not None:
            # Interrupt victim
            victim_core.preempt()


def workload_generator(
    env: simpy.Environment,
    lam: float,
    mu: float,
    scheduler: CPUScheduler,
    simulation_time: float,
):
    """Generate processes according to a Poisson arrival process with exponential service times."""
    pid = 0
    while env.now < simulation_time:
        # Time until next arrival
        inter_arrival = random.expovariate(lam)
        yield env.timeout(inter_arrival)
        arrival_time = env.now
        # Service time
        service_time = random.expovariate(mu)
        p = Process(pid=pid, arrival_time=arrival_time, total_cpu_time=service_time)
        scheduler.add_process(p)
        # Preempt if needed (new Q1 job)
        scheduler.preempt_if_needed(p)
        pid += 1


def simulate(
    num_cpus: int,
    lam: float,
    mu: float,
    q1: float,
    q2: float,
    s_period: float,
    simulation_time: float,
    seed: Optional[int] = None,
    io_probability: float = 0.0,
    io_rate: float = 1.0,
) -> Dict[str, float]:
    """Run an MLFQ simulation with optional I/O modelling.

    This function drives the discrete‑event simulation of a multi‑core CPU
    scheduled by a Multi‑Level Feedback Queue (MLFQ). In addition to the
    standard parameters controlling arrival and service rates, time quanta
    and priority boosting, optional arguments allow modelling of I/O wait
    events: each process may probabilistically yield the CPU for an I/O
    operation, during which it does not contend for CPU time and returns
    to the tail of its current queue once the I/O completes.

    Parameters
    ----------
    num_cpus : int
        Number of CPU cores in the system (c ≥ 1).
    lam : float
        Arrival rate λ of the Poisson process (jobs per second).
    mu : float
        Service rate µ of the exponential distribution (jobs per second per core).
    q1 : float
        Time quantum for the highest priority queue Q1.
    q2 : float
        Time quantum for the second priority queue Q2.
    s_period : float
        Period between priority boosts S.
    simulation_time : float
        Total simulation time (seconds).
    seed : int, optional
        Random seed for reproducibility.
    io_probability : float, optional
        Probability that a running process initiates an I/O wait after a CPU burst.
        When zero, no I/O operations occur and the model reduces to a pure
        CPU scheduler.
    io_rate : float, optional
        Rate parameter (1/mean) of the exponential distribution governing I/O
        wait times. Larger values correspond to shorter average I/O durations.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys `average_wait_time`, `average_turnaround_time`, and
        `num_jobs` indicating the number of jobs completed during the simulation.
    """
    if seed is not None:
        random.seed(seed)
    env = simpy.Environment()
    scheduler = CPUScheduler(
        env,
        num_cpus=num_cpus,
        q1=q1,
        q2=q2,
        s_period=s_period,
        io_probability=io_probability,
        io_rate=io_rate,
    )
    env.process(workload_generator(env, lam, mu, scheduler, simulation_time))
    env.run(until=simulation_time)
    return scheduler.stats.calculate_averages()


if __name__ == "__main__":
    # Example usage: run a small simulation and print results
    result = simulate(
        num_cpus=4,
        lam=2.0,
        mu=1.0,
        q1=0.05,
        q2=0.1,
        s_period=1.0,
        simulation_time=100.0,
        seed=123,
    )
    print("Simulation results:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
