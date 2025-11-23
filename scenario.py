# scenarios.py
"""
Prepare simulated scenarios for MLFQ system.
Each scenario is a dict of parameters to pass into simulate(**params).
"""

# 1. Validation group: nearly same to M/M/c for theoretical formula comparison
scenarios_validation = [
    {
        "name": "VAL1_light_MMc",
        "num_cpus": 4,
        "lam": 1.0,  # light load
        "mu": 1.0,
        "q1": 0.05,
        "q2": 0.1,
        "s_period": 1.0,
        "simulation_time": 100.0,
        "io_probability": 0.0,
        "io_rate": 1.0,
        "seed": 11,
    },
    {
        "name": "VAL2_heavy_MMc",
        "num_cpus": 4,
        "lam": 3.6,  # heavy load, rho ~ 0.9
        "mu": 1.0,
        "q1": 0.05,
        "q2": 0.1,
        "s_period": 1.0,  # very slow boosting
        "simulation_time": 100.0,
        "io_probability": 0.0,
        "io_rate": 1.0,
        "seed": 12,
    },
]

# 2. Quantum & Booster group
scenarios_quantum_boost = [
    {
        "name": "Q1_small_quantum",
        "num_cpus": 4,
        "lam": 2.0,
        "mu": 1.0,
        "q1": 0.01,  # very small quantum
        "q2": 0.02,
        "s_period": 1.0,
        "simulation_time": 100.0,
        "io_probability": 0.0,
        "io_rate": 1.0,
        "seed": 21,
    },
    {
        "name": "Q2_large_quantum",
        "num_cpus": 4,
        "lam": 2.0,
        "mu": 1.0,
        "q1": 0.2,  # large quantum
        "q2": 0.4,
        "s_period": 1.0,
        "simulation_time": 100.0,
        "io_probability": 0.0,
        "io_rate": 1.0,
        "seed": 22,
    },
    {
        "name": "B1_fast_boost",
        "num_cpus": 4,
        "lam": 2.5,
        "mu": 1.0,
        "q1": 0.05,
        "q2": 0.1,
        "s_period": 1.0,  # fast boosting
        "simulation_time": 100.0,
        "io_probability": 0.0,
        "io_rate": 1.0,
        "seed": 23,
    },
    {
        "name": "B2_slow_boost",
        "num_cpus": 4,
        "lam": 2.5,
        "mu": 1.0,
        "q1": 0.05,
        "q2": 0.1,
        "s_period": 10.0,  # slow boosting
        "simulation_time": 100.0,
        "io_probability": 0.0,
        "io_rate": 1.0,
        "seed": 24,
    },
]

# 3. CPU-bound vs I/O-bound group
scenarios_io = [
    {
        "name": "IO1_cpu_bound",
        "num_cpus": 4,
        "lam": 2.0,
        "mu": 1.0,
        "q1": 0.05,
        "q2": 0.1,
        "s_period": 1.0,
        "simulation_time": 100.0,
        "io_probability": 0.05,  # mostly CPU-bound
        "io_rate": 1.0,
        "seed": 31,
    },
    {
        "name": "IO2_io_bound",
        "num_cpus": 4,
        "lam": 2.0,
        "mu": 1.0,
        "q1": 0.05,
        "q2": 0.1,
        "s_period": 1.0,
        "simulation_time": 100.0,
        "io_probability": 0.7,  # often goes to I/O
        "io_rate": 2.0,  # I/O relatively fast
        "seed": 32,
    },
]


# 4. CPU cores count group
def make_cores_scenarios():
    res = []
    for c in [1, 2, 4, 8]:
        res.append(
            {
                "name": f"Cores_{c}",
                "num_cpus": c,
                "lam": 2.5,
                "mu": 1.0,
                "q1": 0.05,
                "q2": 0.1,
                "s_period": 1.0,
                "simulation_time": 100.0,
                "io_probability": 0.0,
                "io_rate": 1.0,
                "seed": 40 + c,
            }
        )
    return res


scenarios_cores = make_cores_scenarios()

# Combine all for convenience
ALL_SCENARIOS = (
    scenarios_validation + scenarios_quantum_boost + scenarios_io + scenarios_cores
)
