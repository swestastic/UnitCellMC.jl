# Simulation 

"""
    SimulationParameters(β, n_thermalization, n_steps, n_unmeasured)

Parameters controlling a Monte Carlo simulation.

## Arguments

- `β`: Inverse temperature.
- `n_thermalization`: Number of thermalization steps.
- `n_steps`: Number of Monte Carlo steps.
- `n_unmeasured`: Number of steps between measurements.
"""
struct SimulationParameters
    β::Float64
    n_thermalization::Int
    n_steps::Int
    n_unmeasured::Int
end