"""
This example code implements the 2D Ising model using the UnitCellMC.jl package. 
It demonstrates how to set up the geometry, initialize the state, and perform 
Metropolis Monte Carlo steps to simulate the system's evolution.
"""

#### Import the necessary modules

import UnitCellMC as ucmc
import LatticeUtilities as lu

#### Initialize Geometry

L = [8] # In this case this is [Lx, Ly] becuase the lattice_vectors are [̂x, ̂y]

lattice_vectors = [[1.0]]
basis_vectors = [[0.0]] # our unit cell is a single site at the origin

unit_cell = lu.UnitCell( 
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true] # periodic boundary conditions along ̂x
)

bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1]) # bond along ̂x
bonds = [bond_1]

geometry = ucmc.Geometry(unit_cell, lattice, bonds)

n_sites = geometry.n_sites

#### Initialize State
J1 = 1.0 # Coupling strength along ̂x
J = [J1]
h = 0.0 # External magnetic field strength

model = ucmc.IsingModel(J, h)
algorithm = ucmc.MetropolisAlgorithm()

state = ucmc.initialize_state(model, geometry) # Generate random spin configuration, calculate the initial energy and magnetization

#### Set simulation parameters

n_thermalization = 10_000 # warmup / thermalization sweeps before measurements
n_measurements = 10_000 # number of sweeps (L Metropolis steps per sweep) to perform measurements
n_unmeasured = 5 # number of unmeasured sweeps between each measurement to reduce autocorrelation
n_bins = 100 # number of bins for jackknife error estimation

function MC_Sweep!(
    algorithm,
    model,
    geometry,
    state,
    container::ucmc.MeasurementContainer,
    T
)
    for _ in 1:n_sites
        ucmc.step!(algorithm, model, geometry, state, T)
    end
end

#### Perform simulation
function run_simulation(algorithm, model, geometry, state, container::ucmc.MeasurementContainer, parameters)
    T = parameters.T

    for _ in 1:parameters.n_thermalization
        MC_Sweep!(algorithm, model, geometry, state, container, T)
    end

    for _ in 1:parameters.n_measurements
        MC_Sweep!(algorithm, model, geometry, state, container, T)
        ucmc.measure!(container, state)

        for __ in 1:parameters.n_unmeasured           
            MC_Sweep!(algorithm, model, geometry, state, container, T)
        end
    end

    return container
end

function sweep_Ts(
    algorithm,
    model,
    geometry,
    state,
    n_thermalization,
    n_measurements,
    n_unmeasured,
    n_bins,
    Ts;
    simulated_annealing = true,
    measurements = Symbol[]
)

    sweep_results = Vector{Any}(undef, length(Ts))
    n_sites = geometry.n_sites
    L = geometry.lattice.L

    for (i, T) in enumerate(Ts)

        parameters = ucmc.SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)

        if !simulated_annealing
            state = ucmc.initialize_state(model, geometry)
        end

        container = ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins; measurements = measurements)
        container = run_simulation(algorithm, model, geometry, state, container, parameters)

        processed_results = ucmc.analyze(container, T, n_sites)
        sweep_results[i] = processed_results
        # ucmc.save_results(model, algorithm, L, T, processed_results, parameters) # Use this to save results for each T separately, if desired
    end

    ucmc.save_sweep_results(
        sweep_results, Ts, model, algorithm, L,
        n_thermalization, n_measurements, n_unmeasured, n_bins
    )

    return sweep_results
end

Ts = collect(0.1:0.1:1.0) # 10 temperatures from T = 0.1 to T = 1.0

sweep_results = sweep_Ts(
    algorithm, model, geometry, state,
    n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
    simulated_annealing = true,
    measurements = [:correlation] # enable correlation measurements
    # measurements = []
)