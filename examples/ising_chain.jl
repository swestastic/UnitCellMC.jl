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

n_sites = lu.nsites(unit_cell, lattice)

#### Initialize State
J1 = 1.0 # Coupling strength along ̂x
J = [J1]
h = 0.0 # External magnetic field strength

model = ucmc.IsingModel(J, h)
algorithm = ucmc.MetropolisAlgorithm()

state = ucmc.initialize_state(model, geometry) # Generate random spin configuration, calculate the initial energy and magnetization

#### Set simulation parameters

n_thermalization = 10_000 # warmup / thermalization sweeps before measurements
n_sweeps = 10_000 # number of sweeps (L Metropolis steps per sweep) to perform measurements
n_unmeasured = 5 # number of unmeasured sweeps between each measurement to reduce autocorrelation
n_bins = 100 # number of bins for jackknife error estimation

function MC_Sweep!(
    algorithm,
    model,
    geometry,
    state,
    container::ucmc.MeasurementContainer,
    β
)
    for _ in 1:n_sites
        ucmc.step!(algorithm, model, geometry, state, β)
    end
end

#### Perform simulation
function run_simulation(algorithm, model, geometry, state, container::ucmc.MeasurementContainer, parameters)
    β = parameters.β

    for _ in 1:parameters.n_thermalization
        MC_Sweep!(algorithm, model, geometry, state, container, β)
    end

    for _ in 1:parameters.n_sweeps
        MC_Sweep!(algorithm, model, geometry, state, container, β)
        ucmc.measure!(container, state)

        for __ in 1:parameters.n_unmeasured           
            MC_Sweep!(algorithm, model, geometry, state, container, β)
        end
    end

    return container
end

function sweep_βs(
    algorithm,
    model,
    geometry,
    state,
    n_thermalization,
    n_sweeps,
    n_unmeasured,
    n_bins,
    βs;
    simulated_annealing = true,
    measurements = Symbol[]
)

    sweep_results = Vector{Any}(undef, length(βs))

    for (i, β) in enumerate(βs)

        parameters = ucmc.SimulationParameters(β, n_thermalization, n_sweeps, n_unmeasured, n_bins)

        if !simulated_annealing
            state = ucmc.initialize_state(model, geometry)
        end

        container = ucmc.MeasurementContainer(model, geometry, n_sweeps, n_bins; measurements = measurements)
        container = run_simulation(algorithm, model, geometry, state, container, parameters)

        processed_results = ucmc.analyze(container, β, n_sites)
        sweep_results[i] = processed_results
        ucmc.save_results(model, algorithm, L, β, processed_results, parameters)
    end

    return sweep_results
end

βs = collect(0.1:0.1:1.0) # 10 temperatures from β = 0.1 to β = 1.0

sweep_results = sweep_βs(
    algorithm, model, geometry, state,
    n_thermalization, n_sweeps, n_unmeasured, n_bins, βs;
    simulated_annealing = true,
    measurements = [:correlation] # enable correlation measurements
    # measurements = []
)