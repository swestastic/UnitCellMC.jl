"""
This example code implements the 2D Ising model using the UnitCellMC.jl package. 
It demonstrates how to set up the geometry, initialize the state, and perform 
Metropolis Monte Carlo steps to simulate the system's evolution.
"""

import UnitCellMC as ucmc
import LatticeUtilities as lu

#### Initialize Geometry

L = [10,10]

lattice_vectors = [[1.0, 0.0], [0.0, 1.0]]
basis_vectors = [[0.0, 0.0]]

unit_cell = lu.UnitCell(
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true, true]
)

bond_1 = lu.Bond(orbitals = (1, 1), displacement = [0, 1])
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [-1, 0])

J1 = 1.0
J2 = 1.0

bonds = [bond_1, bond_2]
geometry = ucmc.Geometry(unit_cell, lattice, bonds)

n_sites = lu.nsites(unit_cell, lattice)

#### Initialize State
J1 = 1.0
J2 = 1.0
J = [J1, J2]
h = 0.0

model = ucmc.IsingModel(J, h)
algorithm = ucmc.Metropolis()

state = ucmc.initialize_state(model, geometry)

#### Set simulation parameters

# β = 0.5
n_thermalization = 10_000
n_steps = 10_000
n_unmeasured = 5
n_bins = 100

#### Perform simulation
function run_simulation(algorithm, model, geometry, state, parameters)
    
    β = parameters.β

    measurements = ucmc.MeasurementContainer(model, parameters.n_steps, parameters.n_bins)

    # thermalization steps
    for _ in 1:parameters.n_thermalization
        ucmc.step!(algorithm, model, geometry, state, β)
    end

    for _ in 1:parameters.n_steps
        ucmc.step!(algorithm, model, geometry, state, β)

        ucmc.measure!(measurements, model, state)

        for __ in 1:parameters.n_unmeasured
            ucmc.step!(algorithm, model, geometry, state, β)
        end

    end

    return measurements
end

function sweep_βs(
    algorithm, 
    model, 
    geometry, 
    state, 
    n_thermalization, 
    n_steps, 
    n_unmeasured, 
    n_bins, 
    βs;
    simulated_annealing = true
    )

    sweep_results = Vector{Any}(undef, length(βs))

    for (i, β) in enumerate(βs)

        parameters = ucmc.SimulationParameters(
            β,
            n_thermalization,
            n_steps,
            n_unmeasured,
            n_bins
        )

        if !simulated_annealing 
            state = ucmc.initialize_state(model, geometry)
        end

        results = run_simulation(
            algorithm,
            model,
            geometry,
            state,
            parameters
        )

        processed_results = ucmc.analyze(results, model, β, n_sites)

        sweep_results[i] = processed_results
    end

    return sweep_results
end

βs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

sweep_results = sweep_βs(
    algorithm, 
    model, 
    geometry, 
    state, 
    n_thermalization, 
    n_steps, 
    n_unmeasured, 
    n_bins, 
    βs;
    simulated_annealing = true
)


#### Plotting

primary_avg(results, name::Symbol) = [r.primary[name][1] for r in results]
primary_err(results, name::Symbol) = [r.primary[name][2] for r in results]

derived_avg(results, name::Symbol) = [getproperty(r.derived, name) for r in results]
derived_err(results, err_name::Symbol) = [getproperty(r.derived, err_name) for r in results]

energies = primary_avg(sweep_results, :energy)
energy_errors = primary_err(sweep_results, :energy)

magnetizations = primary_avg(sweep_results, :magnetization)
magnetization_errors = primary_err(sweep_results, :magnetization)

specific_heats = derived_avg(sweep_results, :specific_heat)
specific_heat_errors = derived_err(sweep_results, :specific_heat_err)

susceptibilities = derived_avg(sweep_results, :susceptibility)
susceptibility_errors = derived_err(sweep_results, :susceptibility_err)

using Plots
energyPlot = plot(
    1.0 ./ βs,
    energies / n_sites,
    yerror = energy_errors / n_sites,
    xlabel = "Temperature (T)",
    ylabel = "Average Energy per Site",
    title = "Average Energy vs Temperature",
    legend = false
)

magnetizationPlot = plot(
    1.0 ./ βs,
    magnetizations / n_sites,
    yerror = magnetization_errors / n_sites,
    xlabel = "Temperature (T)",
    ylabel = "Average Magnetization per Site",
    title = "Average Magnetization vs Temperature",
    legend = false
)

susceptibilityPlot = plot(
    1.0 ./ βs,
    susceptibilities / n_sites,
    yerror = susceptibility_errors / n_sites,
    xlabel = "Temperature (T)",
    ylabel = "⟨ χ ⟩ / N",
    title = "Susceptibility vs Temperature",
    legend = false
)

specificHeatPlot = plot(
    1.0 ./ βs,
    specific_heats / n_sites,
    yerror = specific_heat_errors / n_sites,
    xlabel = "Temperature (T)",
    ylabel = "⟨ C_v ⟩ / N",
    title = "Specific Heat vs Temperature",
    legend = false
)

display(energyPlot)
display(magnetizationPlot)
display(susceptibilityPlot)
display(specificHeatPlot)
