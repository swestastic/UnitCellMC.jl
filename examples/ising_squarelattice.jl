using Revise
using UnitCellMC
using LatticeUtilities: UnitCell, Bond, Lattice, build_neighbor_table, map_neighbor_table, nsites
using Random
using Plots


# Setup lattice and geometry
Lx = 8
Ly = 8

unit_cell = UnitCell(
    lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
    basis_vecs   = [[0.0, 0.0]]
)

lattice = Lattice(
    L        = [Lx, Ly],
    periodic = [true, true]
)

model_geometry = ModelGeometry(unit_cell, lattice)

bond_px = Bond(orbitals = (1,1), displacement = [1, 0])
bond_py = Bond(orbitals = (1,1), displacement = [0, 1])

bonds = [bond_px, bond_py]

UnitCellMC.add_bond!(model_geometry, bond_px)
UnitCellMC.add_bond!(model_geometry, bond_py)

neighbor_table     = build_neighbor_table(bonds, unit_cell, lattice)
neighbor_table_map = map_neighbor_table(neighbor_table)

N_sites   = nsites(unit_cell, lattice)
N_warmup  = 1_000
N_measure = 1_000
N_bins    = 20

@assert mod(N_measure, N_bins) == 0 "N_measure must be divisible by N_bins for proper binning of measurements."

J         = 1.0

model     = IsingModel(J, Lx, Ly)
algorithm = Metropolis() # Can also swap this out for Glauber()

# β sweep
βs = collect(0.20:0.05:2.00)

measurement_container = Dict{String, Vector{Float64}}(
    "Beta"          => Float64[],
    "J"             => Float64[],
    "Energy"        => Float64[],
    "Energy_Sqrd"   => Float64[],
    "Magnetization" => Float64[],
    "Magnetization_Sqrd" => Float64[],
    "Specific_Heat" => Float64[],
    "Magnetic_Susceptibility" => Float64[],
    "Acceptance_Ratio" => Float64[],

    "Energy_Error" => Float64[],
    "Energy_Sqrd_Error" => Float64[],
    "Magnetization_Error" => Float64[],
    "Magnetization_Sqrd_Error" => Float64[],
    "Specific_Heat_Error" => Float64[],
    "Magnetic_Susceptibility_Error" => Float64[]
)


for β in βs
    global measurement_results  # make this accessible for error bars in plots

    measurement_results = Run_Simulation(
        model, algorithm, β, neighbor_table_map, N_warmup, N_measure, N_bins
    )
    push!(measurement_container["Beta"],          β)
    push!(measurement_container["J"],             J)
    push!(measurement_container["Energy"],        measurement_results["Energy"][1] / N_sites)
    push!(measurement_container["Energy_Sqrd"],   measurement_results["Energy_Sqrd"][1] / N_sites)
    push!(measurement_container["Magnetization"], abs(measurement_results["Magnetization"][1]) / N_sites)
    push!(measurement_container["Magnetization_Sqrd"], measurement_results["Magnetization_Sqrd"][1] / N_sites)
    push!(measurement_container["Specific_Heat"], measurement_results["Specific_Heat"][1] / N_sites)
    push!(measurement_container["Magnetic_Susceptibility"], measurement_results["Magnetic_Susceptibility"][1] / N_sites)
    push!(measurement_container["Acceptance_Ratio"], measurement_results["Acceptance_Ratio"][1])

    push!(measurement_container["Energy_Error"], measurement_results["Energy"][2] / N_sites)
    push!(measurement_container["Energy_Sqrd_Error"], measurement_results["Energy_Sqrd"][2] / N_sites)
    push!(measurement_container["Magnetization_Error"], measurement_results["Magnetization"][2] / N_sites)
    push!(measurement_container["Magnetization_Sqrd_Error"], measurement_results["Magnetization_Sqrd"][2] / N_sites)
    push!(measurement_container["Specific_Heat_Error"], measurement_results["Specific_Heat"][2] / N_sites)
    push!(measurement_container["Magnetic_Susceptibility_Error"], measurement_results["Magnetic_Susceptibility"][2] / N_sites)
end

    print(measurement_container)

# Plots
p1 = plot(
    1 ./measurement_container["Beta"],
    measurement_container["Energy"],
    yerror = measurement_container["Energy_Error"],  # error bars for energy
    xlabel = "T",
    ylabel = "⟨E⟩ / site",
    title  = "Ising Model — Energy vs T  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

p2 = plot(
    1 ./measurement_container["Beta"],
    measurement_container["Magnetization"],
    yerror = measurement_container["Magnetization_Error"],  # error bars for magnetization
    xlabel = "β",
    ylabel = "⟨|M|⟩ / site",
    title  = "Ising Model — Magnetization vs β  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

p3 = plot(
    1 ./measurement_container["Beta"],
    measurement_container["Specific_Heat"],
    yerror = measurement_container["Specific_Heat_Error"],  # error bars for specific heat
    xlabel = "T",
    ylabel = "⟨C⟩ / site",
    title  = "Ising Model — Specific Heat vs T  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

p4 = plot(
    1 ./measurement_container["Beta"],
    measurement_container["Magnetic_Susceptibility"],
    yerror = measurement_container["Magnetic_Susceptibility_Error"],  # error bars for magnetic susceptibility
    xlabel = "β",
    ylabel = "⟨χ⟩ / site",
    title  = "Ising Model — Magnetic Susceptibility vs β  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

p5 = plot(
    1 ./measurement_container["Beta"],
    measurement_container["Acceptance_Ratio"],
    # yerror = measurement_container["Magnetic_Susceptibility_Error"],  # error bars for magnetic susceptibility
    xlabel = "β",
    ylabel = "⟨χ⟩ / site",
    title  = "Ising Model — Acceptance Ratio vs T  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

pLattice = UnitCellMC.plot_lattice(
    unit_cell,
    lattice,
    neighbor_table_map,
    bonds
)

png(p1, "plots/ising_energy.png")
png(p2, "plots/ising_magnetization.png")
png(p3, "plots/ising_specific_heat.png")
png(p4, "plots/ising_magnetic_susceptibility.png")
png(p5, "plots/ising_acceptance_ratio.png")
png(pLattice, "plots/lattice.png")