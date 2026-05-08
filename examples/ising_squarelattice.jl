using Revise
using UnitCellMC
using LatticeUtilities: UnitCell, Bond, Lattice, build_neighbor_table, map_neighbor_table
using Random
using Plots


# Setup lattice and geometry
Lx = 4
Ly = 4

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

UnitCellMC.add_bond!(model_geometry, bond_px)
UnitCellMC.add_bond!(model_geometry, bond_py)

neighbor_table     = build_neighbor_table([bond_px, bond_py], unit_cell, lattice)
neighbor_table_map = map_neighbor_table(neighbor_table)

N_sites   = length(neighbor_table_map)
N_warmup  = 5000
N_measure = 5000
J         = 1.0

model     = IsingModel(J, Lx, Ly)
algorithm = Metropolis() # Can also swap this out for Glauber()

# β sweep
βs = collect(0.1:0.1:1.5)

measurement_container = Dict{String, Vector{Float64}}(
    "Beta"          => Float64[],
    "J"             => Float64[],
    "Energy"        => Float64[],
    "Magnetization" => Float64[]
)

for β in βs
    mean_E, mean_M = Run_Simulation(
        model, algorithm, β, neighbor_table_map, N_warmup, N_measure
    )
    push!(measurement_container["Beta"],          β)
    push!(measurement_container["J"],             J)
    push!(measurement_container["Energy"],        mean_E / N_sites)
    push!(measurement_container["Magnetization"], abs(mean_M) / N_sites)
end

# Plots
p1 = plot(
    measurement_container["Beta"],
    measurement_container["Energy"],
    xlabel = "β",
    ylabel = "⟨E⟩ / site",
    title  = "Ising Model — Energy vs β  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

p2 = plot(
    measurement_container["Beta"],
    measurement_container["Magnetization"],
    xlabel = "β",
    ylabel = "⟨|M|⟩ / site",
    title  = "Ising Model — Magnetization vs β  ($(Lx)×$(Ly))",
    lw     = 2,
    marker = :circle,
    legend = false
)

png(p1, "ising_energy.png")     # explicitly save energy-only plot
png(p2, "ising_magnetization.png")
