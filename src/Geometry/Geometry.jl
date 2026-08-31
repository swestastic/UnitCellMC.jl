import LatticeUtilities
import LatticeUtilities: UnitCell, Lattice, Bond

const NeighborInfo = NamedTuple{
    (:bonds, :neighbors),
    Tuple{Vector{Int}, Vector{Int}}
}

struct Geometry{D,T<:AbstractFloat,N}
    unit_cell::UnitCell{D,T,N}
    lattice::Lattice{D}
    bonds::Vector{Bond{D}}
    bond_id_to_template::Vector{Int}
    neighbor_table::Dict{Int,NeighborInfo}
end

function Geometry(
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    bonds::Vector{Bond{D}}
) where {D,T}

    per_template_tables = [
        LatticeUtilities.build_neighbor_table([bond], unit_cell, lattice)
        for bond in bonds
    ]

    bond_id_to_template = reduce(
        vcat,
        [fill(i, size(nt, 2)) for (i, nt) in enumerate(per_template_tables)]
    )

    neighbor_table = reduce(hcat, per_template_tables)
    neighbor_table_map = LatticeUtilities.map_neighbor_table(neighbor_table)

    return Geometry(unit_cell, lattice, bonds, bond_id_to_template, neighbor_table_map)
end

include("Bonds.jl")