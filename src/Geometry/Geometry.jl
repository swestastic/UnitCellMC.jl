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
    neighbor_table::Dict{Int,NeighborInfo}
end


function Geometry(
    unit_cell::UnitCell{D},
    lattice::Lattice{D},
    bonds::Vector{Bond{D}}
) where {D}

    neighbor_table = LatticeUtilities.build_neighbor_table(
        bonds,
        unit_cell,
        lattice
    )

    neighbor_table_map = LatticeUtilities.map_neighbor_table(
        neighbor_table
    )

    return Geometry(
        unit_cell,
        lattice,
        bonds,
        neighbor_table_map
    )
end

include("Bonds.jl")