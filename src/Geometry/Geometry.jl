import LatticeUtilities
import LatticeUtilities: UnitCell, Lattice, Bond

"""
    NeighborInfo

A named tuple type recording, for a given site, the parallel lists of bond
ids (`bonds`) and the neighboring site reached by each bond (`neighbors`).
`bonds[k]` and `neighbors[k]` refer to the same bond/neighbor pair, so the
two vectors always have equal length.
"""
const NeighborInfo = NamedTuple{
    (:bonds, :neighbors),
    Tuple{Vector{Int}, Vector{Int}}
}

"""
    Geometry{D,T<:AbstractFloat,N}

The lattice geometry for a `D`-dimensional model: unit cell, lattice, bonds,
and the neighbor bookkeeping derived from them.

# Fields

- `unit_cell::UnitCell{D,T,N}`: The unit cell (orbital positions, basis vectors).
- `lattice::Lattice{D}`: The lattice (system size, boundary conditions).
- `bonds::Vector{Bond{D}}`: Bond templates defining the model's connectivity.
- `bond_id_to_template::Vector{Int}`: Maps each concrete bond id to the index
  of the bond template (in `bonds`) it was generated from.
- `neighbor_table::Dict{Int,NeighborInfo}`: Maps each site to the bonds and
  neighboring sites connected to it.
- `n_sites::Int`: Total number of sites in the lattice.

Construct with [`Geometry(unit_cell, lattice, bonds)`](@ref).
"""
struct Geometry{D,T<:AbstractFloat,N}
    unit_cell::UnitCell{D,T,N}
    lattice::Lattice{D}
    bonds::Vector{Bond{D}}
    bond_id_to_template::Vector{Int}
    neighbor_table::Dict{Int,NeighborInfo}
    n_sites::Int
end

"""
    Geometry(unit_cell, lattice, bonds)

Build a [`Geometry`](@ref) by expanding each bond template in `bonds` into
concrete bonds and neighbors over `unit_cell` and `lattice`.

For each bond template, a neighbor table is built independently and the
results are concatenated, so `bond_id_to_template` records which template
each resulting bond came from and `neighbor_table` aggregates neighbors
across all templates per site. If `bonds` is empty, every site gets an empty
`NeighborInfo` and `bond_id_to_template` is empty.

# Arguments

- `unit_cell`: The unit cell defining orbitals and local geometry.
- `lattice`: The lattice defining system size and boundary conditions.
- `bonds`: Bond templates to expand into the full neighbor table.

# Returns

A `Geometry{D,T,N}` with `n_sites` set from `unit_cell` and `lattice`.
"""
function Geometry(
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    bonds::Vector{Bond{D}}
) where {D,T}

    n_sites = LatticeUtilities.nsites(unit_cell, lattice)

    if isempty(bonds)
        bond_id_to_template = Int[]
        neighbor_table_map = Dict(i => (bonds=Int[], neighbors=Int[]) for i in 1:n_sites)
        return Geometry(unit_cell, lattice, bonds, bond_id_to_template, neighbor_table_map, n_sites)
    end

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

    return Geometry(unit_cell, lattice, bonds, bond_id_to_template, neighbor_table_map, n_sites)
end

"""
    validate_model_geometry(model::AbstractModel, geometry::Geometry)

Validate the basic compatibility between a model's bond/coupling data and the
lattice geometry. This is intentionally generic: models that carry a vector of
bond couplings (for example `J`) must have the same length as the geometry's
bond templates, and every concrete bond index in `geometry.bond_id_to_template`
must refer to a valid entry in that coupling vector.

Models that do not provide a bond-coupling vector can extend this method to add
more specialized checks.
"""
function validate_model_geometry(model::AbstractModel, geometry::Geometry)
    coupling_names = (:J, :couplings)
    coupling_field = findfirst(name -> hasproperty(model, name), coupling_names)
    coupling_field === nothing && return nothing

    couplings = getproperty(model, coupling_names[coupling_field])
    length(couplings) == length(geometry.bonds) || throw(ArgumentError(
        "length(model.$(coupling_names[coupling_field])) = $(length(couplings)) does not match length(geometry.bonds) = $(length(geometry.bonds))"
    ))

    for (bond_id, template_id) in enumerate(geometry.bond_id_to_template)
        1 <= template_id <= length(couplings) || throw(ArgumentError(
            "bond_id_to_template[$bond_id] = $template_id is out of bounds for model.$(coupling_names[coupling_field]) of length $(length(couplings))"
        ))
    end
    return nothing
end

include("Displacements.jl")