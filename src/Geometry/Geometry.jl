import LatticeUtilities
import LatticeUtilities: UnitCell, Lattice, Bond

"""
    NeighborInfo

A named tuple type recording, for a given site, the parallel lists of bond
ids (`bonds`), the bond template each bond was generated from (`templates`),
and the neighboring site reached by each bond (`neighbors`). `bonds[k]`,
`templates[k]` and `neighbors[k]` all refer to the same bond/neighbor triple,
so the three vectors always have equal length.

`templates[k]` is `bond_id_to_template[bonds[k]]`: the id of the bond
template (in `bonds`) that generated bond `bonds[k]`. It is stored per site
as a per-bond payload index, so that inner loops can use it directly — for
example to index a model's coupling vector — instead of going through
`bond_id_to_template` on every neighbor visit.
"""
const NeighborInfo = NamedTuple{
    (:bonds, :templates, :neighbors),
    Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
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
- `neighbor_table::Vector{NeighborInfo}`: Indexed by site number, each entry
  gives the bonds, their bond templates, and the neighboring sites connected
  to that site.
- `n_sites::Int`: Total number of sites in the lattice.

Construct with [`Geometry(unit_cell, lattice, bonds)`](@ref).
"""
struct Geometry{D,T<:AbstractFloat,N}
    unit_cell::UnitCell{D,T,N}
    lattice::Lattice{D}
    bonds::Vector{Bond{D}}
    bond_id_to_template::Vector{Int}
    neighbor_table::Vector{NeighborInfo}
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
        neighbor_table = [(bonds=Int[], templates=Int[], neighbors=Int[]) for i in 1:n_sites]
        return Geometry(unit_cell, lattice, bonds, bond_id_to_template, neighbor_table, n_sites)
    end

    per_template_tables = [
        LatticeUtilities.build_neighbor_table([bond], unit_cell, lattice)
        for bond in bonds
    ]

    bond_id_to_template = reduce(
        vcat,
        [fill(i, size(nt, 2)) for (i, nt) in enumerate(per_template_tables)]
    )

    neighbor_table_matrix = reduce(hcat, per_template_tables)
    neighbor_table_dict = LatticeUtilities.map_neighbor_table(neighbor_table_matrix)
    neighbor_table = map(1:n_sites) do i
        info = get(neighbor_table_dict, i, (bonds=Int[], neighbors=Int[]))
        return (
            bonds = info.bonds,
            templates = bond_id_to_template[info.bonds],
            neighbors = info.neighbors,
        )
    end

    return Geometry(unit_cell, lattice, bonds, bond_id_to_template, neighbor_table, n_sites)
end

"""
    site_at(geometry, loc, orbital)

Return the site index of orbital `orbital` in the unit cell at location `loc`
(an `NTuple` of cell coordinates). A thin wrapper over
`LatticeUtilities.loc_to_site` so that code outside the Geometry layer does
not need that function's calling convention.

# Returns

The site index (an `Int`).
"""
site_at(geometry::Geometry, loc, orbital) =
    LatticeUtilities.loc_to_site(collect(loc), orbital, geometry.unit_cell, geometry.lattice)

"""
    _cluster_labels(geometry, clusters::Vector{Vector{Int}})

Translate an explicit partition of sites into per-site cluster labels: a
vector `labels` of length `geometry.n_sites` with `labels[i] = c` for site
`i` in `clusters[c]`, and `0` for sites absent from the partition. Entries
of `clusters` with indices above `n_sites` (ghost sites) are ignored.

Shared by the explicit-partition forms of `apply_clusters!`.
"""
function _cluster_labels(geometry, clusters::Vector{Vector{Int}})
    labels = zeros(Int, geometry.n_sites)
    for (c, cluster) in enumerate(clusters), site in cluster
        site <= geometry.n_sites && (labels[site] = c)
    end
    return labels
end

include("Displacements.jl")
include("Momenta.jl")