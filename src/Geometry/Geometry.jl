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
template (in `bonds`) that generated bond `bonds[k]`. It is stored per site so
inner loops can index a model's coupling vector directly instead of calling
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
- `bond_sources::Vector{Int}`: The source site of each concrete bond (the
  site the bond's template displacement points away from); the other endpoint
  is reachable through the neighbor table. Used by measurements that need
  oriented bonds (see [`bond_displacement`](@ref)).
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
    bond_sources::Vector{Int}
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
across all templates per site. If `bonds` is empty — or every template
generates zero concrete bonds, e.g. on an open lattice smaller than every
displacement — every site gets an empty `NeighborInfo` and
`bond_id_to_template` is empty.

# Arguments

- `unit_cell`: The unit cell defining orbitals and local geometry.
- `lattice`: The lattice defining system size and boundary conditions.
- `bonds`: Bond templates to expand into the full neighbor table.

# Throws

`ArgumentError` if a bond template connects a site to itself (equal orbitals,
zero displacement). A self-bond's energy is a constant offset for
fixed-length spins, and every update path skips self-bonds
(`neighbor == site`) while from-scratch energy sums would include them, so
they are rejected here instead of being silently inconsistent.

# Returns

A `Geometry{D,T,N}` with `n_sites` set from `unit_cell` and `lattice`.
"""
function Geometry(
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    bonds::Vector{Bond{D}}
) where {D,T}

    n_sites = LatticeUtilities.nsites(unit_cell, lattice)

    for bond in bonds
        (bond.orbitals[1] == bond.orbitals[2] && iszero(bond.displacement)) &&
            throw(ArgumentError(
                "Self-bond template (orbitals $(bond.orbitals), zero displacement): a bond " *
                "from a site to itself is a constant energy offset and is not supported"
            ))
    end

    if isempty(bonds)
        bond_id_to_template = Int[]
        bond_sources = Int[]
        neighbor_table = [(bonds=Int[], templates=Int[], neighbors=Int[]) for i in 1:n_sites]
        return Geometry(unit_cell, lattice, bonds, bond_id_to_template, bond_sources,
                        neighbor_table, n_sites)
    end

    per_template_tables = [
        LatticeUtilities.build_neighbor_table([bond], unit_cell, lattice)
        for bond in bonds
    ]

    bond_id_to_template = reduce(
        vcat,
        [fill(i, size(nt, 2)) for (i, nt) in enumerate(per_template_tables)]
    )

    # Row 1 of a LatticeUtilities neighbor table is the bond's source site
    # (the site the template's displacement points away from).
    bond_sources = [nt[1, c] for nt in per_template_tables for c in axes(nt, 2)]

    # Every template may fall outside the lattice (e.g. an open lattice too
    # small for any displacement); that degenerates to the empty-bonds case.
    if isempty(bond_id_to_template)
        neighbor_table = [(bonds=Int[], templates=Int[], neighbors=Int[]) for i in 1:n_sites]
        return Geometry(unit_cell, lattice, bonds, bond_id_to_template, bond_sources,
                        neighbor_table, n_sites)
    end

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

    return Geometry(unit_cell, lattice, bonds, bond_id_to_template, bond_sources,
                    neighbor_table, n_sites)
end

"""
    site_at(geometry, loc, orbital)

Return the site index of orbital `orbital` in the unit cell at location `loc`
(an `NTuple` of cell coordinates). A thin wrapper over
`LatticeUtilities.loc_to_site` so that code outside the Geometry layer does
not need that function's calling convention.

# Returns

The site index (an `Int`), or `0` if `loc` is invalid under the lattice's
boundary conditions.
"""
site_at(geometry::Geometry, loc, orbital) =
    LatticeUtilities.loc_to_site(loc, orbital, geometry.unit_cell, geometry.lattice)

"""
    bond_displacement(geometry, template)

Return the real-space displacement vector of bond template `template` (an
index into `geometry.bonds`), from its source orbital to its target orbital.
A thin wrapper over `LatticeUtilities.bond_to_vec`, so measurement code does
not need to call LatticeUtilities conversion routines directly (see
[`site_at`](@ref)). The orientation matches [`bond_sources`](@ref Geometry):
the vector points from `bond_sources[b]` to the other endpoint of each
concrete bond `b` generated by the template.

# Returns

The displacement vector (an `SVector`/`Vector` of length `D`, as returned by
`LatticeUtilities.bond_to_vec`).
"""
bond_displacement(geometry::Geometry, template::Int) =
    LatticeUtilities.bond_to_vec(geometry.bonds[template], geometry.unit_cell)

"""
    geometry_tag(geometry)

Return a short string identifying the geometry's connectivity — its unit
cell, bond templates, and per-dimension boundary conditions, but not its
size — as eight hexadecimal digits. Results directory names include it (see
[`results_dirname`](@ref)) so that two different geometries of the same size
(say a square and a triangular lattice with the same `L`) never share an
output directory. The tag is stable across sessions for the same defining
data, but may change across Julia versions.
"""
function geometry_tag(geometry::Geometry)
    data = (geometry.unit_cell, geometry.bonds, geometry.lattice.periodic)
    return lpad(string(hash(repr(data)) & 0xffffffff, base = 16), 8, '0')
end

include("Displacements.jl")
include("Momenta.jl")