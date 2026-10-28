include("Ising.jl")
include("XY.jl")
include("Clock.jl")

"""
    bond_strength(model, geometry, bond_id)

Return the coupling strength `J` for the bond identified by `bond_id`, looked
up via the bond's template in `geometry`.

This is the checked convenience lookup, intended for setup and analysis code.
Performance-critical inner loops should instead iterate a site's
[`NeighborInfo`](@ref) and index `model.J[template]` directly: the per-site
`templates` vector is parallel to `neighbors`, so neither the bounds check
here nor the `bond_id_to_template` indirection is needed there.

# Arguments

- `model`: The model supplying one coupling per bond template.
- `geometry`: The lattice geometry mapping bond ids to templates.
- `bond_id`: Index of the bond to look up.

# Throws

`ArgumentError` if `bond_id` is out of bounds for
`geometry.bond_id_to_template`.
"""
function bond_strength(model::AbstractModel{T}, geometry::Geometry, bond_id::Int) where {T<:Real}
    templates = geometry.bond_id_to_template
    1 <= bond_id <= length(templates) || throw(ArgumentError(
        "bond_id = $bond_id is out of bounds for geometry.bond_id_to_template of length $(length(templates))"
    ))
    return model.J[templates[bond_id]]
end

function field_strength(model::AbstractModel, geometry::Geometry, site::Int)
    orbital = LatticeUtilities.site_to_orbital(site, geometry.unit_cell)
    return model.h[orbital]
end