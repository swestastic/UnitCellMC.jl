include("Ising.jl")
include("XY.jl")
include("Clock.jl")

"""
    bond_strength(model, geometry, bond_id)

Return the coupling strength `J` for the bond identified by `bond_id`, looked
up via the bond's template in `geometry`.

# Arguments

- `model`: The `IsingModel` supplying bond strengths.
- `geometry`: The lattice geometry mapping bond ids to templates.
- `bond_id`: Index of the bond to look up.
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