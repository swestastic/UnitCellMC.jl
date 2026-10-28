include("Ising.jl")
include("XY.jl")
include("Clock.jl")

"""
    validate_model_geometry(model, geometry)

Validate the compatibility between a model's coupling/field data and the
lattice geometry. The generic method enforces the package-wide layout
conventions for models that carry them: a bond-coupling vector (`model.J`,
or `model.couplings`) must have one entry per bond template in
`geometry.bonds`, every entry of `geometry.bond_id_to_template` must refer
to a valid coupling, and a field vector `model.h` must have one entry per
unit-cell orbital. Models with `J` and `h` fields ([`IsingModel`](@ref),
[`XYModel`](@ref), [`ClockModel`](@ref)) need no method of their own.

Models that store couplings or fields differently should extend this method
with their own checks.

# Throws

`ArgumentError` naming the mismatched lengths or out-of-bounds template id.

# Returns

`nothing`.
"""
function validate_model_geometry(model::AbstractModel, geometry::Geometry)
    coupling_names = (:J, :couplings)
    coupling_field = findfirst(name -> hasproperty(model, name), coupling_names)
    if coupling_field !== nothing
        couplings = getproperty(model, coupling_names[coupling_field])
        length(couplings) == length(geometry.bonds) || throw(ArgumentError(
            "length(model.$(coupling_names[coupling_field])) = $(length(couplings)) does not match length(geometry.bonds) = $(length(geometry.bonds))"
        ))
        for (bond_id, template_id) in enumerate(geometry.bond_id_to_template)
            1 <= template_id <= length(couplings) || throw(ArgumentError(
                "bond_id_to_template[$bond_id] = $template_id is out of bounds for model.$(coupling_names[coupling_field]) of length $(length(couplings))"
            ))
        end
    end

    if hasproperty(model, :h)
        n_orbitals = LatticeUtilities.norbits(geometry.unit_cell)
        length(model.h) == n_orbitals || throw(ArgumentError(
            "length(model.h) = $(length(model.h)) does not match the number of sites in the unit cell = $n_orbitals"
        ))
    end
    return nothing
end

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