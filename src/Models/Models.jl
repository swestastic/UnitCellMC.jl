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

Models that store couplings or fields differently override
[`couplings`](@ref)/[`fields`](@ref) and should extend this method with their
own checks.

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
    couplings(model)

Return the model's bond couplings, one per bond template of the
[`Geometry`](@ref) (see the layout contract under [`AbstractModel`](@ref)).
The default is `model.J`; a model storing couplings differently overrides
this accessor rather than every read site.

Inner loops hoist `J = couplings(model)` once and index `J[template]` with
the per-site `templates` of the neighbor table.
"""
couplings(model::AbstractModel) = model.J

"""
    fields(model)

Return the model's external field strengths, one per unit-cell orbital (see
the layout contract under [`AbstractModel`](@ref)). The default is `model.h`;
a model storing fields differently overrides this accessor.
"""
fields(model::AbstractModel) = model.h

"""
    bond_strength(model, geometry, bond_id)

Return the coupling strength `J` for the bond identified by `bond_id`, looked
up via the bond's template in `geometry` and the model's
[`couplings`](@ref).

This is the checked convenience lookup, intended for setup and analysis code.
Performance-critical inner loops should instead iterate a site's
[`NeighborInfo`](@ref) and index `couplings(model)[template]` directly: the
per-site `templates` vector is parallel to `neighbors`, so neither the bounds
check here nor the `bond_id_to_template` indirection is needed there.

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
    return couplings(model)[templates[bond_id]]
end

"""
    field_strength(model, geometry, site)

Return the external field strength acting on `site`, looked up via the site's
unit-cell orbital in [`fields`](@ref).

No per-call bounds check is needed: the orbital of a valid site is always in
range when [`validate_model_geometry`](@ref) passed at initialization.

# Arguments

- `model`: The model supplying one field per unit-cell orbital.
- `geometry`: The lattice geometry mapping sites to orbitals.
- `site`: Index of the site to look up.
"""
function field_strength(model::AbstractModel, geometry::Geometry, site::Int)
    orbital = LatticeUtilities.site_to_orbital(site, geometry.unit_cell)
    return fields(model)[orbital]
end

"""
    binder_denominator(model)

Return the denominator of the fourth-order Binder cumulant,
`1 - ⟨m⁴⟩ / (binder_denominator * ⟨m²⟩²)`, for `model`'s order parameter:
`3` for a one-component (Ising) order parameter, `2` for a two-component
(XY, or clock with `q >= 3`) one — the high-temperature Gaussian cumulant
ratio. Required by the `:binder_cumulant` optional measurement.

# Throws

`ArgumentError` for models without their own method; define
`binder_denominator(::MyModel)` to enable `:binder_cumulant`.
"""
binder_denominator(model::AbstractModel) = throw(ArgumentError(
    "binder_denominator is not implemented for $(nameof(typeof(model))); " *
    "the :binder_cumulant measurement requires it"
))