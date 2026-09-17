"""
    optional_observables(model, geometry, measurements)

Resolve the optional observable/derived-quantity entries requested by name
in `measurements` into concrete [`Observable`](@ref)/[`DerivedObservable`](@ref)
instances for `model` and `geometry`.

This is where opt-in measurements (those not always computed for every
model) are wired up. Currently only `:correlation` is supported, which adds
the entries built by [`correlation_entry`](@ref). New optional observables
(e.g. a future `:structure_factor`) are added by extending this function
with another `if ... in measurements` branch.

# Arguments

- `model`: The model being measured.
- `geometry`: The lattice geometry.
- `measurements`: Collection of `Symbol` names indicating which optional
  observables/derived quantities to include.

# Returns

An `(obs, der)` tuple:
- `obs::Vector{Observable}`: Raw observables for the requested optional measurements.
- `der::Vector{DerivedObservable}`: Derived quantities for the requested optional measurements.

Both are empty if `measurements` requests none of the currently supported
optional observables.
"""
function optional_observables(model, geometry, measurements)
    obs = Observable[]
    der = DerivedObservable[]

    if :correlation in measurements
        o, d = correlation_entry(model, geometry)
        push!(obs, o)
        push!(der, d)
    end
    # future: :structure_factor in measurements && (push! obs/der similarly)

    return obs, der
end