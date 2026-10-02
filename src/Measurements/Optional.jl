"""
    magnetization_value(state)

Return the scalar magnetization represented by `state`, using the stored total
magnetization for an [`IsingState`](@ref) and the magnitude of the two
components for an [`XYState`](@ref).
"""
magnetization_value(state::IsingState) = state.magnetization
magnetization_value(state::XYState) = hypot(state.magnetization_x, state.magnetization_y)
binder_denominator(::IsingModel) = 3.0
binder_denominator(::XYModel) = 2.0

"""
    abs_magnetization_entry(model, geometry)

Build the optional absolute-magnetization observables for an Ising model.

The resulting raw observables are `:abs_magnetization` and
`:abs_magnetization_squared`, which are useful when the sign-symmetric
magnetization distribution would obscure the finite-size scaling signal.
"""
function abs_magnetization_entry(model, geometry)
    n_sites = geometry.n_sites
    obs = Observable[
        Observable(:abs_magnetization, state -> abs(magnetization_value(state)) / n_sites),
        Observable(:abs_magnetization_squared, state -> (abs(magnetization_value(state)) / n_sites)^2),
    ]
    return obs, DerivedObservable[]
end

"""
    binder_cumulant_entry(model, geometry)

Build the optional fourth-order Binder cumulant for an Ising model.

The routine adds the raw observable `:magnetization_fourth` and the derived
quantity `:binder_cumulant`, computed from the jackknife samples of
`(:magnetization_squared, :magnetization_fourth)` using the standard Binder
cumulant formula, with a zero-second-moment guard to avoid division by zero.
"""
function binder_cumulant_entry(model, geometry)
    n_sites = geometry.n_sites
    denominator = binder_denominator(model)
    obs = Observable[
        Observable(:magnetization_fourth, state -> (magnetization_value(state) / n_sites)^4),
    ]

    compute = function (jk, T, N)
        m2 = jk[:magnetization_squared]
        m4 = jk[:magnetization_fourth]
        value, err = jackknife_stats(
            (m2_i, m4_i) -> begin
                if m2_i == 0.0
                    return 0.0
                end
                return 1.0 - m4_i / (denominator * m2_i^2)
            end,
            m2,
            m4,
        )
        return (binder_cumulant = value, binder_cumulant_err = err)
    end

    der = DerivedObservable[
        DerivedObservable(
            (:binder_cumulant, :binder_cumulant_err),
            compute;
            depends_on = (:magnetization_squared, :magnetization_fourth),
        ),
    ]
    return obs, der
end

"""
    optional_observables(model, geometry, measurements)

Resolve the optional observable/derived-quantity entries requested by name
in `measurements` into concrete [`Observable`](@ref)/[`DerivedObservable`](@ref)
instances for `model` and `geometry`.

Model-specific observables stay in the `measurements` list. Algorithm-specific
summary statistics remain in the `diagnostics` list (see
[`MeasurementContainer`](@ref)).

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

    raw_correlation_requested =
        :raw_correlation in measurements || :connected_correlation in measurements
    if raw_correlation_requested
        o, d = correlation_entry(model, geometry)
        push!(obs, o)
        :connected_correlation in measurements && push!(der, d)
    end
    abs_magnetization_requested =
        :abs_magnetization in measurements || :magnetization_abs in measurements
    if abs_magnetization_requested
        o, d = abs_magnetization_entry(model, geometry)
        append!(obs, o)
        append!(der, d)
    end
    binder_requested = :binder_cumulant in measurements || :binder in measurements
    if binder_requested
        o, d = binder_cumulant_entry(model, geometry)
        append!(obs, o)
        append!(der, d)
    end
    if :magnetization_fourth in measurements && !binder_requested
        o = [Observable(:magnetization_fourth, state -> (magnetization_value(state) / geometry.n_sites)^4)]
        append!(obs, o)
    end
    # future: :structure_factor in measurements && (push! obs/der similarly)

    return obs, der
end