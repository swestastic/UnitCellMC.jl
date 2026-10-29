"""
    abs_magnetization_entry(model, geometry)

Build the optional absolute-magnetization observables for a model.

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

Build the optional fourth-order Binder cumulant for a model.

The routine adds the raw observable `:magnetization_fourth` and the derived
quantity `:binder_cumulant`, computed from the jackknife samples of
`(:magnetization_squared, :magnetization_fourth)` using the standard Binder
cumulant formula, with a zero-second-moment guard to avoid division by zero.
"""
function binder_cumulant_entry(model, geometry)
    denominator = binder_denominator(model)
    # The raw :magnetization_fourth observable comes from its own registry
    # entry, so it is defined in exactly one place.
    obs, _ = optional_observables(model, geometry, Val(:magnetization_fourth))

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
    clock_order_parameter_entry(model, geometry)

Build the optional angular order parameter of a [`ClockModel`](@ref).

The raw observable `:clock_order_parameter` is the instantaneous
`cos(q φ_M)` from [`clock_angle_order`](@ref) (an `O(N)` resum of the
magnetization per measurement), where `φ_M` is the direction of the total
magnetization. Its average `⟨cos(q φ_M)⟩` tends to `1` when the
magnetization locks onto one of the `q` clock directions and to `0` when its
direction is uniformly distributed, so for `q >= 5` it separates the
low-temperature ordered phase from the intermediate critical phase, and its
rise sharpens with system size at the lower transition.

No derived quantity is added: `cos(q φ_M)` is not a sum of local terms, so
a fluctuation "susceptibility" `N * Var / T` would only reflect the trivial
variance `1/2` of a uniformly distributed angle, growing with `N` throughout
the disordered phase.

# Arguments

- `model`: The `ClockModel` being measured; supplies `q`.
- `geometry`: The lattice geometry (unused; kept for the common entry
  signature).

# Returns

An `(obs, der)` tuple: `obs` holds the `:clock_order_parameter` observable
and `der` is empty.

# Throws

`ArgumentError` if `model` is not a [`ClockModel`](@ref).
"""
function clock_order_parameter_entry(model::ClockModel, geometry)
    obs = Observable[
        Observable(:clock_order_parameter, state -> clock_angle_order(model, state)),
    ]
    return obs, DerivedObservable[]
end

clock_order_parameter_entry(model, geometry) = throw(ArgumentError(
    "The :clock_order_parameter measurement requires a ClockModel; got $(nameof(typeof(model)))"
))

"""
    magnetization_per_orbital_entry(model, geometry)

Build the optional per-orbital magnetization measurement.

The raw observable `:magnetization_per_orbital` samples, per measurement, an
`n_components × n_orbitals` matrix whose `(α, o)` entry is component `α` of
the local operator (see [`operator_components`](@ref)) averaged over the
sites of orbital `o` — the magnetization of each sublattice.

The `:connected_correlation` derived quantity needs it: the correct connected
subtraction for a class with orbitals `(oi, oj)` is `⟨S_oi⟩·⟨S_oj⟩` from this
per-orbital observable, not the lattice-averaged `⟨m⟩²`; the two differ
whenever sublattices are inequivalent (per-orbital fields, ferrimagnets,
decorated lattices).

# Returns

An `(obs, der)` tuple: `obs` holds the matrix-valued
`:magnetization_per_orbital` observable; `der` is empty.

# Throws

`ArgumentError` if `model` does not implement [`n_operator_components`](@ref).
"""
function magnetization_per_orbital_entry(model, geometry)
    n_comp = n_operator_components(model)   # fail at container construction
    n_orb = LatticeUtilities.norbits(geometry.unit_cell)
    n_cells = geometry.n_sites ÷ n_orb
    orbitals = [LatticeUtilities.site_to_orbital(site, geometry.unit_cell)
                for site in 1:geometry.n_sites]

    function sample(state)
        m = zeros(Float64, n_comp, n_orb)
        for site in 1:geometry.n_sites
            components = operator_components(model, local_operator(state, site))
            for α in 1:n_comp
                m[α, orbitals[site]] += components[α]
            end
        end
        return m ./= n_cells
    end

    obs = Observable[
        Observable(
            :magnetization_per_orbital,
            sample;
            zero = () -> zeros(Float64, n_comp, n_orb),
            template = () -> Matrix{Float64}[],
        ),
    ]
    return obs, DerivedObservable[]
end

"""
    structure_factor_entry(model, geometry)

Build the optional structure-factor observable for a model.

Precomputes the momentum grid once (via [`build_momentum_classes`](@ref))
and wraps it in a vector-valued, complex [`Observable`](@ref) named
`:structure_factor`, sampled per measurement with
[`structure_factor_instantaneous`](@ref): the orbital-resolved
``S_{ab}(\\mathbf{k})`` for every orbital pair and every momentum on the
finite-lattice grid. A [`DerivedObservable`](@ref) passes the
[`StructureFactorClass`](@ref) labels through to the analysis results as
`:structure_factor_classes`.

The measurement works for any model implementing
[`n_operator_components`](@ref) and [`operator_components`](@ref); no
model-specific code is needed here.

# Arguments

- `model`: The model being measured.
- `geometry`: The lattice geometry.

# Returns

An `(obs, der)` tuple:
- `obs::Vector{Observable}`: The `:structure_factor` observable, sampling a
  `Vector{ComplexF64}` each measurement. Its [`analyze`](@ref) error is
  complex, holding the standard errors of the real and imaginary parts.
- `der::Vector{DerivedObservable}`: Produces `:structure_factor_classes`.

# Throws

`ArgumentError` if `model` does not implement
[`n_operator_components`](@ref).
"""
function structure_factor_entry(model, geometry)
    n_operator_components(model)   # fail at container construction, not mid-run
    momenta = build_momentum_classes(geometry)
    classes = momenta.classes
    n_classes = length(classes)

    obs = Observable[
        Observable(
            :structure_factor,
            state -> structure_factor_instantaneous(model, state, momenta);
            zero     = () -> zeros(ComplexF64, n_classes),
            template = () -> Vector{ComplexF64}[],
        ),
    ]
    der = DerivedObservable[
        DerivedObservable(
            (:structure_factor_classes,),
            (jk, T, N) -> (structure_factor_classes = classes,);
            depends_on = (:structure_factor,),
        ),
    ]
    return obs, der
end

# Accepted aliases, normalized by `canonical_measurement_name` before dispatch.
const _MEASUREMENT_ALIASES = Dict{Symbol,Symbol}(
    :magnetization_abs => :abs_magnetization,
    :binder => :binder_cumulant,
)
canonical_measurement_name(name::Symbol) = get(_MEASUREMENT_ALIASES, name, name)

# Measurement names implicitly included by each optional measurement. The
# per-name `optional_observables` methods already build the implied
# observables; this drives only the construction-time notice.
measurement_implies(::Val{:connected_correlation}) =
    (:raw_correlation, :magnetization_per_orbital)
measurement_implies(::Val{:abs_magnetization}) = (:abs_magnetization_squared,)
measurement_implies(::Val{:binder_cumulant}) = (:magnetization_fourth,)
measurement_implies(::Val) = ()

"""
    optional_observables(model, geometry, ::Val{name})

Build the `(obs, der)` pair for the single optional measurement `name` (see
[`optional_observables`](@ref)`(model, geometry, measurements)`). Each
measurement has its own method — `Val(:raw_correlation)` and
`Val(:connected_correlation)` in `Correlations.jl`, and
`Val(:abs_magnetization)`, `Val(:abs_magnetization_squared)`,
`Val(:binder_cumulant)`, `Val(:magnetization_fourth)`,
`Val(:magnetization_per_orbital)`, `Val(:clock_order_parameter)`, and
`Val(:structure_factor)` here — defined next to its builder, so adding a
measurement never requires editing shared code. This fallback lists the names
accepted by `model`.

# Throws

`ArgumentError` for an unknown name.
"""
optional_observables(model, geometry, ::Val{name}) where {name} = throw(ArgumentError(
    "Unknown optional measurement :$name for $(nameof(typeof(model))); " *
    "available: $(available_measurements(model))"
))

optional_observables(model, geometry, ::Val{:abs_magnetization}) =
    abs_magnetization_entry(model, geometry)

optional_observables(model, geometry, ::Val{:abs_magnetization_squared}) = (
    Observable[Observable(
        :abs_magnetization_squared,
        state -> (abs(magnetization_value(state)) / geometry.n_sites)^2,
    )],
    DerivedObservable[],
)

optional_observables(model, geometry, ::Val{:binder_cumulant}) =
    binder_cumulant_entry(model, geometry)

optional_observables(model, geometry, ::Val{:magnetization_fourth}) = (
    Observable[Observable(
        :magnetization_fourth,
        state -> (magnetization_value(state) / geometry.n_sites)^4,
    )],
    DerivedObservable[],
)

optional_observables(model, geometry, ::Val{:magnetization_per_orbital}) =
    magnetization_per_orbital_entry(model, geometry)

optional_observables(model, geometry, ::Val{:clock_order_parameter}) =
    clock_order_parameter_entry(model, geometry)

optional_observables(model, geometry, ::Val{:structure_factor}) =
    structure_factor_entry(model, geometry)

"""
    optional_observables(model, geometry, measurements)

Resolve the optional observables/derived quantities requested by name in
`measurements` for `model` and `geometry` into concrete
[`Observable`](@ref)/[`DerivedObservable`](@ref) instances.

Each name is canonicalized (aliases such as `:binder` for
`:binder_cumulant`), validated against [`available_measurements`](@ref)`(model)`,
and built by its own `optional_observables(model, geometry, Val(name))`
method, so new measurements and new models extend the registry by adding
methods rather than editing this function. Measurements that implicitly
include others (e.g. `:connected_correlation` includes the raw correlation)
announce the additions at construction.

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

# Throws

`ArgumentError` if a name is not available for `model`.
"""
function optional_observables(model, geometry, measurements)
    requested = Symbol.(collect(measurements))
    unknown = setdiff(requested, available_measurements(model))
    isempty(unknown) || throw(ArgumentError(
        "Optional measurements $unknown are not available for " *
        "$(nameof(typeof(model))); available: $(available_measurements(model))"
    ))

    canonical = unique(canonical_measurement_name.(requested))
    implicit = setdiff(
        unique!(reduce(
            vcat, (collect(measurement_implies(Val(n))) for n in canonical);
            init = Symbol[],
        )),
        requested,
    )
    isempty(implicit) || println(
        "Additional measurements $(implicit) are included by the " *
        "requested measurements $(measurements)."
    )

    obs = Observable[]
    der = DerivedObservable[]
    for name in canonical
        o, d = optional_observables(model, geometry, Val(name))
        append!(obs, o)
        append!(der, d)
    end
    return obs, der
end

# Canonical names of the optional measurements available to every model.
const _COMMON_MEASUREMENTS = (
    :raw_correlation,
    :connected_correlation,
    :abs_magnetization,
    :abs_magnetization_squared,
    :binder_cumulant,
    :magnetization_fourth,
    :magnetization_per_orbital,
    :structure_factor,
)

"""
    extra_measurements(model)

Return a tuple of additional optional measurement names specific to a model,
beyond the common set available to every model (see
[`available_measurements`](@ref)). The default is empty; for example
`extra_measurements(::Type{<:ClockModel}) = (:clock_order_parameter,)`.

A model extending this must also provide the corresponding
`optional_observables(model, geometry, Val(name))` method, or requesting the
name fails at container construction.
"""
extra_measurements(::Type{<:AbstractModel}) = ()
extra_measurements(::Type{<:ClockModel}) = (:clock_order_parameter,)

"""
    available_measurements(model::AbstractModel)

Return a `Vector{Symbol}` listing the optional measurement names accepted by
`model` when constructing a [`MeasurementContainer`](@ref): the common
measurements every model supports, any model-specific extras (see
[`extra_measurements`](@ref)), and the accepted aliases (`:magnetization_abs`
for `:abs_magnetization`, `:binder` for `:binder_cumulant`).

Requests are validated against this list in [`optional_observables`](@ref).
Several names implicitly include others (e.g. `:connected_correlation` also
measures the raw correlation; `:binder_cumulant` also measures
`:magnetization_fourth`); the additions are announced at construction.
"""
available_measurements(::Type{M}) where {M<:AbstractModel} = Symbol[
    _COMMON_MEASUREMENTS...,
    extra_measurements(M)...,
    sort!(collect(keys(_MEASUREMENT_ALIASES)))...,
]

# Instances forward to the type method
available_measurements(::M) where {M<:AbstractModel} = available_measurements(M)
