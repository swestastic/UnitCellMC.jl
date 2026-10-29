"""
    MeasurementContainer{D, N, T, S, Ts, O}

Accumulates binned Monte Carlo measurements and holds the observable
specifications needed to analyze them.

The type parameters `D` (diagnostic names) and `N` (raw observable names) are
tuples of `Symbol`s, and `O` is the concretely typed tuple of
[`Observable`](@ref)s, so that [`measure!`](@ref) is type-stable: observable
values are sampled, named, and accumulated without dynamic dispatch.

Raw measurements are accumulated into `bin_sums` as they're taken and
flushed into `data` once every `bin_size` measurements. `bin_count` tracks
the measurements accumulated in the current bin (see the
accumulation/flushing functions that operate on this container elsewhere in
the codebase).

# Fields

- `data::T`: Per-observable completed bin values (e.g. a `NamedTuple` mapping
  observable name to a vector of per-bin averages), accumulated by
  [`measure!`](@ref) and consumed by [`analyze`](@ref).
- `bin_sums::S`: Per-observable running sums for the bin currently being
  accumulated, reset each time a bin is flushed into `data`.
- `time_series::Ts`: Per-observable unbinned history of every measurement,
  kept for autocorrelation analysis in [`analyze`](@ref). Successive entries
  are `n_unmeasured + 1` sweeps apart. Only the observables selected by the
  constructor's `time_series` keyword are recorded (by default the scalar
  ones), so its keys are a subset of those of `data`.
- `chain_lengths::Vector{Int}`: Lengths of the independent Markov chains
  concatenated in `time_series` when containers are combined with
  [`merge_containers`](@ref); empty for a single run.
- `bin_count::Int`: Number of measurements accumulated in the current bin;
  reset to `0` when the bin is flushed into `data`.
- `bin_size::Int`: Number of measurements per bin (`n_measurements ÷ n_bins`).
- `observables::O`: The raw observables being measured, stored as a
  concretely typed tuple whose names and order match the leading fields of
  `bin_sums` (the trailing fields are the diagnostics named by `D`).
- `derived::Vector{DerivedObservable}`: The derived quantities to compute
  from the raw observables once binning is complete.

Construct with [`MeasurementContainer(model, geometry, n_measurements,
n_bins; measurements)`](@ref).
"""
mutable struct MeasurementContainer{D, N, T, S, Ts, O}
    data::T
    bin_sums::S
    time_series::Ts
    chain_lengths::Vector{Int}
    bin_count::Int
    bin_size::Int
    observables::O
    derived::Vector{DerivedObservable}
end

diagnostic_names(::MeasurementContainer{D}) where {D} = D

"""
    validate_binning(n_measurements, n_bins)

Check that `n_measurements` measurements can be grouped into `n_bins` equal
bins with enough bins for jackknife error estimates. Both
[`SimulationParameters`](@ref) and [`MeasurementContainer`](@ref) apply this
check, so a parameter set and a container built from the same counts are
either both valid or both rejected.

# Throws

`ArgumentError` if `n_measurements` is not positive, if `n_bins` is less
than 2, if `n_measurements` is smaller than `n_bins`, or if `n_measurements`
is not divisible by `n_bins`.

# Returns

`nothing`.
"""
function validate_binning(n_measurements, n_bins)
    n_measurements > 0 || throw(ArgumentError("n_measurements must be positive, got $n_measurements"))
    n_bins >= 2 || throw(ArgumentError("n_bins must be at least 2 for jackknife error estimates, got $n_bins"))
    n_measurements >= n_bins || throw(ArgumentError("n_measurements ($n_measurements) must be at least n_bins ($n_bins)"))
    n_measurements % n_bins == 0 ||
        throw(ArgumentError("n_measurements ($n_measurements) must be divisible by n_bins ($n_bins)"))
    return nothing
end

"""
    MeasurementContainer(model, geometry, n_measurements, n_bins;
      measurements = Symbol[], diagnostics = Symbol[], time_series = :scalars)

Build an empty [`MeasurementContainer`](@ref) for `model`, sized for
`n_measurements` total measurements grouped into `n_bins` bins.

The raw observables to track are the union of `model`/`geometry`'s always-on
observables (from `observables(model, geometry)`) and any optional ones
requested by name via `measurements` (resolved by `optional_observables`).
Derived quantities are collected the same way from `derived_observables` and
`optional_observables`, then checked with [`validate_dependencies`](@ref) to
ensure every derived quantity's inputs are actually being measured. Each raw
observable's initial `data` and `bin_sums` entries are built from its
`template()` and `zero()` constructors, respectively.

# Arguments

- `model`: The model being measured.
- `geometry`: The lattice geometry.
- `n_measurements`: Total number of measurements to be taken; must be
  positive and evenly divisible by `n_bins`.
- `n_bins`: Number of bins to group measurements into; must be at least 2
  (required for jackknife error estimates).
- `measurements`: Names of optional observables/derived quantities to
  include in addition to the model's default set.
- `diagnostics`: Names of per-sweep algorithm diagnostics to include in the
  binned results. These values are supplied to [`measure!`](@ref).
- `time_series`: Which observables keep their full unbinned history for
  autocorrelation analysis (see [`autocorrelation`](@ref)). Binned results
  are unaffected; only the `autocorrelation` entries of [`analyze`](@ref)
  depend on this. One of
  - `:scalars` (default): every scalar observable and diagnostic, but not
    vector-valued observables such as `:correlation`, whose history needs
    `n_measurements × n_displacements` floats and dominates memory on large
    lattices. Scalar histories cost 8 bytes per observable per measurement;
  - `:all` (or `true`): every observable and diagnostic, including vectors;
  - `:none` (or `false`): nothing, disabling autocorrelation analysis;
  - a vector of names, e.g. `[:energy, :magnetization]`.

# Throws

`ArgumentError` if `n_measurements` and `n_bins` fail
[`validate_binning`](@ref) (`n_measurements` not positive, `n_bins` less
than 2, or `n_measurements` not a multiple of `n_bins`), if `measurements`
contains a name not available for `model` (see
[`available_measurements`](@ref)), if any name in `diagnostics` duplicates a
raw observable name, if a requested derived quantity depends on an
observable that isn't being measured (via [`validate_dependencies`](@ref)),
or if `time_series` is not a recognized option or names an observable that
isn't being measured.

# Returns

A `MeasurementContainer` with `bin_count = 0`, ready to accumulate
measurements.
"""
function MeasurementContainer(
    model,
    geometry,
    n_measurements,
    n_bins;
    measurements = Symbol[],
    diagnostics = Symbol[],
    time_series = :scalars
)
    validate_binning(n_measurements, n_bins)

    # Alias canonicalization, availability validation, and the
    # implicit-dependency notice all live in `optional_observables`.
    opt_obs, opt_der = optional_observables(model, geometry, measurements)

    obs = unique(observable -> observable.name, vcat(observables(model, geometry), opt_obs))
    der = unique(derived -> derived.names, vcat(derived_observables(model, geometry), opt_der))

    # Diagnostic names as an NTuple{N,Symbol}; this becomes the type parameter D.
    diag_names = Tuple(Symbol.(diagnostics))
    allunique(diag_names) ||
        throw(ArgumentError("Diagnostic names must be unique: $diag_names"))

    observable_names = [o.name for o in obs]
    isempty(intersect(observable_names, diag_names)) ||
        throw(ArgumentError("Diagnostic names must not duplicate observable names"))

    validate_dependencies(obs, der)

    data = merge(
      NamedTuple(o.name => o.template() for o in obs),
      NamedTuple(name => Float64[] for name in diag_names)
    )
    bin_sums = merge(
      NamedTuple(o.name => o.zero() for o in obs),
      NamedTuple(name => 0.0 for name in diag_names)
    )
    # Unbinned per-measurement values, retained for autocorrelation analysis.
    recorded = time_series_names(time_series, obs, diag_names)
    series = merge(
      NamedTuple(o.name => o.template() for o in obs if o.name in recorded),
      NamedTuple(name => Float64[] for name in diag_names if name in recorded)
    )

    return _build_container(
        Val(diag_names), data, bin_sums, series,
        Int[], 0, n_measurements ÷ n_bins, obs, der
    )
end

function _build_container(::Val{D}, data, bin_sums, series, chain_lengths,
                          bin_count, bin_size, obs, der) where {D}
    # A concretely typed tuple of observables makes measure! type-stable.
    observables = obs isa Tuple ? obs : Tuple(obs)
    obs_names = map(o -> o.name, observables)
    keys(bin_sums)[1:length(observables)] == obs_names || throw(ArgumentError(
        "bin_sums fields $(keys(bin_sums)) must lead with the observable " *
        "names $obs_names in the same order"
    ))
    return MeasurementContainer{D, obs_names, typeof(data), typeof(bin_sums),
                                typeof(series), typeof(observables)}(
        data, bin_sums, series, chain_lengths, bin_count, bin_size, observables, der
    )
end

"""
    time_series_names(selection, obs, diagnostics)

Resolve the `time_series` keyword of [`MeasurementContainer`](@ref) into the
set of observable/diagnostic names whose unbinned history is recorded.
An observable counts as scalar when its `template()` collects `Real` values.
"""
function time_series_names(selection, obs::Vector{Observable}, diagnostics)
    all_names = vcat([o.name for o in obs], collect(diagnostics))
    selection === true && return Set(all_names)
    selection === false && return Set{Symbol}()
    if selection isa Symbol
        selection === :all && return Set(all_names)
        selection === :none && return Set{Symbol}()
        selection === :scalars && return Set(vcat(
            [o.name for o in obs if eltype(o.template()) <: Real],
            collect(diagnostics),
        ))
        throw(ArgumentError(
            "time_series must be :scalars, :all, :none, a Bool, or a vector of " *
            "observable names; got :$selection"
        ))
    end
    selection isa AbstractVector{Symbol} || throw(ArgumentError(
        "time_series must be :scalars, :all, :none, a Bool, or a vector of " *
        "observable names; got $(repr(selection))"
    ))
    unknown = setdiff(selection, all_names)
    isempty(unknown) || throw(ArgumentError(
        "time_series names $unknown are not measured; available: $all_names"
    ))
    return Set(selection)
end

"""
    validate_dependencies(obs::Vector{Observable}, der::Vector{DerivedObservable})

Check that every derived observable's dependencies are covered by the raw
observables in `obs`, throwing a descriptive error if not.

# Arguments

- `obs`: The raw observables that will actually be measured.
- `der`: The derived observables to validate, each with a `depends_on` list
  of required raw observable names and a `names` field used in error
  messages.

# Throws

`ArgumentError` if any derived observable in `der` depends on a name not
present among `obs`, naming the offending derived observable and its
missing dependencies.

# Returns

`nothing`.
"""
function validate_dependencies(obs::Vector{Observable}, der::Vector{DerivedObservable})
    available = Set(o.name for o in obs)
    for d in der
        missing = filter(dep -> dep ∉ available, d.depends_on)
        isempty(missing) && continue
        throw(ArgumentError(
            "Derived observable $(d.names) depends on $(missing), which " *
            (length(missing) == 1 ? "is" : "are") *
            " not among the requested observables. Check `measurements` includes " *
            "whatever raw Observable(s) provide $(missing)."
        ))
    end
    return nothing
end