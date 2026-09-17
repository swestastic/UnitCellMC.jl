"""
    MeasurementContainer{T, S}

Accumulates binned Monte Carlo measurements and holds the observable
specifications needed to analyze them.

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
- `bin_count::Int`: Number of measurements accumulated in the current bin;
  reset to `0` when the bin is flushed into `data`.
- `bin_size::Int`: Number of measurements per bin (`n_measurements ÷ n_bins`).
- `observables::Vector{Observable}`: The raw observables being measured.
- `derived::Vector{DerivedObservable}`: The derived quantities to compute
  from the raw observables once binning is complete.

Construct with [`MeasurementContainer(model, geometry, n_measurements,
n_bins; measurements)`](@ref).
"""
mutable struct MeasurementContainer{T, S}
    data::T
    bin_sums::S
    bin_count::Int
    bin_size::Int
    observables::Vector{Observable}
    derived::Vector{DerivedObservable}
end

"""
    MeasurementContainer(model, geometry, n_measurements, n_bins; measurements = Symbol[])

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

# Throws

`ArgumentError` if `n_measurements` is not positive, if `n_bins` is less
than 2, if `n_measurements` is smaller than `n_bins`, if `n_measurements` is
not divisible by `n_bins`, or if a requested derived quantity depends on an
observable that isn't being measured (via [`validate_dependencies`](@ref)).

# Returns

A `MeasurementContainer` with `bin_count = 0`, ready to accumulate
measurements.
"""
function MeasurementContainer(
    model,
    geometry,
    n_measurements,
    n_bins;
    measurements = Symbol[]
)
    n_measurements > 0 || throw(ArgumentError("n_measurements must be positive, got $n_measurements"))
    n_bins >= 2 || throw(ArgumentError("n_bins must be at least 2 for jackknife error estimates, got $n_bins"))
    n_measurements >= n_bins || throw(ArgumentError("n_measurements ($n_measurements) must be at least n_bins ($n_bins)"))
    n_measurements % n_bins == 0 ||
        throw(ArgumentError("n_measurements ($n_measurements) must be divisible by n_bins ($n_bins)"))

    opt_obs, opt_der = optional_observables(model, geometry, measurements)

    obs = vcat(observables(model, geometry), opt_obs)
    der = vcat(derived_observables(model, geometry), opt_der)

    validate_dependencies(obs, der)

    data     = NamedTuple(o.name => o.template() for o in obs)
    bin_sums = NamedTuple(o.name => o.zero()     for o in obs)

    return MeasurementContainer(data, bin_sums, 0, n_measurements ÷ n_bins, obs, der)
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