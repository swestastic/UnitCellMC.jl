"""
    Observable

Specification of one raw quantity to measure each Monte Carlo step: how to
compute an instantaneous value, and how to initialize its bin accumulator
and its history of completed bins.

# Fields

- `name::Symbol`: The observable's name, used as its key in
  [`MeasurementContainer`](@ref)'s `data`/`bin_sums` and elsewhere.
- `value::Function`: `value(state)` — computes one instantaneous sample from
  the current configuration.
- `zero::Function`: `zero()` — returns the additive identity used to
  initialize/reset the running bin sum (e.g. `0.0` for a scalar, `zeros(n)`
  for a vector).
- `template::Function`: `template()` — returns an empty container to
  accumulate completed bin averages into (e.g. `Float64[]`,
  `Vector{Float64}[]`).

Construct with [`Observable(name, value; zero, template)`](@ref).
"""
struct Observable
    name::Symbol
    value::Function
    zero::Function
    template::Function
end

"""
    Observable(name, value; zero = () -> 0.0, template = () -> Float64[])

Convenience constructor for a scalar-valued [`Observable`](@ref), defaulting
`zero`/`template` to plain `Float64` accumulators.

# Arguments

- `name`: The observable's name.
- `value`: `value(state)` computing one instantaneous sample.
- `zero`: Returns the initial/reset bin sum; defaults to `0.0`.
- `template`: Returns an empty container for completed bin averages;
  defaults to `Float64[]`.

For a vector-valued observable (e.g. a correlation function), pass explicit
`zero`/`template` functions returning the appropriate vector types — see
[`correlation_entry`](@ref) for an example.
"""
Observable(name::Symbol, value::Function; zero::Function = () -> 0.0, template::Function = () -> Float64[]) =
    Observable(name, value, zero, template)

    """
    DerivedObservable

Specification of one or more quantities computed from already-measured raw
observables (and their jackknife samples), rather than sampled directly from
the state.

# Fields

- `names::Tuple{Vararg{Symbol}}`: Name(s) of the quantity/quantities this
  produces (e.g. a value and its paired error).
- `compute::Function`: `compute(jk, T, N)` — given a mapping from raw
  observable name to its jackknife samples, plus temperature `T` and size
  `N`, returns a `NamedTuple` with keys `names`.
- `depends_on::Tuple{Vararg{Symbol}}`: Names of the raw observables this
  quantity requires; checked against the measured set by
  [`validate_dependencies`](@ref).

Construct directly via [`DerivedObservable(names, compute; depends_on)`](@ref)
for full control, or via the [`DerivedObservable(value_name, err_name,
depends_on, f)`](@ref) convenience form for the common
value/jackknife-error pattern.
"""
struct DerivedObservable
    names::Tuple{Vararg{Symbol}}
    compute::Function
    depends_on::Tuple{Vararg{Symbol}}
end

"""
    DerivedObservable(names, compute; depends_on = ())

General constructor for a [`DerivedObservable`](@ref), taking a fully custom
`compute` function.

# Arguments

- `names`: Name(s) of the quantity/quantities produced.
- `compute`: `compute(jk, T, N)` returning a `NamedTuple` with keys `names`.
- `depends_on`: Names of the raw observables `compute` reads from `jk`.
"""
DerivedObservable(names::Tuple{Vararg{Symbol}}, compute::Function; depends_on::Tuple{Vararg{Symbol}} = ()) =
    DerivedObservable(names, compute, depends_on)

    """
    DerivedObservable(value_name, err_name, depends_on, f)

Convenience constructor for the common case of a single derived quantity
computed from raw observables' jackknife samples, with its error obtained
automatically via jackknife resampling.

Builds a `compute(jk, T, N)` function that calls
`jackknife_stats((args...) -> f(args...; T = T, N = N), jk[k]...)` over the
jackknife sample sets for each name in `depends_on`, and packages the
resulting `(value, error)` pair into a `NamedTuple` keyed by `value_name`
and `err_name`.

# Arguments

- `value_name`: Name for the computed value.
- `err_name`: Name for its jackknife error.
- `depends_on`: Names of the raw observables `f` is computed from, in the
  order their jackknife samples are passed to `f`.
- `f`: `f(args...; T, N)` computing the derived quantity from one jackknife
  sample of each observable in `depends_on`, plus temperature `T` and size `N`.

# Returns

A `DerivedObservable` with `names = (value_name, err_name)`.
"""
function DerivedObservable(value_name::Symbol, err_name::Symbol, depends_on::Tuple{Vararg{Symbol}}, f::Function)
    names = (value_name, err_name)
    compute = (jk, T, N) -> begin
        v, err = jackknife_stats((args...) -> f(args...; T = T, N = N), (jk[k] for k in depends_on)...)
        NamedTuple{names}((v, err))
    end
    return DerivedObservable(names, compute, depends_on)
end