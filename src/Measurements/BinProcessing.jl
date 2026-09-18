"""
    average_and_error(values::Vector{Float64})

Compute the bin average and standard error of the mean for a scalar
observable's per-bin values.

The error is estimated as `sqrt(variance / n_bins)`, where `variance` is the
sample variance of `values` (Bessel-corrected, dividing by `n_bins - 1`).

# Arguments

- `values`: Per-bin averages for one scalar observable (e.g. energy,
  magnetization), one entry per bin.

# Throws

`ArgumentError` if fewer than two bins are given.

# Returns

A `(average, error)` tuple.
"""
function average_and_error(values::Vector{Float64})
    n_bins = length(values)
    n_bins > 1 || throw(ArgumentError("At least two bins are required"))

    average = Statistics.mean(values)
    variance = sum((value - average)^2 for value in values) / (n_bins - 1)
    error = sqrt(variance / n_bins)

    return average, error
end

"""
    average_and_error(values::Vector{Vector{Float64}})

Compute the elementwise bin average and standard error of the mean for a
vector-valued observable's per-bin values (e.g. a correlation function
evaluated at each displacement).

Each element of `values` is one bin's vector of per-displacement (or
otherwise per-component) values; these are stacked into an `n_disp × n_bins`
matrix and averaged/errored along the bin dimension.

# Arguments

- `values`: Per-bin value vectors for one vector-valued observable, one
  vector per bin, all of equal length.

# Throws

`ArgumentError` if fewer than two bins are given.

`DimensionMismatch` if the per-bin vectors do not all have the same length.

# Returns

An `(averages, errors)` tuple of vectors, each of the same length as the
per-bin vectors in `values`.
"""
function average_and_error(values::Vector{Vector{Float64}})
    n_bins = length(values)
    n_bins > 1 || throw(ArgumentError("At least two bins are required"))

    stacked = reduce(hcat, values)   # n_disp × n_bins
    averages = vec(Statistics.mean(stacked, dims = 2))
    errs = vec(Statistics.std(stacked, dims = 2)) ./ sqrt(n_bins)

    return averages, errs
end

"""
    jackknife_samples(bin_values::Vector{T}) where T

Compute leave-one-out jackknife samples from per-bin values.

For `n = length(bin_values)` bins with total `total = sum(bin_values)`, the
`i`-th jackknife sample is `(total - bin_values[i]) / (n - 1)`, i.e. the
average of all bins except the `i`-th.

# Arguments

- `bin_values`: Per-bin values (scalar or vector-valued, any type supporting
  `sum`, subtraction, and elementwise division).

# Throws

`ArgumentError` if fewer than two bins are given.

# Returns

A vector of `n` jackknife samples, one per omitted bin. The element type is
determined by the subtraction and division operations.
"""
function jackknife_samples(bin_values::Vector{T}) where T
    n = length(bin_values)
    n > 1 || throw(ArgumentError("At least two bins are required"))
    total = sum(bin_values)
    return [(total .- v) ./ (n - 1) for v in bin_values]
end

"""
    jackknife_stats(f::Function, jk_sample_sets::AbstractVector{<:Real}...)

Compute the jackknife estimate and error of a (possibly nonlinear) function
`f` of one or more primary observables, given their jackknife sample sets.

`f` is evaluated once per jackknife replicate `i`, taking the `i`-th sample
from each set in `jk_sample_sets` (as produced by [`jackknife_samples`](@ref))
as its arguments. The jackknife error accounts for correlations between the
observables and for nonlinearity in `f`, unlike simple error propagation.

# Arguments

- `f`: A function of `length(jk_sample_sets)` real arguments, computing the
  derived quantity from one sample of each underlying observable.
- `jk_sample_sets`: One or more jackknife sample vectors (e.g. from
  [`jackknife_samples`](@ref)), all of equal length.

# Throws

`ArgumentError` if no sample sets are provided, if the sample sets have
different lengths, or if fewer than two jackknife replicates are provided.

# Returns

A `(value, error)` tuple: the jackknife mean of `f` over all replicates, and
its jackknife standard error.
"""
function jackknife_stats(f::Function, jk_sample_sets::AbstractVector{<:Real}...)
  isempty(jk_sample_sets) && throw(ArgumentError("At least one sample set is required"))
    n = length(jk_sample_sets[1])
  n > 1 || throw(ArgumentError("At least two jackknife replicates are required"))
  all(length(samples) == n for samples in jk_sample_sets) ||
    throw(ArgumentError("All jackknife sample sets must have the same length"))
    vals = [f((s[i] for s in jk_sample_sets)...) for i in 1:n]
    v̄ = Statistics.mean(vals)
    variance = (n - 1) / n * sum((v - v̄)^2 for v in vals)
    return v̄, sqrt(variance)
end

"""
    analyze(container::MeasurementContainer, T, N)

Compute primary and derived statistics for all observables recorded in
`container`.

Primary observables (`container.data`) are reduced to `(average, error)`
pairs via [`average_and_error`](@ref) and to jackknife sample sets via
[`jackknife_samples`](@ref). Derived quantities (`container.derived`) are
then computed from the jackknife samples by calling each derived entry's
`compute` function with the jackknife samples, `T`, and `N`, and merging the
results into a single named tuple; this allows derived quantities (e.g.
connected correlations, susceptibilities) to correctly propagate errors from
one or more primary observables via jackknife resampling.

# Arguments

- `container`: A `MeasurementContainer` holding per-bin data for each
  observable (`container.data`) and a list of derived-quantity
  specifications (`container.derived`), each with a `compute(jk, T, N)`
  function.
- `T`: Temperature, passed through to each derived quantity's `compute`.
- `N`: System size (or other size parameter), passed through to each derived
  quantity's `compute`.

# Returns

A named tuple `(primary = ..., derived = ...)`:
- `primary`: `Dict` mapping observable name to its `(average, error)` pair.
- `derived`: Named tuple merging the results of every derived quantity's
  `compute` call.
"""
function analyze(container::MeasurementContainer, T, N)
    bin_means = container.data

    primary = Dict(name => average_and_error(means) for (name, means) in pairs(bin_means))
    jk      = Dict(name => jackknife_samples(means) for (name, means) in pairs(bin_means))

    derived = NamedTuple()
    for d in container.derived
        derived = merge(derived, d.compute(jk, T, N))
    end

    return (primary = primary, derived = derived)
end