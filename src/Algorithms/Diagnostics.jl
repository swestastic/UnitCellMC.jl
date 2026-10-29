"""
    UpdateStatistic

Accumulate a scalar statistic over updates in one sweep. The same type can
represent an acceptance fraction or an average cluster size.
"""
mutable struct UpdateStatistic
    name::Symbol
    total::Float64
    count::Int
end

UpdateStatistic(name::Symbol) = UpdateStatistic(name, 0.0, 0)

function reset!(statistic::UpdateStatistic)
    statistic.total = 0.0
    statistic.count = 0
    return statistic
end

function record!(statistic::UpdateStatistic, value::Real)
    statistic.total += value
    statistic.count += 1
    return statistic
end

function value(statistic::UpdateStatistic)
    statistic.count > 0 || throw(ArgumentError("Cannot read an empty update statistic"))
    return statistic.total / statistic.count
end

"""
    diagnostic_values(algorithm, result)

Turn the [`AbstractUpdateResult`](@ref) returned by one [`step!`](@ref) of
`algorithm` into a `NamedTuple` of per-update diagnostic values (e.g.
`(acceptance_ratio = ...)` for local updates). The keys must match
[`available_diagnostics`](@ref)`(algorithm)`; [`MC_Sweep!`](@ref) accumulates
them into [`UpdateStatistic`](@ref)s and the measurement container records
the sweep averages.

# Throws

`ArgumentError` if no method exists for the algorithm/result pair.
"""
diagnostic_values(::AbstractAlgorithm, ::AbstractUpdateResult) = throw(ArgumentError(
    "diagnostic_values is not implemented for this algorithm/result pair; " *
    "define `diagnostic_values(::MyAlgorithm, ::MyUpdateResult)` and list the " *
    "keys in `available_diagnostics(::Type{<:MyAlgorithm})`"
))

diagnostic_values(::MetropolisAlgorithm, result::MetropolisUpdateResult) =
    (acceptance_ratio = result.accepted ? 1.0 : 0.0,)
diagnostic_values(::GlauberAlgorithm, result::GlauberUpdateResult) =
    (acceptance_ratio = result.accepted ? 1.0 : 0.0,)
diagnostic_values(::WolffAlgorithm, result::WolffUpdateResult) =
    (cluster_size = result.cluster_size,)
diagnostic_values(::SwendsenWangAlgorithm, result::SwendsenWangUpdateResult) = (
    cluster_count = result.cluster_count,
    largest_cluster_fraction = result.largest_cluster_fraction,
)

"""
    available_diagnostics(algorithm::Type{<:AbstractAlgorithm})

Return a `Vector{Symbol}` listing the diagnostic names accepted by `algorithm`
when constructing a [`MeasurementContainer`](@ref).

# Returns

- `MetropolisAlgorithm` → `[:acceptance_ratio]`
- `GlauberAlgorithm` → `[:acceptance_ratio]`
- `WolffAlgorithm` → `[:cluster_size]`
- `SwendsenWangAlgorithm` → `[:cluster_count, :largest_cluster_fraction]`
"""
available_diagnostics(::Type{<:MetropolisAlgorithm})   = [:acceptance_ratio]
available_diagnostics(::Type{<:GlauberAlgorithm})      = [:acceptance_ratio]
available_diagnostics(::Type{<:WolffAlgorithm})        = [:cluster_size]
available_diagnostics(::Type{<:SwendsenWangAlgorithm}) = [:cluster_count, :largest_cluster_fraction]

# Instances forward to the type methods
available_diagnostics(alg::AbstractAlgorithm) = available_diagnostics(typeof(alg))