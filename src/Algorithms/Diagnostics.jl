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

diagnostic_values(::MetropolisAlgorithm, result::MetropolisUpdateResult) =
    (acceptance_ratio = result.accepted ? 1.0 : 0.0,)
diagnostic_values(::WolffAlgorithm, result::WolffUpdateResult) =
    (cluster_size = result.cluster_size,)
diagnostic_values(::SwendsenWangAlgorithm, result::SwendsenWangUpdateResult) = (
    cluster_count = result.cluster_count,
    largest_cluster_fraction = result.largest_cluster_fraction,
)