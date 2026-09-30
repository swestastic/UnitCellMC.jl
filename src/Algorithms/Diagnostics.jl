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

function record!(statistic::UpdateStatistic, value::Real)
    statistic.total += value
    statistic.count += 1
    return statistic
end

function value(statistic::UpdateStatistic)
    statistic.count > 0 || throw(ArgumentError("Cannot read an empty update statistic"))
    return statistic.total / statistic.count
end
diagnostic_name(::MetropolisAlgorithm) = :acceptance_ratio
diagnostic_value(::MetropolisAlgorithm, result::MetropolisUpdateResult) = result.accepted ? 1.0 : 0.0
diagnostic_name(::WolffAlgorithm) = :cluster_size
diagnostic_value(::WolffAlgorithm, result::WolffUpdateResult) = result.cluster_size
diagnostic_name(::SwendsenWangAlgorithm) = :cluster_count
diagnostic_value(::SwendsenWangAlgorithm, result::SwendsenWangUpdateResult) = result.cluster_count