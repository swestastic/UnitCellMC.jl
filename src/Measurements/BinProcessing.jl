function average_and_error(
    values::Vector{Float64}
)
    n_bins = length(values)

    n_bins > 1 || throw(
        ArgumentError("At least two bins are required")
    )

    average = Statistics.mean(values)

    variance = sum(
        (value - average)^2
        for value in values
    ) / (n_bins - 1)

    error = sqrt(variance / n_bins)

    return average, error
end

function analyze(container::MeasurementContainer, model, β, N)
    bin_means = container.data   # NamedTuple{names, Tuple{Vector{Float64}, ...}}

    primary = Dict(name => average_and_error(means) for (name, means) in pairs(bin_means))
    jk      = Dict(name => jackknife_samples(means) for (name, means) in pairs(bin_means))
    derived = derived_observables(model, β, N, jk)

    return (primary = primary, derived = derived)
end

function jackknife_samples(bin_values::Vector{Float64})
    total = sum(bin_values)
    n = length(bin_values)
    return [(total - v) / (n - 1) for v in bin_values]
end

function jackknife_stats(f::Function, jk_sample_sets::Vector{Float64}...)
    n = length(jk_sample_sets[1])
    vals = [f((s[i] for s in jk_sample_sets)...) for i in 1:n]
    v̄ = Statistics.mean(vals)
    variance = (n - 1) / n * sum((v - v̄)^2 for v in vals)
    return v̄, sqrt(variance)
end