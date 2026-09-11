# --- scalar (energy, magnetization, etc.) ---
function average_and_error(values::Vector{Float64})
    n_bins = length(values)
    n_bins > 1 || throw(ArgumentError("At least two bins are required"))

    average = Statistics.mean(values)
    variance = sum((value - average)^2 for value in values) / (n_bins - 1)
    error = sqrt(variance / n_bins)

    return average, error
end

# --- vector-valued (correlation, etc.) ---
function average_and_error(values::Vector{Vector{Float64}})
    n_bins = length(values)
    n_bins > 1 || throw(ArgumentError("At least two bins are required"))

    stacked = reduce(hcat, values)   # n_disp × n_bins
    averages = vec(Statistics.mean(stacked, dims = 2))
    errs = vec(Statistics.std(stacked, dims = 2)) ./ sqrt(n_bins)

    return averages, errs
end

function jackknife_samples(bin_values::Vector{T}) where T
    total = sum(bin_values)
    n = length(bin_values)
    return [(total .- v) ./ (n - 1) for v in bin_values]
end

function jackknife_stats(f::Function, jk_sample_sets::AbstractVector{<:Real}...)
    n = length(jk_sample_sets[1])
    vals = [f((s[i] for s in jk_sample_sets)...) for i in 1:n]
    v̄ = Statistics.mean(vals)
    variance = (n - 1) / n * sum((v - v̄)^2 for v in vals)
    return v̄, sqrt(variance)
end

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