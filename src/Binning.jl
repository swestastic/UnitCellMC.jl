import Statistics

function Process_Bins(
    # We have a Dict of measurement types in bins_container, where each key is a measurement type (e.g. "Energy", "Magnetization")
    # Normalize each bin by the number of measurements per bin (N_measure / N_bins) to get the average value for that bin
    # Finally calculate errorbars (standard error of the mean) across the bins for each measurement type
    
    bins_container::Dict{String, Vector{Float64}}, 
    N_measure::Int, 
    N_bins::Int,
    β::Float64
    )

    processed_results = Dict{String, Vector{Float64}}()

    for (key, values) in bins_container
        Bin_avgs = values / N_measure * N_bins
        Bin_totalavg = Statistics.mean(Bin_avgs)

        processed_results[key] = [Bin_totalavg]

        ErrorBar = 0.0
        for i in 1:N_bins
            ErrorBar +=  (Bin_avgs[i] - Bin_totalavg)^2
        end
        ErrorBar = sqrt(ErrorBar / N_bins) * sqrt(1 / (N_bins - 1)) 
        processed_results[key] = [Bin_totalavg, ErrorBar]
    end

    SpecificHeat, SpecificHeatError = Calc_Specific_Heat(bins_container, N_measure, N_bins, β)
    MagneticSusceptibility, MagneticSusceptibilityError = Calc_Magnetic_Susceptibility(bins_container, N_measure, N_bins, β)
    processed_results["Specific_Heat"] = [SpecificHeat, SpecificHeatError]
    processed_results["Magnetic_Susceptibility"] = [MagneticSusceptibility, MagneticSusceptibilityError]
    return processed_results
end