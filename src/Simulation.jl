function Run_Simulation(
    model::IsingModel,
    algorithm::Union{Metropolis, Glauber},
    β::Float64,
    neighbor_table_map,
    N_warmup::Int,
    N_measure::Int,
    N_bins::Int
)
    N_sites = length(model.spins)

    bins_container = Dict{String, Vector{Float64}}(
        "Energy"        => zeros(Float64, N_bins),
        "Magnetization" => zeros(Float64, N_bins),
        "SpecificHeat"  => zeros(Float64, N_bins)
        )

    # fresh spin configuration for this β
    Initialize_Spins!(model)

    # warmup — thermalize, no measurements
    for _ in 1:N_warmup
        MC_Step!(model, algorithm, β, neighbor_table_map)
    end

    # measurement loop
    for meas in 1:N_measure
        MC_Step!(model, algorithm, β, neighbor_table_map)
        bins_container["Energy"][meas % N_bins + 1] += Calc_Energy(model, neighbor_table_map)
        bins_container["Magnetization"][meas % N_bins + 1] += Calc_Magnetization(model)
    end

    measurement_results = Process_Bins(bins_container, N_measure, N_bins)


    # println("β = $β  |  ⟨E⟩/site = $(round(mean_E/N_sites, digits=4))  |  ⟨|M|⟩/site = $(round(abs(mean_M)/N_sites, digits=4))")
    println("β = $β | $measurement_results")
    return measurement_results
end