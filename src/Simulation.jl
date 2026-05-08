function Run_Simulation(
    model::IsingModel,
    algorithm::Union{Metropolis, Glauber},
    β::Float64,
    neighbor_table_map,
    N_warmup::Int,
    N_measure::Int
)
    N_sites = length(model.spins)

    # fresh spin configuration for this β
    Initialize_Spins!(model)

    # warmup — thermalize, no measurements
    for _ in 1:N_warmup
        MC_Step!(model, algorithm, β, neighbor_table_map)
    end

    # measurement loop
    measurements = Measurements(0.0, 0.0)
    for _ in 1:N_measure
        MC_Step!(model, algorithm, β, neighbor_table_map)
        measurements.E += Calc_Energy(model, neighbor_table_map)
        measurements.M += Calc_Magnetization(model)
    end

    mean_E = measurements.E / N_measure
    mean_M = measurements.M / N_measure

    println("β = $β  |  ⟨E⟩/site = $(round(mean_E/N_sites, digits=4))  |  ⟨|M|⟩/site = $(round(abs(mean_M)/N_sites, digits=4))")

    return mean_E, mean_M
end