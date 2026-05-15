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
        "Energy"                  => zeros(Float64, N_bins),
        "Energy_Sqrd"             => zeros(Float64, N_bins),
        "Magnetization"           => zeros(Float64, N_bins),
        "Magnetization_Sqrd"      => zeros(Float64, N_bins),
        "Specific_Heat"           => zeros(Float64, N_bins),
        "Magnetic_Susceptibility" => zeros(Float64, N_bins),
        )

    # fresh spin configuration for this β — if spins are all zeros initialize
    # print(model.spins)
    if all(model.spins .== 0)
        Initialize_Spins!(model)
        println("Initialized spins for β = $β")
    end

    # warmup — thermalize, no measurements
    for _ in 1:N_warmup
        MC_Step!(model, algorithm, β, neighbor_table_map)
    end

    # measurement loop
    for meas in 1:N_measure
        MC_Step!(model, algorithm, β, neighbor_table_map)
        Energy = Calc_Energy(model, neighbor_table_map)
        Magnetization = Calc_Magnetization(model)
        bins_container["Energy"][meas % N_bins + 1] += Energy
        bins_container["Energy_Sqrd"][meas % N_bins + 1] += Energy^2
        bins_container["Magnetization"][meas % N_bins + 1] += Magnetization
        bins_container["Magnetization_Sqrd"][meas % N_bins + 1] += Magnetization^2
    end

    measurement_results = Process_Bins(bins_container, N_measure, N_bins, β)


    # println("β = $β  |  ⟨E⟩/site = $(round(mean_E/N_sites, digits=4))  |  ⟨|M|⟩/site = $(round(abs(mean_M)/N_sites, digits=4))")
    println("β = $β | $measurement_results")
    return measurement_results
end