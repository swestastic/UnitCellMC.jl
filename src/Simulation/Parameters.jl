struct SimulationParameters{T<:Real}
    β::T
    n_thermalization::Int
    n_sweeps::Int
    n_unmeasured::Int
    n_bins::Int
end