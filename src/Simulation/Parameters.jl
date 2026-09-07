struct SimulationParameters{T<:Real}
    β::T
    n_thermalization::Int
    n_measurements::Int
    n_unmeasured::Int
    n_bins::Int
end