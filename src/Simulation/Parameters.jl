struct SimulationParameters{T<:Real}
    T::T
    n_thermalization::Int
    n_measurements::Int
    n_unmeasured::Int
    n_bins::Int
end