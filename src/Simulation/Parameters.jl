struct SimulationParameters{T<:Real}
    T::T
    n_thermalization::Int
    n_measurements::Int
    n_unmeasured::Int
    n_bins::Int

    function SimulationParameters{T}(T_val, n_thermalization, n_measurements, n_unmeasured, n_bins) where {T<:Real}
        T_val >= 0 || throw(ArgumentError("T must be >= 0, got $T_val"))
        n_thermalization >= 0 || throw(ArgumentError("n_thermalization must be >= 0, got $n_thermalization"))
        n_measurements >= 0 || throw(ArgumentError("n_measurements must be >= 0, got $n_measurements"))
        n_unmeasured >= 0 || throw(ArgumentError("n_unmeasured must be >= 0, got $n_unmeasured"))
        n_bins >= 0 || throw(ArgumentError("n_bins must be >= 0, got $n_bins"))
        new{T}(T_val, n_thermalization, n_measurements, n_unmeasured, n_bins)
    end
end

SimulationParameters(T_val::T, n_thermalization, n_measurements, n_unmeasured, n_bins) where {T<:Real} =
    SimulationParameters{T}(T_val, n_thermalization, n_measurements, n_unmeasured, n_bins)