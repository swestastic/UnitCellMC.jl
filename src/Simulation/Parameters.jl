"""
    SimulationParameters{T<:Real}

Validated parameters controlling a single Monte Carlo simulation run: the
temperature and the sweep counts for thermalization, measurement, and binning.

# Fields

- `T::T`: Temperature; must be strictly positive.
- `n_thermalization::Int`: Number of thermalization (equilibration) sweeps
  before measurements begin; must be non-negative.
- `n_measurements::Int`: Total number of measurement sweeps; must be
  non-negative.
- `n_unmeasured::Int`: Number of sweeps between measurements (decorrelation
  steps); must be non-negative.
- `n_bins::Int`: Number of bins measurements are grouped into for error
  estimation; must be at least 1.

Construct with [`SimulationParameters(T, n_thermalization, n_measurements,
n_unmeasured, n_bins)`](@ref); all fields are validated at construction time.
"""
struct SimulationParameters{T<:Real}
    T::T
    n_thermalization::Int
    n_measurements::Int
    n_unmeasured::Int
    n_bins::Int

    function SimulationParameters{T}(T_val, n_thermalization, n_measurements, n_unmeasured, n_bins) where {T<:Real}
        T_val > 0 || throw(ArgumentError("T must be > 0, got $T_val"))
        n_thermalization >= 0 || throw(ArgumentError("n_thermalization must be >= 0, got $n_thermalization"))
        n_measurements >= 0 || throw(ArgumentError("n_measurements must be >= 0, got $n_measurements"))
        n_unmeasured >= 0 || throw(ArgumentError("n_unmeasured must be >= 0, got $n_unmeasured"))
        n_bins >= 1 || throw(ArgumentError("n_bins must be >= 1, got $n_bins"))
        new{T}(T_val, n_thermalization, n_measurements, n_unmeasured, n_bins)
    end
end

"""
    SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)

Construct [`SimulationParameters`](@ref), inferring the element type `T`
from the temperature argument.

# Arguments

- `T`: Temperature; must be strictly positive.
- `n_thermalization`: Number of thermalization sweeps; must be non-negative.
- `n_measurements`: Number of measurement sweeps; must be non-negative.
- `n_unmeasured`: Number of decorrelation sweeps between measurements; must
  be non-negative.
- `n_bins`: Number of bins for error estimation; must be at least 1.

# Throws

`ArgumentError` if `T` is not positive, or if any of `n_thermalization`,
`n_measurements`, `n_unmeasured` is negative, or if `n_bins` is less than 1.
"""
SimulationParameters(T_val::T, n_thermalization, n_measurements, n_unmeasured, n_bins) where {T<:Real} =
    SimulationParameters{T}(T_val, n_thermalization, n_measurements, n_unmeasured, n_bins)