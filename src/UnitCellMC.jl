module UnitCellMC

using LatticeUtilities
using Statistics

include("Core.jl")
include("Geometry/Geometry.jl")
include("State/State.jl")
include("Algorithms/Algorithms.jl")
include("Models/Models.jl")
include("Measurements/Measurements.jl")
include("Simulation/Simulation.jl")
include("IO/IO.jl")

export Geometry, initialize_state
export IsingModel, XYModel
export MetropolisAlgorithm, WolffAlgorithm, SwendsenWangAlgorithm
export SimulationParameters, MeasurementContainer
export MC_Sweep!, run_simulation, sweep_Ts, analyze, autotune_binning
export save_results, save_sweep_results
export available_measurements, available_diagnostics
export AutocorrelationResult, autocorrelation, autocorrelation_function
export integrated_autocorrelation_time, tau_lower_bound, recommended_binning

end