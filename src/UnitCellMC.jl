module UnitCellMC

using LatticeUtilities
using Random
using Statistics

include("Core.jl")
include("Geometry/Geometry.jl")
include("State/State.jl")
include("Algorithms/Algorithms.jl")
include("Models/Models.jl")
include("State/Initialization.jl")
include("Measurements/Measurements.jl")
include("Simulation/Simulation.jl")
include("IO/IO.jl")

export Geometry, initialize_state
export IsingModel, XYModel, ClockModel
export MetropolisAlgorithm, GlauberAlgorithm, WolffAlgorithm, SwendsenWangAlgorithm
export RandomProposal, SequentialProposal
export SimulationParameters, MeasurementContainer
export MC_Sweep!, run_simulation, sweep_Ts, analyze, autotune_binning
export set_global_seed!, global_seed
export save_results, save_sweep_results
export available_measurements, available_diagnostics
export AutocorrelationResult, autocorrelation, autocorrelation_function
export integrated_autocorrelation_time, tau_lower_bound, recommended_binning

end