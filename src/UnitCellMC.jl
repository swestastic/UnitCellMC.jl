
module UnitCellMC

# export ModelGeometry, IsingModel, XYModel, PottsModel, Metropolis, Run_Simulation
export ModelGeometry, IsingModel, Metropolis, Glauber, Run_Simulation, Process_Bins

include("Geometry.jl")

include("Models/Ising.jl")
# include("Models/XY.jl")
# include("Models/Potts.jl")

include("Algorithms/MetropolisHastings.jl")
include("Algorithms/Glauber.jl")

include("Simulation.jl")
include("Binning.jl")

end # module
