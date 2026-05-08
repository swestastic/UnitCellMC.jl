
module UnitCellMC

# export ModelGeometry, IsingModel, XYModel, PottsModel, Metropolis, Run_Simulation
export ModelGeometry, IsingModel, Metropolis, Run_Simulation

include("Geometry.jl")
include("Simulation.jl")

include("Models/Ising.jl")
# include("Models/XY.jl")
# include("Models/Potts.jl")

include("Algorithms/MetropolisHastings.jl")

end # module
