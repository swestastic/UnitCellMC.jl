
module UnitCellMC

# export ModelGeometry, IsingModel, XYModel, PottsModel, Metropolis, Run_Simulation
export ModelGeometry
export IsingModel
export Metropolis, Glauber
export Run_Simulation
export Process_Bins
export plot_lattice, get_lattice_points

include("Geometry.jl")

include("Models/Ising.jl")
# include("Models/XY.jl")
# include("Models/Potts.jl")

include("Algorithms/MetropolisHastings.jl")
include("Algorithms/Glauber.jl")

include("Binning.jl")
include("Simulation.jl")
include("Display.jl")

end # module
