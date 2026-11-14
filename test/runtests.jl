using Test
using Random
using Statistics
import UnitCellMC as ucmc
import LatticeUtilities as lu

include("test_helpers.jl")

# test_models.jl defines the PairingTestModel/State fixtures also used by
# test_optional_measurements.jl, so keep it earlier in this list.
include("test_geometry.jl")
include("test_models.jl")
include("test_states.jl")
include("test_metropolis.jl")
include("test_glauber_heatbath.jl")
include("test_cluster_algorithms.jl")
include("test_proposals.jl")
include("test_measurements.jl")
include("test_optional_measurements.jl")
include("test_structure_factor.jl")
include("test_simulation.jl")
include("test_io.jl")
include("test_clock.jl")
include("test_tempering.jl")
include("test_helicity.jl")
