abstract type AbstractAlgorithm end

struct Metropolis <: AbstractAlgorithm end

function step! end

include("Metropolis.jl")