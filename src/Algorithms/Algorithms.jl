"""
    AbstractUpdateResult

Abstract supertype for the result returned by each Monte Carlo update
algorithm.
"""
abstract type AbstractUpdateResult end

"""
    MetropolisUpdateResult

Result of one Metropolis update, recording whether the proposed move was
accepted.

# Fields

- `accepted::Bool`: Whether the proposed update was applied to the state.
"""
struct MetropolisUpdateResult <: AbstractUpdateResult
    accepted::Bool
end

"""
    GlauberUpdateResult

Result of one Glauber update, recording whether the proposed move was
accepted.

# Fields

- `accepted::Bool`: Whether the proposed update was applied to the state.
"""
struct GlauberUpdateResult <: AbstractUpdateResult
    accepted::Bool
end

"""
    steps_per_sweep(algorithm, geometry)

Return the number of [`step!`](@ref) updates that constitute one Monte Carlo
sweep for `algorithm` on `geometry`.

For local-update algorithms (the default, e.g. [`MetropolisAlgorithm`](@ref)
and [`GlauberAlgorithm`](@ref))
a sweep is `geometry.n_sites` single-site updates, so that on average every
site is updated once per sweep — exactly once with
[`SequentialProposal`](@ref), which visits the sites in order. For cluster
algorithms
([`WolffAlgorithm`](@ref), [`SwendsenWangAlgorithm`](@ref)) a single cluster
update already moves a macroscopic fraction of the lattice, so a sweep is
exactly one call to [`step!`](@ref).

Extend this function to change the sweep size for a custom algorithm.
"""
steps_per_sweep(::AbstractAlgorithm, geometry::Geometry) = geometry.n_sites

include("Proposals.jl")
include("Metropolis.jl")
include("Glauber.jl")
include("ClusterWorkspace.jl")
include("Wolff.jl")
include("SwendsenWang.jl")
include("Diagnostics.jl")
