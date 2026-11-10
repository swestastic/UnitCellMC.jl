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
    HeatbathUpdateResult

Result of one heatbath update, recording whether the redrawn value of the
visited site differs from its previous value.

A heatbath update has no acceptance test — the new value is drawn from the
site's exact conditional distribution (see [`HeatbathAlgorithm`](@ref)) — so
the only way a step can leave the site unchanged is by drawing its current
value, which for discrete models happens with nonzero probability.

# Fields

- `changed::Bool`: Whether the site's value after the update differs from
  the one before.
"""
struct HeatbathUpdateResult <: AbstractUpdateResult
    changed::Bool
end

"""
    OverrelaxationUpdateResult

Result of one overrelaxation update. The microcanonical reflection is
deterministic and always applied (see [`OverrelaxationAlgorithm`](@ref)), so
there is nothing to record; the type exists so every [`step!`](@ref) returns
an [`AbstractUpdateResult`](@ref).
"""
struct OverrelaxationUpdateResult <: AbstractUpdateResult end

"""
    steps_per_sweep(algorithm, geometry)

Return the number of [`step!`](@ref) updates that constitute one Monte Carlo
sweep for `algorithm` on `geometry`.

For local-update algorithms (the default, e.g. [`MetropolisAlgorithm`](@ref)
[`GlauberAlgorithm`](@ref), [`HeatbathAlgorithm`](@ref), and
[`OverrelaxationAlgorithm`](@ref))
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
include("AcceptanceCache.jl")
include("Metropolis.jl")
include("Glauber.jl")
include("Heatbath.jl")
include("Overrelaxation.jl")
include("ClusterWorkspace.jl")
include("Wolff.jl")
include("SwendsenWang.jl")
include("Diagnostics.jl")
include("Sweeps.jl")
