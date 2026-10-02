"""
    SpinUpdate{T}

A proposed single-site update: change the spin at `site` to `new_value`.

# Fields

- `site::Int`: Index of the site to update.
- `new_value::T`: The proposed value for that site.
"""
struct SpinUpdate{T}
    site::Int
    new_value::T
end

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

include("Metropolis.jl")
include("Wolff.jl")
include("SwendsenWang.jl")
include("Diagnostics.jl")