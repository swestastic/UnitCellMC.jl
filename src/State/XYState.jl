"""
    XYState{S,T<:Real} <: AbstractState

The mutable configuration of an XY model simulation. Each site stores an angle
in radians, while energy and the two Cartesian magnetization components are
cached for incremental updates.

The running totals are maintained incrementally by [`apply_update!`](@ref),
[`apply_cluster!`](@ref), and [`apply_clusters!`](@ref), so they must be kept
consistent with `angles` by construction (see [`initialize_state`](@ref)) and
by every code path that mutates `angles`.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct XYState{S,T<:Real} <: AbstractState
    angles::S
    energy::T
    magnetization_x::T
    magnetization_y::T
end
