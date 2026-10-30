"""
    XYState{S,T<:Real} <: AbstractState

The mutable configuration of an XY model simulation. Each site stores its
planar spin as a unit vector, split into the two component arrays `cosines`
and `sines` (`cosines[i] = cos(θ_i)`, `sines[i] = sin(θ_i)` for the site's
angle `θ_i`); energy and the two Cartesian magnetization components are
cached for incremental updates.

Storing the spin components rather than the angle keeps the hot inner loops
([`energy_difference`](@ref), cluster reflections) free of trigonometric
calls: bond energies are dot products `cos(θ_i - θ_j) = c_i c_j + s_i s_j`,
and the Wolff reflection is the linear map `v ↦ v - 2 (v·r̂) r̂`. Accepted
single-site updates *replace* a site's components with a fresh `sincos`, so
`cosines[i]^2 + sines[i]^2 == 1` holds to rounding indefinitely. Recover the
angle of site `i` with `atan(sines[i], cosines[i])` when needed.

The running totals are maintained incrementally by [`apply_update!`](@ref),
[`apply_cluster!`](@ref), and [`apply_clusters!`](@ref), so they must be kept
consistent with the components by construction (see [`initialize_state`](@ref))
and by every code path that mutates them.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct XYState{S,T<:Real} <: AbstractState
    cosines::S
    sines::S
    energy::T
    magnetization_x::T
    magnetization_y::T
end

# The proposal carries the new components as new_value and their change as
# (cos(new) - cos(old), sin(new) - sin(old)).
function apply_update!(state::XYState, proposal::SpinUpdate, ΔE::Real)
    state.cosines[proposal.site] = proposal.new_value[1]
    state.sines[proposal.site] = proposal.new_value[2]
    state.energy += ΔE
    state.magnetization_x += proposal.magnetization_change[1]
    state.magnetization_y += proposal.magnetization_change[2]
    return nothing
end
