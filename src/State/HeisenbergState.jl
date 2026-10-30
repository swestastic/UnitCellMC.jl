"""
    HeisenbergState{S,T<:Real} <: AbstractState

The mutable configuration of a [`HeisenbergModel`](@ref) simulation. Each site
stores its unit spin vector split into the three component arrays `xs`, `ys`,
`zs`; energy and the three Cartesian magnetization components are cached for
incremental updates.

Storing the Cartesian components keeps the hot inner loops
([`energy_difference`](@ref), cluster reflections) free of trigonometric
calls: bond energies are dot products, and the Wolff reflection is the linear
map `v ↦ v - 2 (v·r̂) r̂`. Accepted single-site updates *replace* a site's
components with a freshly drawn unit vector, so `xs[i]^2 + ys[i]^2 +
zs[i]^2 == 1` holds to rounding indefinitely.

The running totals are maintained incrementally by [`apply_update!`](@ref),
[`apply_cluster!`](@ref), and [`apply_clusters!`](@ref), so they must be kept
consistent with the components by construction (see [`initialize_state`](@ref))
and by every code path that mutates them.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct HeisenbergState{S,T<:Real} <: AbstractState
    xs::S
    ys::S
    zs::S
    energy::T
    magnetization_x::T
    magnetization_y::T
    magnetization_z::T
end

# The proposal carries the new components as new_value and their change as
# (x_new - x_old, y_new - y_old, z_new - z_old).
function apply_update!(state::HeisenbergState, proposal::SpinUpdate, ΔE::Real)
    state.xs[proposal.site] = proposal.new_value[1]
    state.ys[proposal.site] = proposal.new_value[2]
    state.zs[proposal.site] = proposal.new_value[3]
    state.energy += ΔE
    state.magnetization_x += proposal.magnetization_change[1]
    state.magnetization_y += proposal.magnetization_change[2]
    state.magnetization_z += proposal.magnetization_change[3]
    return nothing
end
