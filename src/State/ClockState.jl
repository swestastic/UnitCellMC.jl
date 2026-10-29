"""
    ClockState{S,T<:Real} <: AbstractState

The mutable configuration of a [`ClockModel`](@ref) simulation. Each site
stores an integer clock index `n ∈ 0:q-1`, representing the planar spin at
angle `θ = 2π n / q`; energy and the two Cartesian magnetization components
are cached for incremental updates.

The running totals are maintained incrementally by [`apply_update!`](@ref),
[`apply_cluster!`](@ref), and [`apply_clusters!`](@ref), so they must be kept
consistent with `spins` by construction (see [`initialize_state`](@ref)) and
by every code path that mutates `spins`.

# Fields

- `spins::S`: Per-site clock indices, each in `0:q-1`.
- `energy::T`: Total energy of the current configuration.
- `magnetization_x::T`: Total `x` magnetization, `Σ_i cos(2π n_i / q)`.
- `magnetization_y::T`: Total `y` magnetization, `Σ_i sin(2π n_i / q)`.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct ClockState{S,T<:Real} <: AbstractState
    spins::S
    energy::T
    magnetization_x::T
    magnetization_y::T
end

# The proposal carries `(cos(θ_new) - cos(θ_old), sin(θ_new) - sin(θ_old))`,
# computed by `propose` from the model's lookup tables.
function apply_update!(state::ClockState, proposal::SpinUpdate, ΔE::Real)
    state.spins[proposal.site] = proposal.new_value
    state.energy += ΔE
    state.magnetization_x += proposal.magnetization_change[1]
    state.magnetization_y += proposal.magnetization_change[2]
    return nothing
end
