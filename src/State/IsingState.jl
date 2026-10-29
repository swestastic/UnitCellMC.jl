"""
    IsingState{S,T<:Real,M<:Integer} <: AbstractState

The mutable configuration of an Ising model simulation: per-site spins and
the running totals (energy, magnetization) kept in sync with them as updates
are applied.

The running totals are maintained incrementally by [`apply_update!`](@ref)
rather than recomputed from scratch each step, so they must be kept
consistent with `spins` by construction (see [`initialize_state`](@ref)) and
by every code path that mutates `spins`.

# Fields

- `spins::S`: Per-site spin values (typically `±1`).
- `energy::T`: Total energy of the current configuration.
- `magnetization::M`: Total magnetization (sum of `spins`) of the current
  configuration.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct IsingState{S,T<:Real,M<:Integer} <: AbstractState
    spins::S
    energy::T
    magnetization::M
end

# The proposal's carried magnetization change is `new_value - old spin`.
function apply_update!(state::IsingState, proposal::SpinUpdate, ΔE::Real)
    state.spins[proposal.site] = proposal.new_value
    state.energy += ΔE
    state.magnetization += proposal.magnetization_change
    return nothing
end
