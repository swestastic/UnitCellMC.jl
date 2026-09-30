"""
    MetropolisAlgorithm

Marker type selecting the Metropolis-Hastings update algorithm. Used to
dispatch [`propose`](@ref) and [`step!`](@ref) to the standard
single-flip Metropolis acceptance rule.
"""
struct MetropolisAlgorithm <: AbstractAlgorithm end

"""
    step!(alg, model, geometry, state, T)

Attempt one Monte Carlo update on `state` using `alg`, and apply it if
accepted according to the Metropolis criterion at temperature `T`.

A trial move is generated with [`propose`](@ref), its energy change is
computed with [`energy_difference`](@ref), and it is accepted unconditionally
if `ΔE <= 0`, or with probability `exp(-ΔE / T)` otherwise. Accepted moves are
applied in place via [`apply_update!`](@ref).

# Arguments

- `alg`: The update algorithm to use (e.g. `MetropolisAlgorithm()`).
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The current configuration, mutated in place if the move is accepted.
- `T`: Temperature used in the Metropolis acceptance probability.

# Throws

`ArgumentError` if `T` is not strictly positive.

# Returns

A [`MetropolisUpdateResult`](@ref) describing whether the proposal was
accepted.
"""
function step!(
    alg::MetropolisAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    T::Real
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))

    proposal = propose(alg, model, geometry, state)
    ΔE = energy_difference(model, geometry, state, proposal)

    if ΔE <= 0.0 || log(rand()) < - ΔE / T
        apply_update!(state, proposal, ΔE)
        return MetropolisUpdateResult(true)
    end

    return MetropolisUpdateResult(false)
end

