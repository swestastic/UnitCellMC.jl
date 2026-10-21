"""
    GlauberAlgorithm

Marker type selecting the Glauber single-site update algorithm. Used to
dispatch [`propose`](@ref) and [`step!`](@ref) to the Glauber acceptance
rule, which accepts a trial move with probability
`1 / (1 + exp(ΔE / T))` instead of the Metropolis `min(1, exp(-ΔE / T))`.

Every model proposes the same symmetric single-site trial moves as for
[`MetropolisAlgorithm`](@ref), so detailed balance holds with respect to the
Boltzmann distribution.
"""
struct GlauberAlgorithm <: AbstractAlgorithm end

"""
    step!(alg::GlauberAlgorithm, model, geometry, state, T; rng = Random.default_rng())

Attempt one Glauber update on `state` at temperature `T`, and apply it if
accepted.

A trial move is generated with [`propose`](@ref), its energy change is
computed with [`energy_difference`](@ref), and it is accepted with the Glauber
probability `1 / (1 + exp(ΔE / T))`. Unlike Metropolis, moves with `ΔE <= 0`
are not accepted unconditionally; a move with `ΔE = 0` is accepted with
probability `1/2`. Accepted moves are applied in place via
[`apply_update!`](@ref).

# Arguments

- `alg`: The update algorithm, `GlauberAlgorithm()`.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The current configuration, mutated in place if the move is accepted.
- `T`: Temperature used in the Glauber acceptance probability.
- `rng`: Random number generator used for the proposal and the acceptance
  test. Defaults to the task-local global RNG, so `Random.seed!` makes runs
  reproducible; pass an explicit `AbstractRNG` for an independent stream.

# Throws

`ArgumentError` if `T` is not strictly positive.

# Returns

A [`GlauberUpdateResult`](@ref) describing whether the proposal was accepted.
"""
function step!(
    alg::GlauberAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    T::Real;
    rng::AbstractRNG = Random.default_rng()
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))

    proposal = propose(rng, alg, model, geometry, state)
    ΔE = energy_difference(model, geometry, state, proposal)

    if rand(rng) < glauber_probability(ΔE, T)
        apply_update!(state, proposal, ΔE)
        return GlauberUpdateResult(true)
    end

    return GlauberUpdateResult(false)
end

"""
    glauber_probability(ΔE, T)

Return the Glauber acceptance probability `1 / (1 + exp(ΔE / T))` for a move
with energy change `ΔE` at temperature `T`.

The logistic form is evaluated so that it neither overflows nor loses
precision: for `ΔE > 0` it is rewritten as `exp(-ΔE / T) / (1 + exp(-ΔE / T))`,
which tends to `0` instead of evaluating `exp` of a large argument.

# Returns

A probability in `[0, 1]`.
"""
function glauber_probability(ΔE::Real, T::Real)
    x = ΔE / T
    if x > 0
        w = exp(-x)
        return w / (1 + w)
    end
    return 1 / (1 + exp(x))
end
