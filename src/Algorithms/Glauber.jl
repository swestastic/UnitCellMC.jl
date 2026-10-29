"""
    GlauberAlgorithm(proposal = RandomProposal())

The Glauber single-site update algorithm: the same trial moves as
[`MetropolisAlgorithm`](@ref), but a trial move is accepted with probability
`1 / (1 + exp(ΔE / T))` instead of the Metropolis `min(1, exp(-ΔE / T))`.

By default every model proposes the same symmetric single-site trial moves as
for [`MetropolisAlgorithm`](@ref) — a generic fallback [`propose`](@ref)
method forwards to it — so detailed balance holds with respect to the
Boltzmann distribution. A model may override the fallback with a
Glauber-specific proposal.

# Fields

- `proposal::AbstractProposalScheme`: Which site each update visits —
  [`RandomProposal`](@ref) (the default) or [`SequentialProposal`](@ref).
"""
struct GlauberAlgorithm{P<:AbstractProposalScheme} <: AbstractAlgorithm
    proposal::P
end

GlauberAlgorithm() = GlauberAlgorithm(RandomProposal())

Base.show(io::IO, alg::GlauberAlgorithm) =
    print(io, "GlauberAlgorithm(", alg.proposal, ")")

"""
    propose(rng, algorithm::GlauberAlgorithm, model, geometry, state; workspace = nothing)

Generate a Glauber trial move: pick a site with the algorithm's
[`AbstractProposalScheme`](@ref) (see [`propose_site`](@ref)), then call the
model's `propose(rng, algorithm, model, geometry, state, site)` method to
choose the new value there. By default that method falls back to the model's
Metropolis proposal at the same site; a model can override it with a
Glauber-specific proposal.

# Arguments

- `workspace`: The scheme's per-run state, built by
  [`algorithm_workspace`](@ref). Required for stateful schemes (see
  [`SequentialProposal`](@ref)); the drivers pass it automatically.

# Returns

A [`SpinUpdate`](@ref) describing the site and its proposed new value.
"""
function propose(
    rng::AbstractRNG,
    alg::GlauberAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState;
    workspace = nothing
)
    site = propose_site(rng, alg.proposal, geometry, workspace)
    return propose(rng, alg, model, geometry, state, site)
end

# By default a model's Glauber value proposal is its Metropolis one (the same
# symmetric trial move at the chosen site); a model can override this
# fallback with a Glauber-specific proposal.
propose(rng::AbstractRNG, ::GlauberAlgorithm, model::AbstractModel, geometry, state::AbstractState, site::Int) =
    propose(rng, MetropolisAlgorithm(), model, geometry, state, site)

algorithm_workspace(alg::GlauberAlgorithm, model, geometry, state) =
    proposal_workspace(alg.proposal, geometry)

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
- `workspace`: Per-run scratch space from [`algorithm_workspace`](@ref),
  threaded through [`MC_Sweep!`](@ref) by the drivers. Required when `alg`
  uses a stateful proposal scheme (see [`SequentialProposal`](@ref)).

# Throws

`ArgumentError` if `T` is not strictly positive, or if `alg` uses a stateful
proposal scheme and no `workspace` is given.

# Returns

A [`GlauberUpdateResult`](@ref) describing whether the proposal was accepted.
"""
function step!(
    alg::GlauberAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    T::Real;
    rng::AbstractRNG = Random.default_rng(),
    workspace = nothing
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))

    proposal = propose(rng, alg, model, geometry, state; workspace = workspace)
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
