"""
    MetropolisAlgorithm(proposal = RandomProposal())

The Metropolis-Hastings single-site update algorithm: a trial move is
generated with [`propose`](@ref) and accepted unconditionally when it lowers
the energy, or with probability `exp(-ΔE / T)` otherwise (see [`step!`](@ref)).

# Fields

- `proposal::AbstractProposalScheme`: Which site each update visits —
  [`RandomProposal`](@ref) (the default), [`SequentialProposal`](@ref),
  [`CheckerboardProposal`](@ref), or [`ColoredProposal`](@ref). The model's
  [`propose`](@ref) method decides the new value at that site.
"""
struct MetropolisAlgorithm{P<:AbstractProposalScheme} <: AbstractAlgorithm
    proposal::P
end

MetropolisAlgorithm() = MetropolisAlgorithm(RandomProposal())

Base.show(io::IO, alg::MetropolisAlgorithm) =
    print(io, "MetropolisAlgorithm(", alg.proposal, ")")

"""
    propose(rng, algorithm, model, geometry, state; workspace = nothing)

Generate a Metropolis trial move: pick a site with the algorithm's
[`AbstractProposalScheme`](@ref) (see [`propose_site`](@ref)), then call the
model's `propose(rng, algorithm, model, geometry, state, site)` method to
choose the new value there.

# Arguments

- `workspace`: The scheme's per-run state, built by
  [`algorithm_workspace`](@ref). Required for stateful schemes (see
  [`SequentialProposal`](@ref)); the drivers pass it automatically.

# Returns

A [`SpinUpdate`](@ref) describing the site and its proposed new value.
"""
function propose(
    rng::AbstractRNG,
    alg::MetropolisAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState;
    workspace = nothing
)
    site = propose_site(rng, alg.proposal, geometry, workspace)
    return propose(rng, alg, model, geometry, state, site)
end

algorithm_workspace(alg::MetropolisAlgorithm, model, geometry, state) =
    proposal_workspace(alg.proposal, geometry)

"""
    step!(alg, model, geometry, state, T; rng = Random.default_rng())

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

A [`MetropolisUpdateResult`](@ref) describing whether the proposal was
accepted.
"""
function step!(
    alg::MetropolisAlgorithm,
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

    if ΔE <= 0.0 || log(rand(rng)) < - ΔE / T
        apply_update!(state, proposal, ΔE)
        return MetropolisUpdateResult(true)
    end

    return MetropolisUpdateResult(false)
end
