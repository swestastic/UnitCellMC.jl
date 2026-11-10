"""
    OverrelaxationAlgorithm(proposal = RandomProposal())

The overrelaxation (microcanonical) single-site update algorithm for
continuous spin models: the visited site's spin is reflected about its local
field `H` (the bond-weighted neighbor sum plus the external field),
`v ↦ 2 (v·û) û - v` with `û = H / |H|`. The reflection preserves `v · H`
exactly, so the update conserves the site's energy, is always applied, and
is an involution — there is no acceptance test, no randomness beyond the
site selection, and no temperature dependence (`T` is validated for
interface uniformity but otherwise unused). A site with a vanishing local
field is left unchanged.

Overrelaxation is *not ergodic by itself*: energy conservation confines the
chain to a constant-energy shell of the configuration space. It samples the
canonical ensemble only when alternated with a thermalizing algorithm — the
standard recipe is one [`HeatbathAlgorithm`](@ref) sweep followed by a few
overrelaxation sweeps, which decorrelates XY and Heisenberg configurations
much faster than heatbath alone near criticality. Mix sweeps by calling
[`MC_Sweep!`](@ref) with each algorithm in turn (the drivers accept one
algorithm per run).

Only models with a continuous spin support the reflection; the model hook
[`overrelax`](@ref) is implemented for [`XYModel`](@ref) and
[`HeisenbergModel`](@ref) and throws for discrete models ([`IsingModel`](@ref),
[`ClockModel`](@ref)), where the reflection is the identity or does not map
onto the state space. There are no per-update diagnostics
([`available_diagnostics`](@ref) is empty): the update is deterministic, so
there is no acceptance rate to record.

# Fields

- `proposal::AbstractProposalScheme`: Which site each update visits —
  [`RandomProposal`](@ref) (the default), [`SequentialProposal`](@ref),
  [`CheckerboardProposal`](@ref), or [`ColoredProposal`](@ref).
"""
struct OverrelaxationAlgorithm{P<:AbstractProposalScheme} <: AbstractAlgorithm
    proposal::P
end

OverrelaxationAlgorithm() = OverrelaxationAlgorithm(RandomProposal())

Base.show(io::IO, alg::OverrelaxationAlgorithm) =
    print(io, "OverrelaxationAlgorithm(", alg.proposal, ")")

"""
    overrelax(model, geometry, state, site)

Return the overrelaxation (microcanonical) update of `site` as a
[`SpinUpdate`](@ref): the reflection of the site's spin about its local
field, `v ↦ 2 (v·û) û - v` (see [`OverrelaxationAlgorithm`](@ref)). The
reflection is deterministic and needs neither a temperature nor an `rng`.
The update is applied unconditionally by [`step!`](@ref), so the returned
`SpinUpdate` must carry the magnetization change the same way a
[`propose`](@ref) result does.

Each continuous-spin model implements its own method (currently
[`XYModel`](@ref) and [`HeisenbergModel`](@ref)); this fallback throws so
that a model without one — including the discrete models, for which the
reflection is not defined — fails loudly.

# Throws

`ArgumentError` if `model` does not define its own method. To support a
custom continuous-spin model, define
`overrelax(model::MyModel, geometry, state::MyState, site::Int)` returning
the reflected [`SpinUpdate`](@ref).

# Returns

A [`SpinUpdate`](@ref) carrying the reflected value and the magnetization
change.
"""
function overrelax(model::AbstractModel, geometry::Geometry, state::AbstractState, site::Int)
    throw(ArgumentError(
        "overrelax is not implemented for $(nameof(typeof(model))); the " *
        "microcanonical reflection requires a continuous spin (it is defined " *
        "for XYModel and HeisenbergModel; for a custom model define " *
        "`overrelax(model::$(nameof(typeof(model))), geometry, state, site::Int)`)"
    ))
end

shared_workspace(alg::OverrelaxationAlgorithm, model, geometry) =
    visiting_order(alg.proposal, geometry)

algorithm_workspace(alg::OverrelaxationAlgorithm, model, geometry, state,
                    order = shared_workspace(alg, model, geometry)) =
    LocalUpdateWorkspace(alg.proposal, model, geometry, order)

"""
    step!(alg::OverrelaxationAlgorithm, model, geometry, state, T; rng = Random.default_rng(), workspace = nothing)

Perform one overrelaxation update on `state`: reflect the spin of one site
about its local field (see [`overrelax`](@ref)) and apply it.

The site is chosen with the algorithm's [`AbstractProposalScheme`](@ref)
(see [`propose_site`](@ref)), the reflection is computed by the model's
[`overrelax`](@ref) method, and the update is applied in place via
[`apply_update!`](@ref). The exact energy change — zero to rounding, since
the reflection conserves the site's energy — is still evaluated with
[`energy_difference`](@ref) so the state's running total stays consistent
with from-scratch sums. `T` is unused by the update itself.

# Arguments

- `alg`: The update algorithm, `OverrelaxationAlgorithm()`.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The current configuration, mutated in place.
- `T`: Temperature; validated but unused by the microcanonical update.
- `rng`: Random number generator used for the site selection only. Defaults
  to the task-local global RNG, so `Random.seed!` makes runs reproducible;
  pass an explicit `AbstractRNG` for an independent stream.
- `workspace`: Per-run scratch space from [`algorithm_workspace`](@ref),
  threaded through [`MC_Sweep!`](@ref) by the drivers. Required when `alg`
  uses a stateful proposal scheme (see [`SequentialProposal`](@ref)).

# Throws

`ArgumentError` if `T` is not strictly positive, if `alg` uses a stateful
proposal scheme and no `workspace` is given, or if `model` does not support
overrelaxation (see [`overrelax`](@ref)).

# Returns

An [`OverrelaxationUpdateResult`](@ref).
"""
function step!(
    alg::OverrelaxationAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    T::Real;
    rng::AbstractRNG = Random.default_rng(),
    workspace = nothing
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))

    site = propose_site(rng, alg.proposal, geometry, _proposal_workspace(workspace))
    proposal = overrelax(model, geometry, state, site)
    ΔE = energy_difference(model, geometry, state, proposal)
    apply_update!(state, proposal, ΔE)

    return OverrelaxationUpdateResult()
end
