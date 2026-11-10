"""
    HeatbathAlgorithm(proposal = RandomProposal())

The heatbath single-site update algorithm: the visited site is assigned a new
value drawn directly from its single-site conditional distribution — the
Boltzmann distribution of the site's degree of freedom in the local field of
its frozen neighbors — so there is no acceptance test and every update is
applied. The model hook [`heatbath_draw`](@ref) performs the draw;
[`step!`](@ref) then evaluates the exact energy change with
[`energy_difference`](@ref) and applies the update with
[`apply_update!`](@ref).

Since the new value is sampled from the exact conditional, each update leaves
the Boltzmann distribution invariant (the conditional is the restricted
Boltzmann distribution itself), no matter what value the site held before.

For the two-state [`IsingModel`](@ref) the conditional is a single logistic
probability, so a heatbath update has exactly the transition kernel of a
[`GlauberAlgorithm`](@ref) update and the two algorithms coincide; the Ising
hook reuses [`glauber_probability`](@ref) (and the [`AcceptanceCache`](@ref)
when one is carried). The other models get genuine conditional samplers:
direct enumeration of the `q` max-shifted Boltzmann weights for the
[`ClockModel`](@ref), a von Mises draw about the local-field direction for
the [`XYModel`](@ref) (see [`rand_von_mises`](@ref)), and a Fisher (Langevin)
draw about the local-field direction for the [`HeisenbergModel`](@ref).

# Fields

- `proposal::AbstractProposalScheme`: Which site each update visits —
  [`RandomProposal`](@ref) (the default), [`SequentialProposal`](@ref),
  [`CheckerboardProposal`](@ref), or [`ColoredProposal`](@ref).
"""
struct HeatbathAlgorithm{P<:AbstractProposalScheme} <: AbstractAlgorithm
    proposal::P
end

HeatbathAlgorithm() = HeatbathAlgorithm(RandomProposal())

Base.show(io::IO, alg::HeatbathAlgorithm) =
    print(io, "HeatbathAlgorithm(", alg.proposal, ")")

"""
    heatbath_draw(rng, model, geometry, state, site, T, cache)
    heatbath_draw(rng, model, geometry, state, site, T)

Draw the new value of `site` from its single-site conditional distribution at
temperature `T` — the model-specific core of a [`HeatbathAlgorithm`](@ref)
update — and return it as a [`SpinUpdate`](@ref) carrying the site's new
value and the magnetization change, exactly like a [`propose`](@ref) result.
[`step!`](@ref) evaluates the energy change of the drawn value with
[`energy_difference`](@ref) and applies it unconditionally with
[`apply_update!`](@ref), so the returned `SpinUpdate` must keep the state's
running totals exact the same way a proposal does.

Each model implements its own method (the conditional sampler is the physics
of the update); the generic method throws so that a model without one fails
loudly, like the [`state_type`](@ref) fallback. The six-argument call passes
`cache = nothing`.

# Arguments

- `rng`: Random number generator the conditional draw samples from.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry, giving the site's neighbor table.
- `state`: The current configuration (the site's neighbors are frozen).
- `site`: The site to redraw, chosen by the algorithm's
  [`AbstractProposalScheme`](@ref).
- `T`: Temperature of the conditional Boltzmann distribution.
- `cache`: The [`AcceptanceCache`](@ref) of the run's
  [`LocalUpdateWorkspace`](@ref) when [`tabulate_acceptance`](@ref) is true
  for the model, `nothing` otherwise. Models with a discrete conditional
  spectrum (currently [`IsingModel`](@ref)) use it to memoize the conditional
  probability; continuous models ignore it.

# Throws

`ArgumentError` if `model` does not define its own method. To support a
custom model, define
`heatbath_draw(rng, model::MyModel, geometry, state::MyState, site::Int, T::Real, cache)`
returning a [`SpinUpdate`](@ref) sampled from the site's conditional.

# Returns

A [`SpinUpdate`](@ref) carrying the drawn value and the magnetization change.
"""
function heatbath_draw(
    rng::AbstractRNG,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    site::Int,
    T::Real,
    cache,
)
    throw(ArgumentError(
        "heatbath_draw is not implemented for $(nameof(typeof(model))); " *
        "define `heatbath_draw(rng, model::$(nameof(typeof(model))), geometry, " *
        "state, site::Int, T::Real, cache)` returning a SpinUpdate sampled from " *
        "the site's single-site conditional distribution"
    ))
end

# Six-argument convenience form: no acceptance cache (e.g. direct calls
# outside a driver run).
heatbath_draw(rng::AbstractRNG, model::AbstractModel, geometry::Geometry,
              state::AbstractState, site::Int, T::Real) =
    heatbath_draw(rng, model, geometry, state, site, T, nothing)

shared_workspace(alg::HeatbathAlgorithm, model, geometry) =
    visiting_order(alg.proposal, geometry)

algorithm_workspace(alg::HeatbathAlgorithm, model, geometry, state,
                    order = shared_workspace(alg, model, geometry)) =
    LocalUpdateWorkspace(alg.proposal, model, geometry, order)

"""
    step!(alg::HeatbathAlgorithm, model, geometry, state, T; rng = Random.default_rng(), workspace = nothing)

Perform one heatbath update on `state` at temperature `T`: redraw the value
of one site from its single-site conditional distribution and apply it.

The site is chosen with the algorithm's [`AbstractProposalScheme`](@ref) (see
[`propose_site`](@ref)), the new value is drawn by the model's
[`heatbath_draw`](@ref) method, its energy change is computed with
[`energy_difference`](@ref), and the update is applied in place via
[`apply_update!`](@ref). There is no acceptance test — the draw is already
the exact conditional — so a heatbath update that returns the current value
is the only way a step can leave the site unchanged; the result records
whether the value changed.

# Arguments

- `alg`: The update algorithm, `HeatbathAlgorithm()`.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The current configuration, mutated in place.
- `T`: Temperature of the conditional distribution.
- `rng`: Random number generator used for the site selection and the
  conditional draw. Defaults to the task-local global RNG, so
  `Random.seed!` makes runs reproducible; pass an explicit `AbstractRNG` for
  an independent stream.
- `workspace`: Per-run scratch space from [`algorithm_workspace`](@ref),
  threaded through [`MC_Sweep!`](@ref) by the drivers. Required when `alg`
  uses a stateful proposal scheme (see [`SequentialProposal`](@ref)); for
  models with a discrete conditional spectrum it carries the
  [`AcceptanceCache`](@ref) (see [`tabulate_acceptance`](@ref)).

# Throws

`ArgumentError` if `T` is not strictly positive, or if `alg` uses a stateful
proposal scheme and no `workspace` is given.

# Returns

A [`HeatbathUpdateResult`](@ref) describing whether the site's value changed.
"""
function step!(
    alg::HeatbathAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    T::Real;
    rng::AbstractRNG = Random.default_rng(),
    workspace = nothing
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))

    site = propose_site(rng, alg.proposal, geometry, _proposal_workspace(workspace))
    proposal = heatbath_draw(rng, model, geometry, state, site, T, _acceptance_cache(workspace))

    changed = proposal.new_value != local_operator(state, site)
    ΔE = energy_difference(model, geometry, state, proposal)
    apply_update!(state, proposal, ΔE)

    return HeatbathUpdateResult(changed)
end
