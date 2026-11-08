"""
    ACCEPTANCE_CACHE_CAPACITY

Maximum number of memoized `ΔE → exp(-ΔE / T)` entries held by an
[`AcceptanceCache`](@ref), so the memo table is bounded and its scan stays
short. The typical models selected by [`tabulate_acceptance`](@ref) have far
fewer distinct `ΔE` values than this; a wider spectrum keeps the first
entries it meets and computes further values directly.
"""
const ACCEPTANCE_CACHE_CAPACITY = 64

"""
    AcceptanceCache

Per-run memo table of Boltzmann factors `u = exp(-ΔE / T)` for the
local-update acceptance tests, built by [`algorithm_workspace`](@ref) as part
of a [`LocalUpdateWorkspace`](@ref) when [`tabulate_acceptance`](@ref) is
true for the model. The Metropolis test `rand(rng) < exp(-ΔE / T)` and the
Glauber acceptance probability (equivalently the Ising heatbath conditional
probability, see [`heatbath_draw`](@ref)) both become table lookups once an entry is
memoized.

Entries are keyed by the exact `Float64` value of each computed `ΔE`, so a
hit reproduces a direct `exp(-ΔE / T)` evaluation bit for bit. The table is
cleared when `T` changes between updates (the drivers use a fixed `T` per
run, so this happens at most once) and holds at most
[`ACCEPTANCE_CACHE_CAPACITY`](@ref) entries; once full, new values are
computed directly while already-memoized values are still served. Like
[`ClusterWorkspace`](@ref), it
carries mutable per-run state and must not be shared between concurrent runs.
"""
mutable struct AcceptanceCache
    T::Float64  # temperature the current entries were memoized at
    delta_Es::Vector{Float64}  # exact ΔE keys, parallel to boltzmann_factors
    boltzmann_factors::Vector{Float64}
end

AcceptanceCache() = AcceptanceCache(
    NaN,
    sizehint!(Float64[], ACCEPTANCE_CACHE_CAPACITY),
    sizehint!(Float64[], ACCEPTANCE_CACHE_CAPACITY),
)

"""
    reset!(cache::AcceptanceCache, T)

Clear the memoized entries of `cache` and re-target it at temperature `T`.
Called automatically by [`cached_boltzmann_factor`](@ref) whenever the
temperature changes between updates.
"""
function reset!(cache::AcceptanceCache, T::Float64)
    cache.T = T
    empty!(cache.delta_Es)
    empty!(cache.boltzmann_factors)
    return cache
end

"""
    cached_boltzmann_factor(cache, ΔE, T)

Return `exp(-ΔE / T)`, memoized in `cache` keyed by the exact `Float64`
value of `ΔE` (see [`AcceptanceCache`](@ref)). The first occurrence of each
distinct `ΔE` at temperature `T` pays for the `exp`; later occurrences are a
short linear scan. Once the cache holds [`ACCEPTANCE_CACHE_CAPACITY`](@ref)
entries, new values are computed directly without being memoized.
"""
@inline function cached_boltzmann_factor(cache::AcceptanceCache, ΔE::Real, T::Float64)
    cache.T == T || reset!(cache, T)
    delta_Es = cache.delta_Es
    factors = cache.boltzmann_factors
    @inbounds for k in eachindex(delta_Es)
        delta_Es[k] == ΔE && return factors[k]
    end
    u = exp(-ΔE / T)
    if length(delta_Es) < ACCEPTANCE_CACHE_CAPACITY
        push!(delta_Es, Float64(ΔE))
        push!(factors, u)
    end
    return u
end

"""
    tabulate_acceptance(model)

Return `true` if the model's single-site [`energy_difference`](@ref) takes
values from a small discrete set, so that local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref),
[`HeatbathAlgorithm`](@ref)) benefit from
memoizing Boltzmann factors per temperature in an [`AcceptanceCache`](@ref).
Memoization is exact whenever the spectrum is discrete, since computed `ΔE`
values repeat bit for bit.

The default is `false`, appropriate for continuous-spectrum models
([`XYModel`](@ref), [`HeisenbergModel`](@ref)) whose `ΔE` values essentially
never repeat, and for [`ClockModel`](@ref), whose spectrum is discrete but
grows with `q` times the coordination number, so beyond small `q` the scan
outweighs the saved `exp` (measured). Extend it for a custom model with a
small discrete spectrum.
"""
tabulate_acceptance(::AbstractModel) = false

"""
    LocalUpdateWorkspace

Per-run workspace for the local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref),
[`HeatbathAlgorithm`](@ref)), built by
[`algorithm_workspace`](@ref) and threaded through [`MC_Sweep!`](@ref) into
[`step!`](@ref). Bundles the site-selection scheme's per-run state with the
acceptance-probability memo table.

# Fields

- `proposal`: The [`AbstractProposalScheme`](@ref)'s per-run state —
  `nothing` for [`RandomProposal`](@ref), a
  [`SequentialProposalWorkspace`](@ref) or [`ColoredProposalWorkspace`](@ref)
  for the stateful schemes.
- `acceptance`: An [`AcceptanceCache`](@ref) when
  [`tabulate_acceptance`](@ref) is true for the model, `nothing` otherwise.

Like [`ClusterWorkspace`](@ref), it carries mutable state between steps and
must not be shared between concurrent runs.
"""
struct LocalUpdateWorkspace{P,A}
    proposal::P
    acceptance::A
end

LocalUpdateWorkspace(
    scheme::AbstractProposalScheme, model::AbstractModel, geometry::Geometry
) = LocalUpdateWorkspace(scheme, model, geometry, visiting_order(scheme, geometry))

# `order` reuses a precomputed `visiting_order(scheme, geometry)`; see
# `proposal_workspace`.
LocalUpdateWorkspace(
    scheme::AbstractProposalScheme, model::AbstractModel, geometry::Geometry, order
) = LocalUpdateWorkspace(
    proposal_workspace(scheme, geometry, order),
    tabulate_acceptance(model) ? AcceptanceCache() : nothing,
)

# Unwrap helpers. `step!`/`propose` also accept a bare scheme workspace (or
# `nothing`) in place of a LocalUpdateWorkspace, so code that builds the
# proposal scheme's workspace directly keeps working.
_proposal_workspace(workspace::LocalUpdateWorkspace) = workspace.proposal
_proposal_workspace(workspace) = workspace

_acceptance_cache(workspace::LocalUpdateWorkspace) = workspace.acceptance
_acceptance_cache(workspace) = nothing
