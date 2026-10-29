"""
    RandomProposal

Site-selection scheme for local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)): every
[`step!`](@ref) update draws a site uniformly at random from `rng`. This is
the default scheme.
"""
struct RandomProposal <: AbstractProposalScheme end

"""
    SequentialProposal

Site-selection scheme for local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)): [`step!`](@ref)
updates visit the sites in site-index order, so one sweep
([`steps_per_sweep`](@ref) updates) attempts every site exactly once instead
of `n_sites` random draws.

`LatticeUtilities.jl` numbers sites in column-major unit-cell order (the
first lattice direction varies fastest), so consecutive updates target
neighboring sites — on a square lattice the sweep proceeds column by column
(equivalently row by row, up to transposition).

Each single-site update still satisfies detailed balance, so the completed
sweep leaves the Boltzmann distribution invariant; only the visiting order
differs from [`RandomProposal`](@ref).

The cursor is mutable per-run state held in a
[`SequentialProposalWorkspace`](@ref), built once per run by
[`algorithm_workspace`](@ref) and threaded into [`step!`](@ref) by the
drivers. Calling [`step!`](@ref) or [`propose`](@ref) directly without a
`workspace` throws an `ArgumentError`, since a fresh cursor would restart at
site 1 on every call.
"""
struct SequentialProposal <: AbstractProposalScheme end

"""
    SequentialProposalWorkspace

Per-run cursor for the [`SequentialProposal`](@ref) scheme: `next` is the
site the next [`step!`](@ref) update visits. Built by
[`proposal_workspace`](@ref) via [`algorithm_workspace`](@ref). Like
[`ClusterWorkspace`](@ref), it carries mutable state between steps and must
not be shared between concurrent runs.
"""
mutable struct SequentialProposalWorkspace
    next::Int
end

"""
    propose_site(rng, scheme, geometry, workspace)

Return the site the next local [`step!`](@ref) update visits under `scheme`.
[`RandomProposal`](@ref) draws uniformly from `rng` and ignores `workspace`;
[`SequentialProposal`](@ref) returns the cursor of its
[`SequentialProposalWorkspace`](@ref) and advances it, wrapping after
`geometry.n_sites`.

# Throws

`ArgumentError` if a stateful scheme is called without its workspace.
"""
propose_site(rng::AbstractRNG, ::RandomProposal, geometry::Geometry, workspace) =
    rand(rng, 1:geometry.n_sites)

function propose_site(
    ::AbstractRNG,
    ::SequentialProposal,
    geometry::Geometry,
    workspace::SequentialProposalWorkspace
)
    site = workspace.next
    workspace.next = mod1(site + 1, geometry.n_sites)
    return site
end

function propose_site(::AbstractRNG, scheme::SequentialProposal, ::Geometry, ::Nothing)
    throw(ArgumentError(
        "$(nameof(typeof(scheme))) requires its workspace: build one with " *
        "`algorithm_workspace(algorithm, model, geometry, state)` and pass it via " *
        "the `workspace` keyword of `step!`/`propose` (the drivers " *
        "`run_simulation`/`MC_Sweep!` do this automatically)"
    ))
end

"""
    proposal_workspace(scheme, geometry)

Return the per-run site-selection state for `scheme`: `nothing` for
stateless schemes ([`RandomProposal`](@ref)), a
[`SequentialProposalWorkspace`](@ref) for [`SequentialProposal`](@ref).
[`algorithm_workspace`](@ref) forwards to this for local-update algorithms,
and [`run_simulation`](@ref) threads the result through [`MC_Sweep!`](@ref)
into [`step!`](@ref). Extend it for custom schemes with per-run state.
"""
proposal_workspace(::RandomProposal, geometry) = nothing
proposal_workspace(::SequentialProposal, geometry) = SequentialProposalWorkspace(1)
