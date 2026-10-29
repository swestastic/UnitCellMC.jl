struct IsingModel{T<:Real} <: AbstractModel{T}
    J::Vector{T}   # one entry per bond template, same order as geometry.bonds
    h::Vector{T}    # one value per site in the unit cell

    function IsingModel{T}(J::Vector{T}, h::Vector{T}) where {T<:Real}
        return new{T}(J, h)
    end
end

"""
    IsingModel(J, h)

An Ising model with bond strengths `J` and external field `h`.

# Arguments

- `J`: Bond strengths, one value per bond template. Its length is checked
    against `geometry.bonds` by [`initialize_state`](@ref), since the
    constructor does not receive a geometry.
- `h`: External magnetic field, one value per site in the unit cell.
"""
function IsingModel(J::Vector{<:Real}, h::AbstractVector{<:Real})
    T = promote_type(eltype(J), eltype(h))
    return IsingModel{T}(convert(Vector{T}, J), convert(Vector{T}, h))
end

"""
    local_operator(state, i)

Return the value of the local spin operator at site `i` for `state`.
"""
local_operator(state::IsingState, i) = state.spins[i]

state_type(::IsingModel) = IsingState

magnetization_value(state::IsingState) = state.magnetization

binder_denominator(::IsingModel) = 3.0

"""
    operator_product(model, a, b)

Return the two-body coupling contribution `a * b` between a pair of local
operator values, as used when computing bond energies for `model`.
"""
operator_product(model::IsingModel, a::Real, b::Real) = a * b

n_operator_components(::IsingModel) = 1
operator_components(::IsingModel, s::Real) = (s,)

"""
    propose(rng, ::MetropolisAlgorithm, model, geometry, state)

Generate a Metropolis trial move for `model`: pick a random site, drawn from
`rng`, and propose flipping its spin.

# Returns

A `SpinUpdate` describing the site and its proposed (flipped) value.
"""
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, ::IsingModel, geometry, state)
    site = rand(rng, eachindex(state.spins))
    s = state.spins[site]
    return SpinUpdate(site, -s, -2s)
end

"""
    energy_difference(model, geometry, state, proposal)

Compute the change in total energy `ΔE` that would result from changing the
spin at `proposal.site` to `proposal.new_value`, given the couplings and
field in `model`.

# Arguments

- `model`: The `IsingModel` providing bond strengths and field `h`.
- `geometry`: Lattice geometry giving each site's neighbors and bonds.
- `state`: Current spin configuration.
- `proposal`: A `SpinUpdate` describing the trial move.

# Returns

The energy difference `ΔE` between the proposed and current configurations.
For current spin `s`, proposed spin `s_new`, and local field
`H = h + sum(J * neighbor_spin)`, this is `(s - s_new) * H`.
"""
function energy_difference(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    proposal::SpinUpdate
)
    site = proposal.site
    s = state.spins[site]
    info = geometry.neighbor_table[site]

    J = couplings(model)
    weighted_sum = zero(promote_type(eltype(state.spins), eltype(J)))
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        weighted_sum += J[template] * state.spins[neighbor]
    end

    return (s - proposal.new_value) * (weighted_sum + field_strength(model, geometry, site))
end

"""
    cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T)

Return the Wolff bond-activation probability for a ferromagnetic Ising bond,
given `bond_template`, the index of the bond's template in `geometry.bonds`
(taken from `neighbor_table[site].templates`). Only equal neighboring spins
can join the cluster.

The coupling signs are checked once per update by
[`validate_cluster_model`](@ref) (cluster updates require non-negative,
ferromagnetic couplings), not per bond visit.
"""
function cluster_bond_probability(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    site::Int,
    neighbor::Int,
    bond_template::Int,
    T::Real
)
    J = couplings(model)[bond_template]
    state.spins[site] == state.spins[neighbor] || return 0.0
    return -expm1(-2 * J / T)
end

mutable struct IsingClusterContext
    ghost_attached::Bool
end

cluster_context(::AbstractRNG, ::IsingModel, ::Geometry, ::IsingState, ::Real) =
    IsingClusterContext(false)

cluster_ghost_count(::IsingModel, ::Geometry, ::IsingState, ::IsingClusterContext) = 2

function cluster_ghost_probability(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    site::Int,
    ghost::Int,
    T::Real,
    ::IsingClusterContext,
)
    ghost in 1:2 || throw(ArgumentError("Ising ghost index must be 1 or 2; got $ghost"))
    ghost_spin = ghost == 1 ? 1 : -1
    field = field_strength(model, geometry, site)
    state.spins[site] == ghost_spin || return 0.0
    return -expm1(-2 * abs(field) / T) * (sign(field) == ghost_spin)
end

function cluster_ghost_attached!(context::IsingClusterContext, ::Int)
    context.ghost_attached = true
    return nothing
end

"""
    apply_cluster!(model, geometry, state, cluster)

Flip a Wolff cluster and update the cached Ising observables incrementally.
Only bonds crossing the cluster boundary change energy: with pre-flip spins,
a boundary bond `(i, j)` with `i` in the cluster contributes
`ΔE = 2 * J * spins[i] * spins[j]`, and each flipped site contributes a field
term `ΔE = 2 * h * spins[i]` and `ΔM = -2 * spins[i]`.

This explicit form allocates a membership mask per call and serves as the
readable reference implementation (exercised by the tests); the Wolff driver
passes down the mask it already built.
"""
function apply_cluster!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    cluster::Vector{Int}
)
    in_cluster = falses(geometry.n_sites)
    for site in cluster
        in_cluster[site] = true
    end
    _flip_ising_cluster!(model, geometry, state, cluster, in_cluster)
    return nothing
end

# `in_cluster` must be true exactly on the cluster's sites. Wolff's `step!`
# passes down the mask it built while growing the cluster.
function _flip_ising_cluster!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    cluster::Vector{Int},
    in_cluster::AbstractVector{Bool},
)
    ΔE = zero(state.energy)
    spin_change = zero(state.magnetization)
    J = couplings(model)

    for site in cluster
        s = state.spins[site]
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || in_cluster[neighbor]) && continue
            ΔE += 2 * J[template] * s * state.spins[neighbor]
        end
        ΔE += 2 * field_strength(model, geometry, site) * s
        state.spins[site] = -s
        spin_change += 2 * state.spins[site]
    end

    state.energy += ΔE
    state.magnetization += spin_change
    return nothing
end

function apply_cluster!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    cluster::Vector{Int},
    context::IsingClusterContext,
    in_cluster::BitVector,
)
    context.ghost_attached || _flip_ising_cluster!(
        model, geometry, state, cluster, in_cluster
    )
    return nothing
end

"""
    apply_clusters!(rng, model, geometry, state, clusters::Vector{Vector{Int}})

Explicit-partition form of [`apply_clusters!`](@ref): flip each cluster in
`clusters` independently with probability 1/2. Allocates its own label map;
the Swendsen-Wang driver uses the allocation-free workspace form below. This
explicit form is the readable reference implementation exercised by the
tests.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    clusters::Vector{Vector{Int}},
)
    flip = [rand(rng, Bool) for _ in clusters]
    _flip_ising_clusters!(model, geometry, state, _cluster_labels(geometry, clusters), flip)
    return nothing
end

"""
    apply_clusters!(rng, model, geometry, state, workspace, context)

Randomly orient each Swendsen-Wang cluster and update the cached Ising
observables incrementally via [`_flip_ising_clusters!`](@ref). The
orientation decisions are drawn by [`draw_cluster_flips!`](@ref) — one coin
flip per cluster, in the first-encounter order of `workspace.roots`, with
ghost-attached clusters never flipped — and recorded per root-id label in
`workspace.flip`.

`context` is unused for the Ising model: ghost attachment is read from the
labels alone.

A bond only changes energy when exactly one of its endpoints sits in a
flipped cluster, so the energy update touches only bonds incident to flipped
sites rather than resumming the whole lattice.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    workspace::ClusterWorkspace,
    context = nothing,
)
    flip = draw_cluster_flips!(rng, workspace, geometry)
    _flip_ising_clusters!(model, geometry, state, workspace.labels, flip)
    return nothing
end

"""
    _flip_ising_clusters!(model, geometry, state, labels, flip)

Flip every lattice site whose cluster label is marked in `flip`, accumulating
the exact energy and magnetization change. `labels[i]` is the union-find root
of site `i` (see [`step!`](@ref) with [`SwendsenWangAlgorithm`](@ref)) and
doubles as the site→cluster map: a bond incident to a flipped site is
internal (both endpoints flip, no change) when `flip[labels[neighbor]]` is
true, external (never visited) when neither endpoint flips, and crossing
(counted once, from the flipped side, with pre-flip spins) otherwise.
"""
function _flip_ising_clusters!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    labels::Vector{Int},
    flip::Vector{Bool},
)
    ΔE = zero(state.energy)
    ΔM = zero(state.magnetization)
    J = couplings(model)

    for site in 1:geometry.n_sites
        flip[labels[site]] || continue
        s = state.spins[site]
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || flip[labels[neighbor]]) && continue
            ΔE += 2 * J[template] * s * state.spins[neighbor]
        end
        ΔE += 2 * field_strength(model, geometry, site) * s
        state.spins[site] = -s
        ΔM -= 2 * s
    end

    state.energy += ΔE
    state.magnetization += ΔM
    return nothing
end