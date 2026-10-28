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

function validate_model_geometry(model::IsingModel, geometry::Geometry)
    invoke(validate_model_geometry, Tuple{AbstractModel, Geometry}, model, geometry)

    n_orbitals = LatticeUtilities.norbits(geometry.unit_cell)
    length(model.h) == n_orbitals || throw(ArgumentError(
        "length(model.h) = $(length(model.h)) does not match the number of sites in the unit cell = $n_orbitals"
    ))
    return nothing
end

"""
    local_operator(state, i)

Return the value of the local spin operator at site `i` for `state`.
"""
local_operator(state::IsingState, i) = state.spins[i]

state_type(::IsingModel) = IsingState

magnetization_value(state::IsingState) = state.magnetization

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
    return SpinUpdate(site, -state.spins[site])
end

"""
    propose(rng, ::GlauberAlgorithm, model::IsingModel, geometry, state)

Generate a Glauber trial move for `model`: pick a random site, drawn from
`rng`, and propose flipping its spin. This is the same symmetric proposal as
for [`MetropolisAlgorithm`](@ref); only the acceptance rule in
[`step!`](@ref) differs.

# Returns

A `SpinUpdate` describing the site and its proposed (flipped) value.
"""
function propose(rng::AbstractRNG, ::GlauberAlgorithm, model::IsingModel, geometry, state)
    return propose(rng, MetropolisAlgorithm(), model, geometry, state)
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

    J = model.J
    weighted_sum = zero(promote_type(eltype(state.spins), eltype(J)))
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        weighted_sum += J[template] * state.spins[neighbor]
    end

    return (s - proposal.new_value) * (weighted_sum + field_strength(model, geometry, site))
end

"""
    apply_update!(state, proposal, ΔE)

Apply `proposal` to `state` in place, updating the spin at the proposed site
along with the running total energy and magnetization.

# Arguments

- `state`: The `IsingState` to mutate.
- `proposal`: A `SpinUpdate` describing the site and its new value.
- `ΔE`: The precomputed energy change from applying `proposal`.
"""
function apply_update!(
    state::IsingState,
    proposal::SpinUpdate,
    ΔE::Real
)
    site = proposal.site
    s = state.spins[site]
    state.spins[site] = proposal.new_value
    state.energy += ΔE
    state.magnetization += proposal.new_value - s
    return nothing
end

"""
    cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T)

Return the Wolff bond-activation probability for a ferromagnetic Ising bond,
given `bond_template`, the index of the bond's template in `geometry.bonds`
(taken from `neighbor_table[site].templates`). Only equal neighboring spins
can join the cluster.

# Throws

`ArgumentError` if the bond coupling is negative; Wolff cluster updates
require non-negative (ferromagnetic) couplings.
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
    J = model.J[bond_template]
    J >= 0 || throw(ArgumentError(
        "WolffAlgorithm requires non-negative Ising couplings; got J = $J"
    ))
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

function validate_cluster_model(
    model::IsingModel,
    ::Geometry,
    ::IsingState,
    ::Real
)
    all(>=(0), model.J) || throw(ArgumentError(
        "Ising cluster algorithms require non-negative couplings; got J = $(model.J)"
    ))
    return nothing
end

"""
    apply_cluster!(model, geometry, state, cluster)

Flip a Wolff cluster and update the cached Ising observables incrementally.
Only bonds crossing the cluster boundary change energy: with pre-flip spins,
a boundary bond `(i, j)` with `i` in the cluster contributes
`ΔE = 2 * J * spins[i] * spins[j]`, and each flipped site contributes a field
term `ΔE = 2 * h * spins[i]` and `ΔM = -2 * spins[i]`.
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
    J = model.J

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
the Swendsen-Wang driver uses the workspace form below.
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

Randomly orient each Swendsen-Wang cluster (one coin flip per cluster, drawn
from `rng` in the first-encounter order of `workspace.roots`) and update the
cached Ising observables incrementally via [`_flip_ising_clusters!`](@ref).
Clusters are described by root-id labels: `workspace.labels[i]` is the label
of site `i`, and labels above `geometry.n_sites` belong to ghost sites, whose
clusters are never flipped. The orientation decisions are recorded per label
in `workspace.flip`.

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
    labels = workspace.labels
    flip = workspace.flip
    for root in workspace.roots
        flip[root] = rand(rng, Bool)
    end
    for ghost in (geometry.n_sites + 1):length(labels)
        flip[labels[ghost]] = false
    end
    _flip_ising_clusters!(model, geometry, state, labels, flip)
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
    J = model.J

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