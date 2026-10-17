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

function field_strength(model::IsingModel, geometry::Geometry, site::Int)
    return model.h[LatticeUtilities.site_to_orbital(site, geometry.unit_cell)]
end

"""
    bond_strength(model, geometry, bond_id)

Return the coupling strength `J` for the bond identified by `bond_id`, looked
up via the bond's template in `geometry`.

# Arguments

- `model`: The `IsingModel` supplying bond strengths.
- `geometry`: The lattice geometry mapping bond ids to templates.
- `bond_id`: Index of the bond to look up.
"""
function bond_strength(model::IsingModel, geometry::Geometry, bond_id::Int)
    1 <= bond_id <= length(geometry.bond_id_to_template) || throw(ArgumentError(
        "bond_id = $bond_id is out of bounds for geometry.bond_id_to_template of length $(length(geometry.bond_id_to_template))"
    ))
    template_id = geometry.bond_id_to_template[bond_id]
    return model.J[template_id]
end

"""
    local_operator(state, i)

Return the value of the local spin operator at site `i` for `state`.
"""
local_operator(state::IsingState, i) = state.spins[i]

"""
    operator_product(model, a, b)

Return the two-body coupling contribution `a * b` between a pair of local
operator values, as used when computing bond energies for `model`.
"""
operator_product(model::IsingModel, a::Real, b::Real) = a * b

"""
    propose(::MetropolisAlgorithm, model, geometry, state)

Generate a Metropolis trial move for `model`: pick a random site and propose
flipping its spin.

# Returns

A `SpinUpdate` describing the site and its proposed (flipped) value.
"""
function propose(::MetropolisAlgorithm, ::IsingModel, geometry, state)
    site = rand(eachindex(state.spins))
    return SpinUpdate(site, -state.spins[site])
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

    weighted_sum = zero(promote_type(eltype(state.spins), eltype(model.J)))
    for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
        neighbor == site && continue
        weighted_sum += bond_strength(model, geometry, bond_id) * state.spins[neighbor]
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
    cluster_bond_probability(model, geometry, state, site, neighbor, bond_id, T)

Return the Wolff bond-activation probability for a ferromagnetic Ising bond.
Only equal neighboring spins can join the cluster.

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
    bond_id::Int,
    T::Real
)
    J = bond_strength(model, geometry, bond_id)
    J >= 0 || throw(ArgumentError(
        "WolffAlgorithm requires non-negative Ising couplings; got J = $J"
    ))
    state.spins[site] == state.spins[neighbor] || return 0.0
    return -expm1(-2 * J / T)
end

mutable struct IsingClusterContext
    ghost_attached::Bool
end

cluster_context(::IsingModel, ::Geometry, ::IsingState, ::Real) =
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

    ΔE = zero(state.energy)
    spin_change = zero(eltype(state.spins))

    for site in cluster
        s = state.spins[site]
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            (neighbor == site || in_cluster[neighbor]) && continue
            ΔE += 2 * bond_strength(model, geometry, bond_id) * s *
                state.spins[neighbor]
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
)
    context.ghost_attached || apply_cluster!(model, geometry, state, cluster)
    return nothing
end

"""
    apply_clusters!(model, geometry, state, clusters)

Randomly orient each Swendsen-Wang cluster and update the cached Ising
observables incrementally. A bond only changes energy when exactly one of its
endpoints sits in a flipped cluster, so the energy update touches only bonds
incident to flipped sites rather than resumming the whole lattice.
"""
function apply_clusters!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    clusters::Vector{Vector{Int}}
)
    flip = [rand(Bool) for _ in clusters]
    _flip_ising_clusters!(model, geometry, state, clusters, flip)
    return nothing
end

function apply_clusters!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    clusters::Vector{Vector{Int}},
    ::IsingClusterContext,
)
    flip = [
        rand(Bool) && !any(site -> site > geometry.n_sites, cluster)
        for cluster in clusters
    ]
    _flip_ising_clusters!(model, geometry, state, clusters, flip)
    return nothing
end

"""
    _flip_ising_clusters!(model, geometry, state, clusters, flip)

Flip every cluster `c` with `flip[c] == true`, accumulating the exact energy
and magnetization change. A site→cluster index map identifies each bond
incident to a flipped site as internal (both endpoints flip, no change),
external (neither flips, never visited), or crossing (counted once, from the
flipped side, with pre-flip spins).
"""
function _flip_ising_clusters!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    clusters::Vector{Vector{Int}},
    flip::Vector{Bool},
)
    cluster_id = zeros(Int, geometry.n_sites)
    for (c, cluster) in enumerate(clusters), site in cluster
        site <= geometry.n_sites && (cluster_id[site] = c)
    end

    ΔE = zero(state.energy)
    ΔM = zero(state.magnetization)

    for (c, cluster) in enumerate(clusters)
        flip[c] || continue
        for site in cluster
            s = state.spins[site]
            info = geometry.neighbor_table[site]
            for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
                (neighbor == site || flip[cluster_id[neighbor]]) && continue
                ΔE += 2 * bond_strength(model, geometry, bond_id) * s *
                    state.spins[neighbor]
            end
            ΔE += 2 * field_strength(model, geometry, site) * s
            state.spins[site] = -s
            ΔM -= 2 * s
        end
    end

    state.energy += ΔE
    state.magnetization += ΔM
    return nothing
end

apply_clusters!(model, geometry, state, clusters, ::Nothing) =
    apply_clusters!(model, geometry, state, clusters)