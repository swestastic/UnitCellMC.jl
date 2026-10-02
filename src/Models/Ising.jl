struct IsingModel{T<:Real} <: AbstractModel
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

iszero_field(h::AbstractVector) = all(iszero, h)

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

function validate_cluster_model(
    model::IsingModel,
    ::Geometry,
    ::IsingState,
    ::Real
)
    iszero_field(model.h) || throw(ArgumentError(
        "WolffAlgorithm requires a zero external field; got h = $(model.h)"
    ))
    return nothing
end

"""Flip a Wolff cluster and update the cached Ising observables."""
function apply_cluster!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    cluster::Vector{Int}
)
    spin_change = zero(eltype(state.spins))

    for site in cluster
        state.spins[site] = -state.spins[site]
        spin_change += 2 * state.spins[site]
    end

    new_energy = zero(state.energy)
    for site in 1:geometry.n_sites
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            new_energy += -0.5 * bond_strength(model, geometry, bond_id) *
                state.spins[site] * state.spins[neighbor]
        end
        new_energy += -field_strength(model, geometry, site) * state.spins[site]
    end

    state.energy = new_energy
    state.magnetization += spin_change
    return nothing
end

"""Randomly orient Swendsen-Wang clusters and refresh Ising observables."""
function apply_clusters!(
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    clusters::Vector{Vector{Int}}
)
    for cluster in clusters
        rand(Bool) || continue
        for site in cluster
            state.spins[site] = -state.spins[site]
        end
    end

    state.energy = zero(state.energy)
    for site in 1:geometry.n_sites
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            state.energy += -0.5 * bond_strength(model, geometry, bond_id) *
                state.spins[site] * state.spins[neighbor]
        end
        state.energy += -field_strength(model, geometry, site) * state.spins[site]
    end
    state.magnetization = sum(state.spins)
    return nothing
end

apply_clusters!(model, geometry, state, clusters, ::Nothing) =
    apply_clusters!(model, geometry, state, clusters)