struct IsingModel{T<:Real} <: AbstractModel
    J::Vector{T}   # one entry per bond template, same order as geometry.bonds
    h::T
end

"""
    IsingModel(J, h)

An Ising model with bond strengths `J` and external field `h`.

# Arguments

- `J`: Bond strengths, one value per bond template. Its length is checked
    against `geometry.bonds` by [`initialize_state`](@ref), since the
    constructor does not receive a geometry.
- `h`: External magnetic field.
"""
function IsingModel(J::Vector{<:Real}, h::Real)
    T = promote_type(eltype(J), typeof(h))
    return IsingModel(convert(Vector{T}, J), convert(T, h))
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

    weighted_sum = sum(
        bond_strength(model, geometry, bond_id) * state.spins[j]
         for (bond_id, j) in zip(info.bonds, info.neighbors)
         if j != site;
        init = zero(eltype(state.spins))
    )

    return (s - proposal.new_value) * (weighted_sum + model.h)
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