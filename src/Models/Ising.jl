struct IsingModel{T<:Real} <: AbstractModel
    J::Vector{T}   # one entry per bond template, same order as geometry.bonds
    h::T
end

# Promotes J and h to a common type so e.g. IsingModel([1, 2], 0.5) still works
# (Int J, Float64 h) without requiring the caller to match types by hand.
function IsingModel(J::Vector{<:Real}, h::Real)
    T = promote_type(eltype(J), typeof(h))
    return IsingModel(convert(Vector{T}, J), convert(T, h))
end

function bond_strength(model::IsingModel, geometry::Geometry, bond_id::Int)
    template_id = geometry.bond_id_to_template[bond_id]
    return model.J[template_id]
end

local_operator(state::IsingState, i) = state.spins[i]
operator_product(model::IsingModel, a::Real, b::Real) = a * b

function propose(::MetropolisAlgorithm, ::IsingModel, geometry, state)
    site = rand(eachindex(state.spins))
    return SpinUpdate(site, -state.spins[site])
end

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
    )

    return 2 * s * (weighted_sum + model.h)
end

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