"""
    XYModel{T<:Real} <: AbstractModel

The classical XY model with bond strengths `J` and site-resolved external
fields `h`. The field points along the x direction, so its contribution is
`-h[i] * cos(angle[i])`.
"""
struct XYModel{T<:Real} <: AbstractModel
    J::Vector{T}
    h::Vector{T}

    function XYModel{T}(J::Vector{T}, h::Vector{T}) where {T<:Real}
        return new{T}(J, h)
    end
end

function XYModel(J::Vector{<:Real}, h::AbstractVector{<:Real})
    T = promote_type(eltype(J), eltype(h))
    return XYModel{T}(convert(Vector{T}, J), convert(Vector{T}, h))
end

initialize_state(model::XYModel, geometry::Geometry) =
    _initialize_xy_state(model, geometry)

function validate_model_geometry(model::XYModel, geometry::Geometry)
    invoke(validate_model_geometry, Tuple{AbstractModel, Geometry}, model, geometry)

    n_orbitals = LatticeUtilities.norbits(geometry.unit_cell)
    length(model.h) == n_orbitals || throw(ArgumentError(
        "length(model.h) = $(length(model.h)) does not match the number of sites in the unit cell = $n_orbitals"
    ))
    return nothing
end

field_strength(model::XYModel, geometry::Geometry, site::Int) =
    model.h[LatticeUtilities.site_to_orbital(site, geometry.unit_cell)]

bond_strength(model::XYModel, geometry::Geometry, bond_id::Int) = begin
    1 <= bond_id <= length(geometry.bond_id_to_template) || throw(ArgumentError(
        "bond_id = $bond_id is out of bounds for geometry.bond_id_to_template of length $(length(geometry.bond_id_to_template))"
    ))
    model.J[geometry.bond_id_to_template[bond_id]]
end

local_operator(state::XYState, site) = state.angles[site]
operator_product(::XYModel, first_angle::Real, second_angle::Real) =
    cos(first_angle - second_angle)

function propose(::MetropolisAlgorithm, ::XYModel, geometry, state::XYState)
    site = rand(eachindex(state.angles))
    return SpinUpdate(site, 2π * rand())
end

function energy_difference(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    proposal::SpinUpdate
)
    site = proposal.site
    angle = state.angles[site]
    info = geometry.neighbor_table[site]
    weighted_sum = zero(promote_type(eltype(state.angles), eltype(model.J)))
    for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
        neighbor == site && continue
        weighted_sum += bond_strength(model, geometry, bond_id) *
            (cos(proposal.new_value - state.angles[neighbor]) - cos(angle - state.angles[neighbor]))
    end
    return -weighted_sum - field_strength(model, geometry, site) *
        (cos(proposal.new_value) - cos(angle))
end

function apply_update!(state::XYState, proposal::SpinUpdate, ΔE::Real)
    site = proposal.site
    old_angle = state.angles[site]
    new_angle = proposal.new_value
    state.angles[site] = new_angle
    state.energy += ΔE
    state.magnetization_x += cos(new_angle) - cos(old_angle)
    state.magnetization_y += sin(new_angle) - sin(old_angle)
    return nothing
end

validate_cluster_model(model::XYModel, ::Geometry, ::XYState, ::Real) = begin
    iszero_field(model.h) || throw(ArgumentError(
        "XY cluster algorithms require a zero external field; got h = $(model.h)"
    ))
    all(>=(0), model.J) || throw(ArgumentError(
        "XY cluster algorithms require non-negative couplings; got J = $(model.J)"
    ))
    nothing
end

cluster_context(::XYModel, ::Geometry, ::XYState, ::Real) = 2π * rand()

function cluster_bond_probability(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    site::Int,
    neighbor::Int,
    bond_id::Int,
    T::Real,
    reflection_axis::Real,
)
    projection_product = cos(state.angles[site] - reflection_axis) *
        cos(state.angles[neighbor] - reflection_axis)
    projection_product > 0 || return 0.0
    return -expm1(-2 * bond_strength(model, geometry, bond_id) * projection_product / T)
end

function apply_cluster!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    cluster::Vector{Int},
    reflection_axis::Real,
)
    for site in cluster
        state.angles[site] = mod(2 * reflection_axis + π - state.angles[site], 2π)
    end
    _refresh_xy_state!(model, geometry, state)
    return nothing
end

function apply_clusters!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    clusters::Vector{Vector{Int}},
    reflection_axis::Real,
)
    for cluster in clusters
        rand(Bool) || continue
        for site in cluster
            state.angles[site] = mod(2 * reflection_axis + π - state.angles[site], 2π)
        end
    end
    _refresh_xy_state!(model, geometry, state)
    return nothing
end

function _refresh_xy_state!(model::XYModel, geometry::Geometry, state::XYState)
    state.energy = zero(state.energy)
    state.magnetization_x = zero(state.magnetization_x)
    state.magnetization_y = zero(state.magnetization_y)
    for site in 1:geometry.n_sites
        angle = state.angles[site]
        state.magnetization_x += cos(angle)
        state.magnetization_y += sin(angle)
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            state.energy += -0.5 * bond_strength(model, geometry, bond_id) *
                cos(angle - state.angles[neighbor])
        end
        state.energy += -field_strength(model, geometry, site) * cos(angle)
    end
    return nothing
end
