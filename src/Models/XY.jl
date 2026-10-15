"""
    XYModel{T<:Real} <: AbstractModel

The classical XY model with bond strengths `J` (one value per bond template)
and external fields `h` (one value per site in the unit cell). The field
points along the x direction, so its contribution is
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
    all(>=(0), model.J) || throw(ArgumentError(
        "XY cluster algorithms require non-negative couplings; got J = $(model.J)"
    ))
    nothing
end

mutable struct XYClusterContext
    reflection_axis::Float64
    ghost_attached::Bool
end

cluster_context(::XYModel, ::Geometry, ::XYState, ::Real) =
    XYClusterContext(2π * rand(), false)

cluster_ghost_count(::XYModel, ::Geometry, ::XYState, ::XYClusterContext) = 2

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

function cluster_bond_probability(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    site::Int,
    neighbor::Int,
    bond_id::Int,
    T::Real,
    context::XYClusterContext,
)
    return cluster_bond_probability(
        model, geometry, state, site, neighbor, bond_id, T,
        context.reflection_axis
    )
end

function cluster_ghost_probability(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    site::Int,
    ghost::Int,
    T::Real,
    context::XYClusterContext,
)
    ghost in 1:2 || throw(ArgumentError("XY ghost index must be 1 or 2; got $ghost"))
    field = field_strength(model, geometry, site)
    field == 0 && return 0.0
    sign(field) == (ghost == 1 ? 1 : -1) || return 0.0
    ghost_angle = ghost == 1 ? 0.0 : π
    projection_product = cos(state.angles[site] - context.reflection_axis) *
        cos(ghost_angle - context.reflection_axis)
    projection_product > 0 || return 0.0
    return -expm1(-2 * abs(field) * projection_product / T)
end

function cluster_ghost_attached!(context::XYClusterContext, ::Int)
    context.ghost_attached = true
    return nothing
end

"""
    apply_cluster!(model, geometry, state, cluster, reflection_axis)

Reflect a Wolff cluster about `reflection_axis` and update the cached XY
observables incrementally. With `p_i = cos(angles[i] - reflection_axis)`
evaluated pre-reflection, a boundary bond `(i, j)` with `i` in the cluster
contributes `ΔE = 2 * J * p_i * p_j` (via
`cos(2φ - θ_i - θ_j) + cos(θ_i - θ_j) = 2 * p_i * p_j`), and each reflected
site contributes `ΔE = 2 * h * cos(φ) * p_i`, `ΔM_x = -2 * cos(φ) * p_i`, and
`ΔM_y = -2 * sin(φ) * p_i`.
"""
function apply_cluster!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    cluster::Vector{Int},
    reflection_axis::Real,
)
    in_cluster = falses(geometry.n_sites)
    for site in cluster
        in_cluster[site] = true
    end

    cos_axis = cos(reflection_axis)
    sin_axis = sin(reflection_axis)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)

    for site in cluster
        angle = state.angles[site]
        projection = cos(angle - reflection_axis)
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            (neighbor == site || in_cluster[neighbor]) && continue
            ΔE += 2 * bond_strength(model, geometry, bond_id) * projection *
                cos(state.angles[neighbor] - reflection_axis)
        end
        ΔE += 2 * field_strength(model, geometry, site) * cos_axis * projection
        ΔMx -= 2 * cos_axis * projection
        ΔMy -= 2 * sin_axis * projection
        state.angles[site] = mod(2 * reflection_axis + π - angle, 2π)
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    return nothing
end

function apply_cluster!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    cluster::Vector{Int},
    context::XYClusterContext,
)
    context.ghost_attached || apply_cluster!(
        model, geometry, state, cluster, context.reflection_axis
    )
    return nothing
end

"""
    apply_clusters!(model, geometry, state, clusters, reflection_axis)

Randomly reflect each Swendsen-Wang cluster about `reflection_axis` and update
the cached XY observables incrementally. A bond only changes energy when
exactly one of its endpoints sits in a reflected cluster, so the energy update
touches only bonds incident to reflected sites rather than resumming the
whole lattice.
"""
function apply_clusters!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    clusters::Vector{Vector{Int}},
    reflection_axis::Real,
)
    flip = [rand(Bool) for _ in clusters]
    _reflect_xy_clusters!(
        model, geometry, state, clusters, flip, reflection_axis
    )
    return nothing
end

function apply_clusters!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    clusters::Vector{Vector{Int}},
    context::XYClusterContext,
)
    flip = [
        rand(Bool) && !any(site -> site > geometry.n_sites, cluster)
        for cluster in clusters
    ]
    _reflect_xy_clusters!(
        model, geometry, state, clusters, flip, context.reflection_axis
    )
    return nothing
end

"""
    _reflect_xy_clusters!(model, geometry, state, clusters, flip, reflection_axis)

Reflect every cluster `c` with `flip[c] == true`, accumulating the exact
energy and magnetization change. A site→cluster index map identifies each
bond incident to a reflected site as internal (both endpoints reflect, no
change), external (neither reflects, never visited), or crossing (counted
once, from the reflected side, with pre-reflection angles).
"""
function _reflect_xy_clusters!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    clusters::Vector{Vector{Int}},
    flip::Vector{Bool},
    reflection_axis::Real,
)
    cluster_id = zeros(Int, geometry.n_sites)
    for (c, cluster) in enumerate(clusters), site in cluster
        site <= geometry.n_sites && (cluster_id[site] = c)
    end

    cos_axis = cos(reflection_axis)
    sin_axis = sin(reflection_axis)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)

    for (c, cluster) in enumerate(clusters)
        flip[c] || continue
        for site in cluster
            angle = state.angles[site]
            projection = cos(angle - reflection_axis)
            info = geometry.neighbor_table[site]
            for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
                (neighbor == site || flip[cluster_id[neighbor]]) && continue
                ΔE += 2 * bond_strength(model, geometry, bond_id) * projection *
                    cos(state.angles[neighbor] - reflection_axis)
            end
            ΔE += 2 * field_strength(model, geometry, site) * cos_axis * projection
            ΔMx -= 2 * cos_axis * projection
            ΔMy -= 2 * sin_axis * projection
            state.angles[site] = mod(2 * reflection_axis + π - angle, 2π)
        end
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    return nothing
end
