"""
    XYModel{T<:Real} <: AbstractModel

The classical XY model with bond strengths `J` (one value per bond template)
and external fields `h` (one value per site in the unit cell). The field
points along the x direction, so its contribution is
`-h[i] * cos(angle[i])`.
"""
struct XYModel{T<:Real} <: AbstractModel{T}
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

local_operator(state::XYState, site) = state.angles[site]

state_type(::XYModel) = XYState

magnetization_value(state::XYState) = hypot(state.magnetization_x, state.magnetization_y)

binder_denominator(::XYModel) = 2.0

operator_product(::XYModel, first_angle::Real, second_angle::Real) =
    cos(first_angle - second_angle)

n_operator_components(::XYModel) = 2
operator_components(::XYModel, angle::Real) = (cos(angle), sin(angle))

"""
    propose(rng, ::MetropolisAlgorithm, model::XYModel, geometry, state, site)

Generate a Metropolis trial move for the XY model at `site` (chosen by the
algorithm's [`AbstractProposalScheme`](@ref)): propose a new angle drawn
uniformly from `[0, 2π)` via `rng`.

# Returns

A [`SpinUpdate`](@ref) for the site and new angle, carrying the
magnetization change.
"""
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, ::XYModel, geometry, state::XYState, site::Int)
    old = state.angles[site]
    new = 2π * rand(rng)
    return SpinUpdate(site, new, (cos(new) - cos(old), sin(new) - sin(old)))
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
    J = couplings(model)
    weighted_sum = zero(promote_type(eltype(state.angles), eltype(J)))
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        weighted_sum += J[template] *
            (cos(proposal.new_value - state.angles[neighbor]) - cos(angle - state.angles[neighbor]))
    end
    return -weighted_sum - field_strength(model, geometry, site) *
        (cos(proposal.new_value) - cos(angle))
end

mutable struct XYClusterContext
    reflection_axis::Float64
    ghost_attached::Bool
end

cluster_context(rng::AbstractRNG, ::XYModel, ::Geometry, ::XYState, ::Real) =
    XYClusterContext(2π * rand(rng), false)

cluster_ghost_count(::XYModel, ::Geometry, ::XYState, ::XYClusterContext) = 2

function cluster_bond_probability(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    site::Int,
    neighbor::Int,
    bond_template::Int,
    T::Real,
    reflection_axis::Real,
)
    projection_product = cos(state.angles[site] - reflection_axis) *
        cos(state.angles[neighbor] - reflection_axis)
    projection_product > 0 || return 0.0
    return -expm1(-2 * couplings(model)[bond_template] * projection_product / T)
end

function cluster_bond_probability(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    site::Int,
    neighbor::Int,
    bond_template::Int,
    T::Real,
    context::XYClusterContext,
)
    return cluster_bond_probability(
        model, geometry, state, site, neighbor, bond_template, T,
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
This explicit form allocates a membership mask per call and is the
readable reference implementation exercised by the tests; the Wolff
driver passes down the mask it already built.
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
    _reflect_xy_cluster!(model, geometry, state, cluster, in_cluster, reflection_axis)
    return nothing
end

# `in_cluster` must be true exactly on the cluster's sites. Wolff's `step!`
# passes down the mask it built while growing the cluster.
function _reflect_xy_cluster!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    cluster::Vector{Int},
    in_cluster::AbstractVector{Bool},
    reflection_axis::Real,
)
    cos_axis = cos(reflection_axis)
    sin_axis = sin(reflection_axis)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    J = couplings(model)

    for site in cluster
        angle = state.angles[site]
        projection = cos(angle - reflection_axis)
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || in_cluster[neighbor]) && continue
            ΔE += 2 * J[template] * projection *
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
    in_cluster::BitVector,
)
    context.ghost_attached || _reflect_xy_cluster!(
        model, geometry, state, cluster, in_cluster, context.reflection_axis
    )
    return nothing
end

"""
    apply_clusters!(rng, model, geometry, state, clusters::Vector{Vector{Int}}, reflection_axis)

Explicit-partition form of [`apply_clusters!`](@ref): reflect each cluster in
`clusters` about `reflection_axis` independently with probability 1/2.
Allocates its own label map; the Swendsen-Wang driver uses the allocation-free
workspace form below. This
explicit form is the readable reference implementation exercised by the
tests.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    clusters::Vector{Vector{Int}},
    reflection_axis::Real,
)
    flip = [rand(rng, Bool) for _ in clusters]
    _reflect_xy_clusters!(
        model, geometry, state, _cluster_labels(geometry, clusters), flip, reflection_axis
    )
    return nothing
end

"""
    apply_clusters!(rng, model, geometry, state, workspace, context)

Randomly reflect each Swendsen-Wang cluster about `context.reflection_axis`
and update the cached XY observables incrementally via
[`_reflect_xy_clusters!`](@ref). The orientation decisions are drawn by
[`draw_cluster_flips!`](@ref) — one coin flip per cluster, in the
first-encounter order of `workspace.roots`, with ghost-attached clusters
never reflected — and recorded per root-id label in `workspace.flip`.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    workspace::ClusterWorkspace,
    context::XYClusterContext,
)
    flip = draw_cluster_flips!(rng, workspace, geometry)
    _reflect_xy_clusters!(
        model, geometry, state, workspace.labels, flip, context.reflection_axis
    )
    return nothing
end

"""
    _reflect_xy_clusters!(model, geometry, state, labels, flip, reflection_axis)

Reflect every lattice site whose cluster label is marked in `flip`,
accumulating the exact energy and magnetization change. `labels[i]` is the
union-find root of site `i` (see [`step!`](@ref) with
[`SwendsenWangAlgorithm`](@ref)) and doubles as the site→cluster map: a bond
incident to a reflected site is internal (both endpoints reflect, no change)
when `flip[labels[neighbor]]` is true, external (never visited) when neither
endpoint reflects, and crossing (counted once, from the reflected side, with
pre-reflection angles) otherwise.
"""
function _reflect_xy_clusters!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    labels::Vector{Int},
    flip::Vector{Bool},
    reflection_axis::Real,
)
    cos_axis = cos(reflection_axis)
    sin_axis = sin(reflection_axis)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    J = couplings(model)

    for site in 1:geometry.n_sites
        flip[labels[site]] || continue
        angle = state.angles[site]
        projection = cos(angle - reflection_axis)
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || flip[labels[neighbor]]) && continue
            ΔE += 2 * J[template] * projection *
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
