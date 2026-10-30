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

# The local operator value of an XY site is its unit vector (cos θ, sin θ),
# read straight from the state's component arrays.
local_operator(state::XYState, site) = (state.cosines[site], state.sines[site])

state_type(::XYModel) = XYState

magnetization_value(state::XYState) = hypot(state.magnetization_x, state.magnetization_y)

binder_denominator(::XYModel) = 2.0

"""
    operator_product(model, a, b)

Return the two-body coupling contribution between a pair of local operator
values, as used when computing bond energies for `model`. For the XY model
the values are the unit vectors `(cos θ, sin θ)` and the product is their
dot product, `cos(θ_a - θ_b) = c_a c_b + s_a s_b`.
"""
operator_product(::XYModel, a::NTuple{2,Real}, b::NTuple{2,Real}) = a[1] * b[1] + a[2] * b[2]

n_operator_components(::XYModel) = 2
operator_components(::XYModel, v::NTuple{2,Real}) = v

"""
    propose(rng, ::MetropolisAlgorithm, model::XYModel, geometry, state, site)

Generate a Metropolis trial move for the XY model at `site` (chosen by the
algorithm's [`AbstractProposalScheme`](@ref)): propose a new spin drawn
uniformly from the unit circle via `rng`.

# Returns

A [`SpinUpdate`](@ref) carrying the proposed components `(cos θ, sin θ)` as
`new_value` and their change from the current components as
`magnetization_change`.
"""
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, ::XYModel, geometry, state::XYState, site::Int)
    s_new, c_new = sincos(2π * rand(rng))
    return SpinUpdate(
        site, (c_new, s_new),
        (c_new - state.cosines[site], s_new - state.sines[site]),
    )
end

"""
    energy_difference(model, geometry, state, proposal)

Compute the change in total energy `ΔE` that would result from changing the
spin at `proposal.site` to `proposal.new_value`, given the couplings and
field in `model`.

Written with the spin components `v_i = (c_i, s_i)` directly, so the neighbor
loop is free of trigonometric calls: with `Δv = v_new - v_old` and the local
field sums `C = Σ J c_j`, `S = Σ J s_j` over the site's bonds,
`ΔE = -(Δc * C + Δs * S) - h * Δc`.

# Returns

The energy difference `ΔE` between the proposed and current configurations.
"""
function energy_difference(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    proposal::SpinUpdate
)
    site = proposal.site
    info = geometry.neighbor_table[site]
    J = couplings(model)
    T = promote_type(eltype(state.cosines), eltype(J))
    C = zero(T)
    S = zero(T)
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        C += J[template] * state.cosines[neighbor]
        S += J[template] * state.sines[neighbor]
    end
    # The carried change is (cos(new) - cos(old), sin(new) - sin(old)).
    dc, ds = proposal.magnetization_change
    return -(dc * C + ds * S) - field_strength(model, geometry, site) * dc
end

"""
    XYClusterContext

Per-update state of an XY cluster move (Wolff or Swendsen-Wang): the random
reflection axis `r̂ = (axis_cos, axis_sin)` (stored both as the angle
`reflection_axis` and as precomputed components, so projections
`p_i = v_i · r̂` need no trigonometric calls), and whether the growing Wolff
cluster has bonded to a ghost spin (in which case it is not reflected).
"""
mutable struct XYClusterContext
    reflection_axis::Float64
    axis_cos::Float64
    axis_sin::Float64
    ghost_attached::Bool
end

function cluster_context(rng::AbstractRNG, ::XYModel, ::Geometry, ::XYState, ::Real)
    reflection_axis = 2π * rand(rng)
    axis_sin, axis_cos = sincos(reflection_axis)
    return XYClusterContext(reflection_axis, axis_cos, axis_sin, false)
end

cluster_ghost_count(::XYModel, ::Geometry, ::XYState, ::XYClusterContext) = 2

# Component of a site's spin along the reflection axis r̂ = (axis_cos, axis_sin).
_projection(state::XYState, site::Int, axis_cos::Real, axis_sin::Real) =
    state.cosines[site] * axis_cos + state.sines[site] * axis_sin

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
    axis_sin, axis_cos = sincos(reflection_axis)
    projection_product = _projection(state, site, axis_cos, axis_sin) *
        _projection(state, neighbor, axis_cos, axis_sin)
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
    projection_product = _projection(state, site, context.axis_cos, context.axis_sin) *
        _projection(state, neighbor, context.axis_cos, context.axis_sin)
    projection_product > 0 || return 0.0
    return -expm1(-2 * couplings(model)[bond_template] * projection_product / T)
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
    # The ghost spins point along ±x, so their axis components are ±cos(φ).
    ghost_projection = ghost == 1 ? context.axis_cos : -context.axis_cos
    projection_product = _projection(state, site, context.axis_cos, context.axis_sin) *
        ghost_projection
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
observables incrementally. With `p_i = v_i · r̂` the pre-reflection component
of site `i`'s spin along the axis `r̂ = (cos φ, sin φ)`, a boundary bond
`(i, j)` with `i` in the cluster contributes `ΔE = 2 * J * p_i * p_j`, and
each reflected site contributes `ΔE = 2 * h * cos(φ) * p_i`,
`ΔM_x = -2 * cos(φ) * p_i`, and `ΔM_y = -2 * sin(φ) * p_i`. The reflection
itself is the linear map `v ↦ v - 2 (v·r̂) r̂`.

This explicit form allocates a membership mask per call and serves as the
readable reference implementation (exercised by the tests); the Wolff driver
passes down the mask it already built.
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
    sin_axis, cos_axis = sincos(reflection_axis)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    J = couplings(model)

    for site in cluster
        c = state.cosines[site]
        s = state.sines[site]
        projection = c * cos_axis + s * sin_axis
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || in_cluster[neighbor]) && continue
            ΔE += 2 * J[template] * projection *
                _projection(state, neighbor, cos_axis, sin_axis)
        end
        ΔE += 2 * field_strength(model, geometry, site) * cos_axis * projection
        ΔMx -= 2 * cos_axis * projection
        ΔMy -= 2 * sin_axis * projection
        state.cosines[site] = c - 2 * projection * cos_axis
        state.sines[site] = s - 2 * projection * sin_axis
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
incident to a reflected site is internal (both endpoints reflect, and the
reflection `v ↦ v - 2 (v·r̂) r̂` preserves their dot product) when
`flip[labels[neighbor]]` is true, external (never visited) when neither
endpoint reflects, and crossing (counted once, from the reflected side, with
pre-reflection components) otherwise.
"""
function _reflect_xy_clusters!(
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    labels::Vector{Int},
    flip::Vector{Bool},
    reflection_axis::Real,
)
    sin_axis, cos_axis = sincos(reflection_axis)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    J = couplings(model)

    for site in 1:geometry.n_sites
        flip[labels[site]] || continue
        c = state.cosines[site]
        s = state.sines[site]
        projection = c * cos_axis + s * sin_axis
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || flip[labels[neighbor]]) && continue
            ΔE += 2 * J[template] * projection *
                _projection(state, neighbor, cos_axis, sin_axis)
        end
        ΔE += 2 * field_strength(model, geometry, site) * cos_axis * projection
        ΔMx -= 2 * cos_axis * projection
        ΔMy -= 2 * sin_axis * projection
        state.cosines[site] = c - 2 * projection * cos_axis
        state.sines[site] = s - 2 * projection * sin_axis
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    return nothing
end
