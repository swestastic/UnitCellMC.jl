"""
    HeisenbergModel{T<:Real} <: AbstractModel

The classical Heisenberg model: unit spin vectors ``v_i \\in S^2`` with bond
strengths `J` (one value per bond template) and external fields `h` (one
value per site in the unit cell), with Hamiltonian

    H = -Σ_⟨ij⟩ J_ij (v_i · v_j) - Σ_i h_i x_i

The field points along the `x` direction, as for the [`XYModel`](@ref) and
[`ClockModel`](@ref). Spin components are stored directly in
[`HeisenbergState`](@ref), so bond energies are plain dot products and the
Wolff/Swendsen-Wang cluster updates use the standard O(3) reflection
embedding `v ↦ v - 2 (v·r̂) r̂` about a random unit vector `r̂`.

Construct with `HeisenbergModel(J, h)`; see [`Geometry`](@ref) for how bond
templates map to the entries of `J`.
"""
struct HeisenbergModel{T<:Real} <: AbstractModel{T}
    J::Vector{T}
    h::Vector{T}

    function HeisenbergModel{T}(J::Vector{T}, h::Vector{T}) where {T<:Real}
        return new{T}(J, h)
    end
end

"""
    HeisenbergModel(J, h)

A Heisenberg model with bond strengths `J` and external field `h`.

# Arguments

- `J`: Bond strengths, one value per bond template. Its length is checked
    against `geometry.bonds` by [`initialize_state`](@ref), since the
    constructor does not receive a geometry.
- `h`: External field along `x`, one value per site in the unit cell.
"""
function HeisenbergModel(J::Vector{<:Real}, h::AbstractVector{<:Real})
    T = promote_type(eltype(J), eltype(h))
    return HeisenbergModel{T}(convert(Vector{T}, J), convert(Vector{T}, h))
end

# The local operator value of a Heisenberg site is its unit vector (x, y, z),
# read straight from the state's component arrays.
local_operator(state::HeisenbergState, site) =
    (state.xs[site], state.ys[site], state.zs[site])

state_type(::HeisenbergModel) = HeisenbergState

magnetization_value(state::HeisenbergState) =
    hypot(state.magnetization_x, state.magnetization_y, state.magnetization_z)

# Three-component order parameter: the high-temperature Gaussian cumulant
# ratio is (n + 2) / n = 5/3.
binder_denominator(::HeisenbergModel) = 5 / 3

"""
    operator_product(model::HeisenbergModel, a, b)

Return the two-body coupling contribution between a pair of local operator
values, as used when computing bond energies. For the Heisenberg model the
values are the unit vectors `(x, y, z)` and the product is their dot product.
"""
operator_product(::HeisenbergModel, a::NTuple{3,Real}, b::NTuple{3,Real}) =
    a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

n_operator_components(::HeisenbergModel) = 3
operator_components(::HeisenbergModel, v::NTuple{3,Real}) = v

"""
    propose(rng, ::MetropolisAlgorithm, model::HeisenbergModel, geometry, state, site)

Generate a Metropolis trial move for the Heisenberg model at `site` (chosen
by the algorithm's [`AbstractProposalScheme`](@ref)): propose a new spin
drawn uniformly from the unit sphere via [`random_unit_vector`](@ref).

# Returns

A [`SpinUpdate`](@ref) carrying the proposed components `(x, y, z)` as
`new_value` and their change from the current components as
`magnetization_change`.
"""
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, ::HeisenbergModel, geometry, state::HeisenbergState, site::Int)
    new = random_unit_vector(rng)
    return SpinUpdate(
        site, new,
        (new[1] - state.xs[site], new[2] - state.ys[site], new[3] - state.zs[site]),
    )
end

"""
    energy_difference(model::HeisenbergModel, geometry, state, proposal)

Compute the change in total energy `ΔE` that would result from changing the
spin at `proposal.site` to `proposal.new_value`, given the couplings and
field in `model`.

Written with the Cartesian components directly, so the neighbor loop is free
of trigonometric calls: with `Δv = v_new - v_old` and the local field sums
`C_α = Σ J v_j,α` over the site's bonds,
`ΔE = -Δv · C - h * Δx`.

# Returns

The energy difference `ΔE` between the proposed and current configurations.
"""
function energy_difference(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    proposal::SpinUpdate
)
    site = proposal.site
    info = geometry.neighbor_table[site]
    J = couplings(model)
    T = promote_type(eltype(state.xs), eltype(J))
    Cx = zero(T)
    Cy = zero(T)
    Cz = zero(T)
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        Cx += J[template] * state.xs[neighbor]
        Cy += J[template] * state.ys[neighbor]
        Cz += J[template] * state.zs[neighbor]
    end
    # The carried change is (x_new - x_old, y_new - y_old, z_new - z_old).
    dx, dy, dz = proposal.magnetization_change
    return -(dx * Cx + dy * Cy + dz * Cz) - field_strength(model, geometry, site) * dx
end

"""
    HeisenbergClusterContext

Per-update state of a Heisenberg cluster move (Wolff or Swendsen-Wang): the
random reflection axis `axis` (a unit vector `r̂`; projections
`p_i = v_i · r̂` are then plain dot products), and whether the growing Wolff
cluster has bonded to a ghost spin (in which case it is not reflected).
"""
mutable struct HeisenbergClusterContext
    axis::NTuple{3,Float64}
    ghost_attached::Bool
end

cluster_context(rng::AbstractRNG, ::HeisenbergModel, ::Geometry, ::HeisenbergState, ::Real) =
    HeisenbergClusterContext(random_unit_vector(rng), false)

cluster_ghost_count(::HeisenbergModel, ::Geometry, ::HeisenbergState, ::HeisenbergClusterContext) = 2

# Component of a site's spin along the reflection axis r̂.
_projection(state::HeisenbergState, site::Int, axis::NTuple{3,Real}) =
    state.xs[site] * axis[1] + state.ys[site] * axis[2] + state.zs[site] * axis[3]

# Normalize a proposed reflection axis; the O(3) Wolff embedding requires a
# unit vector for the acceptance probabilities to satisfy detailed balance.
function _unit_axis(axis::NTuple{3,Real})
    norm = hypot(axis[1], axis[2], axis[3])
    norm > 0 || throw(ArgumentError("The reflection axis must be nonzero, got $axis"))
    return (axis[1] / norm, axis[2] / norm, axis[3] / norm)
end

function cluster_bond_probability(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    site::Int,
    neighbor::Int,
    bond_template::Int,
    T::Real,
    axis::NTuple{3,Real},
)
    unit = _unit_axis(axis)
    projection_product = _projection(state, site, unit) * _projection(state, neighbor, unit)
    projection_product > 0 || return 0.0
    return -expm1(-2 * couplings(model)[bond_template] * projection_product / T)
end

function cluster_bond_probability(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    site::Int,
    neighbor::Int,
    bond_template::Int,
    T::Real,
    context::HeisenbergClusterContext,
)
    projection_product = _projection(state, site, context.axis) *
        _projection(state, neighbor, context.axis)
    projection_product > 0 || return 0.0
    return -expm1(-2 * couplings(model)[bond_template] * projection_product / T)
end

function cluster_ghost_probability(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    site::Int,
    ghost::Int,
    T::Real,
    context::HeisenbergClusterContext,
)
    ghost in 1:2 || throw(ArgumentError("Heisenberg ghost index must be 1 or 2; got $ghost"))
    field = field_strength(model, geometry, site)
    field == 0 && return 0.0
    sign(field) == (ghost == 1 ? 1 : -1) || return 0.0
    # The ghost spins point along ±x, so their axis components are ±r̂_x.
    ghost_projection = ghost == 1 ? context.axis[1] : -context.axis[1]
    projection_product = _projection(state, site, context.axis) * ghost_projection
    projection_product > 0 || return 0.0
    return -expm1(-2 * abs(field) * projection_product / T)
end

function cluster_ghost_attached!(context::HeisenbergClusterContext, ::Int)
    context.ghost_attached = true
    return nothing
end

"""
    apply_cluster!(model, geometry, state, cluster, axis)

Reflect a Wolff cluster about the unit vector `axis` (a nonzero 3-vector,
normalized on entry) and update the cached Heisenberg observables
incrementally. With `p_i = v_i · r̂` the pre-reflection component of site
`i`'s spin along the axis `r̂`, a boundary bond `(i, j)` with `i` in the
cluster contributes `ΔE = 2 * J * p_i * p_j`, and each reflected site
contributes `ΔE = 2 * h * r̂_x * p_i` and `ΔM_α = -2 * r̂_α * p_i`. The
reflection itself is the linear map `v ↦ v - 2 (v·r̂) r̂`.

This explicit form allocates a membership mask per call and serves as the
readable reference implementation (exercised by the tests); the Wolff driver
passes down the mask it already built.
"""
function apply_cluster!(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    cluster::Vector{Int},
    axis::NTuple{3,Real},
)
    in_cluster = falses(geometry.n_sites)
    for site in cluster
        in_cluster[site] = true
    end
    _reflect_heisenberg_cluster!(model, geometry, state, cluster, in_cluster, _unit_axis(axis))
    return nothing
end

# `in_cluster` must be true exactly on the cluster's sites. Wolff's `step!`
# passes down the mask it built while growing the cluster; `axis` must already
# be a unit vector.
function _reflect_heisenberg_cluster!(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    cluster::Vector{Int},
    in_cluster::AbstractVector{Bool},
    axis::NTuple{3,Real},
)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    ΔMz = zero(state.magnetization_z)
    J = couplings(model)

    for site in cluster
        x = state.xs[site]
        y = state.ys[site]
        z = state.zs[site]
        projection = x * axis[1] + y * axis[2] + z * axis[3]
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || in_cluster[neighbor]) && continue
            ΔE += 2 * J[template] * projection * _projection(state, neighbor, axis)
        end
        ΔE += 2 * field_strength(model, geometry, site) * axis[1] * projection
        ΔMx -= 2 * axis[1] * projection
        ΔMy -= 2 * axis[2] * projection
        ΔMz -= 2 * axis[3] * projection
        state.xs[site] = x - 2 * projection * axis[1]
        state.ys[site] = y - 2 * projection * axis[2]
        state.zs[site] = z - 2 * projection * axis[3]
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    state.magnetization_z += ΔMz
    return nothing
end

function apply_cluster!(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    cluster::Vector{Int},
    context::HeisenbergClusterContext,
    in_cluster::BitVector,
)
    context.ghost_attached || _reflect_heisenberg_cluster!(
        model, geometry, state, cluster, in_cluster, context.axis
    )
    return nothing
end

"""
    apply_clusters!(rng, model, geometry, state, clusters::Vector{Vector{Int}}, axis)

Explicit-partition form of [`apply_clusters!`](@ref): reflect each cluster in
`clusters` about `axis` (a nonzero 3-vector, normalized on entry)
independently with probability 1/2. Allocates its own label map; the
Swendsen-Wang driver uses the allocation-free workspace form below. This
explicit form is the readable reference implementation exercised by the
tests.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    clusters::Vector{Vector{Int}},
    axis::NTuple{3,Real},
)
    flip = [rand(rng, Bool) for _ in clusters]
    _reflect_heisenberg_clusters!(
        model, geometry, state, _cluster_labels(geometry, clusters), flip, _unit_axis(axis)
    )
    return nothing
end

"""
    apply_clusters!(rng, model, geometry, state, workspace, context)

Randomly reflect each Swendsen-Wang cluster about `context.axis` and update
the cached Heisenberg observables incrementally via
[`_reflect_heisenberg_clusters!`](@ref). The orientation decisions are drawn
by [`draw_cluster_flips!`](@ref) — one coin flip per cluster, in the
first-encounter order of `workspace.roots`, with ghost-attached clusters
never reflected — and recorded per root-id label in `workspace.flip`.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    workspace::ClusterWorkspace,
    context::HeisenbergClusterContext,
)
    flip = draw_cluster_flips!(rng, workspace, geometry)
    _reflect_heisenberg_clusters!(
        model, geometry, state, workspace.labels, flip, context.axis
    )
    return nothing
end

"""
    _reflect_heisenberg_clusters!(model, geometry, state, labels, flip, axis)

Reflect every lattice site whose cluster label is marked in `flip` about the
unit vector `axis`, accumulating the exact energy and magnetization change.
`labels[i]` is the union-find root of site `i` (see [`step!`](@ref) with
[`SwendsenWangAlgorithm`](@ref)) and doubles as the site→cluster map: a bond
incident to a reflected site is internal (both endpoints reflect, and the
reflection `v ↦ v - 2 (v·r̂) r̂` preserves their dot product) when
`flip[labels[neighbor]]` is true, external (never visited) when neither
endpoint reflects, and crossing (counted once, from the reflected side, with
pre-reflection components) otherwise.
"""
function _reflect_heisenberg_clusters!(
    model::HeisenbergModel,
    geometry::Geometry,
    state::HeisenbergState,
    labels::Vector{Int},
    flip::Vector{Bool},
    axis::NTuple{3,Real},
)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    ΔMz = zero(state.magnetization_z)
    J = couplings(model)

    for site in 1:geometry.n_sites
        flip[labels[site]] || continue
        x = state.xs[site]
        y = state.ys[site]
        z = state.zs[site]
        projection = x * axis[1] + y * axis[2] + z * axis[3]
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || flip[labels[neighbor]]) && continue
            ΔE += 2 * J[template] * projection * _projection(state, neighbor, axis)
        end
        ΔE += 2 * field_strength(model, geometry, site) * axis[1] * projection
        ΔMx -= 2 * axis[1] * projection
        ΔMy -= 2 * axis[2] * projection
        ΔMz -= 2 * axis[3] * projection
        state.xs[site] = x - 2 * projection * axis[1]
        state.ys[site] = y - 2 * projection * axis[2]
        state.zs[site] = z - 2 * projection * axis[3]
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    state.magnetization_z += ΔMz
    return nothing
end
