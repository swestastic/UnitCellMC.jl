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
    propose_value(rng, model::XYModel, geometry, state, site)

Propose the new value at `site` (chosen by the algorithm's
[`AbstractProposalScheme`](@ref)) for a local-update trial move: a new spin
drawn uniformly from the unit circle via `rng`.

# Returns

A [`SpinUpdate`](@ref) carrying the proposed components `(cos θ, sin θ)` as
`new_value` and their change from the current components as
`magnetization_change`.
"""
function propose_value(rng::AbstractRNG, ::XYModel, geometry::Geometry, state::XYState, site::Int)
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
    rand_von_mises(rng, κ)

Draw an angle in `(-π, π]` from the von Mises distribution
`p(θ) ∝ exp(κ cos θ)`, the single-site conditional of the [`XYModel`](@ref)
under [`HeatbathAlgorithm`](@ref) up to rotation by the local-field
direction.

Sampling uses the Best–Fisher rejection algorithm: with
`a = 1 + sqrt(1 + 4κ²)`, `b = (a - sqrt(2a)) / (2κ)`, and
`r = (1 + b²) / (2b)`, draw `z = cos(π u₁)`, set
`f = (1 + r z) / (r + z)` and `c = κ (r - f)`, and accept
`θ = ±acos(f)` (sign fair) when `c (2 - c) > u₂`, else when
`log(c / u₂) + 1 ≥ c`. For `κ < 1e-8` the distribution is uniform to
`O(κ)`, so a uniform angle is returned directly (the `κ → 0` limit of the
general path is `0 / 0`).

# Throws

`ArgumentError` if `κ < 0`.

# Returns

An angle in `(-π, π]` centered on direction `0`.
"""
function rand_von_mises(rng::AbstractRNG, κ::Real)
    κ >= 0 || throw(ArgumentError(
        "von Mises concentration must be nonnegative, got κ = $κ"))
    κ < 1e-8 && return 2π * rand(rng) - π

    a = 1 + sqrt(1 + 4κ^2)
    b = (a - sqrt(2a)) / (2κ)
    r = (1 + b^2) / (2b)
    while true
        z = cospi(rand(rng))
        f = (1 + r * z) / (r + z)
        c = κ * (r - f)
        u2 = rand(rng)
        if c * (2 - c) - u2 > 0 || log(c / u2) + 1 - c >= 0
            return (rand(rng) < 0.5 ? -1 : 1) * acos(clamp(f, -1.0, 1.0))
        end
    end
end

"""
    heatbath_draw(rng, model::XYModel, geometry, state, site, T, cache)

Draw the new spin at `site` from its single-site conditional distribution at
temperature `T`: with the local field `H = (C + h, S)` assembled from the
bond sums `C = Σ J cos(θ_j)`, `S = Σ J sin(θ_j)`, the conditional angle
distribution is the von Mises law `p(θ) ∝ exp(κ cos(θ - φ))` with
concentration `κ = |H| / T` about the local-field direction `φ`.

The angle is drawn as `θ = δ + φ` with `δ` from [`rand_von_mises`](@ref), and
the rotation by `φ` is applied through `cos φ = (C + h) / |H|`,
`sin φ = S / |H|`, so no `atan` call is needed and the result is a unit
vector by construction. A vanishing local field (`|H| = 0`, e.g. no bonds
and no field) gives a uniform angle. `cache` goes unused: the draw does not
reduce to a memoized Boltzmann factor.

# Returns

A [`SpinUpdate`](@ref) carrying the drawn components `(cos θ, sin θ)` as
`new_value` and their change from the current components as
`magnetization_change`.
"""
function heatbath_draw(
    rng::AbstractRNG,
    model::XYModel,
    geometry::Geometry,
    state::XYState,
    site::Int,
    T::Real,
    cache,
)
    info = geometry.neighbor_table[site]
    J = couplings(model)
    Hx = Float64(field_strength(model, geometry, site))
    Hy = zero(Hx)
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        Hx += J[template] * state.cosines[neighbor]
        Hy += J[template] * state.sines[neighbor]
    end

    R = hypot(Hx, Hy)
    sin_δ, cos_δ = sincos(rand_von_mises(rng, R / T))
    if R == 0
        c_new, s_new = cos_δ, sin_δ
    else
        ux, uy = Hx / R, Hy / R
        c_new = cos_δ * ux - sin_δ * uy
        s_new = sin_δ * ux + cos_δ * uy
    end

    return SpinUpdate(
        site, (c_new, s_new),
        (c_new - state.cosines[site], s_new - state.sines[site]),
    )
end

"""
    overrelax(model::XYModel, geometry, state, site)

Return the overrelaxation update of the XY spin at `site`: its reflection
about the local field `H = (C + h, S)` assembled from the bond sums
`C = Σ J cos(θ_j)`, `S = Σ J sin(θ_j)`. On the circle this is the angle map
`θ ↦ 2φ - θ` with `φ` the local-field direction, written in components as
`v ↦ 2 (v·H) H / |H|² - v` so no trigonometric calls are needed; the
reflected components remain a unit vector to rounding. The reflection
preserves `v · H`, so it conserves the site's energy exactly. A vanishing
local field (`|H| = 0`) leaves the spin unchanged.

# Returns

A [`SpinUpdate`](@ref) carrying the reflected components as `new_value` and
their change from the current components as `magnetization_change`.
"""
function overrelax(model::XYModel, geometry::Geometry, state::XYState, site::Int)
    info = geometry.neighbor_table[site]
    J = couplings(model)
    Hx = Float64(field_strength(model, geometry, site))
    Hy = zero(Hx)
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        Hx += J[template] * state.cosines[neighbor]
        Hy += J[template] * state.sines[neighbor]
    end

    c_old = state.cosines[site]
    s_old = state.sines[site]
    R2 = Hx * Hx + Hy * Hy
    R2 == 0 && return SpinUpdate(site, (c_old, s_old), (zero(R2), zero(R2)))

    projection = (c_old * Hx + s_old * Hy) / R2
    c_new = 2 * projection * Hx - c_old
    s_new = 2 * projection * Hy - s_old
    return SpinUpdate(
        site, (c_new, s_new), (c_new - c_old, s_new - s_old),
    )
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

This form allocates a membership mask per call; the Wolff driver instead
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
Allocates its own label map; the Swendsen-Wang driver uses the workspace
form below.
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

# The planar-spin observable sets, shared by the two-component models
# ([`XYModel`](@ref) here and [`ClockModel`](@ref) in `Clock.jl`): per-site
# energy, both Cartesian magnetization components, the magnetization
# magnitude, and the square of each — plus the fluctuation-formula derived
# quantities. A new planar-spin model reuses these two helpers instead of
# duplicating the lists.
function _planar_observables(geometry)
    n_sites = geometry.n_sites
    [
        Observable(:energy, state -> total_energy(state) / n_sites),
        Observable(:energy_squared, state -> (total_energy(state) / n_sites)^2),
        Observable(:magnetization_x, state -> state.magnetization_x / n_sites),
        Observable(:magnetization_x_squared, state -> (state.magnetization_x / n_sites)^2),
        Observable(:magnetization_y, state -> state.magnetization_y / n_sites),
        Observable(:magnetization_y_squared, state -> (state.magnetization_y / n_sites)^2),
        Observable(:magnetization, state -> hypot(state.magnetization_x, state.magnetization_y) / n_sites),
        Observable(:magnetization_squared, state -> (
            (state.magnetization_x^2 + state.magnetization_y^2) / n_sites^2
        )),
    ]
end

_planar_derived_observables(geometry) = [
    DerivedObservable(:specific_heat, :specific_heat_err, (:energy, :energy_squared),
        (e, e2; T, N) -> N * (e2 - e^2) / T^2),
    DerivedObservable(:susceptibility_x, :susceptibility_x_err,
        (:magnetization_x, :magnetization_x_squared),
        (mx, mx2; T, N) -> N * (mx2 - mx^2) / T),
    DerivedObservable(:susceptibility_y, :susceptibility_y_err,
        (:magnetization_y, :magnetization_y_squared),
        (my, my2; T, N) -> N * (my2 - my^2) / T),
    DerivedObservable(:susceptibility, :susceptibility_err,
        (:magnetization_x, :magnetization_x_squared,
            :magnetization_y, :magnetization_y_squared),
        (mx, mx2, my, my2; T, N) -> N * (mx2 - mx^2 + my2 - my^2) / T),
]

"""
    observables(::XYModel, geometry)

Return the raw per-measurement observables tracked for every [`XYModel`](@ref)
simulation (shared with the other planar-spin models, see
[`observables`](@ref)): per-site energy, both Cartesian magnetization
components, the magnetization magnitude `hypot(M_x, M_y) / N`, and the square
of each.

# Returns

A `Vector{Observable}` with `:energy`, `:energy_squared`, `:magnetization_x`,
`:magnetization_x_squared`, `:magnetization_y`, `:magnetization_y_squared`,
`:magnetization`, and `:magnetization_squared`.
"""
observables(::XYModel, geometry) = _planar_observables(geometry)

"""
    derived_observables(::XYModel, geometry)

Return the always-on derived quantities for the [`XYModel`](@ref) (shared
with the other planar-spin models): the specific heat, the component
susceptibilities `:susceptibility_x` and `:susceptibility_y`, and the total
`:susceptibility = N * (⟨m_x²⟩ - ⟨m_x⟩² + ⟨m_y²⟩ - ⟨m_y⟩²) / T`, each with its
jackknife error.
"""
derived_observables(::XYModel, geometry) = _planar_derived_observables(geometry)

# The XY model supports the helicity-modulus measurement (see
# `src/Measurements/HelicityModulus.jl` for the entry builder).
extra_measurements(::Type{<:XYModel}) = (:helicity_modulus,)
