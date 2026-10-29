"""
    ClockModel{T<:Real} <: AbstractModel

The `q`-state clock model: planar spins restricted to the `q` angles
`θ = 2π n / q`, `n ∈ 0:q-1`, with Hamiltonian

    H = -Σ_⟨ij⟩ J_ij cos(θ_i - θ_j) - Σ_i h_i cos(θ_i)

with bond strengths `J` (one value per bond template) and external fields `h`
(one value per site in the unit cell). As for the [`XYModel`](@ref), the field
points along the `x` direction. `q = 2` is the Ising model and `q → ∞`
approaches the XY model.

# Fields

- `q::Int`: Number of clock states.
- `J::Vector{T}`: Bond strengths, one per bond template.
- `h::Vector{T}`: External fields, one per site in the unit cell.
- `cosines::Vector{Float64}`: Lookup table, `cosines[n + 1] = cos(2π n / q)`.
- `sines::Vector{Float64}`: Lookup table, `sines[n + 1] = sin(2π n / q)`.

Construct with [`ClockModel(q, J, h)`](@ref).
"""
struct ClockModel{T<:Real} <: AbstractModel{T}
    q::Int
    J::Vector{T}
    h::Vector{T}
    cosines::Vector{Float64}
    sines::Vector{Float64}

    function ClockModel{T}(q::Integer, J::Vector{T}, h::Vector{T}) where {T<:Real}
        q >= 2 || throw(ArgumentError("ClockModel requires q >= 2; got q = $q"))
        # cospi/sinpi give exact zeros and ±1 (e.g. cos(π/2) == 0 for q = 4).
        cosines = [cospi(2n / q) for n in 0:(q - 1)]
        sines = [sinpi(2n / q) for n in 0:(q - 1)]
        return new{T}(Int(q), J, h, cosines, sines)
    end
end

"""
    ClockModel(q, J, h)

A `q`-state clock model with bond strengths `J` and external field `h`.

# Arguments

- `q`: Number of clock states; spins take the angles `2π n / q`,
    `n ∈ 0:q-1`.
- `J`: Bond strengths, one value per bond template. Its length is checked
    against `geometry.bonds` by [`initialize_state`](@ref), since the
    constructor does not receive a geometry.
- `h`: External field along `x`, one value per site in the unit cell.

# Throws

`ArgumentError` if `q < 2`.
"""
function ClockModel(q::Integer, J::Vector{<:Real}, h::AbstractVector{<:Real})
    T = promote_type(eltype(J), eltype(h))
    return ClockModel{T}(q, convert(Vector{T}, J), convert(Vector{T}, h))
end

Base.show(io::IO, model::ClockModel{T}) where {T} =
    print(io, "ClockModel{", T, "}(", model.q, ", ", model.J, ", ", model.h, ")")

"""
    clock_cos(model::ClockModel, n)
    clock_sin(model::ClockModel, n)

Return `cos(2π n / q)` (resp. `sin(2π n / q)`) from the lookup tables of
`model`. `n` may be any integer; it is reduced modulo `q`, so differences of
clock indices can be passed directly, e.g. `clock_cos(model, n_i - n_j)` for
the bond term `cos(θ_i - θ_j)`.
"""
clock_cos(model::ClockModel, n::Integer) = @inbounds model.cosines[mod(n, model.q) + 1]
clock_sin(model::ClockModel, n::Integer) = @inbounds model.sines[mod(n, model.q) + 1]

local_operator(state::ClockState, site) = state.spins[site]

state_type(::ClockModel) = ClockState

magnetization_value(state::ClockState) = hypot(state.magnetization_x, state.magnetization_y)


# For q = 2 the magnetization only has an x component (Ising universality);
# for q >= 3 the clock states are isotropic to second order, so the
# high-temperature magnetization is a two-component Gaussian, as for XY.
binder_denominator(model::ClockModel) = model.q == 2 ? 3.0 : 2.0

"""
    clock_angle_order(model::ClockModel, state::ClockState)

Return the instantaneous angular order parameter `cos(q φ_M)` of `state`,
where `φ_M = atan(M_y, M_x)` is the direction of the total magnetization.

The magnetization is resummed from `state.spins` (`O(N)`) rather than read
from the running totals. On small lattices `M = 0` is a common
configuration, and there the running totals hold only rounding residue whose
direction is meaningless and grows with run length. A fresh sum is accurate
to about `N * eps()`, so `|M| <= √eps()` is treated as zero, for which the
direction is undefined and `0.0` is returned.
"""
function clock_angle_order(model::ClockModel, state::ClockState)
    mx = my = 0.0
    for n in state.spins
        mx += clock_cos(model, n)
        my += clock_sin(model, n)
    end
    hypot(mx, my) <= sqrt(eps(Float64)) && return 0.0
    return cos(model.q * atan(my, mx))
end

operator_product(model::ClockModel, first_index::Integer, second_index::Integer) =
    clock_cos(model, first_index - second_index)

n_operator_components(::ClockModel) = 2
operator_components(model::ClockModel, n::Integer) = (clock_cos(model, n), clock_sin(model, n))

"""
    propose(rng, ::MetropolisAlgorithm, model::ClockModel, geometry, state)

Generate a Metropolis trial move for the clock model: pick a random site and
propose one of the other `q - 1` clock states uniformly at random (a
symmetric proposal), both drawn from `rng`.

# Returns

A [`SpinUpdate`](@ref) for the chosen site and new clock index, carrying the magnetization change.
"""
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, model::ClockModel, geometry, state::ClockState)
    site = rand(rng, eachindex(state.spins))
    old = state.spins[site]
    new = mod(old + rand(rng, 1:(model.q - 1)), model.q)
    return SpinUpdate(
        site, new,
        (clock_cos(model, new) - clock_cos(model, old),
            clock_sin(model, new) - clock_sin(model, old)),
    )
end

"""
    propose(rng, ::GlauberAlgorithm, model::ClockModel, geometry, state)

Generate a Glauber trial move for the clock model: pick a random site and
propose one of the other `q - 1` clock states uniformly at random, both drawn
from `rng`. This is the same symmetric proposal as for
[`MetropolisAlgorithm`](@ref); only the acceptance rule in [`step!`](@ref)
differs.

# Returns

A [`SpinUpdate`](@ref) for the chosen site and new clock index, carrying the magnetization change.
"""
function propose(rng::AbstractRNG, ::GlauberAlgorithm, model::ClockModel, geometry, state::ClockState)
    return propose(rng, MetropolisAlgorithm(), model, geometry, state)
end

function energy_difference(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    proposal::SpinUpdate
)
    site = proposal.site
    old = state.spins[site]
    new = proposal.new_value
    info = geometry.neighbor_table[site]
    J = model.J
    weighted_sum = zero(promote_type(Float64, eltype(J)))
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        n_j = state.spins[neighbor]
        weighted_sum += J[template] *
            (clock_cos(model, new - n_j) - clock_cos(model, old - n_j))
    end
    return -weighted_sum - field_strength(model, geometry, site) * proposal.magnetization_change[1]
end

validate_cluster_model(model::ClockModel, ::Geometry, ::ClockState, ::Real) = begin
    all(>=(0), model.J) || throw(ArgumentError(
        "Clock cluster algorithms require non-negative couplings; got J = $(model.J)"
    ))
    nothing
end

"""
    ClockClusterContext

Per-update state of a clock-model cluster move (Wolff or Swendsen-Wang).

The move reflects spins across one of the `q` mirror lines of the clock,
`n ↦ mod(mirror - n, q)`, which is the XY reflection `θ ↦ 2φ + π - θ` about
the axis `φ = π mirror / q - π/2`; it maps clock states onto clock states.

# Fields

- `mirror::Int`: Mirror index in `0:q-1`.
- `projections::Vector{Float64}`: `projections[n + 1] = cos(2π n / q - φ)`,
  the component of clock state `n` along the reflection axis.
- `ghost_projection::Float64`: `cos(φ)`, the axis component of the `+x`
  ghost spin (the `-x` ghost has `-cos(φ)`).
- `ghost_attached::Bool`: Whether a Wolff cluster has bonded to a ghost spin,
  in which case it is not reflected.
"""
mutable struct ClockClusterContext
    mirror::Int
    projections::Vector{Float64}
    ghost_projection::Float64
    ghost_attached::Bool
end

function ClockClusterContext(model::ClockModel, mirror::Integer)
    q = model.q
    # cos(2πn/q - φ) with φ = πm/q - π/2 equals -sin(π(2n - m)/q).
    projections = [-sinpi((2n - mirror) / q) for n in 0:(q - 1)]
    return ClockClusterContext(mirror, projections, sinpi(mirror / q), false)
end

function cluster_context(rng::AbstractRNG, model::ClockModel, ::Geometry, ::ClockState, ::Real)
    # For q = 2 the even mirror is the identity (both states lie on it), so
    # always use the spin-flip mirror; for q >= 3 every mirror moves some state.
    mirror = model.q == 2 ? 1 : rand(rng, 0:(model.q - 1))
    return ClockClusterContext(model, mirror)
end

cluster_ghost_count(::ClockModel, ::Geometry, ::ClockState, ::ClockClusterContext) = 2

function cluster_bond_probability(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    site::Int,
    neighbor::Int,
    bond_template::Int,
    T::Real,
    context::ClockClusterContext,
)
    projection_product = context.projections[state.spins[site] + 1] *
        context.projections[state.spins[neighbor] + 1]
    projection_product > 0 || return 0.0
    return -expm1(-2 * model.J[bond_template] * projection_product / T)
end

function cluster_ghost_probability(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    site::Int,
    ghost::Int,
    T::Real,
    context::ClockClusterContext,
)
    ghost in 1:2 || throw(ArgumentError("Clock ghost index must be 1 or 2; got $ghost"))
    field = field_strength(model, geometry, site)
    field == 0 && return 0.0
    sign(field) == (ghost == 1 ? 1 : -1) || return 0.0
    ghost_projection = ghost == 1 ? context.ghost_projection : -context.ghost_projection
    projection_product = context.projections[state.spins[site] + 1] * ghost_projection
    projection_product > 0 || return 0.0
    return -expm1(-2 * abs(field) * projection_product / T)
end

function cluster_ghost_attached!(context::ClockClusterContext, ::Int)
    context.ghost_attached = true
    return nothing
end

"""
    apply_cluster!(model::ClockModel, geometry, state, cluster, mirror::Integer)

Reflect a Wolff cluster across clock mirror `mirror`
(`n ↦ mod(mirror - n, q)`, see [`ClockClusterContext`](@ref)) and update the
cached clock observables incrementally via
[`_reflect_clock_sites!`](@ref).
"""
function apply_cluster!(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    cluster::Vector{Int},
    mirror::Integer,
)
    reflected = falses(geometry.n_sites)
    for site in cluster
        reflected[site] = true
    end
    _reflect_clock_sites!(model, geometry, state, cluster, reflected, mirror)
    return nothing
end

function apply_cluster!(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    cluster::Vector{Int},
    context::ClockClusterContext,
    in_cluster::BitVector,
)
    # Wolff's step! passes down the mask it built while growing the cluster.
    context.ghost_attached || _reflect_clock_sites!(
        model, geometry, state, cluster, in_cluster, context.mirror
    )
    return nothing
end

"""
    apply_clusters!(rng, model::ClockModel, geometry, state, clusters::Vector{Vector{Int}}, mirror::Integer)

Explicit-partition form of [`apply_clusters!`](@ref): reflect each cluster in
`clusters` across clock mirror `mirror` with probability 1/2. Allocates its
own label map (via [`_cluster_labels`](@ref)); the Swendsen-Wang driver uses
the workspace form below.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    clusters::Vector{Vector{Int}},
    mirror::Integer,
)
    flip = [rand(rng, Bool) for _ in clusters]
    _reflect_clock_clusters!(
        model, geometry, state, _cluster_labels(geometry, clusters), flip, mirror
    )
    return nothing
end

"""
    apply_clusters!(rng, model::ClockModel, geometry, state, workspace, context)

Reflect each Swendsen-Wang cluster across `context.mirror` and update the
cached clock observables incrementally via
[`_reflect_clock_clusters!`](@ref). The orientation decisions are drawn by
[`draw_cluster_flips!`](@ref) — one coin flip per cluster, in the
first-encounter order of `workspace.roots`, with ghost-attached clusters
never reflected — and recorded per root-id label in `workspace.flip`.
"""
function apply_clusters!(
    rng::AbstractRNG,
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    workspace::ClusterWorkspace,
    context::ClockClusterContext,
)
    flip = draw_cluster_flips!(rng, workspace, geometry)
    _reflect_clock_clusters!(
        model, geometry, state, workspace.labels, flip, context.mirror
    )
    return nothing
end

"""
    _reflect_clock_sites!(model, geometry, state, sites, reflected, mirror)

Apply the clock reflection `n ↦ mod(mirror - n, q)` to every site in `sites`
and accumulate the exact energy and magnetization change from the lookup
tables. `reflected[i]` must be `true` exactly for the sites in `sites`.

Bonds with both endpoints reflected keep their energy
(`cos(θ_i - θ_j)` is invariant under a common reflection), so only bonds
crossing the boundary of the reflected set are visited; their far endpoint
is never mutated, so pre- and post-reflection energies are evaluated
directly as `-J (cos(θ'_i - θ_j) - cos(θ_i - θ_j))`.
"""
function _reflect_clock_sites!(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    sites,
    reflected::AbstractVector{Bool},
    mirror::Integer,
)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    J = model.J

    for site in sites
        old = state.spins[site]
        new = mod(mirror - old, model.q)
        new == old && continue
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || reflected[neighbor]) && continue
            n_j = state.spins[neighbor]
            ΔE -= J[template] *
                (clock_cos(model, new - n_j) - clock_cos(model, old - n_j))
        end
        Δcos = clock_cos(model, new) - clock_cos(model, old)
        ΔE -= field_strength(model, geometry, site) * Δcos
        ΔMx += Δcos
        ΔMy += clock_sin(model, new) - clock_sin(model, old)
        state.spins[site] = new
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    return nothing
end

"""
    _reflect_clock_clusters!(model, geometry, state, labels, flip, mirror)

Reflect every lattice site whose cluster label is marked in `flip` across
clock mirror `mirror` (`n -> mod(mirror - n, q)`), accumulating the exact
energy and magnetization change from the lookup tables. `labels[i]` is the
union-find root of site `i` (see [`step!`](@ref) with
[`SwendsenWangAlgorithm`](@ref)) and doubles as the site-to-cluster map: a
bond incident to a reflected site is internal (both endpoints reflect, and
reflection preserves `cos(θ_i - θ_j)`) when `flip[labels[neighbor]]`
is true, external (never visited) when neither endpoint reflects, and
crossing (counted once, from the reflected side, with pre-reflection indices)
otherwise. Sites lying on the mirror (`new == old`) do not move, so their
incident bonds are unchanged and are skipped together with the site itself.
"""
function _reflect_clock_clusters!(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    labels::Vector{Int},
    flip::Vector{Bool},
    mirror::Integer,
)
    ΔE = zero(state.energy)
    ΔMx = zero(state.magnetization_x)
    ΔMy = zero(state.magnetization_y)
    J = model.J

    for site in 1:geometry.n_sites
        flip[labels[site]] || continue
        old = state.spins[site]
        new = mod(mirror - old, model.q)
        new == old && continue
        info = geometry.neighbor_table[site]
        for (template, neighbor) in zip(info.templates, info.neighbors)
            (neighbor == site || flip[labels[neighbor]]) && continue
            n_j = state.spins[neighbor]
            ΔE -= J[template] *
                (clock_cos(model, new - n_j) - clock_cos(model, old - n_j))
        end
        Δcos = clock_cos(model, new) - clock_cos(model, old)
        ΔE -= field_strength(model, geometry, site) * Δcos
        ΔMx += Δcos
        ΔMy += clock_sin(model, new) - clock_sin(model, old)
        state.spins[site] = new
    end

    state.energy += ΔE
    state.magnetization_x += ΔMx
    state.magnetization_y += ΔMy
    return nothing
end
