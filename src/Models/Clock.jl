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

initialize_state(model::ClockModel, geometry::Geometry; rng::AbstractRNG = Random.default_rng()) =
    _initialize_clock_state(model, geometry, rng)

function validate_model_geometry(model::ClockModel, geometry::Geometry)
    invoke(validate_model_geometry, Tuple{AbstractModel, Geometry}, model, geometry)

    n_orbitals = LatticeUtilities.norbits(geometry.unit_cell)
    length(model.h) == n_orbitals || throw(ArgumentError(
        "length(model.h) = $(length(model.h)) does not match the number of sites in the unit cell = $n_orbitals"
    ))
    return nothing
end

field_strength(model::ClockModel, geometry::Geometry, site::Int) =
    model.h[LatticeUtilities.site_to_orbital(site, geometry.unit_cell)]

bond_strength(model::ClockModel, geometry::Geometry, bond_id::Int) = begin
    1 <= bond_id <= length(geometry.bond_id_to_template) || throw(ArgumentError(
        "bond_id = $bond_id is out of bounds for geometry.bond_id_to_template of length $(length(geometry.bond_id_to_template))"
    ))
    model.J[geometry.bond_id_to_template[bond_id]]
end

local_operator(state::ClockState, site) = state.spins[site]
operator_product(model::ClockModel, first_index::Integer, second_index::Integer) =
    clock_cos(model, first_index - second_index)

n_operator_components(::ClockModel) = 2
operator_components(model::ClockModel, n::Integer) = (clock_cos(model, n), clock_sin(model, n))

"""
    ClockUpdate

A proposed single-site clock move: change the clock index at `site` to
`new_value`. The proposal carries the magnetization change of the move,
computed by [`propose`](@ref) from the model's lookup tables, so that
[`apply_update!`](@ref) can keep the cached totals exact without access to
the model.

# Fields

- `site::Int`: Index of the site to update.
- `new_value::Int`: The proposed clock index, in `0:q-1`.
- `magnetization_x_change::Float64`: `cos(θ_new) - cos(θ_old)`.
- `magnetization_y_change::Float64`: `sin(θ_new) - sin(θ_old)`.
"""
struct ClockUpdate
    site::Int
    new_value::Int
    magnetization_x_change::Float64
    magnetization_y_change::Float64
end

"""
    propose(rng, ::MetropolisAlgorithm, model::ClockModel, geometry, state)

Generate a Metropolis trial move for the clock model: pick a random site and
propose one of the other `q - 1` clock states uniformly at random (a
symmetric proposal), both drawn from `rng`.

# Returns

A [`ClockUpdate`](@ref) for the chosen site and new clock index.
"""
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, model::ClockModel, geometry, state::ClockState)
    site = rand(rng, eachindex(state.spins))
    old = state.spins[site]
    new = mod(old + rand(rng, 1:(model.q - 1)), model.q)
    return ClockUpdate(
        site, new,
        clock_cos(model, new) - clock_cos(model, old),
        clock_sin(model, new) - clock_sin(model, old),
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

A [`ClockUpdate`](@ref) for the chosen site and new clock index.
"""
function propose(rng::AbstractRNG, ::GlauberAlgorithm, model::ClockModel, geometry, state::ClockState)
    return propose(rng, MetropolisAlgorithm(), model, geometry, state)
end

function energy_difference(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    proposal::ClockUpdate
)
    site = proposal.site
    old = state.spins[site]
    new = proposal.new_value
    info = geometry.neighbor_table[site]
    weighted_sum = zero(promote_type(Float64, eltype(model.J)))
    for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
        neighbor == site && continue
        n_j = state.spins[neighbor]
        weighted_sum += bond_strength(model, geometry, bond_id) *
            (clock_cos(model, new - n_j) - clock_cos(model, old - n_j))
    end
    return -weighted_sum - field_strength(model, geometry, site) * proposal.magnetization_x_change
end

function apply_update!(state::ClockState, proposal::ClockUpdate, ΔE::Real)
    state.spins[proposal.site] = proposal.new_value
    state.energy += ΔE
    state.magnetization_x += proposal.magnetization_x_change
    state.magnetization_y += proposal.magnetization_y_change
    return nothing
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
    bond_id::Int,
    T::Real,
    context::ClockClusterContext,
)
    projection_product = context.projections[state.spins[site] + 1] *
        context.projections[state.spins[neighbor] + 1]
    projection_product > 0 || return 0.0
    return -expm1(-2 * bond_strength(model, geometry, bond_id) * projection_product / T)
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
)
    context.ghost_attached || apply_cluster!(model, geometry, state, cluster, context.mirror)
    return nothing
end

"""
    apply_clusters!(rng, model::ClockModel, geometry, state, clusters, mirror::Integer)

Reflect each Swendsen-Wang cluster across clock mirror `mirror` with
probability 1/2 (coin flips drawn from `rng`) and update the cached clock
observables incrementally via [`_reflect_clock_sites!`](@ref).
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
    _reflect_clock_clusters!(model, geometry, state, clusters, flip, mirror)
    return nothing
end

function apply_clusters!(
    rng::AbstractRNG,
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    clusters::Vector{Vector{Int}},
    context::ClockClusterContext,
)
    flip = [
        rand(rng, Bool) && !any(site -> site > geometry.n_sites, cluster)
        for cluster in clusters
    ]
    _reflect_clock_clusters!(model, geometry, state, clusters, flip, context.mirror)
    return nothing
end

function _reflect_clock_clusters!(
    model::ClockModel,
    geometry::Geometry,
    state::ClockState,
    clusters::Vector{Vector{Int}},
    flip::Vector{Bool},
    mirror::Integer,
)
    reflected = falses(geometry.n_sites)
    for (c, cluster) in enumerate(clusters)
        flip[c] || continue
        for site in cluster
            site <= geometry.n_sites && (reflected[site] = true)
        end
    end
    _reflect_clock_sites!(model, geometry, state, findall(reflected), reflected, mirror)
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

    for site in sites
        old = state.spins[site]
        new = mod(mirror - old, model.q)
        new == old && continue
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            (neighbor == site || reflected[neighbor]) && continue
            n_j = state.spins[neighbor]
            ΔE -= bond_strength(model, geometry, bond_id) *
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
