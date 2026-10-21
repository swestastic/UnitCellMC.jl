"""
    ClockState{S,T<:Real} <: AbstractState

The mutable configuration of a [`ClockModel`](@ref) simulation. Each site
stores an integer clock index `n ∈ 0:q-1`, representing the planar spin at
angle `θ = 2π n / q`; energy and the two Cartesian magnetization components
are cached for incremental updates.

The running totals are maintained incrementally by [`apply_update!`](@ref),
[`apply_cluster!`](@ref), and [`apply_clusters!`](@ref), so they must be kept
consistent with `spins` by construction (see [`initialize_state`](@ref)) and
by every code path that mutates `spins`.

# Fields

- `spins::S`: Per-site clock indices, each in `0:q-1`.
- `energy::T`: Total energy of the current configuration.
- `magnetization_x::T`: Total `x` magnetization, `Σ_i cos(2π n_i / q)`.
- `magnetization_y::T`: Total `y` magnetization, `Σ_i sin(2π n_i / q)`.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct ClockState{S,T<:Real} <: AbstractState
    spins::S
    energy::T
    magnetization_x::T
    magnetization_y::T
end

function _initialize_clock_state(model, geometry::Geometry, rng::AbstractRNG = Random.default_rng())
    validate_model_geometry(model, geometry)

    spins = rand(rng, 0:(model.q - 1), geometry.n_sites)
    T = promote_type(Float64, eltype(model.J), eltype(model.h))
    energy = zero(T)
    magnetization_x = zero(T)
    magnetization_y = zero(T)

    for site in 1:geometry.n_sites
        n = spins[site]
        magnetization_x += clock_cos(model, n)
        magnetization_y += clock_sin(model, n)
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            energy += -0.5 * bond_strength(model, geometry, bond_id) *
                clock_cos(model, n - spins[neighbor])
        end
        energy += -field_strength(model, geometry, site) * clock_cos(model, n)
    end

    return ClockState(spins, energy, magnetization_x, magnetization_y)
end
