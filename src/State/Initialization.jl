# Per-model `initialize_state` methods, included from `src/UnitCellMC.jl`
# after `Models/Models.jl`: the state structs must exist before the model
# files are included, while these signatures name the model types. New models
# add their method here (or to a sibling file included at the same point);
# the fallback in `State.jl` throws for models without one.

"""
    initialize_state(model::IsingModel, geometry::Geometry; rng = Random.default_rng())

Build a random initial [`IsingState`](@ref) for `model` on `geometry`, with
energy and magnetization computed from scratch to match the random spins.

Each site is assigned a uniformly random spin of `±1`. Total energy is
computed by summing `-0.5 * J * spins[i] * spins[j]` over every bond incident
to every site (the `0.5` avoids double-counting since each bond is visited
from both endpoints) plus the field contribution `-h * spins[i]` per site;
magnetization is simply the sum of all spins.

# Arguments

- `model`: The [`IsingModel`](@ref) supplying bond strengths `J` and field `h`.
- `geometry`: The lattice geometry, giving `n_sites`, `bonds`, and the
  neighbor table used to enumerate each site's bonds.
- `rng`: Random number generator for the initial spins.

# Throws

`ArgumentError` from [`validate_model_geometry`](@ref) if `model` and
`geometry` are incompatible (coupling/field lengths versus the bond templates
and unit-cell orbitals).

# Returns

A randomly initialized [`IsingState`](@ref) with `energy` and `magnetization`
consistent with its `spins`.
"""
function initialize_state(
    model::IsingModel,
    geometry::Geometry;
    rng::AbstractRNG = Random.default_rng()
)
    validate_model_geometry(model, geometry)

    n_sites = geometry.n_sites
    spins = rand(rng, Int8[-1, 1], n_sites)

    T = promote_type(eltype(couplings(model)), eltype(fields(model)))
    energy = zero(T)

    for i in 1:n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * spins[i] * spins[j]
        end
        energy += -field_strength(model, geometry, i) * spins[i]
    end

    magnetization = sum(Int, spins)

    return IsingState(spins, energy, magnetization)
end

"""
    initialize_state(model::XYModel, geometry::Geometry; rng = Random.default_rng())

Build a random initial [`XYState`](@ref) for `model` on `geometry`, with
energy and both Cartesian magnetization components computed from scratch.

Each site is assigned a uniformly random unit vector (angle uniform in
`[0, 2π)`). Total energy is `-0.5 * J * (c_i c_j + s_i s_j)` summed over
every bond incident to every site (the `0.5` avoids double-counting) plus
the field contribution `-h * c_i` per site, with `(c, s)` the stored spin
components.

# Arguments

- `model`: The [`XYModel`](@ref) supplying bond strengths `J` and field `h`.
- `geometry`: The lattice geometry, giving `n_sites` and the neighbor table.
- `rng`: Random number generator for the initial angles.

# Throws

`ArgumentError` from [`validate_model_geometry`](@ref) if `model` and
`geometry` are incompatible.

# Returns

A randomly initialized [`XYState`](@ref) with `energy`, `magnetization_x`,
and `magnetization_y` consistent with its spin components.
"""
function initialize_state(
    model::XYModel,
    geometry::Geometry;
    rng::AbstractRNG = Random.default_rng()
)
    validate_model_geometry(model, geometry)

    angles = 2π .* rand(rng, geometry.n_sites)
    cosines = cos.(angles)
    sines = sin.(angles)
    T = promote_type(Float64, eltype(couplings(model)), eltype(fields(model)))
    energy = zero(T)
    magnetization_x = zero(T)
    magnetization_y = zero(T)

    for site in 1:geometry.n_sites
        c = cosines[site]
        s = sines[site]
        magnetization_x += c
        magnetization_y += s
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            energy += -0.5 * bond_strength(model, geometry, bond_id) *
                (c * cosines[neighbor] + s * sines[neighbor])
        end
        energy += -field_strength(model, geometry, site) * c
    end

    return XYState(cosines, sines, energy, magnetization_x, magnetization_y)
end

"""
    initialize_state(model::ClockModel, geometry::Geometry; rng = Random.default_rng())

Build a random initial [`ClockState`](@ref) for `model` on `geometry`, with
energy and both Cartesian magnetization components computed from scratch.

Each site is assigned a uniformly random clock index in `0:q-1`. Total
energy is `-0.5 * J * cos(θ_i - θ_j)` summed over every bond incident to
every site (the `0.5` avoids double-counting) plus the field contribution
`-h * cos(θ_i)` per site, with `θ = 2π n / q` evaluated through
[`clock_cos`](@ref).

# Arguments

- `model`: The [`ClockModel`](@ref) supplying `q`, bond strengths `J`, and
  field `h`.
- `geometry`: The lattice geometry, giving `n_sites` and the neighbor table.
- `rng`: Random number generator for the initial clock indices.

# Throws

`ArgumentError` from [`validate_model_geometry`](@ref) if `model` and
`geometry` are incompatible.

# Returns

A randomly initialized [`ClockState`](@ref) with `energy`, `magnetization_x`,
and `magnetization_y` consistent with its `spins`.
"""
function initialize_state(
    model::ClockModel,
    geometry::Geometry;
    rng::AbstractRNG = Random.default_rng()
)
    validate_model_geometry(model, geometry)

    spins = rand(rng, 0:(model.q - 1), geometry.n_sites)
    T = promote_type(Float64, eltype(couplings(model)), eltype(fields(model)))
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

"""
    random_unit_vector(rng)

Draw a uniformly distributed unit vector in ``\\mathbb{R}^3`` (a random point
on the unit sphere): with `z` uniform on `[-1, 1]` and `φ` uniform on
`[0, 2π)`, the vector `(√(1 - z²) cos φ, √(1 - z²) sin φ, z)` is uniform on
the sphere. Shared by [`initialize_state`](@ref) and the Metropolis
[`propose`](@ref) of the [`HeisenbergModel`](@ref), so both consume the same
two draws per vector from `rng`.

# Returns

A `(x, y, z)` tuple with `x² + y² + z² == 1` to rounding.
"""
function random_unit_vector(rng::AbstractRNG)
    z = 1 - 2 * rand(rng)
    s, c = sincos(2π * rand(rng))
    r = sqrt(1 - z^2)
    return (r * c, r * s, z)
end

"""
    initialize_state(model::HeisenbergModel, geometry::Geometry; rng = Random.default_rng())

Build a random initial [`HeisenbergState`](@ref) for `model` on `geometry`,
with energy and all three Cartesian magnetization components computed from
scratch.

Each site is assigned a uniformly random unit vector (see
[`random_unit_vector`](@ref)). Total energy is `-0.5 * J * (v_i · v_j)`
summed over every bond incident to every site (the `0.5` avoids
double-counting) plus the field contribution `-h * x_i` per site, with the
field pointing along the `x` direction.

# Arguments

- `model`: The [`HeisenbergModel`](@ref) supplying bond strengths `J` and
  field `h`.
- `geometry`: The lattice geometry, giving `n_sites` and the neighbor table.
- `rng`: Random number generator for the initial spin vectors.

# Throws

`ArgumentError` from [`validate_model_geometry`](@ref) if `model` and
`geometry` are incompatible.

# Returns

A randomly initialized [`HeisenbergState`](@ref) with `energy`,
`magnetization_x`, `magnetization_y`, and `magnetization_z` consistent with
its spin components.
"""
function initialize_state(
    model::HeisenbergModel,
    geometry::Geometry;
    rng::AbstractRNG = Random.default_rng()
)
    validate_model_geometry(model, geometry)

    vectors = [random_unit_vector(rng) for _ in 1:geometry.n_sites]
    xs = [v[1] for v in vectors]
    ys = [v[2] for v in vectors]
    zs = [v[3] for v in vectors]
    T = promote_type(Float64, eltype(couplings(model)), eltype(fields(model)))
    energy = zero(T)
    magnetization_x = zero(T)
    magnetization_y = zero(T)
    magnetization_z = zero(T)

    for site in 1:geometry.n_sites
        x = xs[site]
        y = ys[site]
        z = zs[site]
        magnetization_x += x
        magnetization_y += y
        magnetization_z += z
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            energy += -0.5 * bond_strength(model, geometry, bond_id) *
                (x * xs[neighbor] + y * ys[neighbor] + z * zs[neighbor])
        end
        energy += -field_strength(model, geometry, site) * x
    end

    return HeisenbergState(xs, ys, zs, energy, magnetization_x, magnetization_y, magnetization_z)
end
