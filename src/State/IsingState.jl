import LatticeUtilities as lu

"""
    IsingState{S,T<:Real,M<:Integer} <: AbstractState

The mutable configuration of an Ising model simulation: per-site spins and
the running totals (energy, magnetization) kept in sync with them as updates
are applied.

The running totals are maintained incrementally by [`apply_update!`](@ref)
rather than recomputed from scratch each step, so they must be kept
consistent with `spins` by construction (see [`initialize_state`](@ref)) and
by every code path that mutates `spins`.

# Fields

- `spins::S`: Per-site spin values (typically `±1`).
- `energy::T`: Total energy of the current configuration.
- `magnetization::M`: Total magnetization (sum of `spins`) of the current
  configuration.

Construct with [`initialize_state(model, geometry)`](@ref).
"""
mutable struct IsingState{S,T<:Real,M<:Integer} <: AbstractState
    spins::S
    energy::T
    magnetization::M
end

"""
    initialize_state(model, geometry::Geometry)

Build a random initial [`IsingState`](@ref) for `model` on `geometry`, with
energy and magnetization computed from scratch to match the random spins.

Each site is assigned a uniformly random spin of `±1`. Total energy is
computed by summing `-0.5 * J * spins[i] * spins[j]` over every bond incident
to every site (the `0.5` avoids double-counting since each bond is visited
from both endpoints) plus the field contribution `-model.h * spins[i]` per
site; magnetization is simply the sum of all spins.

# Arguments

- `model`: The `IsingModel` supplying bond strengths `J` and field `h`.
- `geometry`: The lattice geometry, giving `n_sites`, `bonds`, and the
  neighbor table used to enumerate each site's bonds.

# Throws

`ArgumentError` if `length(model.J)` does not match `length(geometry.bonds)`.

# Returns

A randomly initialized `IsingState` with `energy` and `magnetization`
consistent with its `spins`.
"""
function initialize_state(
    model,
    geometry::Geometry
    )

length(model.J) == length(geometry.bonds) ||
    throw(ArgumentError("length(model.J) = $(length(model.J)) does not match length(geometry.bonds) = $(length(geometry.bonds))"))

    n_sites = geometry.n_sites
    spins = rand([-1, 1], n_sites)

    T = promote_type(eltype(model.J), typeof(model.h))
    energy = zero(T)

    for i in 1:n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * spins[i] * spins[j]
        end
        energy += -model.h * spins[i]
    end

    magnetization = sum(spins)

    return IsingState(
        spins,
        energy,
        magnetization
    )
end