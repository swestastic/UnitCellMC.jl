import LatticeUtilities as lu

mutable struct IsingState{S,T<:Real,M<:Integer} <: AbstractState
    spins::S
    energy::T
    magnetization::M
end

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