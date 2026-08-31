mutable struct IsingState{S} <: AbstractState
    spins::S
    energy::Float64
    magnetization::Int64
end

function initialize_state(
    model,
    geometry
    )

    @assert length(model.J) == length(geometry.bonds) "length(model.J) = $(length(model.J)) does not match length(geometry.bonds) = $(length(geometry.bonds))"

    N = prod(geometry.lattice.L)
    spins = rand([-1, 1], N)
    energy = 0.0

    for i in 1:N
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