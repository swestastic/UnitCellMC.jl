mutable struct IsingState{S} <: AbstractState
    spins::S
    energy::Float64
    magnetization::Int64
end

function initialize_state(
    model,
    geometry
    )
    
    N = prod(geometry.lattice.L)
    spins = rand([-1, 1], N)
    energy = 0.0

    for i in 1:N
        for j in geometry.neighbor_table[i].neighbors
            energy += -0.5 * model.J * spins[i] * spins[j]
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