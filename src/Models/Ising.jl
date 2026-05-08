struct IsingModel
    spins::Vector{Int}
    J::Float64
    Lx::Int
    Ly::Int
end

function IsingModel(J::Float64, Lx::Int, Ly::Int)
    model = IsingModel(zeros(Int, Lx * Ly), J, Lx, Ly)
    Initialize_Spins!(model)
    return model
end

function Initialize_Spins!(model::IsingModel)
    model.spins .= rand([-1, 1], length(model.spins))
end

function Calc_Energy(model::IsingModel, neighbor_table_map)
    energy = 0.0
    for i in 1:length(model.spins)
        for nbr in neighbor_table_map[i].neighbors
            energy += -model.J * model.spins[i] * model.spins[nbr]
        end
    end
    return energy / 2   # correct for double-counting
end

function Calc_Magnetization(model::IsingModel)
    return Float64(sum(model.spins))
end

mutable struct Measurements
    E::Float64
    M::Float64
end
