mutable struct IsingModel
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

function Calc_Specific_Heat(
    bins_container::Dict{String, Vector{Float64}},
    N_measure::Int,
    N_bins::Int,
    β::Float64,
    )

    Energy_avgs = bins_container["Energy"] / N_measure * N_bins
    Energy_Sqrd_avgs = bins_container["Energy_Sqrd"] / N_measure * N_bins
    
    SpecificHeat_avgs = (Energy_Sqrd_avgs - Energy_avgs.^2) * β^2 / length(Energy_avgs)
    ErrorBar = 0.0
    for i in 1:N_bins
        ErrorBar +=  (SpecificHeat_avgs[i] - Statistics.mean(SpecificHeat_avgs))^2
    end
    SpecificHeat = Statistics.mean(SpecificHeat_avgs)
    ErrorBar = sqrt(ErrorBar / N_bins) * sqrt(1 / (N_bins - 1))

    return (SpecificHeat, ErrorBar)
end

function Calc_Magnetic_Susceptibility(
    bins_container::Dict{String, Vector{Float64}},
    N_measure::Int,
    N_bins::Int,
    β::Float64,
    )

    Magnetization_avgs = bins_container["Magnetization"] / N_measure * N_bins
    Magnetization_Sqrd_avgs = bins_container["Magnetization_Sqrd"] / N_measure * N_bins

    MagneticSusceptibility_avgs = (Magnetization_Sqrd_avgs - Magnetization_avgs.^2) * β / length(Magnetization_avgs)
    ErrorBar = 0.0
    for i in 1:N_bins
        ErrorBar +=  (MagneticSusceptibility_avgs[i] - Statistics.mean(MagneticSusceptibility_avgs))^2
    end
    MagneticSusceptibility = Statistics.mean(MagneticSusceptibility_avgs)
    ErrorBar = sqrt(ErrorBar / N_bins) * sqrt(1 / (N_bins - 1))

    return (MagneticSusceptibility, ErrorBar)
end