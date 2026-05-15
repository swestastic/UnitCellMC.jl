struct Metropolis end

function MC_Step!(
    model::IsingModel,
    ::Metropolis,
    β::Float64,
    neighbor_table_map
)
    Acceptance_On_Sweep = 0
    N_sites = length(model.spins)
    for i in 1:N_sites
        s            = model.spins[i]
        neighbor_sum = sum(model.spins[nbr] for nbr in neighbor_table_map[i].neighbors)
        ΔE           = 2 * model.J * s * neighbor_sum
        if ΔE <= 0 || rand() < exp(-β * ΔE)
            model.spins[i] = -s
            Acceptance_On_Sweep += 1
        end
    end
    return Acceptance_On_Sweep
end