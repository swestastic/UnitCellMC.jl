struct Glauber end

function MC_Step!(
    model::IsingModel,
    ::Glauber,
    β::Float64,
    neighbor_table_map
)
    N_sites = length(model.spins)
    for i in 1:N_sites
        s            = model.spins[i]
        neighbor_sum = sum(model.spins[nbr] for nbr in neighbor_table_map[i].neighbors)
        ΔE           = 2 * model.J * s * neighbor_sum
        if rand() < 1 / (1 + exp(β * ΔE))
            model.spins[i] = -s
        end
    end
end