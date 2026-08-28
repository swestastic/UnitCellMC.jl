abstract type AbstractModel end

struct IsingModel <: AbstractModel
    J::Float64
    h::Float64
end

function energy_change(
    model::IsingModel,
    state,
    lattice,
    site
)
    s = state.configuration[site]

    neighbor_sum = sum(
        state.configuration[j]
        for j in neighbors(lattice, site)
    )

    ΔE = 2s * (model.J * neighbor_sum + model.h)

    return ΔE
end

function update!(
    ::IsingModel,
    state,
    lattice,
    site,
    ΔE
)
    s = state.configuration[site]

    state.configuration[site] = -s

    state.energy += ΔE
    state.magnetization -= 2s

    return nothing
end