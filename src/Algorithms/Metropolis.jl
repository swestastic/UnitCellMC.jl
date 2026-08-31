function step!(
    ::Metropolis,
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    β::Float64
)
    site = rand(eachindex(state.spins))
    s = state.spins[site]

    info = geometry.neighbor_table[site]

    # Weighted sum: J for each neighbor's bond type × that neighbor's spin
    weighted_sum = sum(
        bond_strength(model, geometry, bond_id) * state.spins[j]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
    )

    ΔE = 2 * s * (weighted_sum + model.h)

    if ΔE <= 0.0 || rand() < exp(-β * ΔE)
        state.spins[site] = -s
        state.energy += ΔE
        state.magnetization -= 2 * s
        return true
    end

    return false
end