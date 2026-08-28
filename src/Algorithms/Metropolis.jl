function step!(
    ::Metropolis,
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    β::Float64
)
    # Choose random site
    site = rand(eachindex(state.spins))

    # Current spin
    s = state.spins[site]

    # Sum neighboring spins
    neighbor_sum = sum(
        state.spins[j]
        for j in geometry.neighbor_table[site].neighbors
    )

    # Energy change from s -> -s
    ΔE = 2 * s * (
        model.J * neighbor_sum +
        model.h
    )

    # Metropolis acceptance
    if ΔE <= 0.0 || rand() < exp(-β * ΔE)

        # Flip spin
        state.spins[site] = -s

        # Update state
        state.energy += ΔE
        state.magnetization -= 2 * s

        return true
    end

    return false
end