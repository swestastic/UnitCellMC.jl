# Fused fast paths for specific model/algorithm pairs.
#
# A fused path is algorithm mechanics specialized to one model, so it lives in
# the algorithm layer — but it names a concrete model type, so this file is
# included from src/UnitCellMC.jl after Models/Models.jl rather than from
# Algorithms/Algorithms.jl. Every override of `sweep_steps!` must satisfy the
# contract in its docstring: visit sites and draw from `rng` exactly as the
# generic per-step loop, keep the state's running totals exact, and record
# identical diagnostics. Acceptance decisions go through the algorithm's
# shared kernel (e.g. `metropolis_accept`) so the fused path can never drift
# from `step!`.

# Fused fast path for the common case of a single-orbital Ising lattice swept
# with uniform-random site selection: with the field uniform across sites,
# the `field_strength` lookup, the `SpinUpdate` struct, and the per-step
# function layering all drop out of the inner loop. The loop draws from `rng`
# exactly as the generic path (one site draw per update, one acceptance draw
# per uphill proposal) and mirrors the `energy_difference` arithmetic term
# for term, so a seeded run follows the same trajectory either way.
function sweep_steps!(
    alg::MetropolisAlgorithm{RandomProposal},
    model::IsingModel,
    geometry::Geometry,
    state::IsingState,
    T, stats::Tuple, diagnostics::Val{D};
    rng::AbstractRNG, workspace
) where {D}
    if LatticeUtilities.norbits(geometry.unit_cell) == 1
        return _ising_metropolis_sweep!(
            alg, model, geometry, state, T, stats, diagnostics, rng,
            _acceptance_cache(workspace)
        )
    end
    return _generic_sweep_steps!(
        alg, model, geometry, state, T, stats, diagnostics; rng = rng, workspace = workspace
    )
end

function _ising_metropolis_sweep!(
    alg::MetropolisAlgorithm, model::IsingModel, geometry::Geometry,
    state::IsingState, T, stats::Tuple, ::Val{D}, rng::AbstractRNG, cache
) where {D}
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))
    validate_model_geometry(model, geometry)

    n = geometry.n_sites
    table = geometry.neighbor_table
    J = couplings(model)
    h = fields(model)[1]  # single orbital: the field is uniform
    spins = state.spins
    length(spins) == n || throw(ArgumentError(
        "length(state.spins) = $(length(spins)) does not match geometry.n_sites = $n"
    ))

    for _ in 1:steps_per_sweep(alg, geometry)
        site = rand(rng, 1:n)
        @inbounds s = spins[site]
        @inbounds info = table[site]
        weighted_sum = zero(promote_type(eltype(spins), eltype(J)))
        @inbounds for k in eachindex(info.neighbors)
            neighbor = info.neighbors[k]
            neighbor == site && continue
            weighted_sum += J[info.templates[k]] * spins[neighbor]
        end
        ΔE = (s - (-s)) * (weighted_sum + h)
        accepted = metropolis_accept(rng, ΔE, T, cache)
        if accepted
            @inbounds spins[site] = -s
            state.energy += ΔE
            state.magnetization += -2 * s
        end
        record_all!(stats, diagnostic_values(alg, MetropolisUpdateResult(accepted)), Val(D))
    end
    return stats
end
