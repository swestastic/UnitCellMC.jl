@inline function record_all!(stats::Tuple, values, ::Val{D}) where {D}
    ntuple(i -> record!(stats[i], getfield(values, D[i])), Val(length(D)))
    return nothing
end

"""
    sweep_steps!(algorithm, model, geometry, state, T, stats, diagnostics;
        rng, workspace)

Run the [`steps_per_sweep`](@ref) [`step!`](@ref) updates that make up one
Monte Carlo sweep of `algorithm` at temperature `T`, accumulating the
per-update diagnostics selected by `diagnostics` (a `Val` wrapping the
requested keys, from the [`MeasurementContainer`](@ref) type parameter) into
the [`UpdateStatistic`](@ref)s of `stats`. Called by [`MC_Sweep!`](@ref)
after it resets `stats`.

The default method loops over [`step!`](@ref), recording
[`diagnostic_values`](@ref) after each update. A model/algorithm pair with a
fused fast path — currently single-orbital [`IsingModel`](@ref) lattices
under [`MetropolisAlgorithm`](@ref) with [`RandomProposal`](@ref) — may
override it, but an override must visit sites and draw from `rng` exactly as
the generic loop does, keep the state's running totals exact, and record
identical diagnostics.
"""
function sweep_steps!(
    alg, model, geometry, state, T, stats::Tuple, ::Val{D};
    rng::AbstractRNG, workspace
) where {D}
    return _generic_sweep_steps!(
        alg, model, geometry, state, T, stats, Val(D); rng = rng, workspace = workspace
    )
end

# The per-step loop shared by every algorithm without a fused fast path,
# split on the presence of a workspace so custom algorithms whose `step!`
# does not accept the keyword never see it.
function _generic_sweep_steps!(
    alg, model, geometry, state, T, stats::Tuple, ::Val{D};
    rng::AbstractRNG, workspace
) where {D}
    if workspace === nothing
        for _ in 1:steps_per_sweep(alg, geometry)
            result = step!(alg, model, geometry, state, T; rng = rng)
            record_all!(stats, diagnostic_values(alg, result), Val(D))
        end
    else
        for _ in 1:steps_per_sweep(alg, geometry)
            result = step!(alg, model, geometry, state, T; rng = rng, workspace = workspace)
            record_all!(stats, diagnostic_values(alg, result), Val(D))
        end
    end
    return stats
end
