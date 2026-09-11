struct MetropolisAlgorithm <: AbstractAlgorithm end

function step!(
    alg::MetropolisAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    T::Real
)
    proposal = propose(alg, model, geometry, state)
    ΔE = energy_difference(model, geometry, state, proposal)

    if ΔE <= 0.0 || log(rand()) < - ΔE / T
        apply_update!(state, proposal, ΔE)
        return true
    end

    return false
end

