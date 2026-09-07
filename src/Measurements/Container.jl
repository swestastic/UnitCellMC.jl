mutable struct MeasurementContainer{T, S}
    data::T
    bin_sums::S
    bin_count::Int
    bin_size::Int
    observables::Vector{Observable}
    derived::Vector{DerivedObservable}
end

function MeasurementContainer(
    model,
    geometry,
    n_steps,
    n_bins;
    measurements = Symbol[]
)
    n_steps > 0 || throw(ArgumentError("n_steps must be positive, got $n_steps"))
    n_bins >= 2 || throw(ArgumentError("n_bins must be at least 2 for jackknife error estimates, got $n_bins"))
    n_steps >= n_bins || throw(ArgumentError("n_steps ($n_steps) must be at least n_bins ($n_bins)"))
    n_steps % n_bins == 0 ||
        throw(ArgumentError("n_steps ($n_steps) must be divisible by n_bins ($n_bins)"))

    opt_obs, opt_der = optional_observables(model, geometry, measurements)

    obs = vcat(observables(model, geometry), opt_obs)
    der = vcat(derived_observables(model, geometry), opt_der)

    validate_dependencies(obs, der)

    data     = NamedTuple(o.name => o.template() for o in obs)
    bin_sums = NamedTuple(o.name => o.zero()     for o in obs)

    return MeasurementContainer(data, bin_sums, 0, n_steps ÷ n_bins, obs, der)
end

function validate_dependencies(obs::Vector{Observable}, der::Vector{DerivedObservable})
    available = Set(o.name for o in obs)
    for d in der
        missing = filter(dep -> dep ∉ available, d.depends_on)
        isempty(missing) && continue
        throw(ArgumentError(
            "Derived observable $(d.names) depends on $(missing), which " *
            (length(missing) == 1 ? "is" : "are") *
            " not among the requested observables. Check `measurements` includes " *
            "whatever raw Observable(s) provide $(missing)."
        ))
    end
    return nothing
end