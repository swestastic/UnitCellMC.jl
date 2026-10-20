"""
    autotune_binning(algorithm, model, geometry, state, Ts;
                     n_thermalization = 1_000, n_pilot = 5_000,
                     measurements = Symbol[], observables = nothing,
                     bin_factor = 10, n_unmeasured = nothing,
                     window = 6.0, max_lag = nothing)

Choose `n_unmeasured` and the bin size from the data instead of guessing:
run a short pilot simulation at each temperature in `Ts`, measuring after
*every* sweep, estimate the integrated autocorrelation time `τ` (in sweeps)
of each observable with [`autocorrelation`](@ref), and convert the slowest
one into `(bin_size, n_unmeasured)` with [`recommended_binning`](@ref).

Autocorrelation times grow strongly near a phase transition (critical
slowing down, `τ ∼ L^z`) and, for local updates, at low temperature, so pass
every temperature of a planned [`sweep_Ts`](@ref) (or at least those near
the critical point and the lowest one). The pilot visits `Ts` in the order
given and carries `state` from one temperature to the next, like
[`sweep_Ts`](@ref) with `simulated_annealing = true`; `state` is mutated.

# Keywords

- `n_thermalization`: Warmup sweeps before each pilot measurement run.
- `n_pilot`: Pilot measurement sweeps per temperature. The default maximum
  lag (`n_pilot ÷ 10`) resolves `τ` up to about `n_pilot / 60` sweeps.
- `measurements`: Optional measurements to include in the pilot, as for
  [`MeasurementContainer`](@ref) (e.g. `[:abs_magnetization]`).
- `observables`: Names of the observables whose `τ` determines the
  recommendation. Only these keep an unbinned history during the pilot.
  Default: all scalar observables (vector-valued ones such as
  `:correlation` are memory-hungry and must be listed explicitly). For the zero-field
  Ising model you may want to exclude the signed `:magnetization`, whose
  `τ` at low temperature is the (often astronomically long) time to tunnel
  between the `±m` ground states rather than the relaxation time of the
  physical quantities.
- `bin_factor`, `n_unmeasured`: Passed to [`recommended_binning`](@ref).
- `window`, `max_lag`: Passed to [`autocorrelation`](@ref).

# Returns

A named tuple with fields

- `bin_size`, `n_unmeasured`: The recommendation. Use
  `n_measurements = bin_size * n_bins` in production runs.
- `tau`: The slowest integrated autocorrelation time used (in sweeps).
- `T`, `observable`: Where `tau` was found.
- `per_temperature`: For each `T` in `Ts`, a `Dict` mapping observable name
  to its [`AutocorrelationResult`](@ref) (times in sweeps).

If an observable's `τ` could not be resolved within the pilot, a warning is
emitted and its lower bound ([`tau_lower_bound`](@ref)) is used, so the
recommendation may still be too small: increase `n_pilot` in that case.
Observables that did not change at all during the pilot (frozen dynamics)
are ignored with a warning.

# Throws

`ArgumentError` for invalid arguments, or if no selected observable
fluctuated during the pilot (so no autocorrelation time could be estimated).
"""
function autotune_binning(
    algorithm::AbstractAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    Ts;
    n_thermalization::Integer = 1_000,
    n_pilot::Integer = 5_000,
    measurements::AbstractVector{Symbol} = Symbol[],
    observables::Union{AbstractVector{Symbol}, Nothing} = nothing,
    bin_factor::Real = 10,
    n_unmeasured::Union{Integer, Nothing} = nothing,
    window::Real = 6.0,
    max_lag::Union{Integer, Nothing} = nothing,
)
    temperatures = Ts isa Real ? [Ts] : collect(Ts)
    isempty(temperatures) && throw(ArgumentError("Ts must contain at least one temperature"))
    n_pilot >= 2 || throw(ArgumentError("n_pilot must be at least 2, got $n_pilot"))
    observables === nothing || !isempty(observables) ||
        throw(ArgumentError("observables must name at least one observable"))
    n_thermalization >= 0 ||
        throw(ArgumentError("n_thermalization must be >= 0, got $n_thermalization"))

    n_pilot_bins = findfirst(d -> n_pilot % d == 0, 2:n_pilot) + 1   # smallest divisor ≥ 2
    per_temperature = Dict{Symbol, AutocorrelationResult}[]
    unresolved = Tuple{Float64, Symbol}[]
    frozen = Tuple{Float64, Symbol}[]
    best = (tau = -Inf, T = NaN, observable = :none)

    for T in temperatures
        # One measurement per sweep (n_unmeasured = 0), so τ is in sweeps.
        # Bins are irrelevant here, so use as few as the container allows.
        pilot = MeasurementContainer(model, geometry, n_pilot, n_pilot_bins;
            measurements = measurements,
            time_series = something(observables, :scalars),
        )
        selected = collect(keys(pilot.time_series))

        parameters = SimulationParameters(T, n_thermalization, n_pilot, 0, n_pilot_bins)
        run_simulation(algorithm, model, geometry, state, pilot, parameters)

        results = Dict{Symbol, AutocorrelationResult}(
            name => autocorrelation(pilot.time_series[name]; window = window, max_lag = max_lag)
            for name in selected
        )
        push!(per_temperature, results)

        for (name, result) in results
            result.status === :unresolved && push!(unresolved, (Float64(T), name))
            result.status === :constant && push!(frozen, (Float64(T), name))
            tau = tau_lower_bound(result)
            isfinite(tau) || continue
            tau > best.tau && (best = (tau = tau, T = Float64(T), observable = name))
        end
    end

    isempty(unresolved) || @warn(
        "Autocorrelation time not resolved within the pilot run for " *
        join(("$name at T = $T" for (T, name) in unresolved), ", ") *
        "; using a lower bound. Increase `n_pilot` (or exclude these observables " *
        "via `observables`) for a reliable recommendation."
    )
    isempty(frozen) || @warn(
        "No fluctuations during the pilot run for " *
        join(("$name at T = $T" for (T, name) in frozen), ", ") *
        "; excitations are too rare to be sampled on the pilot time scale (e.g. " *
        "no Metropolis move accepted at very low T), so these were ignored. Error " *
        "bars at such temperatures will be zero or unreliable."
    )
    isfinite(best.tau) || throw(ArgumentError(
        "No selected observable fluctuated during the pilot run, so no " *
        "autocorrelation time could be estimated"
    ))

    recommendation = recommended_binning(best.tau; bin_factor = bin_factor, n_unmeasured = n_unmeasured)
    return (
        bin_size = recommendation.bin_size,
        n_unmeasured = recommendation.n_unmeasured,
        tau = best.tau,
        T = best.T,
        observable = best.observable,
        per_temperature = per_temperature,
    )
end
