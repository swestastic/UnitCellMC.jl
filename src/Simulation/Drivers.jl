"""
    steps_per_sweep(algorithm, geometry)

Return the number of [`step!`](@ref) updates that constitute one Monte Carlo
sweep for `algorithm` on `geometry`.

For local-update algorithms (the default, e.g. [`MetropolisAlgorithm`](@ref))
a sweep is `geometry.n_sites` single-site updates, so that on average every
site is updated once per sweep. For cluster algorithms
([`WolffAlgorithm`](@ref), [`SwendsenWangAlgorithm`](@ref)) a single cluster
update already moves a macroscopic fraction of the lattice, so a sweep is
exactly one call to [`step!`](@ref).

Extend this function to change the sweep size for a custom algorithm.
"""
steps_per_sweep(::AbstractAlgorithm, geometry::Geometry) = geometry.n_sites
steps_per_sweep(::WolffAlgorithm, ::Geometry) = 1
steps_per_sweep(::SwendsenWangAlgorithm, ::Geometry) = 1

"""
    MC_Sweep!(algorithm, model, geometry, state, container, T, stats)

Perform one Monte Carlo sweep on `state` at temperature `T`, accumulating
per-update algorithm diagnostics into `stats`.

The number of [`step!`](@ref) calls per sweep is determined by
[`steps_per_sweep`](@ref): `geometry.n_sites` for local-update algorithms,
one cluster update for cluster algorithms. After each update,
[`diagnostic_values`](@ref) is evaluated and the values matching
`container.diagnostics` are recorded into the corresponding
[`UpdateStatistic`](@ref)s in `stats`, which must be one per diagnostic (see
[`run_simulation`](@ref)).

# Arguments

- `algorithm`: The update algorithm to use.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The current configuration, mutated in place.
- `container`: A [`MeasurementContainer`](@ref); only its `diagnostics` field
  is used to select which diagnostics to accumulate.
- `T`: Temperature passed to [`step!`](@ref).
- `stats`: Vector of [`UpdateStatistic`](@ref)s, one per entry in
  `container.diagnostics`. Reset at the start of the sweep.

# Returns

`stats`, containing the sweep-averaged diagnostic values.
"""
function MC_Sweep!(algorithm, model, geometry, state, container::MeasurementContainer, T, stats)
    foreach(reset!, stats)
    for _ in 1:steps_per_sweep(algorithm, geometry)
        result = step!(algorithm, model, geometry, state, T)
        values = diagnostic_values(algorithm, result)
        for (index, name) in enumerate(container.diagnostics)
            record!(stats[index], getproperty(values, name))
        end
    end
    return stats
end

"""
    run_simulation(algorithm, model, geometry, state, container, parameters)

Run a single-temperature Monte Carlo simulation and return `container` filled
with the binned measurements described by `parameters`.

The sequence of sweeps is:

1. `parameters.n_thermalization` warmup sweeps (simply calls to
   [`MC_Sweep!`](@ref)), with no measurements taken;
2. `parameters.n_measurements` measurement sweeps, each followed by a call to
   [`measure!`](@ref) with the sweep-averaged diagnostics;
3. `parameters.n_unmeasured` decorrelation sweeps after each measurement to
   reduce autocorrelation.

One sweep consists of [`steps_per_sweep`](@ref) calls to [`step!`](@ref):
`geometry.n_sites` single-site updates for `MetropolisAlgorithm`, or a single
cluster update for `WolffAlgorithm` and `SwendsenWangAlgorithm`.

# Arguments

- `algorithm`: The update algorithm to use (e.g. `MetropolisAlgorithm()`).
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The current configuration, mutated in place.
- `container`: The [`MeasurementContainer`](@ref) to fill, mutated in place.
- `parameters`: A [`SimulationParameters`](@ref) controlling temperature and
  sweep counts.

# Returns

`container`, ready for post-processing with [`analyze`](@ref).
"""
function run_simulation(
    algorithm::AbstractAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    container::MeasurementContainer,
    parameters::SimulationParameters
)
    T = parameters.T
    stats = [UpdateStatistic(name) for name in container.diagnostics]

    for _ in 1:parameters.n_thermalization
        MC_Sweep!(algorithm, model, geometry, state, container, T, stats)
    end

    for _ in 1:parameters.n_measurements
        MC_Sweep!(algorithm, model, geometry, state, container, T, stats)
        diagnostics = NamedTuple(
            name => value(stats[index])
            for (index, name) in enumerate(container.diagnostics)
        )
        measure!(container, state; diagnostics)

        for __ in 1:parameters.n_unmeasured
            MC_Sweep!(algorithm, model, geometry, state, container, T, stats)
        end
    end

    return container
end

"""
    sweep_Ts(algorithm, model, geometry, state,
        n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
        simulated_annealing = true, measurements = Symbol[],
        diagnostics = Symbol[], root = "results")

Run a temperature sweep: one [`run_simulation`](@ref) per temperature in
`Ts`, followed by [`analyze`](@ref) at each temperature, and save the
combined results with [`save_sweep_results`](@ref).

A new [`SimulationParameters`](@ref) and [`MeasurementContainer`](@ref) are
created for every temperature. If `simulated_annealing` is `true` (default),
`state` is carried over between temperatures (best practice: order `Ts` from
highest to lowest); otherwise a fresh random state is initialized for each
temperature via [`initialize_state`](@ref).

# Arguments

- `algorithm`: The update algorithm to use.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The starting configuration; reused across temperatures when
  `simulated_annealing = true`.
- `n_thermalization`: Warmup sweeps per temperature.
- `n_measurements`: Measurement sweeps per temperature.
- `n_unmeasured`: Decorrelation sweeps between measurements.
- `n_bins`: Number of bins for jackknife error estimation.
- `Ts`: Temperatures to simulate.
- `simulated_annealing`: Carry `state` between temperatures (default `true`).
- `measurements`: Optional measurement names to enable in each container.
- `diagnostics`: Algorithm diagnostic names to record per sweep.
- `root`: Base directory passed to [`save_sweep_results`](@ref).

# Returns

A vector with one `analyze` result per temperature in `Ts`.
"""
function sweep_Ts(
    algorithm::AbstractAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    n_thermalization::Int,
    n_measurements::Int,
    n_unmeasured::Int,
    n_bins::Int,
    Ts;
    simulated_annealing::Bool = true,
    measurements::Vector{Symbol} = Symbol[],
    diagnostics::Vector{Symbol} = Symbol[],
    root::AbstractString = "results"
)
    sweep_results = Vector{Any}(undef, length(Ts))
    n_sites = geometry.n_sites
    L = geometry.lattice.L

    for (i, T) in enumerate(Ts)
        parameters = SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)

        if !simulated_annealing
            state = initialize_state(model, geometry)
        end

        container = MeasurementContainer(model, geometry, n_measurements, n_bins;
            measurements = measurements,
            diagnostics = diagnostics
        )
        container = run_simulation(algorithm, model, geometry, state, container, parameters)

        sweep_results[i] = analyze(container, T, n_sites)
    end

    save_sweep_results(
        sweep_results, Ts, model, algorithm, L,
        n_thermalization, n_measurements, n_unmeasured, n_bins;
        root = root
    )

    return sweep_results
end
