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
    MC_Sweep!(algorithm, model, geometry, state, container, T, stats;
        rng = Random.default_rng()) on `state` at temperature `T`, accumulating
per-update algorithm diagnostics into `stats`.

The number of [`step!`](@ref) calls per sweep is determined by
[`steps_per_sweep`](@ref): `geometry.n_sites` for local-update algorithms,
one cluster update for cluster algorithms. After each update,
`diagnostic_values` is evaluated and the values matching
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
- `rng`: Random number generator passed to every [`step!`](@ref).

# Returns

`stats`, containing the sweep-averaged diagnostic values.
"""
function MC_Sweep!(
    algorithm, model, geometry, state, container::MeasurementContainer, T, stats;
    rng::AbstractRNG = Random.default_rng()
)
    foreach(reset!, stats)
    for _ in 1:steps_per_sweep(algorithm, geometry)
        result = step!(algorithm, model, geometry, state, T; rng = rng)
        values = diagnostic_values(algorithm, result)
        for (index, name) in enumerate(container.diagnostics)
            record!(stats[index], getproperty(values, name))
        end
    end
    return stats
end

"""
    run_simulation(algorithm, model, geometry, state, container, parameters;
        rng = Random.default_rng()) Monte Carlo simulation and return `container` filled
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
- `rng`: Random number generator for every update. Defaults to the
  task-local global RNG, so calling `Random.seed!(seed)` beforehand makes the
  run reproducible; pass e.g. `Random.Xoshiro(seed)` for an explicit stream.

# Returns

`container`, ready for post-processing with [`analyze`](@ref).
"""
function run_simulation(
    algorithm::AbstractAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    state::AbstractState,
    container::MeasurementContainer,
    parameters::SimulationParameters;
    rng::AbstractRNG = Random.default_rng()
)
    T = parameters.T
    stats = [UpdateStatistic(name) for name in container.diagnostics]

    for _ in 1:parameters.n_thermalization
        MC_Sweep!(algorithm, model, geometry, state, container, T, stats; rng = rng)
    end

    for _ in 1:parameters.n_measurements
        MC_Sweep!(algorithm, model, geometry, state, container, T, stats; rng = rng)
        diagnostics = NamedTuple(
            name => value(stats[index])
            for (index, name) in enumerate(container.diagnostics)
        )
        measure!(container, state; diagnostics)

        for __ in 1:parameters.n_unmeasured
            MC_Sweep!(algorithm, model, geometry, state, container, T, stats; rng = rng)
        end
    end

    return container
end

"""
    merge_containers(containers)

Merge the binned measurement data of several filled
[`MeasurementContainer`](@ref)s — e.g. from `n_runs` independent
[`run_simulation`](@ref)s at the same temperature — into a single container
whose `data` holds the concatenation of all runs' bins. Analysis on the
merged container therefore pools all runs, and error estimates reflect the
variance between independent runs as well as within them.

The containers must record the same observables and diagnostics, and keep
the same unbinned time series (e.g. built with identical
`measurements`/`diagnostics`/`time_series` arguments); an `ArgumentError`
is thrown otherwise. The time series are concatenated too, with each run's
length stored in `chain_lengths` so autocorrelations never straddle runs.

# Returns

A new `MeasurementContainer` with `length(containers) * n_bins` completed
bins, ready for [`analyze`](@ref).
"""
function merge_containers(containers::Vector{<:MeasurementContainer})
    isempty(containers) && throw(ArgumentError("Cannot merge zero containers"))
    first_container = first(containers)
    expected_names = keys(first_container.data)
    series_names = keys(first_container.time_series)
    for container in containers
        keys(container.data) == expected_names ||
            throw(ArgumentError("All containers must record identical observables and diagnostics"))
        keys(container.time_series) == series_names ||
            throw(ArgumentError("All containers must record identical time series"))
    end

    data = NamedTuple(
        name => reduce(vcat, (container.data[name] for container in containers))
        for name in expected_names
    )
    time_series = NamedTuple(
        name => reduce(vcat, (container.time_series[name] for container in containers))
        for name in series_names
    )
    # Keep track of where each run's series starts so that autocorrelations
    # are never computed across the boundary between independent runs.
    chain_lengths = isempty(series_names) ? Int[] : filter(>(0), reduce(vcat, (
        isempty(container.chain_lengths) ?
            [length(first(container.time_series))] : container.chain_lengths
        for container in containers
    )))
    bin_sums = merge(
        NamedTuple(o.name => o.zero() for o in first_container.observables),
        NamedTuple(name => 0.0 for name in first_container.diagnostics)
    )
    return MeasurementContainer(
        data, bin_sums, time_series, chain_lengths, 0, first_container.bin_size,
        first_container.observables, first_container.derived, first_container.diagnostics
    )
end

"""
    sweep_Ts(algorithm, model, geometry, state,
        n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
        simulated_annealing = true, n_runs = 1, measurements = Symbol[],
        diagnostics = Symbol[], time_series = :scalars, root = "results",
        rng = Random.default_rng(), seed = recorded_seed(rng))

Run a temperature sweep: [`run_simulation`](@ref) at each temperature in
`Ts`, followed by [`analyze`](@ref), and save the combined results with
[`save_sweep_results`](@ref).

With `n_runs > 1`, `n_runs` independent simulations are performed at each
temperature and their bins are merged with [`merge_containers`](@ref) before
analysis, so the returned error bars reflect run-to-run variance. The merged
container holds `n_runs * n_bins` bins per temperature. Runs share an
annealing trajectory: with `simulated_annealing = true` (default) each run
carries its own configuration between temperatures (`state` seeds run 1;
best practice: order `Ts` from highest to lowest); with
`simulated_annealing = false` every run is re-initialized from a fresh
random state at every temperature.

The runs execute in parallel over Julia threads: start Julia with
`--threads auto` (or set `JULIA_NUM_THREADS`) to use more than one.

# Reproducibility

Before any threads start, one seed per run is drawn from `rng` and each run
gets its own `Random.Xoshiro` generator, which it keeps for the whole sweep
(state initialization and every update at every temperature). A run therefore
consumes the same random stream whichever thread executes it, so the results
are bit-for-bit identical for any thread count. To reproduce a sweep, call
[`set_global_seed!`](@ref) at the start of the script and keep the inputs,
including the initial `state`, the same. The seed is written to the sweep's
`metadata.txt` (as `unknown` if the global RNG was seeded some other way,
e.g. with `Random.seed!`), so every results folder records how to recreate
it. An explicit generator such as `rng = Random.Xoshiro(seed)` works too;
pass `seed` alongside it to have it recorded.

A new [`SimulationParameters`](@ref) is created for every temperature
(shared by all runs at that temperature), and a new
[`MeasurementContainer`](@ref) is created for every temperature and run.

# Arguments

- `algorithm`: The update algorithm to use.
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `state`: The starting configuration for run 1; reused across temperatures
  when `simulated_annealing = true`.
- `n_thermalization`: Warmup sweeps per run and temperature.
- `n_measurements`: Measurement sweeps per run and temperature.
- `n_unmeasured`: Decorrelation sweeps between measurements.
- `n_bins`: Number of bins per run for jackknife error estimation.
- `Ts`: Temperatures to simulate.
- `simulated_annealing`: Carry each run's state between temperatures
  (default `true`).
- `n_runs`: Number of independent runs averaged at each temperature
  (default `1`); must be at least 1.
- `measurements`: Optional measurement names to enable in each container.
- `diagnostics`: Algorithm diagnostic names to record per sweep.
- `time_series`: Which observables keep their unbinned history for the
  autocorrelation analysis in [`analyze`](@ref); see
  [`MeasurementContainer`](@ref). The default `:scalars` skips the
  correlation-function vector; use `:none` to store nothing.
- `root`: Base directory passed to [`save_sweep_results`](@ref).
- `rng`: Random number generator from which the per-run seeds are drawn.
  Defaults to the task-local global RNG, which [`set_global_seed!`](@ref)
  (or `Random.seed!`) controls.
- `seed`: Seed recorded in `metadata.txt`. Defaults to the
  [`global_seed`](@ref) when `rng` is the global RNG, and to `nothing`
  (recorded as `unknown`) for an explicit `rng`.

# Throws

`ArgumentError` if `n_runs` is less than 1.

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
    n_runs::Int = 1,
    measurements::Vector{Symbol} = Symbol[],
    diagnostics::Vector{Symbol} = Symbol[],
    time_series = :scalars,
    root::AbstractString = "results",
    rng::AbstractRNG = Random.default_rng(),
    seed = recorded_seed(rng)
)
    n_runs >= 1 || throw(ArgumentError("n_runs must be >= 1, got $n_runs"))

    sweep_results = Vector{Any}(undef, length(Ts))
    n_sites = geometry.n_sites
    L = geometry.lattice.L

    # Seed every run's generator up front on the calling task, so each run
    # owns a fixed stream independent of thread scheduling and thread count.
    run_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:n_runs]
    states = AbstractState[state]
    for r in 2:n_runs
        push!(states, initialize_state(model, geometry; rng = run_rngs[r]))
    end

    for (i, T) in enumerate(Ts)
        parameters = SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)

        run_containers = Vector{MeasurementContainer}(undef, n_runs)
        Threads.@threads for r in 1:n_runs
            run_rng = run_rngs[r]
            run_state = states[r]
            if !simulated_annealing
                run_state = initialize_state(model, geometry; rng = run_rng)
                states[r] = run_state
            end

            container = MeasurementContainer(model, geometry, n_measurements, n_bins;
                measurements = measurements,
                diagnostics = diagnostics,
                time_series = time_series
            )
            run_containers[r] = run_simulation(
                algorithm, model, geometry, run_state, container, parameters; rng = run_rng
            )
        end

        sweep_results[i] = analyze(merge_containers(run_containers), T, n_sites)
    end

    save_sweep_results(
        sweep_results, Ts, model, algorithm, L,
        n_thermalization, n_measurements, n_unmeasured, n_bins;
        root = root, seed = seed
    )

    return sweep_results
end
