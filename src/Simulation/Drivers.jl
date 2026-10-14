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
    merge_containers(containers)

Merge the binned measurement data of several filled
[`MeasurementContainer`](@ref)s — e.g. from `n_runs` independent
[`run_simulation`](@ref)s at the same temperature — into a single container
whose `data` holds the concatenation of all runs' bins. Analysis on the
merged container therefore pools all runs, and error estimates reflect the
variance between independent runs as well as within them.

The containers must record the same observables and diagnostics (e.g. built
with identical `measurements`/`diagnostics` arguments); an `ArgumentError`
is thrown otherwise.

# Returns

A new `MeasurementContainer` with `length(containers) * n_bins` completed
bins, ready for [`analyze`](@ref).
"""
function merge_containers(containers::Vector{<:MeasurementContainer})
    isempty(containers) && throw(ArgumentError("Cannot merge zero containers"))
    first_container = first(containers)
    expected_names = keys(first_container.data)
    for container in containers
        keys(container.data) == expected_names ||
            throw(ArgumentError("All containers must record identical observables and diagnostics"))
    end

    data = NamedTuple(
        name => reduce(vcat, (container.data[name] for container in containers))
        for name in expected_names
    )
    bin_sums = merge(
        NamedTuple(o.name => o.zero() for o in first_container.observables),
        NamedTuple(name => 0.0 for name in first_container.diagnostics)
    )
    return MeasurementContainer(
        data, bin_sums, 0, first_container.bin_size,
        first_container.observables, first_container.derived, first_container.diagnostics
    )
end

"""
    sweep_Ts(algorithm, model, geometry, state,
        n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
        simulated_annealing = true, n_runs = 1, measurements = Symbol[],
        diagnostics = Symbol[], root = "results")

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
`--threads auto` (or set `JULIA_NUM_THREADS`) to use more than one. Each
run uses its own thread-local RNG stream, so results are statistically
independent regardless of thread count.

A new [`SimulationParameters`](@ref) and [`MeasurementContainer`](@ref) are
created for every temperature and run.

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
- `root`: Base directory passed to [`save_sweep_results`](@ref).

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
    root::AbstractString = "results"
)
    n_runs >= 1 || throw(ArgumentError("n_runs must be >= 1, got $n_runs"))

    sweep_results = Vector{Any}(undef, length(Ts))
    n_sites = geometry.n_sites
    L = geometry.lattice.L
    states = AbstractState[state]
    for _ in 2:n_runs
        push!(states, initialize_state(model, geometry))
    end

    for (i, T) in enumerate(Ts)
        parameters = SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)

        run_containers = Vector{MeasurementContainer}(undef, n_runs)
        Threads.@threads for r in 1:n_runs
            run_state = states[r]
            if !simulated_annealing
                run_state = initialize_state(model, geometry)
                states[r] = run_state
            end

            container = MeasurementContainer(model, geometry, n_measurements, n_bins;
                measurements = measurements,
                diagnostics = diagnostics
            )
            run_containers[r] = run_simulation(algorithm, model, geometry, run_state, container, parameters)
        end

        sweep_results[i] = analyze(merge_containers(run_containers), T, n_sites)
    end

    save_sweep_results(
        sweep_results, Ts, model, algorithm, L,
        n_thermalization, n_measurements, n_unmeasured, n_bins;
        root = root
    )

    return sweep_results
end
