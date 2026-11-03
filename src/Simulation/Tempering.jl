"""
    SwapStats

Running counters of the swap attempts and accepts of a parallel tempering
ladder, one entry per adjacent temperature pair. Filled by
[`swap_pass!`](@ref) and reported in [`ParallelTemperingResult`](@ref).
"""
mutable struct SwapStats
    attempts::Vector{Int}
    accepts::Vector{Int}
end

SwapStats(n_pairs::Int) = SwapStats(zeros(Int, n_pairs), zeros(Int, n_pairs))

"""
    ParallelTemperingResult{R}

The result of a [`tempering_sweep`](@ref): the per-temperature analysis of
every replica plus the swap statistics of the adjacent temperature pairs.

# Fields

- `results::Vector{R}`: One [`analyze`](@ref) result per temperature, in the
  order of the ladder (`results[i]` corresponds to `temperatures[i]`).
- `temperatures::Vector{Float64}`: The temperature ladder, ascending;
  `temperatures[1]` is the coldest replica.
- `swap_attempts::Vector{Int}`: Swap attempts per adjacent pair;
  `swap_attempts[i]` counts attempted exchanges of the configurations at
  `temperatures[i]` and `temperatures[i + 1]`.
- `swap_accepts::Vector{Int}`: Accepted swaps per adjacent pair, same order.
- `swap_acceptance::Vector{Float64}`: `swap_accepts ./ swap_attempts` per
  adjacent pair (`NaN` for a pair that was never attempted, e.g. when
  `swap_every` exceeds the total number of rounds). Acceptance rates far
  from the `0.2`–`0.5` range indicate the ladder spacing should be adjusted;
  see [`geometric_ladder`](@ref).
"""
struct ParallelTemperingResult{R}
    results::Vector{R}
    temperatures::Vector{Float64}
    swap_attempts::Vector{Int}
    swap_accepts::Vector{Int}
    swap_acceptance::Vector{Float64}
end

function ParallelTemperingResult(results::Vector{R}, temperatures, stats::SwapStats) where {R}
    length(results) == length(temperatures) || throw(ArgumentError(
        "results ($(length(results))) and temperatures ($(length(temperatures))) " *
        "must have the same length"))
    length(stats.attempts) == max(length(temperatures) - 1, 0) || throw(ArgumentError(
        "swap statistics ($(length(stats.attempts)) pairs) must cover the " *
        "$(max(length(temperatures) - 1, 0)) adjacent temperature pairs"))
    acceptance = [a == 0 ? NaN : c / a for (a, c) in zip(stats.attempts, stats.accepts)]
    return ParallelTemperingResult{R}(
        results, temperatures, stats.attempts, stats.accepts, acceptance)
end

function Base.show(io::IO, result::ParallelTemperingResult)
    return print(io, "ParallelTemperingResult(",
        length(result.temperatures), " temperatures, ",
        sum(result.swap_accepts), "/", sum(result.swap_attempts), " swaps accepted)")
end

"""
    swap_pass!(states, betas, stats, pass, rng)

Attempt exchanges of the configurations of neighboring replicas in a
parallel tempering ladder, recording the outcome per adjacent pair in
`stats` (see [`SwapStats`](@ref)).

`states[i]` is the configuration currently simulated at inverse temperature
`betas[i]`; a swap exchanges the *configurations* (the `states` array
entries are exchanged, so each replica keeps its fixed temperature and the
cached `energy` travels with the configuration), accepted with the
Metropolis criterion `min(1, exp(x))` where
`x = (betas[i] - betas[j]) * (states[i].energy - states[j].energy)`,
evaluated as `log(rand(rng)) < x` to avoid overflow. The cached running
energies are used directly, making each attempt `O(1)`; their accumulated
floating-point drift over a run (~1e-10 of the total energy) is irrelevant
to an acceptance probability.

To preserve detailed balance while giving every adjacent pair equal
opportunity, the pass alternates its pairing by `pass` (the count of swap
passes so far, starting at 1): odd passes attempt pairs `(1, 2), (3, 4), …`,
even passes `(2, 3), (4, 5), …`. Draws come only from `rng`, consumed in a
fixed serial order, so results are independent of thread count.

# Returns

`states`, with the accepted exchanges applied in place.
"""
function swap_pass!(states, betas, stats::SwapStats, pass::Int, rng::AbstractRNG)
    start = isodd(pass) ? 1 : 2
    for i in start:2:(length(states) - 1)
        j = i + 1
        x = (betas[i] - betas[j]) * (states[i].energy - states[j].energy)
        stats.attempts[i] += 1
        if log(rand(rng)) < x
            states[i], states[j] = states[j], states[i]
            stats.accepts[i] += 1
        end
    end
    return states
end

"""
    TemperingEnsemble

The mutable per-replica state of a [`tempering_sweep`](@ref): one
configuration, measurement container, diagnostic accumulator, workspace, and
random stream per temperature, plus the serial swap stream and statistics.
Swaps exchange entries of `states` only — every other array stays pinned to
its temperature, keeping temperature-dependent workspace data (the
[`AcceptanceCache`](@ref)) valid.
"""
mutable struct TemperingEnsemble{S, C, P, W, R}
    states::S
    containers::Vector{C}
    stats::Vector{P}
    workspaces::Vector{W}
    temperatures::Vector{Float64}
    betas::Vector{Float64}
    rngs::Vector{R}
    swap_rng::R
    swap_stats::SwapStats
    round::Int
end

# Dispatches on the containers' shared diagnostic set D so the per-replica
# statistic tuples are concretely typed (see run_simulation).
function _tempering_ensemble(algorithm, model, geometry, shared, states,
                             containers::Vector{<:MeasurementContainer{D}},
                             temperatures, betas, replica_rngs, swap_rng) where {D}
    stats = [ntuple(i -> UpdateStatistic(D[i]), Val(length(D))) for _ in containers]
    workspaces = [algorithm_workspace(algorithm, model, geometry, s, shared) for s in states]
    return TemperingEnsemble(
        states, containers, stats, workspaces, temperatures, betas,
        replica_rngs, swap_rng, SwapStats(length(temperatures) - 1), 0)
end

"""
    _tempering_round!(algorithm, model, geometry, ensemble, swap_every)

Advance every replica of `ensemble` by one [`MC_Sweep!`](@ref) at its own
temperature — in parallel over threads, each replica consuming only its own
random stream — then, every `swap_every` rounds, run a serial
[`swap_pass!`](@ref). One such round is the parallel tempering analogue of
a single sweep in [`run_simulation`](@ref).
"""
function _tempering_round!(algorithm, model, geometry, ensemble::TemperingEnsemble, swap_every)
    ensemble.round += 1
    Threads.@threads for r in eachindex(ensemble.states)
        MC_Sweep!(algorithm, model, geometry, ensemble.states[r], ensemble.containers[r],
                  ensemble.temperatures[r], ensemble.stats[r];
                  rng = ensemble.rngs[r], workspace = ensemble.workspaces[r])
    end
    if ensemble.round % swap_every == 0
        swap_pass!(ensemble.states, ensemble.betas, ensemble.swap_stats,
                   ensemble.round ÷ swap_every, ensemble.swap_rng)
    end
    return nothing
end

# Function barrier dispatching on the shared diagnostic set D (as in
# _tempering_ensemble) so the diagnostics NamedTuple is built with a static
# key set, as in run_simulation.
_tempering_measure!(ensemble::TemperingEnsemble) =
    _tempering_measure!(ensemble, first(ensemble.containers))
function _tempering_measure!(ensemble::TemperingEnsemble, ::MeasurementContainer{D}) where {D}
    for r in eachindex(ensemble.states)
        measured = NamedTuple{D}(ntuple(i -> value(ensemble.stats[r][i]), Val(length(D))))
        measure!(ensemble.containers[r], ensemble.states[r]; diagnostics = measured)
    end
    return nothing
end

"""
    tempering_sweep(algorithm, model, geometry, Ts,
        n_thermalization, n_measurements, n_unmeasured, n_bins;
        swap_every = 1, measurements = Symbol[], diagnostics = Symbol[],
        time_series = :scalars, states = nothing, root = "results",
        save = true, rng = Random.default_rng(), seed = recorded_seed(rng))

Run a parallel tempering (replica exchange) simulation over the temperature
ladder `Ts` and return a [`ParallelTemperingResult`](@ref). Unless
`save = false`, the output is saved with [`save_tempering_results`](@ref).

One replica simulates `model` at each temperature with `algorithm` — any
`AbstractAlgorithm` works as the inner update, including the cluster
algorithms. The simulation proceeds in *rounds*: one [`MC_Sweep!`](@ref)
per replica (in parallel over threads), followed every `swap_every` rounds
by a serial [`swap_pass!`](@ref) attempting to exchange the configurations
of neighboring temperatures. The sequence of rounds mirrors
[`run_simulation`](@ref) with "sweep" replaced by "round":

1. `n_thermalization` warmup rounds with no measurements;
2. `n_measurements` measurement rounds, each followed by [`measure!`](@ref)
   of every replica into its own [`MeasurementContainer`](@ref);
3. `n_unmeasured` decorrelation rounds after each measurement.

Swap passes also run during thermalization and decorrelation rounds — they
are part of the dynamics, mixing configurations across the ladder so that
low-temperature replicas equilibrate through excursions to high
temperature. After the run, each replica's container is analyzed with
[`analyze`](@ref) at its own temperature, giving one result per temperature
of the same shape as [`sweep_Ts`](@ref) returns.

`Ts` must be sorted in ascending order: `Ts[1]` is the coldest replica.
Build a ladder with [`geometric_ladder`](@ref) and space it so adjacent
swap acceptance rates (reported in the result) land near `0.2`–`0.5`.

# Reproducibility

Before any threads start, `rng` seeds one `Random.Xoshiro` per replica
(used for that replica's state initialization and every update at every
round) plus one more for the swap passes, which are serial. Each random
stream is therefore consumed in a fixed order independent of thread
scheduling, so the results are bit-for-bit identical for any thread count.
Call [`set_global_seed!`](@ref) before the sweep and keep the inputs the
same to reproduce it; the seed is written to `metadata.txt` (as `unknown`
if the global RNG was seeded some other way). An explicit generator such as
`rng = Random.Xoshiro(seed)` works too; pass `seed` alongside it to have it
recorded.

# Arguments

- `algorithm`: The per-replica update algorithm (e.g.
  [`MetropolisAlgorithm`](@ref)).
- `model`: The model defining couplings/energetics.
- `geometry`: The lattice geometry.
- `Ts`: The temperature ladder; at least 2 temperatures, sorted ascending.
- `n_thermalization`: Warmup rounds before measurements begin.
- `n_measurements`: Measurement rounds; must be divisible by `n_bins`.
- `n_unmeasured`: Decorrelation rounds between measurements. Note
  [`autotune_binning`](@ref) pilots are ordinary single-temperature runs;
  its recommended `n_unmeasured` remains a sensible starting point here,
  since tempering only shortens autocorrelation times.
- `n_bins`: Number of bins per replica for jackknife error estimation.
- `swap_every`: Rounds between swap passes (default `1`).
- `measurements`: Optional measurement names enabled in every container.
- `diagnostics`: Algorithm diagnostic names recorded per replica sweep.
- `time_series`: Which observables keep their unbinned history; see
  [`MeasurementContainer`](@ref).
- `states`: Optional starting configurations, one per temperature, e.g.
  pre-thermalized states from an earlier run. By default every replica
  starts from a fresh random state drawn with its own stream.
- `root`: Base directory passed to [`save_tempering_results`](@ref).
- `save`: Save the results under `root` (default `true`). With
  `save = false` nothing is written; the returned result is the only output.
- `rng`: Random number generator from which the per-replica and swap seeds
  are drawn. Defaults to the task-local global RNG, which
  [`set_global_seed!`](@ref) (or `Random.seed!`) controls.
- `seed`: Seed recorded in `metadata.txt`. Defaults to the
  [`global_seed`](@ref) when `rng` is the global RNG, and to `nothing`
  (recorded as `unknown`) for an explicit `rng`.

# Throws

`ArgumentError` if fewer than 2 temperatures are given, if `Ts` is not
sorted ascending or contains a non-positive temperature, if
`swap_every < 1`, if the sweep counts fail [`SimulationParameters`](@ref)
validation, if `states` does not hold exactly one configuration per
temperature or one of them is not the state type expected by `model` (see
[`validate_model_state`](@ref)), or if `measurements`, `diagnostics`, or
`time_series` are rejected by [`MeasurementContainer`](@ref) or
[`validate_run`](@ref). All checks run on the calling task before any
thread starts, so they are thrown directly rather than wrapped in a
`CompositeException`.

# Returns

A [`ParallelTemperingResult`](@ref): one [`analyze`](@ref) result per
temperature plus the swap attempts, accepts, and acceptance rate of each
adjacent temperature pair.
"""
function tempering_sweep(
    algorithm::AbstractAlgorithm,
    model::AbstractModel,
    geometry::Geometry,
    Ts,
    n_thermalization::Int,
    n_measurements::Int,
    n_unmeasured::Int,
    n_bins::Int;
    swap_every::Int = 1,
    measurements::Vector{Symbol} = Symbol[],
    diagnostics::Vector{Symbol} = Symbol[],
    time_series = :scalars,
    states = nothing,
    root::AbstractString = "results",
    save::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
    seed = recorded_seed(rng)
)
    temperatures = Float64.(collect(Ts))
    n_replicas = length(temperatures)
    n_replicas >= 2 || throw(ArgumentError(
        "tempering_sweep needs at least 2 temperatures, got $n_replicas"))
    issorted(temperatures) || throw(ArgumentError(
        "Temperatures must be sorted in ascending order (Ts[1] is the coldest " *
        "replica), got $temperatures"))
    swap_every >= 1 || throw(ArgumentError("swap_every must be >= 1, got $swap_every"))
    # Validates every temperature and the binning up front, on the calling task.
    parameters_by_T = [SimulationParameters(T, n_thermalization, n_measurements,
                                            n_unmeasured, n_bins)
                       for T in temperatures]
    # Deterministic per-geometry workspace data is built once — here on the
    # calling task — and shared by every replica (see sweep_Ts).
    shared = shared_workspace(algorithm, model, geometry)
    if states !== nothing
        length(states) == n_replicas || throw(ArgumentError(
            "states ($(length(states))) must hold one starting configuration per " *
            "temperature ($n_replicas)"))
        foreach(s -> validate_model_state(model, s), states)
    end

    # Seed every replica's generator plus the serial swap stream up front on
    # the calling task, so each stream is consumed in a fixed order
    # independent of thread scheduling and thread count.
    replica_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:n_replicas]
    swap_rng = Random.Xoshiro(rand(rng, UInt64))

    states = states === nothing ?
        [initialize_state(model, geometry; rng = replica_rngs[r]) for r in 1:n_replicas] :
        collect(states)
    containers = [MeasurementContainer(model, geometry, n_measurements, n_bins;
        measurements = measurements, diagnostics = diagnostics, time_series = time_series)
        for _ in 1:n_replicas]
    foreach(r -> validate_run(algorithm, containers[r], parameters_by_T[r]), 1:n_replicas)

    ensemble = _tempering_ensemble(algorithm, model, geometry, shared, states, containers,
                                   temperatures, 1.0 ./ temperatures, replica_rngs, swap_rng)

    for _ in 1:n_thermalization
        _tempering_round!(algorithm, model, geometry, ensemble, swap_every)
    end
    for _ in 1:n_measurements
        _tempering_round!(algorithm, model, geometry, ensemble, swap_every)
        _tempering_measure!(ensemble)
        for _ in 1:n_unmeasured
            _tempering_round!(algorithm, model, geometry, ensemble, swap_every)
        end
    end

    results = [analyze(containers[r], temperatures[r], geometry.n_sites)
               for r in 1:n_replicas]
    result = ParallelTemperingResult(results, temperatures, ensemble.swap_stats)

    save && save_tempering_results(result, model, algorithm, geometry,
        n_thermalization, n_measurements, n_unmeasured, n_bins;
        root = root, seed = seed, swap_every = swap_every,
        measurements = measurements, time_series = time_series)

    return result
end

"""
    geometric_ladder(T_min, T_max, n)

Return `n` geometrically spaced temperatures from `T_min` to `T_max`
(inclusive), sorted ascending — the standard starting point for the
temperature ladder of a [`tempering_sweep`](@ref). Geometric spacing puts
equal ratio between neighbors, which roughly equalizes swap acceptance
rates when the heat capacity varies slowly; refine from the reported
acceptances when it does not.

# Throws

`ArgumentError` if `T_min` is not positive, if `T_max <= T_min`, or if
`n < 2`.

# Returns

A `Vector{Float64}` of length `n` with `Ts[1] == T_min` and
`Ts[end] == T_max`.
"""
function geometric_ladder(T_min::Real, T_max::Real, n::Integer)
    T_min > 0 || throw(ArgumentError("T_min must be positive, got $T_min"))
    T_max > T_min || throw(ArgumentError("T_max must exceed T_min, got T_max = $T_max, T_min = $T_min"))
    n >= 2 || throw(ArgumentError("n must be at least 2, got $n"))
    return exp.(range(log(T_min), log(T_max), n))
end
