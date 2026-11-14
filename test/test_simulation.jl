@testset "SimulationParameters validation" begin
    @test_throws ArgumentError ucmc.SimulationParameters(0.0, 0, 0, 0, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, -1, 0, 0, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, -1, 0, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, 0, -1, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, 0, 0, 0)
end

@testset "SimulationParameters and MeasurementContainer agree on binning" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])

    # (n_measurements, n_bins) pairs: both constructors accept or both reject
    valid = [(2, 2), (8, 4), (20, 4), (9, 3)]
    invalid = [(0, 2), (-4, 2), (4, 1), (4, 0), (1, 2), (10, 4), (95, 10)]
    for (n_measurements, n_bins) in valid
        @test ucmc.SimulationParameters(1.0, 0, n_measurements, 0, n_bins) isa ucmc.SimulationParameters
        @test ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins).bin_size ==
            n_measurements ÷ n_bins
    end
    for (n_measurements, n_bins) in invalid
        @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, n_measurements, 0, n_bins)
        @test_throws ArgumentError ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins)
    end
end

@testset "steps_per_sweep" begin
    geometry = square_geometry(4)
    @test ucmc.steps_per_sweep(ucmc.MetropolisAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.GlauberAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.HeatbathAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.OverrelaxationAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.WolffAlgorithm(), geometry) == 1
    @test ucmc.steps_per_sweep(ucmc.SwendsenWangAlgorithm(), geometry) == 1
end

@testset "MC_Sweep! update counts" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    n_sites = geometry.n_sites

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:acceptance_ratio])
    stats = (ucmc.UpdateStatistic(:acceptance_ratio),)
    ucmc.MC_Sweep!(ucmc.MetropolisAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == n_sites
    @test 0.0 <= ucmc.value(stats[1]) <= 1.0

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:change_ratio])
    stats = (ucmc.UpdateStatistic(:change_ratio),)
    ucmc.MC_Sweep!(ucmc.HeatbathAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == n_sites
    @test 0.0 <= ucmc.value(stats[1]) <= 1.0

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:cluster_size])
    stats = (ucmc.UpdateStatistic(:cluster_size),)
    ucmc.MC_Sweep!(ucmc.WolffAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == 1
    @test 1 <= ucmc.value(stats[1]) <= n_sites

    container = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:cluster_count, :largest_cluster_fraction])
    stats = (ucmc.UpdateStatistic(:cluster_count), ucmc.UpdateStatistic(:largest_cluster_fraction))
    ucmc.MC_Sweep!(ucmc.SwendsenWangAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == 1
    @test stats[2].count == 1
    @test ucmc.value(stats[1]) >= 1
    @test 0.0 < ucmc.value(stats[2])
end

@testset "Algorithms reject non-positive temperature" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    for alg in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(), ucmc.WolffAlgorithm(),
                ucmc.HeatbathAlgorithm(), ucmc.OverrelaxationAlgorithm(),
                ucmc.SwendsenWangAlgorithm())
        for T in (0.0, -1.0, -Inf)
            @test_throws ArgumentError ucmc.step!(alg, model, geometry, state, T)
        end
    end
end

@testset "run_simulation fills bins" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    parameters = ucmc.SimulationParameters(2.0, 5, 20, 2, 4)

    container = ucmc.MeasurementContainer(model, geometry, 20, 4; diagnostics = [:acceptance_ratio])
    result = ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    @test result === container
    @test length(container.data.energy) == 4
    @test length(container.data.acceptance_ratio) == 4
    @test all(v -> 0.0 <= v <= 1.0, container.data.acceptance_ratio)

    # empty diagnostics path
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(model, geometry, 20, 4)
    ucmc.run_simulation(ucmc.WolffAlgorithm(), model, geometry, state, container, parameters)
    @test length(container.data.energy) == 4
    @test !haskey(pairs(container.data), :acceptance_ratio)
end

@testset "run_simulation rejects a container that does not match the parameters" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()

    # container bins 100 measurements into 10 bins of 10
    for parameters in (
        ucmc.SimulationParameters(2.0, 5, 90, 0, 10),   # bin_size 9
        ucmc.SimulationParameters(2.0, 5, 100, 0, 5),   # bin_size 20
        ucmc.SimulationParameters(2.0, 5, 50, 0, 10),   # bin_size 5
    )
        state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
        spins = copy(state.spins)
        container = ucmc.MeasurementContainer(model, geometry, 100, 10)
        @test_throws ArgumentError ucmc.run_simulation(
            algorithm, model, geometry, state, container, parameters; rng = Xoshiro(2)
        )
        # rejected before any sweep: neither the state nor the container changed
        @test state.spins == spins
        @test isempty(container.data.energy)
        @test container.bin_count == 0
    end

    # matching counts fill exactly n_bins complete bins
    parameters = ucmc.SimulationParameters(2.0, 5, 100, 0, 10)
    container = ucmc.MeasurementContainer(model, geometry, 100, 10)
    ucmc.run_simulation(algorithm, model, geometry, ucmc.initialize_state(model, geometry),
        container, parameters)
    @test length(container.data.energy) == 10
    @test container.bin_count == 0

    # diagnostics the algorithm does not provide are rejected up front
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(model, geometry, 100, 10; diagnostics = [:cluster_size])
    err = try
        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("cluster_size", err.msg)
    @test isempty(container.data.energy)
end

@testset "sweep_Ts end-to-end" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    Ts = [3.0, 1.0]
    root = mktempdir()

    sweep_results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, state,
        3, 6, 1, 3, Ts;
        simulated_annealing = true,
        diagnostics = [:cluster_size],
        root = root
    )

    @test length(sweep_results) == length(Ts)
    @test all(r -> haskey(r.primary, :energy), sweep_results)
    @test all(r -> haskey(r.primary, :cluster_size), sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry; root = root)
    @test isdir(dir)
    @test isfile(joinpath(dir, "measurement_results.csv"))
end

@testset "merge_containers pools run bins" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    parameters = ucmc.SimulationParameters(2.0, 2, 8, 1, 4)

    containers = map(1:2) do _
        state = ucmc.initialize_state(model, geometry)
        container = ucmc.MeasurementContainer(model, geometry, 8, 4; diagnostics = [:acceptance_ratio])
        ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    end

    merged = ucmc.merge_containers(containers)
    @test length(merged.data.energy) == 8 # 2 runs * 4 bins per run
    @test merged.data.energy == vcat(containers[1].data.energy, containers[2].data.energy)
    @test length(merged.data.acceptance_ratio) == 8

    # analysis works on the pooled bins
    results = ucmc.analyze(merged, 2.0, geometry.n_sites)
    @test haskey(results.primary, :energy)

    # mismatched containers and empty input are rejected, with the differing
    # keys named in the message
    mismatched = ucmc.MeasurementContainer(model, geometry, 8, 4; diagnostics = [:cluster_size])
    err = try
        ucmc.merge_containers([first(containers), mismatched])
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("cluster_size", err.msg)
    @test_throws ArgumentError ucmc.merge_containers(ucmc.MeasurementContainer[])

    # key order is not significant: the same measurements in a different
    # request order merge fine (the first container's layout is kept)
    fwd = ucmc.MeasurementContainer(model, geometry, 8, 4;
        measurements = [:raw_correlation, :magnetization_per_orbital])
    rev = ucmc.MeasurementContainer(model, geometry, 8, 4;
        measurements = [:magnetization_per_orbital, :raw_correlation])
    @test keys(fwd.data) != keys(rev.data)  # order differs
    reordered = ucmc.merge_containers([fwd, rev])
    @test keys(reordered.data) == keys(fwd.data)

    # mismatched bin sizes are rejected: pooling unequal-size bins would
    # silently weight unequal-quality averages equally (both containers hold
    # the same observables, so the bin-size check is what fires)
    other_bin_size = ucmc.MeasurementContainer(model, geometry, 12, 4;  # bin_size 3, not 2
                                               diagnostics = [:acceptance_ratio])
    err = try
        ucmc.merge_containers([first(containers), other_bin_size])
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("bin size", err.msg)

    # the derived quantities must match too: identical raw data keys with a
    # different derived set would silently gain/lose quantities in the pooled
    # analysis (binder_cumulant adds a derived entry on top of the same raw
    # magnetization_fourth observable)
    plain = ucmc.MeasurementContainer(model, geometry, 8, 4; measurements = [:magnetization_fourth])
    binder = ucmc.MeasurementContainer(model, geometry, 8, 4; measurements = [:binder_cumulant])
    @test keys(plain.data) == keys(binder.data)  # the trap: raw keys match
    err = try
        ucmc.merge_containers([plain, binder])
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("derived", err.msg)
    @test occursin("binder_cumulant", err.msg)
    @test_throws ArgumentError ucmc.merge_containers([binder, plain])

    # same for the correlation labels: raw + per-orbital magnetization has the
    # same raw keys as connected_correlation but lacks its derived entries
    raw_mpo = ucmc.MeasurementContainer(model, geometry, 8, 4;
        measurements = [:raw_correlation, :magnetization_per_orbital])
    connected = ucmc.MeasurementContainer(model, geometry, 8, 4;
        measurements = [:connected_correlation])
    @test_throws ArgumentError ucmc.merge_containers([raw_mpo, connected])

    # same bin size with different bin counts merges fine: equal-size bins
    # stay equal-weight in the pooled jackknife (this is how an
    # already-merged container pools with a fresh run)
    parameters_long = ucmc.SimulationParameters(2.0, 2, 10, 1, 5)
    longer = ucmc.MeasurementContainer(model, geometry, 10, 5;  # bin_size 2, 5 bins
                                       diagnostics = [:acceptance_ratio])
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry,
                        ucmc.initialize_state(model, geometry), longer, parameters_long)
    pooled = ucmc.merge_containers([first(containers), longer])
    @test length(pooled.data.energy) == 4 + 5
    @test pooled.bin_size == 2

    # a single-container merge copies instead of aliasing: a later measure!
    # on either container must not mutate the other
    solo = ucmc.MeasurementContainer(model, geometry, 8, 4;
                                     diagnostics = [:acceptance_ratio], time_series = :all)
    solo_state = ucmc.initialize_state(model, geometry)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, solo_state, solo, parameters)
    merged_solo = ucmc.merge_containers([solo])
    @test merged_solo.data.energy == solo.data.energy
    @test merged_solo.data.energy !== solo.data.energy
    @test merged_solo.time_series.energy !== solo.time_series.energy

    n_bins_before = length(merged_solo.data.energy)
    n_series_before = length(merged_solo.time_series.energy)
    # two more measurements (bin_size = 2) flush one new bin into `solo`
    ucmc.measure!(solo, solo_state; diagnostics = (acceptance_ratio = 0.5,))
    ucmc.measure!(solo, solo_state; diagnostics = (acceptance_ratio = 0.5,))
    @test length(solo.data.energy) == n_bins_before + 1
    @test length(merged_solo.data.energy) == n_bins_before
    @test length(merged_solo.time_series.energy) == n_series_before

    # merge_containers is exported (it is documented in the API reference)
    @test :merge_containers in names(ucmc)
end

@testset "sweep_Ts ensemble averaging" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    Ts = [3.0, 1.0]
    root = mktempdir()

    sweep_results = ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, state,
        3, 6, 1, 3, Ts;
        simulated_annealing = false,
        n_runs = 2,
        diagnostics = [:acceptance_ratio],
        root = root
    )

    @test length(sweep_results) == length(Ts)
    @test all(r -> haskey(r.primary, :energy), sweep_results)
    @test all(r -> haskey(r.primary, :acceptance_ratio), sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))

    @test_throws ArgumentError ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, state,
        3, 6, 1, 3, Ts; n_runs = 0, root = root
    )
end

@testset "sweep_Ts validates inputs before the threaded loop" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()

    # Each call must throw the ArgumentError itself, not a CompositeException
    # from inside Threads.@threads, and must fail before any simulation runs:
    # the starting state is untouched and no results directory is written.
    function sweep_error(args...; kwargs...)
        state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
        spins = copy(state.spins)
        root = mktempdir()
        err = try
            ucmc.sweep_Ts(algorithm, model, geometry, state, args...;
                n_runs = 2, root = root, kwargs...)
            nothing
        catch e
            e
        end
        untouched = state.spins == spins && isempty(readdir(root))
        return err, untouched
    end

    cases = [
        ((3, 6, 1, 1, [2.0]), (;)),                                  # n_bins < 2
        ((3, 0, 1, 2, [2.0]), (;)),                                  # n_measurements = 0
        ((3, 7, 1, 2, [2.0]), (;)),                                  # not divisible
        ((3, 6, 1, 3, [2.0, 1.0, -1.0]), (;)),                       # bad last temperature
        ((3, 6, 1, 3, [2.0]), (; diagnostics = [:cluster_size])),    # wrong algorithm diagnostic
        ((3, 6, 1, 3, [2.0]), (; time_series = [:not_measured])),    # container rejects
    ]
    for (args, kwargs) in cases, annealing in (true, false)
        err, untouched = sweep_error(args...; simulated_annealing = annealing, kwargs...)
        @test err isa ArgumentError
        @test untouched
    end
end

@testset "RNG reproducibility" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    xy_model = ucmc.XYModel([1.0, 1.0], [0.0])
    clock_model = ucmc.ClockModel(5, [1.0, 1.0], [0.3])
    heisenberg_model = ucmc.HeisenbergModel([1.0, 1.0], [0.0])
    parameters = ucmc.SimulationParameters(2.0, 5, 20, 1, 4)

    # explicit rng: identical streams give identical states and bins
    for m in (model, xy_model, clock_model, heisenberg_model),
        algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(), ucmc.WolffAlgorithm(),
                      ucmc.HeatbathAlgorithm(), ucmc.SwendsenWangAlgorithm())
        containers = map(1:2) do _
            rng = Xoshiro(2024)
            state = ucmc.initialize_state(m, geometry; rng = rng)
            container = ucmc.MeasurementContainer(m, geometry, 20, 4)
            ucmc.run_simulation(algorithm, m, geometry, state, container, parameters; rng = rng)
        end
        @test isequal(containers[1].data, containers[2].data)
    end

    # the global RNG is the default, so Random.seed! reproduces a run
    runs = map(1:2) do _
        Random.seed!(99)
        state = ucmc.initialize_state(model, geometry)
        container = ucmc.MeasurementContainer(model, geometry, 20, 4)
        ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    end
    @test isequal(runs[1].data, runs[2].data)

    # threaded sweep_Ts runs are reproducible from the global seed
    root = mktempdir()
    primaries(results) = [r.primary for r in results]
    sweep(state; kwargs...) = ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, state,
        5, 20, 1, 4, [3.0, 2.0]; n_runs = 4, time_series = :none, root = root, kwargs...
    )
    seeded_sweep(seed; kwargs...) = (Random.seed!(seed);
        sweep(ucmc.initialize_state(model, geometry); kwargs...))
    for annealing in (true, false)
        @test isequal(
            primaries(seeded_sweep(7; simulated_annealing = annealing)),
            primaries(seeded_sweep(7; simulated_annealing = annealing))
        )
    end
    @test !isequal(primaries(seeded_sweep(7)), primaries(seeded_sweep(8)))

    # an explicit rng behaves like the seeded global RNG
    explicit_sweep(seed) = (rng = Xoshiro(seed);
        sweep(ucmc.initialize_state(model, geometry; rng = rng); rng = rng))
    @test isequal(primaries(explicit_sweep(5)), primaries(explicit_sweep(5)))
end

@testset "autotune_binning" begin
    Random.seed!(11)
    geometry = square_geometry(6)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    Ts = [5.0, 2.3]

    tuning = ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, Ts;
        n_thermalization = 200, n_pilot = 4_000,
        measurements = [:abs_magnetization],
        observables = [:energy, :abs_magnetization],
    )
    @test length(tuning.per_temperature) == length(Ts)
    @test all(d -> Set(keys(d)) == Set([:energy, :abs_magnetization]), tuning.per_temperature)
    @test tuning.T in Ts
    @test tuning.observable in (:energy, :abs_magnetization)
    # the recommendation is driven by the slowest observable/temperature
    @test tuning.tau == maximum(ucmc.tau_lower_bound(r) for d in tuning.per_temperature for r in values(d))
    @test (bin_size = tuning.bin_size, n_unmeasured = tuning.n_unmeasured) ==
        ucmc.recommended_binning(tuning.tau)
    # Metropolis dynamics are correlated over more than one sweep, and much
    # slower close to T_c ≈ 2.269 than deep in the paramagnet (critical slowing down)
    @test tuning.tau > 1.0
    @test tuning.per_temperature[2][:abs_magnetization].tau > tuning.per_temperature[1][:abs_magnetization].tau

    # cluster updates decorrelate the order parameter much faster than Metropolis at T_c
    wolff = ucmc.autotune_binning(
        ucmc.SwendsenWangAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry), 2.3;
        n_thermalization = 200, n_pilot = 4_000,
        measurements = [:abs_magnetization], observables = [:abs_magnetization],
    )
    @test wolff.tau < tuning.per_temperature[2][:abs_magnetization].tau

    # with a fixed n_unmeasured only the bin size is tuned
    fixed = ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, 3.0;
        n_thermalization = 50, n_pilot = 2_000, n_unmeasured = 0,
    )
    @test fixed.n_unmeasured == 0
    @test fixed.bin_size == ceil(Int, 10 * fixed.tau)

    @test_throws ArgumentError ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0;
        n_pilot = 100, observables = [:not_measured],
    )
    @test_throws ArgumentError ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, Float64[]
    )
    @test_throws ArgumentError ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0; observables = Symbol[]
    )

    # by default only scalar observables are tracked in the pilot; vectors on request
    pilot = ucmc.autotune_binning(
        ucmc.WolffAlgorithm(), model, geometry, state, 2.5;
        n_thermalization = 10, n_pilot = 501, measurements = [:raw_correlation],
    )
    @test !haskey(pilot.per_temperature[1], :correlation)
    @test haskey(pilot.per_temperature[1], :energy)
    pilot = ucmc.autotune_binning(
        ucmc.WolffAlgorithm(), model, geometry, state, 2.5;
        n_thermalization = 10, n_pilot = 501, measurements = [:raw_correlation],
        observables = [:correlation],
    )
    @test collect(keys(pilot.per_temperature[1])) == [:correlation]
end
