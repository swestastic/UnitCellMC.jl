@testset "Sweep save edge cases" begin
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()
    geometry = square_geometry(2)
    root = mktempdir()

    empty_dir = ucmc.save_sweep_results([], Float64[], model, algorithm, geometry, 0, 0, 0, 1; root = root)
    @test isdir(empty_dir)
    @test isfile(joinpath(empty_dir, "metadata.txt"))
    @test !isfile(joinpath(empty_dir, "measurement_results.csv"))
    @test !isfile(joinpath(empty_dir, "correlation_results.csv"))

    @test_throws ArgumentError ucmc.save_sweep_results([Dict()], [1.0, 2.0], model, algorithm, geometry, 0, 0, 0, 1; root = root)

    # CSV rows are deterministic: the primary (Dict) entries come out sorted
    fake = (primary = Dict(:z => (1.0, 0.1), :a => (2.0, 0.2), :m => (3.0, 0.3), :e => (4.0, 0.4)),
            derived = NamedTuple())
    names, values, errors = ucmc.measurement_rows(fake)
    @test names == ["a", "e", "m", "z"]
    @test values == [2.0, 4.0, 3.0, 1.0]
    @test errors == [0.2, 0.4, 0.3, 0.1]
end

@testset "sweep_Ts save keyword" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    root = mktempdir()

    # save = false: pure computation, nothing written
    results = ucmc.sweep_Ts(ucmc.MetropolisAlgorithm(), model, geometry, state,
        1, 4, 0, 2, [2.0]; root = root, save = false)
    @test length(results) == 1
    @test haskey(results[1].primary, :energy)
    @test isempty(readdir(root))

    # default (save = true) writes the sweep directory
    ucmc.sweep_Ts(ucmc.MetropolisAlgorithm(), model, geometry, state,
        1, 4, 0, 2, [2.0]; root = root)
    dir = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))
end

@testset "seed recorded in metadata" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    metadata(dir) = readlines(joinpath(dir, "metadata.txt"))
    sweep_dir(root) = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry; root = root)
    sweep(root; kwargs...) = ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        2, 4, 0, 2, [2.0]; n_runs = 2, time_series = :none, root = root, kwargs...
    )

    # set_global_seed! seeds the global RNG exactly like Random.seed!
    @test ucmc.set_global_seed!(42) == 42
    @test ucmc.global_seed() == 42
    a = rand(3)
    Random.seed!(42)
    @test rand(3) == a

    # sweep_Ts records the global seed and Julia version
    root = mktempdir()
    ucmc.set_global_seed!(42)
    sweep(root)
    lines = metadata(sweep_dir(root))
    @test "seed = 42" in lines
    @test "julia_version = $VERSION" in lines

    # ... and the run configuration needed to recreate the sweep
    @test "n_runs = 2" in lines
    @test "simulated_annealing = true" in lines
    @test "measurements = Symbol[]" in lines
    @test "time_series = none" in lines

    root2 = mktempdir()
    sweep(root2; simulated_annealing = false, measurements = [:raw_correlation])
    lines2 = metadata(sweep_dir(root2))
    @test "n_runs = 2" in lines2
    @test "simulated_annealing = false" in lines2
    @test "measurements = [:raw_correlation]" in lines2

    # direct save_sweep_results callers that omit the configuration get "unknown"
    root3 = mktempdir()
    dir3 = ucmc.save_sweep_results([], [], model, ucmc.MetropolisAlgorithm(), geometry,
                                   2, 4, 0, 2; root = root3)
    lines3 = metadata(dir3)
    @test "n_runs = unknown" in lines3
    @test "simulated_annealing = unknown" in lines3
    @test "measurements = unknown" in lines3
    @test "time_series = unknown" in lines3

    # the recorded seed recreates the saved results
    first_csv = read(joinpath(sweep_dir(root), "measurement_results.csv"), String)
    ucmc.set_global_seed!(42)
    sweep(root)
    @test read(joinpath(sweep_dir(root), "measurement_results.csv"), String) == first_csv

    # without an argument a random seed is drawn, returned, and recorded
    seed = ucmc.set_global_seed!()
    @test ucmc.global_seed() == seed
    sweep(root)
    @test "seed = $seed" in metadata(sweep_dir(root))

    # an explicit rng has no recoverable seed unless one is passed alongside it
    sweep(root; rng = Xoshiro(5))
    @test "seed = unknown" in metadata(sweep_dir(root))
    sweep(root; rng = Xoshiro(5), seed = 5)
    @test "seed = 5" in metadata(sweep_dir(root))

    # single-temperature saves record the seed too
    ucmc.set_global_seed!(7)
    state = ucmc.initialize_state(model, geometry)
    parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
    container = ucmc.MeasurementContainer(model, geometry, 4, 2)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    results = ucmc.analyze(container, 2.0, geometry.n_sites)
    single = ucmc.save_results(model, ucmc.MetropolisAlgorithm(), geometry, 2.0,
        results, parameters; root = root)
    @test "seed = 7" in metadata(single)
    single = ucmc.save_results(model, ucmc.MetropolisAlgorithm(), geometry, 2.0,
        results, parameters; root = root, seed = nothing)
    @test "seed = unknown" in metadata(single)
end

@testset "raw_correlation results are saved" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    rng = Xoshiro(5)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    parameters = ucmc.SimulationParameters(2.0, 5, 20, 1, 4)

    container = ucmc.MeasurementContainer(model, geometry, 20, 4;
                                          measurements = [:raw_correlation])
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state,
                        container, parameters; rng = rng)
    results = ucmc.analyze(container, 2.0, geometry.n_sites)

    # the raw correlation carries its displacement-class labels through
    # analyze, so correlation_rows can label its columns (bug: the data used
    # to be silently dropped on the save path)
    @test haskey(results.primary, :correlation)
    @test hasproperty(results.derived, :correlation_classes)
    @test !hasproperty(results.derived, :correlation_connected)

    classes = results.derived.correlation_classes
    df = ucmc.correlation_rows(results)
    @test df !== nothing
    @test propertynames(df) == [:disp_1, :disp_2, :orbital_i, :orbital_j,
                                :raw_value, :raw_error]
    @test size(df, 1) == length(classes)
    @test df.raw_value == first(results.primary[:correlation])
    @test df.orbital_i == [c.orbitals[1] for c in classes]
    @test df.disp_2 == [c.displacement[2] for c in classes]

    # and it is written to disk on the single-run save path
    root = mktempdir()
    dir = ucmc.save_results(model, ucmc.MetropolisAlgorithm(), geometry, 2.0,
                            results, parameters; root = root)
    @test isfile(joinpath(dir, "correlation_results.csv"))

    # no classes, no frame
    @test ucmc.correlation_rows((primary = Dict(), derived = NamedTuple())) === nothing

    # the connected measurement still adds its two columns
    container = ucmc.MeasurementContainer(model, geometry, 20, 4;
                                          measurements = [:connected_correlation])
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry,
                        ucmc.initialize_state(model, geometry; rng = rng),
                        container, parameters; rng = rng)
    df = ucmc.correlation_rows(ucmc.analyze(container, 2.0, geometry.n_sites))
    @test propertynames(df) == [:disp_1, :disp_2, :orbital_i, :orbital_j,
                                :raw_value, :raw_error,
                                :connected_value, :connected_error]

    # requesting both raw and connected emits the labels exactly once
    container = ucmc.MeasurementContainer(model, geometry, 20, 4;
        measurements = [:raw_correlation, :connected_correlation])
    @test count(d -> :correlation_classes in d.names, container.derived) == 1
end

@testset "magnetization_per_orbital results are saved" begin
    geometry = two_orbital_geometry(2)
    model = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(5)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    parameters = ucmc.SimulationParameters(2.0, 5, 20, 1, 4)

    container = ucmc.MeasurementContainer(model, geometry, 20, 4;
                                          measurements = [:magnetization_per_orbital])
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state,
                        container, parameters; rng = rng)
    results = ucmc.analyze(container, 2.0, geometry.n_sites)

    # the matrix-valued observable is analyzed but used to be dropped on the
    # save path; the rows unroll it with one row per (orbital, component)
    @test haskey(results.primary, :magnetization_per_orbital)
    df = ucmc.magnetization_per_orbital_rows(results)
    @test df !== nothing
    @test propertynames(df) == [:orbital, :component, :value, :error]
    avg, err = results.primary[:magnetization_per_orbital]
    n_comp, n_orb = size(avg)
    @test (n_comp, n_orb) == (1, 2)  # Ising on the two-orbital geometry
    @test size(df, 1) == n_comp * n_orb
    @test df.orbital == [1, 2]
    @test df.component == [1, 1]
    @test df.value == vec(permutedims(avg, (2, 1)))
    @test df.error == vec(permutedims(err, (2, 1)))

    # written to disk on the single-run path...
    root = mktempdir()
    dir = ucmc.save_results(model, ucmc.MetropolisAlgorithm(), geometry, 2.0,
                            results, parameters; root = root)
    path = joinpath(dir, "magnetization_per_orbital_results.csv")
    @test isfile(path)
    csv = read(path, String)
    @test occursin("orbital,component,value,error", csv)
    @test length(split(chomp(csv), "\n")) == 1 + n_comp * n_orb

    # ...and on the sweep path, including via :connected_correlation's
    # implicit inclusion of :magnetization_per_orbital
    sroot = mktempdir()
    ucmc.sweep_Ts(ucmc.MetropolisAlgorithm(), model, geometry,
                  ucmc.initialize_state(model, geometry), 2, 8, 0, 2, [2.0];
                  measurements = [:connected_correlation], time_series = :none, root = sroot)
    sdir = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry; root = sroot)
    @test isfile(joinpath(sdir, "magnetization_per_orbital_results.csv"))

    # no measurement, no frame
    @test ucmc.magnetization_per_orbital_rows((primary = Dict(),)) === nothing
end

@testset "autocorrelation results are saved" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    root = mktempdir()
    Ts = [3.0, 2.0]

    sweep_results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, state,
        10, 40, 0, 4, Ts;
        n_runs = 2, diagnostics = [:cluster_size], root = root
    )
    @test all(r -> haskey(r.autocorrelation, :energy), sweep_results)
    @test all(r -> r.autocorrelation[:energy].n == 80, sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry; root = root)
    path = joinpath(dir, "autocorrelation_results.csv")
    @test isfile(path)
    lines = readlines(path)
    @test lines[1] == "temperature,observable,tau,tau_err,window,n_samples,status"
    n_names = length(sweep_results[1].autocorrelation)
    @test length(lines) == 1 + length(Ts) * n_names

    rows = ucmc.autocorrelation_rows(sweep_results[1])
    @test rows.observable == sort(rows.observable)
    @test ucmc.autocorrelation_rows((primary = Dict(), derived = NamedTuple())) === nothing

    single = ucmc.save_results(model, ucmc.WolffAlgorithm(), geometry, 3.0,
        sweep_results[1], ucmc.SimulationParameters(3.0, 10, 40, 0, 4); root = root)
    @test isfile(joinpath(single, "autocorrelation_results.csv"))
end
