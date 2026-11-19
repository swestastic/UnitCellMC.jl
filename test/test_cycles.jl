@testset "AlgorithmCycle construction" begin
    wolff = ucmc.WolffAlgorithm()
    metropolis = ucmc.MetropolisAlgorithm()

    cycle = ucmc.AlgorithmCycle(wolff, metropolis)
    @test cycle isa ucmc.AbstractAlgorithm
    @test cycle.algorithms == (wolff, metropolis)
    @test cycle.counts == (1, 1)

    # `algorithm => n_sweeps` entries, and container forms
    counted = ucmc.AlgorithmCycle(ucmc.HeatbathAlgorithm(), ucmc.OverrelaxationAlgorithm() => 3)
    @test counted.counts == (1, 3)
    @test ucmc.AlgorithmCycle((wolff, metropolis)).algorithms == (wolff, metropolis)
    @test ucmc.AlgorithmCycle([wolff, metropolis]).algorithms == (wolff, metropolis)

    @test_throws ArgumentError ucmc.AlgorithmCycle()
    @test_throws ArgumentError ucmc.AlgorithmCycle(ucmc.MetropolisAlgorithm() => 0)
    @test_throws ArgumentError ucmc.AlgorithmCycle(:not_an_algorithm)
    # Two entries of the same algorithm type share a diagnostic prefix
    @test_throws ArgumentError ucmc.AlgorithmCycle(ucmc.MetropolisAlgorithm(), ucmc.MetropolisAlgorithm())
    @test_throws ArgumentError ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.WolffAlgorithm())
    # ...but distinct types with the same bare diagnostic name are fine
    @test ucmc.AlgorithmCycle(ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm()) isa ucmc.AlgorithmCycle

    @test string(counted) ==
        "AlgorithmCycle($(counted.algorithms[1]), $(counted.algorithms[2]) => 3)"
end

@testset "AlgorithmCycle diagnostics" begin
    @test ucmc.diagnostic_prefix(ucmc.MetropolisAlgorithm()) === :metropolis
    @test ucmc.diagnostic_prefix(ucmc.SwendsenWangAlgorithm()) === :swendsen_wang

    cycle = ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.MetropolisAlgorithm())
    @test ucmc.available_diagnostics(cycle) == [:wolff_cluster_size, :metropolis_acceptance_ratio]

    # Overrelaxation has no diagnostics, so it contributes no names
    relaxed = ucmc.AlgorithmCycle(ucmc.HeatbathAlgorithm(), ucmc.OverrelaxationAlgorithm() => 2)
    @test ucmc.available_diagnostics(relaxed) == [:heatbath_change_ratio]
end

@testset "AlgorithmCycle steps_per_sweep and workspaces" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    cycle = ucmc.AlgorithmCycle(ucmc.WolffAlgorithm() => 2, ucmc.MetropolisAlgorithm())
    @test ucmc.steps_per_sweep(cycle, geometry) == 2 * 1 + geometry.n_sites

    # shared/per-run workspaces are per-entry tuples
    shared = ucmc.shared_workspace(cycle, model, geometry)
    @test shared isa Tuple && length(shared) == 2
    workspace = ucmc.algorithm_workspace(cycle, model, geometry, state, shared)
    @test workspace isa Tuple && length(workspace) == 2
    @test workspace[1] isa ucmc.ClusterWorkspace
    @test workspace[2] isa ucmc.LocalUpdateWorkspace

    # a colored entry shares its visiting order through the shared tuple
    colored = ucmc.AlgorithmCycle(ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal()), ucmc.WolffAlgorithm())
    shared_colored = ucmc.shared_workspace(colored, model, geometry)
    @test shared_colored[1] isa Vector{Int}
    @test shared_colored[2] === nothing
    w1 = ucmc.algorithm_workspace(colored, model, geometry, state, shared_colored)
    @test w1[1].proposal.order === shared_colored[1]
end

@testset "AlgorithmCycle MC_Sweep! update counts" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    n_sites = geometry.n_sites

    cycle = ucmc.AlgorithmCycle(ucmc.WolffAlgorithm() => 2, ucmc.MetropolisAlgorithm())
    container = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:wolff_cluster_size, :metropolis_acceptance_ratio])
    stats = (ucmc.UpdateStatistic(:wolff_cluster_size), ucmc.UpdateStatistic(:metropolis_acceptance_ratio))
    workspace = ucmc.algorithm_workspace(cycle, model, geometry, state)
    ucmc.MC_Sweep!(cycle, model, geometry, state, container, 2.0, stats; workspace = workspace)
    @test stats[1].count == 2
    @test 1 <= ucmc.value(stats[1]) <= n_sites
    @test stats[2].count == n_sites
    @test 0.0 <= ucmc.value(stats[2]) <= 1.0

    # diagnostics may be requested for a subset of the entries
    partial = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:metropolis_acceptance_ratio])
    stats_partial = (ucmc.UpdateStatistic(:metropolis_acceptance_ratio),)
    ucmc.MC_Sweep!(cycle, model, geometry, state, partial, 2.0, stats_partial; workspace = workspace)
    @test stats_partial[1].count == n_sites

    # without a workspace every entry allocates its own per step
    stats_noworkspace = (ucmc.UpdateStatistic(:wolff_cluster_size),
                         ucmc.UpdateStatistic(:metropolis_acceptance_ratio))
    ucmc.MC_Sweep!(cycle, model, geometry, state, container, 2.0, stats_noworkspace)
    @test stats_noworkspace[1].count == 2
    @test stats_noworkspace[2].count == n_sites
    # a stateful proposal scheme keeps its cursor inside the cycle's
    # workspace: every sweep visits each site exactly once, then wraps
    sequential = ucmc.AlgorithmCycle(ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()))
    seq_workspace = ucmc.algorithm_workspace(sequential, model, geometry, state)
    seq_stats = (ucmc.UpdateStatistic(:metropolis_acceptance_ratio),)
    ucmc.MC_Sweep!(sequential, model, geometry, state, partial, 2.0, seq_stats; workspace = seq_workspace)
    ucmc.MC_Sweep!(sequential, model, geometry, state, partial, 2.0, seq_stats; workspace = seq_workspace)
    @test seq_stats[1].count == n_sites  # reset at the start of each sweep
    @test seq_workspace[1].proposal.next == 1  # wrapped after visiting every site

    @test state.energy ≈ spin_energy(model, geometry, state.spins)
    @test state.magnetization == sum(state.spins)
end

@testset "AlgorithmCycle of one matches the bare algorithm" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.5])
    parameters = ucmc.SimulationParameters(2.0, 5, 10, 1, 2)

    bare_state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
    bare_container = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:acceptance_ratio])
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, bare_state,
                        bare_container, parameters; rng = Xoshiro(3))

    cycle_state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
    cycle_container = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:metropolis_acceptance_ratio])
    ucmc.run_simulation(ucmc.AlgorithmCycle(ucmc.MetropolisAlgorithm()), model, geometry,
                        cycle_state, cycle_container, parameters; rng = Xoshiro(3))

    @test cycle_state.spins == bare_state.spins
    @test cycle_state.energy == bare_state.energy
    @test cycle_container.data[:metropolis_acceptance_ratio] == bare_container.data[:acceptance_ratio]
end

@testset "run_simulation with an AlgorithmCycle" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(5))
    cycle = ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.MetropolisAlgorithm())

    container = ucmc.MeasurementContainer(model, geometry, 8, 4;
        diagnostics = [:wolff_cluster_size, :metropolis_acceptance_ratio])
    parameters = ucmc.SimulationParameters(2.0, 3, 8, 1, 4)
    ucmc.run_simulation(cycle, model, geometry, state, container, parameters; rng = Xoshiro(7))

    @test length(container.data[:energy]) == 4
    @test length(container.data[:wolff_cluster_size]) == 4
    @test length(container.data[:metropolis_acceptance_ratio]) == 4
    @test all(1 .<= container.data[:wolff_cluster_size] .<= geometry.n_sites)
    @test all(0 .<= container.data[:metropolis_acceptance_ratio] .<= 1)
    @test state.energy ≈ spin_energy(model, geometry, state.spins)
    @test state.magnetization == sum(state.spins)

    # an unprefixed name is rejected with the available prefixed ones listed
    bad = ucmc.MeasurementContainer(model, geometry, 8, 4; diagnostics = [:acceptance_ratio])
    @test_throws ArgumentError ucmc.run_simulation(
        cycle, model, geometry, state, bad, parameters)

    # the classic XY mix: heatbath thermalizes, overrelaxation decorrelates
    xy = ucmc.XYModel([1.0, 1.0], [0.0])
    xy_state = ucmc.initialize_state(xy, geometry; rng = Xoshiro(9))
    relaxed = ucmc.AlgorithmCycle(ucmc.HeatbathAlgorithm(), ucmc.OverrelaxationAlgorithm() => 2)
    xy_container = ucmc.MeasurementContainer(xy, geometry, 8, 4;
        diagnostics = [:heatbath_change_ratio])
    ucmc.run_simulation(relaxed, xy, geometry, xy_state, xy_container, parameters; rng = Xoshiro(11))
    @test length(xy_container.data[:heatbath_change_ratio]) == 4
    @test xy_state.energy ≈ xy_energy(xy, geometry, xy_state)
end

@testset "sweep_Ts with an AlgorithmCycle" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    cycle = ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.MetropolisAlgorithm())
    Ts = [3.0, 2.0]
    diagnostics = [:wolff_cluster_size, :metropolis_acceptance_ratio]

    root = mktempdir()
    ucmc.set_global_seed!(42)
    results = ucmc.sweep_Ts(cycle, model, geometry,
        ucmc.initialize_state(model, geometry), 2, 4, 0, 2, Ts;
        n_runs = 2, diagnostics = diagnostics, time_series = :none, root = root)
    @test length(results) == 2
    @test haskey(results[1].primary, :wolff_cluster_size)
    @test haskey(results[1].primary, :metropolis_acceptance_ratio)

    # same seed reproduces the sweep exactly, thread count notwithstanding
    ucmc.set_global_seed!(42)
    repeat = ucmc.sweep_Ts(cycle, model, geometry,
        ucmc.initialize_state(model, geometry), 2, 4, 0, 2, Ts;
        n_runs = 2, diagnostics = diagnostics, time_series = :none,
        root = mktempdir(), save = false)
    @test [r.primary for r in repeat] == [r.primary for r in results]

    # the results directory is labeled with the composed algorithm
    dir = ucmc.sweep_dirname(model, cycle, geometry; root = root)
    @test isdir(dir)
    @test occursin("WolffAlgorithm+MetropolisAlgorithm", dir)
    lines = readlines(joinpath(dir, "metadata.txt"))
    # names print module-qualified or not depending on the session's imports
    scrubbed = replace.(lines, "UnitCellMC." => "")
    @test "algorithm = AlgorithmCycle(WolffAlgorithm(), MetropolisAlgorithm(RandomProposal()))" in scrubbed
    @test "diagnostics = $diagnostics" in lines

    # a fresh state per temperature (simulated_annealing = false) works too
    fresh = ucmc.sweep_Ts(ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.MetropolisAlgorithm() => 2),
        model, geometry, ucmc.initialize_state(model, geometry), 2, 4, 0, 2, Ts;
        simulated_annealing = false, time_series = :none, root = mktempdir(), save = false)
    @test length(fresh) == 2
end

@testset "algorithm_label for cycles" begin
    @test ucmc.algorithm_label(ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.MetropolisAlgorithm())) ==
        "WolffAlgorithm+MetropolisAlgorithm"
    @test ucmc.algorithm_label(ucmc.AlgorithmCycle(
        ucmc.HeatbathAlgorithm(), ucmc.OverrelaxationAlgorithm() => 3)) ==
        "HeatbathAlgorithm+3xOverrelaxationAlgorithm"
    @test ucmc.algorithm_label(ucmc.AlgorithmCycle(
        ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()), ucmc.WolffAlgorithm())) ==
        "MetropolisAlgorithm-SequentialProposal+WolffAlgorithm"
end

@testset "tempering_sweep and autotune with an AlgorithmCycle" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    cycle = ucmc.AlgorithmCycle(ucmc.WolffAlgorithm(), ucmc.MetropolisAlgorithm())

    result = ucmc.tempering_sweep(cycle, model, geometry, [2.0, 3.0], 2, 4, 0, 2;
        diagnostics = [:metropolis_acceptance_ratio], time_series = :none, save = false)
    @test result isa ucmc.ParallelTemperingResult
    @test length(result.results) == 2
    @test haskey(result.results[1].primary, :metropolis_acceptance_ratio)

    state = ucmc.initialize_state(model, geometry)
    tuning = ucmc.autotune_binning(cycle, model, geometry, state, [2.0];
        n_thermalization = 5, n_pilot = 40)
    @test tuning.bin_size >= 1
    @test tuning.n_unmeasured >= 0
end
