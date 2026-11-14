@testset "swap_pass!" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    s1 = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
    s2 = ucmc.initialize_state(model, geometry; rng = Xoshiro(2))
    s3 = ucmc.initialize_state(model, geometry; rng = Xoshiro(3))
    states = [s1, s2, s3]
    # fabricated energies: pass 1 must swap (1, 2) (x = (2 - 1) * (1) = 1 > 0,
    # always accepted), pass 2 must reject (2, 3) (x = (1 - 2) * (100) = -100)
    s1.energy = -1.0
    s2.energy = -2.0
    s3.energy = -101.0
    betas = [2.0, 1.0, 2.0]
    stats = ucmc.SwapStats(2)

    ucmc.swap_pass!(states, betas, stats, 1, Xoshiro(1))
    @test states[1] === s2 && states[2] === s1 && states[3] === s3
    @test states[1].energy == -2.0  # the cached energy travels with the configuration
    @test stats.attempts == [1, 0] && stats.accepts == [1, 0]

    ucmc.swap_pass!(states, betas, stats, 2, Xoshiro(2))
    @test states[1] === s2 && states[2] === s1 && states[3] === s3  # rejected: unchanged
    @test stats.attempts == [1, 1] && stats.accepts == [1, 0]
end

@testset "ParallelTemperingResult" begin
    res = ucmc.ParallelTemperingResult([:a, :b], [1.0, 2.0], ucmc.SwapStats(1))
    @test isnan(res.swap_acceptance[1])  # never attempted
    @test occursin("2 temperatures", sprint(show, res))
    @test_throws ArgumentError ucmc.ParallelTemperingResult([:a], [1.0, 2.0], ucmc.SwapStats(1))
    @test_throws ArgumentError ucmc.ParallelTemperingResult([:a, :b], [1.0, 2.0], ucmc.SwapStats(2))

    stats = ucmc.SwapStats(2)
    stats.attempts .= [4, 2]
    stats.accepts .= [1, 2]
    res = ucmc.ParallelTemperingResult([:a, :b, :c], [1.0, 2.0, 3.0], stats)
    @test res.swap_acceptance == [0.25, 1.0]
end

@testset "geometric_ladder" begin
    Ts = ucmc.geometric_ladder(1.0, 4.0, 3)
    @test Ts ≈ [1.0, 2.0, 4.0]
    @test issorted(Ts)
    @test_throws ArgumentError ucmc.geometric_ladder(0.0, 1.0, 4)
    @test_throws ArgumentError ucmc.geometric_ladder(2.0, 1.0, 4)
    @test_throws ArgumentError ucmc.geometric_ladder(1.0, 2.0, 1)
end

@testset "tempering_sweep validation" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    alg = ucmc.MetropolisAlgorithm()

    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [2.0], 1, 4, 0, 2; save = false)
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [2.0, 1.0], 1, 4, 0, 2; save = false)
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [1.0, 0.0], 1, 4, 0, 2; save = false)
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [1.0, 2.0], 1, 4, 0, 2; swap_every = 0, save = false)
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [1.0, 2.0], 1, 4, 0, 3; save = false)

    state = ucmc.initialize_state(model, geometry)
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [1.0, 2.0], 1, 4, 0, 2;
        states = [state], save = false)
    bad = ucmc.initialize_state(ucmc.XYModel([1.0, 1.0], [0.0]), geometry)
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [1.0, 2.0], 1, 4, 0, 2;
        states = [state, bad], save = false)

    # diagnostics the algorithm does not provide are rejected up front
    @test_throws ArgumentError ucmc.tempering_sweep(alg, model, geometry, [1.0, 2.0], 1, 4, 0, 2;
        diagnostics = [:cluster_size], save = false)
end

@testset "tempering_sweep end-to-end" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    Ts = [1.0, 2.0, 3.0]
    root = mktempdir()

    result = ucmc.tempering_sweep(ucmc.MetropolisAlgorithm(), model, geometry, Ts,
        5, 8, 1, 4; diagnostics = [:acceptance_ratio], rng = Xoshiro(11), root = root)

    @test result isa ucmc.ParallelTemperingResult
    @test length(result.results) == 3
    @test result.temperatures == Ts
    @test all(r -> haskey(r.primary, :energy), result.results)
    @test all(r -> haskey(r.primary, :acceptance_ratio), result.results)
    @test all(result.swap_accepts .<= result.swap_attempts)
    @test all(a -> 0.0 <= a <= 1.0, result.swap_acceptance)
    # 5 + 8 * (1 + 1) = 21 rounds, one pair per pass alternating: [11, 10]
    @test sum(result.swap_attempts) == 21
    @test result.swap_attempts == [11, 10]

    dir = ucmc.tempering_dirname(model, ucmc.MetropolisAlgorithm(), geometry; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))
    @test isfile(joinpath(dir, "swap_acceptance.csv"))
    @test isfile(joinpath(dir, "metadata.txt"))
    @test any(line -> startswith(line, "swap_every = 1"), readlines(joinpath(dir, "metadata.txt")))

    # swap_every = 3: passes at rounds 3, 6, ..., 21, so 7 passes over 21 rounds
    result3 = ucmc.tempering_sweep(ucmc.MetropolisAlgorithm(), model, geometry, Ts,
        5, 8, 1, 4; swap_every = 3, rng = Xoshiro(11), save = false)
    @test sum(result3.swap_attempts) == 7
    @test result3.swap_attempts == [4, 3]

    # the cluster algorithms work as the inner update too
    wolff = ucmc.tempering_sweep(ucmc.WolffAlgorithm(), model, geometry, [1.0, 3.0],
        2, 4, 0, 2; rng = Xoshiro(3), save = false)
    @test length(wolff.results) == 2
    @test all(r -> haskey(r.primary, :energy), wolff.results)

    # provided starting states are accepted
    starts = [ucmc.initialize_state(model, geometry; rng = Xoshiro(100 + r)) for r in 1:2]
    seeded = ucmc.tempering_sweep(ucmc.MetropolisAlgorithm(), model, geometry, [1.0, 3.0],
        2, 4, 0, 2; states = starts, rng = Xoshiro(3), save = false)
    @test length(seeded.results) == 2
end

@testset "tempering_sweep is reproducible for a fixed rng" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    run() = ucmc.tempering_sweep(ucmc.MetropolisAlgorithm(), model, geometry, [1.0, 2.0],
        3, 6, 1, 3; diagnostics = [:acceptance_ratio], save = false, rng = Xoshiro(99))
    a, b = run(), run()
    @test [r.primary[:energy] for r in a.results] == [r.primary[:energy] for r in b.results]
    @test [r.primary[:acceptance_ratio] for r in a.results] == [r.primary[:acceptance_ratio] for r in b.results]
    @test a.swap_accepts == b.swap_accepts
end

@testset "tempering_sweep matches exact results" begin
    # 2x2 Ising with a field: exact per-site energy by enumeration over the 16
    # configurations, at each temperature of the ladder
    geometry = square_geometry(2)
    N = geometry.n_sites
    model = ucmc.IsingModel([1.0, 0.8], [0.35])
    Ts = [1.5, 3.0]

    exact = map(Ts) do T
        Z = E = 0.0
        for c in 0:(2^N - 1)
            spins = 1 .- 2 .* digits(c; base = 2, pad = N)
            e = spin_energy(model, geometry, spins)
            w = exp(-e / T)
            Z += w
            E += w * e
        end
        E / (Z * N)
    end

    result = ucmc.tempering_sweep(ucmc.MetropolisAlgorithm(), model, geometry, Ts,
        200, 20_000, 0, 20; save = false, rng = Xoshiro(2024))
    for (r, exact_e) in zip(result.results, exact)
        e, e_err = r.primary[:energy]
        @test abs(e - exact_e) < 4 * e_err
    end
end

@testset "reweight" begin
    # Reweighting to the simulated temperature reproduces the direct analysis
    geometry = square_geometry(3)
    N = geometry.n_sites
    T = 2.5
    model = ucmc.IsingModel([1.0, 1.0], [0.0])

    rng = Xoshiro(99)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    parameters = ucmc.SimulationParameters(T, 1_000, 20_000, 0, 40)
    container = ucmc.MeasurementContainer(model, geometry, 20_000, 40)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters; rng = rng)
    analysis = ucmc.analyze(container, T, N)

    rw = ucmc.reweight(container, T, [T], N)
    @test rw isa ucmc.ReweightingResult
    @test rw.T_sim == T
    @test rw.temperatures == [T]
    @test rw.n_sites == N
    @test rw.energy[1] ≈ analysis.primary[:energy][1]
    @test rw.energy_err[1] ≈ analysis.primary[:energy][2]
    @test rw.specific_heat[1] ≈ analysis.derived.specific_heat
    @test rw.specific_heat_err[1] ≈ analysis.derived.specific_heat_err
    @test rw.magnetization[1] ≈ analysis.primary[:magnetization][1]
    @test rw.magnetization_err[1] ≈ analysis.primary[:magnetization][2]
    @test rw.susceptibility[1] ≈ analysis.derived.susceptibility
    @test rw.susceptibility_err[1] ≈ analysis.derived.susceptibility_err
    @test rw.n_eff[1] == 20_000  # uniform weights: every sample counts

    # magnetization = nothing skips the magnetic quantities
    rw_e = ucmc.reweight(container, T, [2.4, T, 2.6], N; magnetization = nothing)
    @test rw_e.magnetization === nothing
    @test rw_e.magnetization_err === nothing
    @test rw_e.susceptibility === nothing
    @test rw_e.susceptibility_err === nothing
    @test length(rw_e.energy) == 3
    @test rw_e.n_eff[2] == 20_000
    @test 1 < rw_e.n_eff[1] < 20_000

    # input validation
    @test_throws ArgumentError ucmc.reweight(container, 0.0, [T], N)
    @test_throws ArgumentError ucmc.reweight(container, T, Float64[], N)
    @test_throws ArgumentError ucmc.reweight(container, T, [2.0, -1.0], N)
    @test_throws ArgumentError ucmc.reweight(container, T, [T], 0)
    @test_throws ArgumentError ucmc.reweight(container, T, [T], N; magnetization = :bogus)

    empty_series = ucmc.MeasurementContainer(model, geometry, 4, 2; time_series = :none)
    @test_throws ArgumentError ucmc.reweight(empty_series, T, [T], N)

    partial = ucmc.MeasurementContainer(model, geometry, 4, 2)
    ucmc.measure!(partial, ucmc.initialize_state(model, geometry))
    @test_throws ArgumentError ucmc.reweight(partial, T, [T], N)
end

@testset "reweight matches exact enumeration" begin
    # 2x2 Ising at h = 0: exact moments by enumeration over the 16
    # configurations, compared with reweighting from one simulation at T_sim
    geometry = square_geometry(2)
    N = geometry.n_sites
    T_sim = 2.0
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    Ts = [1.6, 2.0, 2.4]

    exact = map(Ts) do T
        Z = E = E2 = M2 = 0.0
        for c in 0:(2^N - 1)
            spins = 1 .- 2 .* digits(c; base = 2, pad = N)
            en = spin_energy(model, geometry, spins)
            w = exp(-en / T)
            Z += w
            E += w * en
            E2 += w * en^2
            M2 += w * sum(spins)^2
        end
        e = E / (Z * N)
        c_v = (E2 / Z - (E / Z)^2) / (N * T^2)
        χ = M2 / (Z * N * T)  # ⟨m⟩ = 0 at h = 0
        (e, c_v, χ)
    end

    rng = Xoshiro(2024)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    parameters = ucmc.SimulationParameters(T_sim, 1_000, 100_000, 0, 50)
    container = ucmc.MeasurementContainer(model, geometry, 100_000, 50)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters; rng = rng)

    rw = ucmc.reweight(container, T_sim, Ts, N)
    for (i, T) in enumerate(Ts)
        @test abs(rw.energy[i] - exact[i][1]) < 5 * rw.energy_err[i]
        @test abs(rw.specific_heat[i] - exact[i][2]) < 5 * rw.specific_heat_err[i]
        @test abs(rw.susceptibility[i] - exact[i][3]) < 5 * rw.susceptibility_err[i]
    end
    @test all(rw.n_eff .> 10_000)  # well inside the histogram-overlap window
end
