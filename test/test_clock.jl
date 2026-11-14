@testset "Clock model construction and state" begin
    @test_throws ArgumentError ucmc.ClockModel(1, [1.0], [0.0])
    @test_throws ArgumentError ucmc.ClockModel(0, [1.0], [0.0])

    model = ucmc.ClockModel(4, [1, 2], [0.5])
    @test model isa ucmc.ClockModel{Float64}
    @test model.q == 4
    # lookup tables are exact at the axes
    @test model.cosines == [1.0, 0.0, -1.0, 0.0]
    @test model.sines == [0.0, 1.0, 0.0, -1.0]
    @test ucmc.clock_cos(model, -1) == 0.0
    @test ucmc.clock_sin(model, 7) == -1.0
    @test repr(model) == "ClockModel{Float64}(4, [1.0, 2.0], [0.5])"

    geometry = two_orbital_geometry(2)
    model = ucmc.ClockModel(6, [1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
    @test state isa ucmc.ClockState
    @test length(state.spins) == geometry.n_sites
    @test clock_invariants_hold(model, geometry, state)

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.ClockModel(6, [1.0, 2.0], [0.5]), geometry
    )
    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.ClockModel(6, [1.0], [0.5, -0.25]), geometry
    )
end

@testset "Clock updates keep invariants" begin
    geometry = two_orbital_geometry(2)
    n_sites = geometry.n_sites
    # q = 2 is the Ising special case; q = 3 and q = 6 cover odd and even tables
    for q in (2, 3, 6), h in ([0.0, 0.0], [0.5, -0.25])
        model = ucmc.ClockModel(q, [1.0, 2.0], h)
        for algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
                          ucmc.HeatbathAlgorithm(), ucmc.WolffAlgorithm(),
                          ucmc.SwendsenWangAlgorithm())
            rng = Xoshiro(q)
            state = ucmc.initialize_state(model, geometry; rng = rng)
            invariants = true
            diagnostics_ok = true
            for _ in 1:30
                result = ucmc.step!(algorithm, model, geometry, state, 1.5; rng = rng)
                invariants &= clock_invariants_hold(model, geometry, state)
                values = ucmc.diagnostic_values(algorithm, result)
                diagnostics_ok &= collect(propertynames(values)) ==
                    ucmc.available_diagnostics(algorithm)
            end
            @test invariants
            @test diagnostics_ok
        end
    end

    # Metropolis always proposes a different clock state; for q = 2 it is a flip.
    rng = Xoshiro(3)
    for q in (2, 3, 5)
        model = ucmc.ClockModel(q, [1.0, 1.0], [0.0, 0.0])
        state = ucmc.initialize_state(model, geometry; rng = rng)
        proposals = [ucmc.propose(rng, ucmc.MetropolisAlgorithm(), model, geometry, state) for _ in 1:200]
        @test all(p -> p.new_value != state.spins[p.site] && p.new_value in 0:(q - 1), proposals)
        q == 2 && @test all(p -> p.new_value == 1 - state.spins[p.site], proposals)
        q > 2 && @test length(unique(p.new_value - state.spins[p.site] for p in proposals)) > 1
    end

    # cluster algorithms are ferromagnetic only; Metropolis accepts any sign
    antiferro = ucmc.ClockModel(3, [1.0, -1.0], [0.0, 0.0])
    state = ucmc.initialize_state(antiferro, geometry)
    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        @test_throws ArgumentError ucmc.step!(algorithm, antiferro, geometry, state, 1.0)
    end
    for _ in 1:20
        ucmc.step!(ucmc.MetropolisAlgorithm(), antiferro, geometry, state, 1.0)
        ucmc.step!(ucmc.GlauberAlgorithm(), antiferro, geometry, state, 1.0)
        ucmc.step!(ucmc.HeatbathAlgorithm(), antiferro, geometry, state, 1.0)
    end
    @test clock_invariants_hold(antiferro, geometry, state)
end

@testset "Clock cluster reflection" begin
    geometry = square_geometry(2)
    model = ucmc.ClockModel(4, [1.0, 1.0], [0.3])
    spins = [0, 1, 2, 3]
    make_state(spins) = ucmc.ClockState(
        copy(spins), clock_energy(model, geometry, spins), clock_mx(4, spins), clock_my(4, spins)
    )

    state = make_state(spins)
    ucmc.apply_cluster!(model, geometry, state, [1, 2], 1)   # n ↦ 1 - n
    @test state.spins == [1, 0, 2, 3]
    @test clock_invariants_hold(model, geometry, state)

    # Swendsen-Wang reflects each cluster as a whole, or not at all.
    reflect(n) = mod(2 - n, 4)
    for seed in 1:8
        state = make_state(spins)
        ucmc.apply_clusters!(Xoshiro(seed), model, geometry, state, [[1, 2], [3, 4]], 2)
        for cluster in ([1, 2], [3, 4])
            @test state.spins[cluster] == spins[cluster] ||
                state.spins[cluster] == reflect.(spins[cluster])
        end
        @test clock_invariants_hold(model, geometry, state)
    end

    # The mirror n ↦ m - n is the XY reflection θ ↦ 2φ + π - θ about the axis
    # φ = πm/q - π/2, and the context stores each state's component along φ.
    for q in (3, 4, 5), m in 0:(q - 1)
        context = ucmc.ClockClusterContext(ucmc.ClockModel(q, [1.0], [0.0]), m)
        φ = π * m / q - π / 2
        @test context.projections ≈ [cos(2π * n / q - φ) for n in 0:(q - 1)] atol = 1e-12
        @test context.ghost_projection ≈ cos(φ) atol = 1e-12
        @test all(cis(2π * mod(m - n, q) / q) ≈ cis(2φ + π - 2π * n / q) for n in 0:(q - 1))
    end
end

@testset "Clock q = 2 reduces to Ising" begin
    geometry = two_orbital_geometry(2)
    clock = ucmc.ClockModel(2, [1.0, 2.0], [0.5, -0.25])
    ising = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(4)
    for _ in 1:10
        spins = rand(rng, 0:1, geometry.n_sites)
        @test clock_energy(clock, geometry, spins) ≈ spin_energy(ising, geometry, 1 .- 2 .* spins)
    end

    # cluster moves use the spin-flip mirror with Ising projections ±1
    state = ucmc.initialize_state(clock, geometry; rng = rng)
    context = ucmc.cluster_context(rng, clock, geometry, state, 1.0)
    @test context.mirror == 1
    @test context.projections == [1.0, -1.0]

    @test ucmc.binder_denominator(clock) == 3.0
    @test ucmc.binder_denominator(ucmc.ClockModel(3, [1.0], [0.0])) == 2.0
end

@testset "Clock model matches exact enumeration" begin
    geometry = square_geometry(2)
    N = geometry.n_sites
    q, T = 3, 1.1
    model = ucmc.ClockModel(q, [1.0, 0.8], [0.35])

    Z = E = Mx = 0.0
    for c in 0:(q^N - 1)
        spins = digits(c; base = q, pad = N)
        e = clock_energy(model, geometry, spins)
        w = exp(-e / T)
        Z += w
        E += w * e
        Mx += w * clock_mx(q, spins)
    end
    exact_e, exact_mx = E / (Z * N), Mx / (Z * N)

    parameters = ucmc.SimulationParameters(T, 200, 20_000, 0, 20)
    for algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
                      ucmc.HeatbathAlgorithm(), ucmc.WolffAlgorithm(),
                      ucmc.SwendsenWangAlgorithm())
        rng = Xoshiro(2024)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, 20_000, 20)
        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, T, N)
        e, e_err = result.primary[:energy]
        mx, mx_err = result.primary[:magnetization_x]
        @test abs(e - exact_e) < 4 * e_err
        @test abs(mx - exact_mx) < 4 * mx_err
    end
end

@testset "Clock measurements" begin
    geometry = square_geometry(3)
    N = geometry.n_sites
    model = ucmc.ClockModel(5, [1.0, 1.0], [0.2])

    parameters = ucmc.SimulationParameters(1.5, 2, 8, 0, 2)
    # the measurement pipeline is algorithm-agnostic: one local algorithm
    # (scalar diagnostic) and both cluster flavors (the two diagnostic shapes)
    # suffice; per-algorithm diagnostics are checked in "Clock updates keep
    # invariants"
    for algorithm in (ucmc.MetropolisAlgorithm(), ucmc.WolffAlgorithm(),
                      ucmc.SwendsenWangAlgorithm())
        rng = Xoshiro(6)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, 8, 2;
            measurements = ucmc.available_measurements(model),
            diagnostics = ucmc.available_diagnostics(algorithm))

        connected = only(filter(d -> :correlation_connected in d.names, container.derived))
        @test connected.depends_on == (:correlation, :magnetization_per_orbital)

        values = Dict(o.name => o.value(state) for o in container.observables)
        @test values[:magnetization] ≈ hypot(state.magnetization_x, state.magnetization_y) / N
        @test values[:abs_magnetization] ≈ values[:magnetization]
        @test values[:magnetization_fourth] ≈ values[:magnetization]^4
        @test values[:correlation][1] ≈ 1.0   # on-site class: cos(0)

        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, 1.5, N)
        for name in (:energy, :magnetization_x, :magnetization_y, :magnetization,
                     :abs_magnetization, :magnetization_fourth, :correlation,
                     :structure_factor, :clock_order_parameter,
                     ucmc.available_diagnostics(algorithm)...)
            @test haskey(result.primary, name)
        end
        for name in (:specific_heat, :susceptibility, :susceptibility_x, :susceptibility_y,
                     :binder_cumulant, :correlation_connected, :structure_factor_classes)
            @test haskey(result.derived, name)
        end
        jk_x = ucmc.jackknife_samples(container.data.magnetization_x)
        jk_y = ucmc.jackknife_samples(container.data.magnetization_y)
        n = length(jk_x)
        mx_full = sum(container.data.magnetization_x) / n
        my_full = sum(container.data.magnetization_y) / n
        f_full = 1.0 - mx_full^2 - my_full^2
        v̄ = sum(1.0 - mx^2 - my^2 for (mx, my) in zip(jk_x, jk_y)) / n
        expected = n * f_full - (n - 1) * v̄
        @test result.derived.correlation_connected[1] ≈ expected
    end
end

@testset "Clock sweep_Ts, autotune, and saved results" begin
    geometry = square_geometry(2)
    model = ucmc.ClockModel(4, [1.0, 1.0], [0.0])
    root = mktempdir()
    Ts = [2.0, 1.0]

    results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        3, 6, 1, 3, Ts;
        n_runs = 2,
        measurements = [:connected_correlation, :binder_cumulant, :structure_factor],
        diagnostics = [:cluster_size],
        root = root,
    )
    @test length(results) == length(Ts)
    @test all(r -> haskey(r.primary, :cluster_size), results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry; root = root)
    @test occursin("ClockModel_WolffAlgorithm", dir)
    for file in ("measurement_results.csv", "correlation_results.csv",
                 "structure_factor_results.csv")
        @test isfile(joinpath(dir, file))
    end
    @test "model = ClockModel{Float64}(4, [1.0, 1.0], [0.0])" in
        readlines(joinpath(dir, "metadata.txt"))

    tuning = ucmc.autotune_binning(
        ucmc.SwendsenWangAlgorithm(), model, geometry,
        ucmc.initialize_state(model, geometry), Ts; n_pilot = 200, rng = Xoshiro(2)
    )
    @test tuning.n_unmeasured >= 0
    @test tuning.bin_size >= 1
end

@testset "Clock order parameter measurement" begin
    @test :clock_order_parameter in ucmc.available_measurements(ucmc.ClockModel)
    @test :clock_order_parameter in ucmc.available_measurements(ucmc.ClockModel(6, [1.0], [0.0]))
    @test :clock_order_parameter ∉ ucmc.available_measurements(ucmc.IsingModel)
    @test :clock_order_parameter ∉ ucmc.available_measurements(ucmc.XYModel)
    @test :clock_order_parameter ∉ ucmc.available_measurements(ucmc.HeisenbergModel)

    geometry = square_geometry(2)
    N = geometry.n_sites
    for model in (ucmc.IsingModel([1.0, 1.0], [0.0]), ucmc.XYModel([1.0, 1.0], [0.0]),
                  ucmc.HeisenbergModel([1.0, 1.0], [0.0]))
        @test_throws ArgumentError ucmc.MeasurementContainer(model, geometry, 4, 2;
            measurements = [:clock_order_parameter])
    end

    # cos(q φ_M) for hand-built configurations
    model = ucmc.ClockModel(4, [1.0, 1.0], [0.0])
    make_state(spins) = ucmc.ClockState(
        spins, clock_energy(model, geometry, spins), clock_mx(4, spins), clock_my(4, spins)
    )
    order(spins) = ucmc.clock_angle_order(model, make_state(spins))
    @test order([0, 0, 0, 0]) ≈ 1.0    # locked on a clock axis
    @test order([1, 1, 1, 1]) ≈ 1.0
    @test order([0, 1, 0, 1]) ≈ -1.0   # halfway between axes, φ_M = π/4
    @test order([0, 1, 2, 3]) == 0.0   # M = 0: direction undefined
    @test order([0, 2, 1, 3]) == 0.0
    # rounding residue in the running totals does not leak into the angle
    @test ucmc.clock_angle_order(model, ucmc.ClockState([0, 1, 2, 3], 0.0, 1e-13, -2e-13)) == 0.0
    # odd q: M = 0 only when all states are equally occupied
    q5 = ucmc.ClockModel(5, [1.0], [0.0])
    @test ucmc.clock_angle_order(q5, ucmc.ClockState(collect(0:4), 0.0, 0.0, 0.0)) == 0.0
    # q = 2: the magnetization always lies on the clock axis
    q2 = ucmc.ClockModel(2, [1.0], [0.0])
    @test ucmc.clock_angle_order(q2, ucmc.ClockState([0, 1, 1, 1], 0.0, -2.0, 0.0)) ≈ 1.0

    # the measurement adds one scalar observable and no derived quantity
    model = ucmc.ClockModel(6, [1.0, 1.0], [0.0])
    plain = ucmc.MeasurementContainer(model, geometry, 8, 2)
    container = ucmc.MeasurementContainer(model, geometry, 8, 2;
        measurements = [:clock_order_parameter])
    @test setdiff(keys(container.data), keys(plain.data)) == [:clock_order_parameter]
    @test length(container.derived) == length(plain.derived)
    @test haskey(container.time_series, :clock_order_parameter)

    rng = Xoshiro(12)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    samples = Float64[]
    for _ in 1:8
        ucmc.step!(ucmc.WolffAlgorithm(), model, geometry, state, 1.0; rng = rng)
        mx, my = clock_mx(6, state.spins), clock_my(6, state.spins)
        push!(samples, hypot(mx, my) < 1e-8 ? 0.0 : cos(6 * atan(my, mx)))
        ucmc.measure!(container, state)
    end
    @test container.data.clock_order_parameter ≈ [sum(samples[1:4]) / 4, sum(samples[5:8]) / 4]
    result = ucmc.analyze(container, 1.0, N)
    @test result.primary[:clock_order_parameter][1] ≈ sum(samples) / 8

    # deep in the ordered phase the magnetization is locked to a clock axis
    ordered = ucmc.ClockModel(6, [1.0, 1.0], [0.0])
    square = square_geometry(4)
    rng = Xoshiro(5)
    state = ucmc.initialize_state(ordered, square; rng = rng)
    container = ucmc.MeasurementContainer(ordered, square, 200, 4;
        measurements = [:clock_order_parameter])
    ucmc.run_simulation(ucmc.WolffAlgorithm(), ordered, square, state, container,
        ucmc.SimulationParameters(0.2, 200, 200, 0, 4); rng = rng)
    @test ucmc.analyze(container, 0.2, square.n_sites).primary[:clock_order_parameter][1] > 0.99

    # saved alongside the other scalar results
    root = mktempdir()
    ucmc.sweep_Ts(ucmc.SwendsenWangAlgorithm(), model, geometry,
        ucmc.initialize_state(model, geometry), 2, 4, 0, 2, [1.0];
        measurements = [:clock_order_parameter], root = root)
    dir = ucmc.sweep_dirname(model, ucmc.SwendsenWangAlgorithm(), geometry; root = root)
    saved = [split(line, ",")[2] for line in readlines(joinpath(dir, "measurement_results.csv"))[2:end]]
    @test "clock_order_parameter" in saved
end
