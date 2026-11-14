@testset "Metropolis updates keep invariants" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    for _ in 1:50
        result = ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        @test result.accepted isa Bool
        @test length(state.spins) == geometry.n_sites
        @test all(s in (-1, 1) for s in state.spins)
        @test state.magnetization == sum(state.spins)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end
end

@testset "AcceptanceCache" begin
    cache = ucmc.AcceptanceCache()
    T = 2.0

    # memoized Boltzmann factors match a direct evaluation bit for bit
    for ΔE in (0.5, 4.0, 8.0, -2.0, 0.5, 4.0)
        @test ucmc.cached_boltzmann_factor(cache, ΔE, T) === exp(-ΔE / T)
    end
    @test length(cache.delta_Es) == 4  # the repeats were cache hits

    # changing the temperature clears the memo table
    @test ucmc.cached_boltzmann_factor(cache, 4.0, 3.0) === exp(-4.0 / 3.0)
    @test cache.T == 3.0
    @test length(cache.delta_Es) == 1

    # a spectrum wider than the capacity stops memoizing new values but keeps
    # serving the entries already present
    wide = ucmc.AcceptanceCache()
    for k in 1:(ucmc.ACCEPTANCE_CACHE_CAPACITY + 10)
        @test ucmc.cached_boltzmann_factor(wide, Float64(k), 1.0) === exp(-Float64(k))
    end
    @test length(wide.delta_Es) == ucmc.ACCEPTANCE_CACHE_CAPACITY
    @test ucmc.cached_boltzmann_factor(wide, 1.0, 1.0) === exp(-1.0)
    @test ucmc.cached_boltzmann_factor(wide, 999.0, 1.0) === exp(-999.0)

    # only spectra that are discrete and small enough to scan are tabulated:
    # Ising's handful of values qualifies; the clock spectrum grows with q
    # times the coordination number, so memoizing it does not pay off
    @test ucmc.tabulate_acceptance(ucmc.IsingModel([1.0, 1.0], [0.0]))
    @test !ucmc.tabulate_acceptance(ucmc.ClockModel(6, [1.0, 1.0], [0.0]))
    @test !ucmc.tabulate_acceptance(ucmc.XYModel([1.0, 1.0], [0.0]))
    @test !ucmc.tabulate_acceptance(ucmc.HeisenbergModel([1.0, 1.0], [0.0]))
end

@testset "Single-orbital Metropolis fast path" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.3])
    T = 1.7
    algorithm = ucmc.MetropolisAlgorithm()
    container = ucmc.MeasurementContainer(model, geometry, 10, 2;
                                          diagnostics = [:acceptance_ratio])

    # The fused sweep visits sites, draws, and accepts exactly like the
    # per-step loop, so a seeded run follows the same trajectory either way —
    # with a workspace (tabulated acceptance) or without one.
    for with_workspace in (true, false)
        rng_fast, rng_ref = Xoshiro(123), Xoshiro(123)
        fast_state = ucmc.initialize_state(model, geometry; rng = Xoshiro(99))
        ref_state = ucmc.initialize_state(model, geometry; rng = Xoshiro(99))
        stats = (ucmc.UpdateStatistic(:acceptance_ratio),)

        fast_workspace = with_workspace ?
            ucmc.algorithm_workspace(algorithm, model, geometry, fast_state) : nothing
        ref_workspace = with_workspace ?
            ucmc.algorithm_workspace(algorithm, model, geometry, ref_state) : nothing

        for _ in 1:25
            ucmc.MC_Sweep!(algorithm, model, geometry, fast_state, container, T,
                           stats; rng = rng_fast, workspace = fast_workspace)
        end
        for _ in 1:(25 * geometry.n_sites)
            ucmc.step!(algorithm, model, geometry, ref_state, T;
                       rng = rng_ref, workspace = ref_workspace)
        end

        @test fast_state.spins == ref_state.spins
        @test fast_state.energy == ref_state.energy
        @test fast_state.magnetization == ref_state.magnetization
        @test stats[1].count == geometry.n_sites
        @test 0.0 <= ucmc.value(stats[1]) <= 1.0
    end

    # running totals stay exact through fused sweeps, field included
    plain_container = ucmc.MeasurementContainer(model, geometry, 10, 2)
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(31))
    workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
    rng = Xoshiro(7)
    for _ in 1:50
        ucmc.MC_Sweep!(algorithm, model, geometry, state, plain_container, T, ();
                       rng = rng, workspace = workspace)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
        @test state.magnetization == sum(state.spins)
    end

    # multi-orbital geometries fall back to the generic per-step loop
    geometry2 = two_orbital_geometry(2)
    model2 = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    container2 = ucmc.MeasurementContainer(model2, geometry2, 10, 2)
    state2 = ucmc.initialize_state(model2, geometry2; rng = Xoshiro(5))
    workspace2 = ucmc.algorithm_workspace(algorithm, model2, geometry2, state2)
    rng2 = Xoshiro(13)
    for _ in 1:10
        ucmc.MC_Sweep!(algorithm, model2, geometry2, state2, container2, 2.0, ();
                       rng = rng2, workspace = workspace2)
        @test state2.energy ≈ spin_energy(model2, geometry2, state2.spins)
        @test state2.magnetization == sum(state2.spins)
    end
end

@testset "Single-site update contract" begin
    geometry = square_geometry(2)
    models_and_energy = (
        (ucmc.IsingModel([1.0, 1.0], [0.5]), spin_energy),
        (ucmc.XYModel([1.0, 1.0], [0.5]), xy_energy),
        (ucmc.ClockModel(4, [1.0, 1.0], [0.5]), clock_energy),
        (ucmc.HeisenbergModel([1.0, 1.0], [0.5]), heisenberg_energy),
    )
    mag_totals(s::ucmc.IsingState) = (s.magnetization,)
    mag_totals(s::ucmc.HeisenbergState) =
        (s.magnetization_x, s.magnetization_y, s.magnetization_z)
    mag_totals(s) = (s.magnetization_x, s.magnetization_y)

    for (model, energy) in models_and_energy, algorithm in
            (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
             ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()),
             ucmc.GlauberAlgorithm(ucmc.SequentialProposal()),
             ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal()),
             ucmc.GlauberAlgorithm(ucmc.CheckerboardProposal()),
             ucmc.MetropolisAlgorithm(ucmc.ColoredProposal(2)),
             ucmc.GlauberAlgorithm(ucmc.ColoredProposal(2)))
        rng = Xoshiro(7)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
        for _ in 1:10
            proposal = ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace)

            # one shared proposal type across models and local algorithms
            @test proposal isa ucmc.SpinUpdate

            ΔE = ucmc.energy_difference(model, geometry, state, proposal)
            before_e = state.energy
            before_m = mag_totals(state)
            @test ucmc.apply_update!(state, proposal, ΔE) === nothing

            # the carried delta matches the actual magnetization change, and
            # the running totals stay exact
            carried = proposal.magnetization_change
            carried = carried isa Tuple ? collect(carried) : [carried]
            Δm = collect(mag_totals(state)) - collect(before_m)
            @test all(Δm .≈ carried)
            @test state.energy == before_e + ΔE
            if state isa Union{ucmc.XYState, ucmc.HeisenbergState}
                @test state.energy ≈ energy(model, geometry, state) atol = 1e-10
            else
                @test state.energy ≈ energy(model, geometry, state.spins) atol = 1e-10
            end
        end
    end
end
