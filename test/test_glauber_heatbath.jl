@testset "Glauber acceptance probability" begin
    @test ucmc.glauber_probability(0.0, 1.0) == 0.5
    @test ucmc.glauber_probability(Inf, 1.0) == 0.0
    @test ucmc.glauber_probability(-Inf, 1.0) == 1.0
    @test ucmc.glauber_probability(1e6, 1.0) == 0.0
    @test ucmc.glauber_probability(-1e6, 1.0) == 1.0
    for ΔE in (-3.0, -0.4, 0.7, 2.5), T in (0.5, 1.0, 3.0)
        p = ucmc.glauber_probability(ΔE, T)
        @test p ≈ 1 / (1 + exp(ΔE / T))
        @test p + ucmc.glauber_probability(-ΔE, T) ≈ 1.0
        # detailed balance for a symmetric proposal
        @test p / ucmc.glauber_probability(-ΔE, T) ≈ exp(-ΔE / T)
    end

    # the tabulated form used with an AcceptanceCache agrees with the direct one
    cache = ucmc.AcceptanceCache()
    for ΔE in (-3.0, -0.4, 0.0, 0.7, 2.5), T in (0.5, 1.0, 3.0)
        u = ucmc.cached_boltzmann_factor(cache, ΔE, Float64(T))
        @test (ΔE > 0 ? u / (1 + u) : 1 / (1 + 1 / u)) ≈ ucmc.glauber_probability(ΔE, T)
    end
end

@testset "Glauber updates keep invariants" begin
    geometry = two_orbital_geometry(2)
    algorithm = ucmc.GlauberAlgorithm()
    @test ucmc.available_diagnostics(algorithm) == [:acceptance_ratio]

    # Ising, including fields and an antiferromagnetic bond
    for J in ([1.0, 2.0], [1.0, -1.5])
        model = ucmc.IsingModel(J, [0.5, -0.25])
        rng = Xoshiro(11)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        n_accepted = 0
        for _ in 1:100
            result = ucmc.step!(algorithm, model, geometry, state, 2.0; rng = rng)
            @test result isa ucmc.GlauberUpdateResult
            @test ucmc.diagnostic_values(algorithm, result) ==
                (acceptance_ratio = result.accepted ? 1.0 : 0.0,)
            n_accepted += result.accepted
            @test all(s in (-1, 1) for s in state.spins)
            @test state.magnetization == sum(state.spins)
            @test state.energy ≈ spin_energy(model, geometry, state.spins)
        end
        @test 0 < n_accepted < 100
    end

    # XY
    model = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(12)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    for _ in 1:100
        ucmc.step!(algorithm, model, geometry, state, 2.0; rng = rng)
        @test state.energy ≈ xy_energy(model, geometry, state)
        @test state.magnetization_x ≈ sum(state.cosines)
        @test state.magnetization_y ≈ sum(state.sines)
    end

    # Glauber uses the same trial moves as Metropolis for every model
    for model in (ucmc.IsingModel([1.0, 2.0], [0.5, -0.25]),
                  ucmc.XYModel([1.0, 2.0], [0.5, -0.25]),
                  ucmc.ClockModel(5, [1.0, 2.0], [0.5, -0.25]))
        state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
        glauber = ucmc.propose(Xoshiro(5), algorithm, model, geometry, state)
        metropolis = ucmc.propose(Xoshiro(5), ucmc.MetropolisAlgorithm(), model, geometry, state)
        @test glauber == metropolis
    end

    # with ΔE = 0 for every move (J = h = 0) about half the moves are accepted,
    # whereas Metropolis would accept all of them
    free = ucmc.IsingModel([0.0, 0.0], [0.0, 0.0])
    rng = Xoshiro(13)
    state = ucmc.initialize_state(free, geometry; rng = rng)
    accepted = [ucmc.step!(algorithm, free, geometry, state, 1.0; rng = rng).accepted for _ in 1:4000]
    @test abs(sum(accepted) / length(accepted) - 0.5) < 0.05

    # the same holds with a workspace, where acceptance uses the tabulated
    # Boltzmann factors of the Ising model's AcceptanceCache
    rng = Xoshiro(17)
    state = ucmc.initialize_state(free, geometry; rng = rng)
    workspace = ucmc.algorithm_workspace(algorithm, free, geometry, state)
    accepted = [ucmc.step!(algorithm, free, geometry, state, 1.0;
                           rng = rng, workspace = workspace).accepted for _ in 1:4000]
    @test abs(sum(accepted) / length(accepted) - 0.5) < 0.05

    # and the running totals stay exact on the tabulated path for a model
    # with a nontrivial ΔE spectrum
    model = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(21)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
    for _ in 1:100
        ucmc.step!(algorithm, model, geometry, state, 2.0; rng = rng, workspace = workspace)
        @test state.magnetization == sum(state.spins)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end
end

@testset "Heatbath updates keep invariants" begin
    geometry = two_orbital_geometry(2)
    algorithm = ucmc.HeatbathAlgorithm()
    @test ucmc.available_diagnostics(algorithm) == [:change_ratio]

    # Ising, including fields and an antiferromagnetic bond, both without a
    # workspace and with one (the latter reads the conditional probability
    # off the Ising model's AcceptanceCache)
    for J in ([1.0, 2.0], [1.0, -1.5]), with_workspace in (false, true)
        model = ucmc.IsingModel(J, [0.5, -0.25])
        rng = Xoshiro(11)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        workspace = with_workspace ?
            ucmc.algorithm_workspace(algorithm, model, geometry, state) : nothing
        n_changed = 0
        for _ in 1:100
            result = ucmc.step!(algorithm, model, geometry, state, 2.0;
                                rng = rng, workspace = workspace)
            @test result isa ucmc.HeatbathUpdateResult
            @test ucmc.diagnostic_values(algorithm, result) ==
                (change_ratio = result.changed ? 1.0 : 0.0,)
            n_changed += result.changed
            @test all(s in (-1, 1) for s in state.spins)
            @test state.magnetization == sum(state.spins)
            @test state.energy ≈ spin_energy(model, geometry, state.spins)
        end
        @test 0 < n_changed < 100
    end

    # the workspace is the shared local-update one, with the Ising model's
    # acceptance cache; the continuous-spectrum models get no cache
    model = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(3))
    workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
    @test workspace isa ucmc.LocalUpdateWorkspace
    @test isnothing(workspace.proposal)
    @test workspace.acceptance isa ucmc.AcceptanceCache
    xy = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    xy_state = ucmc.initialize_state(xy, geometry; rng = Xoshiro(4))
    @test isnothing(ucmc.algorithm_workspace(algorithm, xy, geometry, xy_state).acceptance)

    # XY: spins stay on the unit circle and the running totals stay exact
    rng = Xoshiro(12)
    for _ in 1:100
        ucmc.step!(algorithm, xy, geometry, xy_state, 2.0; rng = rng)
        @test all(i -> hypot(xy_state.cosines[i], xy_state.sines[i]) ≈ 1.0,
                  1:geometry.n_sites)
        @test xy_state.energy ≈ xy_energy(xy, geometry, xy_state)
        @test xy_state.magnetization_x ≈ sum(xy_state.cosines)
        @test xy_state.magnetization_y ≈ sum(xy_state.sines)
    end

    # Clock
    clock = ucmc.ClockModel(5, [1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(13)
    clock_state = ucmc.initialize_state(clock, geometry; rng = rng)
    for _ in 1:100
        ucmc.step!(algorithm, clock, geometry, clock_state, 2.0; rng = rng)
        @test clock_invariants_hold(clock, geometry, clock_state)
    end

    # Heisenberg: spins stay on the unit sphere and the totals stay exact
    heisenberg = ucmc.HeisenbergModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(14)
    heisenberg_state = ucmc.initialize_state(heisenberg, geometry; rng = rng)
    for _ in 1:100
        ucmc.step!(algorithm, heisenberg, geometry, heisenberg_state, 2.0; rng = rng)
        @test all(i -> hypot(heisenberg_state.xs[i], heisenberg_state.ys[i],
                             heisenberg_state.zs[i]) ≈ 1.0, 1:geometry.n_sites)
        @test heisenberg_state.energy ≈ heisenberg_energy(heisenberg, geometry, heisenberg_state)
        @test heisenberg_state.magnetization_x ≈ sum(heisenberg_state.xs)
        @test heisenberg_state.magnetization_y ≈ sum(heisenberg_state.ys)
        @test heisenberg_state.magnetization_z ≈ sum(heisenberg_state.zs)
    end

    # free Ising (J = h = 0): the conditional is 1/2 regardless of the current
    # spin, so about half the updates change the spin — on both the direct
    # path and the tabulated (AcceptanceCache) path
    free = ucmc.IsingModel([0.0, 0.0], [0.0, 0.0])
    rng = Xoshiro(15)
    free_state = ucmc.initialize_state(free, geometry; rng = rng)
    changed = [ucmc.step!(algorithm, free, geometry, free_state, 1.0; rng = rng).changed
               for _ in 1:4000]
    @test abs(sum(changed) / length(changed) - 0.5) < 0.05

    rng = Xoshiro(17)
    free_state = ucmc.initialize_state(free, geometry; rng = rng)
    free_workspace = ucmc.algorithm_workspace(algorithm, free, geometry, free_state)
    changed = [ucmc.step!(algorithm, free, geometry, free_state, 1.0;
                          rng = rng, workspace = free_workspace).changed for _ in 1:4000]
    @test abs(sum(changed) / length(changed) - 0.5) < 0.05

    # a stateful scheme without its workspace is an explicit error, and
    # sequential heatbath sweeps keep the running totals exact
    sequential = ucmc.HeatbathAlgorithm(ucmc.SequentialProposal())
    @test_throws ArgumentError ucmc.step!(sequential, model, geometry, state, 2.0;
                                          rng = Xoshiro(1))
    rng = Xoshiro(23)
    seq_state = ucmc.initialize_state(model, geometry; rng = rng)
    seq_workspace = ucmc.algorithm_workspace(sequential, model, geometry, seq_state)
    for _ in 1:(3 * geometry.n_sites)
        ucmc.step!(sequential, model, geometry, seq_state, 2.0;
                   rng = rng, workspace = seq_workspace)
        @test seq_state.energy ≈ spin_energy(model, geometry, seq_state.spins)
        @test seq_state.magnetization == sum(seq_state.spins)
    end

    # the drivers build and thread the workspace automatically
    container = ucmc.MeasurementContainer(model, geometry, 4, 2;
                                          diagnostics = [:change_ratio])
    parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
    ucmc.run_simulation(sequential, model, geometry,
                        ucmc.initialize_state(model, geometry; rng = rng),
                        container, parameters; rng = rng)
    @test all(isfinite, container.data[:energy])
    @test all(isfinite, container.data[:change_ratio])
end

@testset "Heatbath draws match the single-site conditionals" begin
    geometry = two_orbital_geometry(2)
    site = 1
    T = 1.3

    # Ising: with local field H the conditional is p(+) = 1 / (1 + exp(-2H/T))
    model = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(31))
    spins_before = copy(state.spins)
    info = geometry.neighbor_table[site]
    H = ucmc.field_strength(model, geometry, site) +
        sum(model.J[t] * state.spins[j]
            for (t, j) in zip(info.templates, info.neighbors) if j != site)
    p_up = 1 / (1 + exp(-2H / T))
    rng = Xoshiro(32)
    draws = [ucmc.heatbath_draw(rng, model, geometry, state, site, T) for _ in 1:20_000]
    @test all(d -> d.new_value in (-1, 1), draws)
    @test all(d -> d.magnetization_change == d.new_value - state.spins[site], draws)
    @test abs(mean(d -> d.new_value == 1, draws) - p_up) <
        4 * sqrt(p_up * (1 - p_up) / length(draws))
    @test state.spins == spins_before  # drawing does not mutate the state

    # Clock: the conditional over the q states is exactly enumerable
    q = 4
    model = ucmc.ClockModel(q, [1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(33))
    A = ucmc.field_strength(model, geometry, site)
    B = 0.0
    for (t, j) in zip(info.templates, info.neighbors)
        j == site && continue
        A += model.J[t] * ucmc.clock_cos(model, state.spins[j])
        B += model.J[t] * ucmc.clock_sin(model, state.spins[j])
    end
    weights = [exp((A * ucmc.clock_cos(model, n) + B * ucmc.clock_sin(model, n)) / T)
               for n in 0:(q - 1)]
    probs = weights / sum(weights)
    expected_cos = sum(probs[n + 1] * ucmc.clock_cos(model, n) for n in 0:(q - 1))
    expected_sin = sum(probs[n + 1] * ucmc.clock_sin(model, n) for n in 0:(q - 1))
    rng = Xoshiro(34)
    draws = [ucmc.heatbath_draw(rng, model, geometry, state, site, T) for _ in 1:20_000]
    @test all(d -> d.new_value in 0:(q - 1), draws)
    cos_draws = [ucmc.clock_cos(model, d.new_value) for d in draws]
    sin_draws = [ucmc.clock_sin(model, d.new_value) for d in draws]
    @test abs(mean(cos_draws) - expected_cos) < 4 * std(cos_draws) / sqrt(length(draws))
    @test abs(mean(sin_draws) - expected_sin) < 4 * std(sin_draws) / sqrt(length(draws))

    # XY: the conditional is von Mises around the local-field direction; its
    # mean resultant is I1(κ)/I0(κ), computed here by trapezoid quadrature
    model = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(35))
    A = ucmc.field_strength(model, geometry, site)
    B = 0.0
    for (t, j) in zip(info.templates, info.neighbors)
        j == site && continue
        A += model.J[t] * state.cosines[j]
        B += model.J[t] * state.sines[j]
    end
    R = hypot(A, B)
    κ = R / T
    θs = (2π / 720) .* (0:719)
    w = exp.(κ .* cos.(θs))
    expected = sum(cos.(θs) .* w) / sum(w)
    rng = Xoshiro(36)
    draws = [ucmc.heatbath_draw(rng, model, geometry, state, site, T) for _ in 1:20_000]
    @test all(d -> hypot(d.new_value...) ≈ 1.0, draws)
    projections = [(d.new_value[1] * A + d.new_value[2] * B) / R for d in draws]
    @test abs(mean(projections) - expected) <
        4 * std(projections) / sqrt(length(projections))
    transverse = [(-d.new_value[1] * B + d.new_value[2] * A) / R for d in draws]
    @test abs(mean(transverse)) < 4 * std(transverse) / sqrt(length(transverse))

    # no local field (J = h = 0): the conditional is uniform on the circle
    free = ucmc.XYModel([0.0, 0.0], [0.0, 0.0])
    free_state = ucmc.initialize_state(free, geometry; rng = Xoshiro(37))
    rng = Xoshiro(38)
    free_draws = [ucmc.heatbath_draw(rng, free, geometry, free_state, site, T).new_value
                  for _ in 1:20_000]
    @test abs(mean(first.(free_draws))) < 0.02
    @test abs(mean(last.(free_draws))) < 0.02

    # Heisenberg: the conditional is the Fisher distribution about the
    # local-field direction, with mean projection given by the Langevin
    # function L(κ) = coth(κ) - 1/κ and second moment 1 - 2 L(κ) / κ
    model = ucmc.HeisenbergModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(39))
    Hx = ucmc.field_strength(model, geometry, site)
    Hy = Hz = 0.0
    for (t, j) in zip(info.templates, info.neighbors)
        j == site && continue
        Hx += model.J[t] * state.xs[j]
        Hy += model.J[t] * state.ys[j]
        Hz += model.J[t] * state.zs[j]
    end
    R = hypot(Hx, Hy, Hz)
    κ = R / T
    expected = coth(κ) - 1 / κ
    rng = Xoshiro(40)
    draws = [ucmc.heatbath_draw(rng, model, geometry, state, site, T) for _ in 1:20_000]
    @test all(d -> hypot(d.new_value...) ≈ 1.0, draws)
    projections = [(d.new_value[1] * Hx + d.new_value[2] * Hy + d.new_value[3] * Hz) / R
                   for d in draws]
    @test abs(mean(projections) - expected) <
        4 * std(projections) / sqrt(length(projections))
    @test abs(mean(abs2, projections) - (1 - 2 * expected / κ)) <
        4 * std(abs2.(projections)) / sqrt(length(projections))

    # no local field: the conditional is uniform on the sphere
    free = ucmc.HeisenbergModel([0.0, 0.0], [0.0, 0.0])
    free_state = ucmc.initialize_state(free, geometry; rng = Xoshiro(41))
    rng = Xoshiro(42)
    free_draws = [ucmc.heatbath_draw(rng, free, geometry, free_state, site, T).new_value
                  for _ in 1:20_000]
    @test abs(mean(first.(free_draws))) < 0.02
    @test abs(mean(d -> d[2], free_draws)) < 0.02
    @test abs(mean(d -> d[3], free_draws)) < 0.02
end

@testset "Glauber and Heatbath match exact results" begin
    # both algorithms satisfy detailed balance with the same target measure,
    # so they share the exactness harness
    geometry = square_geometry(2)
    N = geometry.n_sites
    T = 2.0
    model = ucmc.IsingModel([1.0, 0.8], [0.35])

    Z = E = M = 0.0
    for c in 0:(2^N - 1)
        spins = 1 .- 2 .* digits(c; base = 2, pad = N)
        e = spin_energy(model, geometry, spins)
        w = exp(-e / T)
        Z += w
        E += w * e
        M += w * sum(spins)
    end
    exact_e, exact_m = E / (Z * N), M / (Z * N)

    # XY on a periodic two-site chain: exact moments by quadrature (the
    # trapezoidal rule is spectrally accurate for periodic integrands)
    chain = ucmc.Geometry(
        lu.UnitCell(lattice_vecs = [[1.0]], basis_vecs = [[0.0]]),
        lu.Lattice(L = [2], periodic = [true]),
        [lu.Bond(orbitals = (1, 1), displacement = [1])],
    )
    xy = ucmc.XYModel([1.0], [0.4])
    T_xy = 0.8
    n_grid = 400
    θs = (2π / n_grid) .* (0:(n_grid - 1))
    Z = E = Mx = 0.0
    for a in θs, b in θs
        e = xy_energy(xy, chain, cos.([a, b]), sin.([a, b]))
        w = exp(-e / T_xy)
        Z += w
        E += w * e
        Mx += w * (cos(a) + cos(b))
    end
    exact_e_xy, exact_mx = E / (2Z), Mx / (2Z)

    for algorithm in (ucmc.GlauberAlgorithm(), ucmc.HeatbathAlgorithm())
        parameters = ucmc.SimulationParameters(T, 200, 20_000, 0, 20)
        rng = Xoshiro(2024)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, 20_000, 20)
        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, T, N)
        e, e_err = result.primary[:energy]
        m, m_err = result.primary[:magnetization]
        @test abs(e - exact_e) < 4 * e_err
        @test abs(m - exact_m) < 4 * m_err

        parameters = ucmc.SimulationParameters(T_xy, 200, 20_000, 0, 20)
        rng = Xoshiro(7)
        state = ucmc.initialize_state(xy, chain; rng = rng)
        container = ucmc.MeasurementContainer(xy, chain, 20_000, 20)
        ucmc.run_simulation(algorithm, xy, chain, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, T_xy, chain.n_sites)
        e, e_err = result.primary[:energy]
        mx, mx_err = result.primary[:magnetization_x]
        @test abs(e - exact_e_xy) < 4 * e_err
        @test abs(mx - exact_mx) < 4 * mx_err
    end
end

@testset "Overrelaxation keeps invariants and conserves energy" begin
    geometry = two_orbital_geometry(2)
    algorithm = ucmc.OverrelaxationAlgorithm()
    @test ucmc.available_diagnostics(algorithm) == Symbol[]

    # XY: the reflection conserves the energy (microcanonical) and keeps the
    # running totals exact
    xy = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(21)
    state = ucmc.initialize_state(xy, geometry; rng = rng)
    E0 = state.energy
    for _ in 1:100
        result = ucmc.step!(algorithm, xy, geometry, state, 2.0; rng = rng)
        @test result isa ucmc.OverrelaxationUpdateResult
        @test ucmc.diagnostic_values(algorithm, result) == NamedTuple()
        @test all(i -> hypot(state.cosines[i], state.sines[i]) ≈ 1.0,
                  1:geometry.n_sites)
        @test state.energy ≈ E0
        @test state.energy ≈ xy_energy(xy, geometry, state)
        @test state.magnetization_x ≈ sum(state.cosines)
        @test state.magnetization_y ≈ sum(state.sines)
    end

    # the reflection is v' = 2 (v·H) H / |H|² - v about the local field, and
    # an involution: the site's own move leaves its local field unchanged
    state = ucmc.initialize_state(xy, geometry; rng = Xoshiro(22))
    site = 1
    info = geometry.neighbor_table[site]
    Hx = ucmc.field_strength(xy, geometry, site)
    Hy = 0.0
    for (t, j) in zip(info.templates, info.neighbors)
        j == site && continue
        Hx += xy.J[t] * state.cosines[j]
        Hy += xy.J[t] * state.sines[j]
    end
    v = (state.cosines[site], state.sines[site])
    projection = (v[1] * Hx + v[2] * Hy) / (Hx^2 + Hy^2)
    proposal = ucmc.overrelax(xy, geometry, state, site)
    @test collect(proposal.new_value) ≈ [2 * projection * Hx - v[1], 2 * projection * Hy - v[2]]
    @test collect(proposal.magnetization_change) ≈ collect(proposal.new_value .- v)
    ucmc.apply_update!(state, proposal, ucmc.energy_difference(xy, geometry, state, proposal))
    @test collect(ucmc.overrelax(xy, geometry, state, site).new_value) ≈ collect(v)

    # Heisenberg: the same invariants on the sphere
    heisenberg = ucmc.HeisenbergModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(23)
    hstate = ucmc.initialize_state(heisenberg, geometry; rng = rng)
    E0 = hstate.energy
    for _ in 1:100
        ucmc.step!(algorithm, heisenberg, geometry, hstate, 2.0; rng = rng)
        @test all(i -> hypot(hstate.xs[i], hstate.ys[i], hstate.zs[i]) ≈ 1.0,
                  1:geometry.n_sites)
        @test hstate.energy ≈ E0
        @test hstate.energy ≈ heisenberg_energy(heisenberg, geometry, hstate)
        @test hstate.magnetization_x ≈ sum(hstate.xs)
        @test hstate.magnetization_y ≈ sum(hstate.ys)
        @test hstate.magnetization_z ≈ sum(hstate.zs)
    end

    # discrete models have no reflection update
    ising = ucmc.IsingModel([1.0, 2.0], [0.0, 0.0])
    ising_state = ucmc.initialize_state(ising, geometry; rng = Xoshiro(24))
    @test_throws ArgumentError ucmc.step!(algorithm, ising, geometry, ising_state, 1.0)
    clock = ucmc.ClockModel(3, [1.0, 2.0], [0.0, 0.0])
    clock_state = ucmc.initialize_state(clock, geometry; rng = Xoshiro(24))
    @test_throws ArgumentError ucmc.step!(algorithm, clock, geometry, clock_state, 1.0)

    # a vanishing local field (J = h = 0) leaves the spin unchanged
    free = ucmc.XYModel([0.0, 0.0], [0.0, 0.0])
    free_state = ucmc.initialize_state(free, geometry; rng = Xoshiro(25))
    before = (copy(free_state.cosines), copy(free_state.sines))
    rng = Xoshiro(26)
    for _ in 1:geometry.n_sites
        ucmc.step!(algorithm, free, geometry, free_state, 1.0; rng = rng)
    end
    @test free_state.cosines == before[1]
    @test free_state.sines == before[2]

    # a stateful scheme without its workspace is an explicit error, and
    # sequential overrelaxation sweeps still conserve the energy
    sequential = ucmc.OverrelaxationAlgorithm(ucmc.SequentialProposal())
    @test_throws ArgumentError ucmc.step!(sequential, xy, geometry, state, 2.0)
    rng = Xoshiro(27)
    seq_state = ucmc.initialize_state(xy, geometry; rng = rng)
    seq_workspace = ucmc.algorithm_workspace(sequential, xy, geometry, seq_state)
    E0 = seq_state.energy
    for _ in 1:(3 * geometry.n_sites)
        ucmc.step!(sequential, xy, geometry, seq_state, 2.0;
                   rng = rng, workspace = seq_workspace)
        @test seq_state.energy ≈ E0
        @test seq_state.energy ≈ xy_energy(xy, geometry, seq_state)
    end

    # run_simulation works with the empty diagnostics list, and requesting a
    # diagnostic fails up front
    parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
    container = ucmc.MeasurementContainer(xy, geometry, 4, 2)
    ucmc.run_simulation(algorithm, xy, geometry,
                        ucmc.initialize_state(xy, geometry; rng = rng),
                        container, parameters; rng = rng)
    @test all(isfinite, container.data[:energy])
    @test_throws ArgumentError ucmc.run_simulation(
        algorithm, xy, geometry, ucmc.initialize_state(xy, geometry; rng = rng),
        ucmc.MeasurementContainer(xy, geometry, 4, 2; diagnostics = [:acceptance_ratio]),
        parameters; rng = rng)

    # seeded runs are reproducible
    containers = map(1:2) do _
        rng = Xoshiro(2024)
        ucmc.run_simulation(algorithm, xy, geometry,
                            ucmc.initialize_state(xy, geometry; rng = rng),
                            ucmc.MeasurementContainer(xy, geometry, 20, 4),
                            ucmc.SimulationParameters(2.0, 5, 20, 1, 4); rng = rng)
    end
    @test isequal(containers[1].data, containers[2].data)
end

@testset "Heatbath + overrelaxation reproduces the canonical ensemble" begin
    # XY on a periodic two-site chain, exact moments by quadrature (the
    # trapezoidal rule is spectrally accurate for periodic integrands)
    chain = ucmc.Geometry(
        lu.UnitCell(lattice_vecs = [[1.0]], basis_vecs = [[0.0]]),
        lu.Lattice(L = [2], periodic = [true]),
        [lu.Bond(orbitals = (1, 1), displacement = [1])],
    )
    xy = ucmc.XYModel([1.0], [0.4])
    T = 0.8
    n_grid = 400
    θs = (2π / n_grid) .* (0:(n_grid - 1))
    Z = E = Mx = 0.0
    for a in θs, b in θs
        e = xy_energy(xy, chain, cos.([a, b]), sin.([a, b]))
        w = exp(-e / T)
        Z += w
        E += w * e
        Mx += w * (cos(a) + cos(b))
    end
    exact_e, exact_mx = E / (2Z), Mx / (2Z)

    # one heatbath sweep followed by three overrelaxation sweeps per
    # measurement — overrelaxation alone is not ergodic, the mixture is
    rng = Xoshiro(2024)
    state = ucmc.initialize_state(xy, chain; rng = rng)
    container = ucmc.MeasurementContainer(xy, chain, 20_000, 20)
    heatbath = ucmc.HeatbathAlgorithm()
    overrelax = ucmc.OverrelaxationAlgorithm()
    heatbath_workspace = ucmc.algorithm_workspace(heatbath, xy, chain, state)
    overrelax_workspace = ucmc.algorithm_workspace(overrelax, xy, chain, state)
    stats = ()
    for _ in 1:200
        ucmc.MC_Sweep!(heatbath, xy, chain, state, container, T, stats;
                       rng = rng, workspace = heatbath_workspace)
    end
    for _ in 1:20_000
        ucmc.MC_Sweep!(heatbath, xy, chain, state, container, T, stats;
                       rng = rng, workspace = heatbath_workspace)
        for _ in 1:3
            ucmc.MC_Sweep!(overrelax, xy, chain, state, container, T, stats;
                           rng = rng, workspace = overrelax_workspace)
        end
        ucmc.measure!(container, state)
    end
    result = ucmc.analyze(container, T, chain.n_sites)
    e, e_err = result.primary[:energy]
    mx, mx_err = result.primary[:magnetization_x]
    @test abs(e - exact_e) < 4 * e_err
    @test abs(mx - exact_mx) < 4 * mx_err
end
