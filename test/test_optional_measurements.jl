@testset "Optional correlation observables" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    raw_container = ucmc.MeasurementContainer(
        model,
        geometry,
        4,
        2;
        measurements = [:raw_correlation],
    )
    @test :correlation in keys(raw_container.data)
    @test !any(:correlation_connected in derived.names for derived in raw_container.derived)

    connected_container = ucmc.MeasurementContainer(
        model,
        geometry,
        4,
        2;
        measurements = [:connected_correlation],
    )
    @test :correlation in keys(connected_container.data)
    @test any(:correlation_connected in derived.names for derived in connected_container.derived)

    state = ucmc.initialize_state(model, geometry)
    for _ in 1:4
        result = ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        ucmc.measure!(connected_container, state)
    end

    result = ucmc.analyze(connected_container, 2.0, geometry.n_sites)
    @test haskey(result.primary, :correlation)
    @test haskey(result.derived, :correlation_connected)
    @test haskey(connected_container.data, :magnetization_per_orbital)
end

@testset "Connected correlation requires periodic boundaries" begin
    # The connected subtraction uses whole-lattice per-orbital magnetization
    # products, which match the exact per-class valid-cell subtraction only
    # under full translation invariance; open lattices must be refused rather
    # than silently inexact (confirmed by exact enumeration on an open 3×3
    # two-orbital Ising lattice, where the error flips signs).
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0]],
    )
    bonds = [lu.Bond(orbitals = (1, 1), displacement = [1, 0]),
             lu.Bond(orbitals = (1, 1), displacement = [0, 1])]
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    for periodic in ([false, false], [true, false])
        open_geometry = ucmc.Geometry(
            unit_cell, lu.Lattice(L = [3, 3], periodic = periodic), bonds)
        @test_throws ArgumentError ucmc.MeasurementContainer(
            model, open_geometry, 4, 2; measurements = [:connected_correlation])
        # the raw correlation is exact either way and stays available
        raw = ucmc.MeasurementContainer(
            model, open_geometry, 4, 2; measurements = [:raw_correlation])
        @test :correlation in keys(raw.data)
    end
    # fully periodic lattices are unaffected
    periodic_container = ucmc.MeasurementContainer(
        model, square_geometry(2), 4, 2; measurements = [:connected_correlation])
    @test :correlation in keys(periodic_container.data)
end

@testset "Connected correlation is per-orbital" begin
    # Two decoupled sublattices under opposite fields have inequivalent
    # sublattice magnetizations, so the per-orbital subtraction differs from
    # the lattice-averaged one.
    geometry = two_orbital_geometry(2)
    N = geometry.n_sites
    n_orb = lu.norbits(geometry.unit_cell)
    n_cells = N ÷ n_orb
    T = 1.0
    model = ucmc.IsingModel([1.0, 1.0], [0.5, -0.25])
    disp = ucmc.build_displacement_classes(geometry)
    L = geometry.lattice.L

    # Exact per-class correlations and per-orbital magnetizations by enumeration.
    Z = 0.0
    m_exact = zeros(n_orb)
    C_exact = zeros(length(disp.displacements))
    for c in 0:(2^N - 1)
        spins = 2 .* digits(c; base = 2, pad = N) .- 1
        w = exp(-spin_energy(model, geometry, spins) / T)
        Z += w
        for site in 1:N
            m_exact[lu.site_to_orbital(site, geometry.unit_cell)] += w * spins[site]
        end
        for r in eachindex(disp.displacements)
            Δl = disp.displacements[r]
            oi, oj = disp.orbitals[r]
            acc = 0.0
            for loc_i in Iterators.product(disp.valid_ranges[r]...)
                i = ucmc.site_at(geometry, loc_i, oi)
                loc_j = ntuple(d -> mod(loc_i[d] + Δl[d], L[d]), length(L))
                acc += spins[i] * spins[ucmc.site_at(geometry, loc_j, oj)]
            end
            C_exact[r] += w * acc / disp.counts[r]
        end
    end
    m_exact ./= Z * n_cells
    C_exact ./= Z
    connected_exact = [
        C_exact[r] - m_exact[oi] * m_exact[oj]
        for (r, (oi, oj)) in enumerate(disp.orbitals)
    ]

    rng = Xoshiro(11)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    container = ucmc.MeasurementContainer(model, geometry, 20_000, 20;
        measurements = [:connected_correlation])
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container,
        ucmc.SimulationParameters(T, 500, 20_000, 0, 20); rng = rng)
    result = ucmc.analyze(container, T, N)

    # the per-orbital magnetization observable matches enumeration
    m_meas, m_err = result.primary[:magnetization_per_orbital]
    @test size(m_meas) == (1, n_orb)
    @test all(abs.(m_meas .- reshape(m_exact, 1, n_orb)) .< 4 .* m_err .+ 5e-3)

    # the connected correlation subtracts ⟨S_oi⟩⟨S_oj⟩ per orbital pair
    conn, conn_err = result.derived.correlation_connected, result.derived.correlation_connected_err
    @test all(abs.(conn .- connected_exact) .< 4 .* conn_err .+ 5e-3)

    # and the two sublattices really are inequivalent here, so the old
    # lattice-averaged subtraction would give a resolvably wrong answer
    wrong = [C_exact[r] - (sum(m_exact) / n_orb)^2 for r in eachindex(disp.displacements)]
    cross = [r for (r, (oi, oj)) in enumerate(disp.orbitals) if oi != oj]
    @test maximum(abs.(connected_exact[cross] .- wrong[cross])) > 0.02
end

@testset "Optional magnetization and Binder observables" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(
        model,
        geometry,
        4,
        2;
        measurements = [:abs_magnetization, :binder_cumulant],
        diagnostics = [:acceptance_ratio],
    )

    for _ in 1:4
        result = ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        ucmc.measure!(container, state; diagnostics = (acceptance_ratio = result.accepted ? 1.0 : 0.0,))
    end

    @test :abs_magnetization in keys(container.data)
    @test :magnetization_fourth in keys(container.data)
    result = ucmc.analyze(container, 2.0, geometry.n_sites)
    @test haskey(result.primary, :abs_magnetization)
    @test haskey(result.derived, :binder_cumulant)

    duplicate_container = ucmc.MeasurementContainer(
        model,
        geometry,
        4,
        2;
        measurements = [
            :abs_magnetization,
            :abs_magnetization_squared,
            :binder_cumulant,
            :magnetization_fourth,
        ],
    )
    @test length(duplicate_container.observables) == length(unique(
        observable.name for observable in duplicate_container.observables
    ))
end

@testset "Optional measurement registry" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])

    # unknown names throw instead of being silently ignored
    @test_throws ArgumentError ucmc.MeasurementContainer(model, geometry, 4, 2;
        measurements = [:not_a_measurement])
    @test_throws ArgumentError ucmc.optional_observables(model, geometry, Val(:not_a_measurement))

    # aliases canonicalize to the same measurement
    derived_names(c) = sort(Symbol[n for d in c.derived for n in d.names])
    plain = ucmc.MeasurementContainer(model, geometry, 4, 2; measurements = [:binder_cumulant])
    aliased = ucmc.MeasurementContainer(model, geometry, 4, 2; measurements = [:binder])
    @test sort(collect(keys(plain.data))) == sort(collect(keys(aliased.data)))
    @test derived_names(plain) == derived_names(aliased)

    abs_alias = ucmc.MeasurementContainer(model, geometry, 4, 2; measurements = [:magnetization_abs])
    @test haskey(abs_alias.data, :abs_magnetization)

    # requesting an implied observable by name adds it exactly once
    both = ucmc.MeasurementContainer(model, geometry, 4, 2;
        measurements = [:abs_magnetization, :abs_magnetization_squared])
    @test haskey(both.data, :abs_magnetization_squared)
    names = [o.name for o in both.observables]
    @test length(names) == length(unique(names))

    # every advertised name builds for each model
    for m in (model, ucmc.XYModel([1.0, 1.0], [0.0]), ucmc.ClockModel(4, [1.0, 1.0], [0.0]))
        container = ucmc.MeasurementContainer(m, geometry, 4, 2;
            measurements = ucmc.available_measurements(m))
        @test !isempty(keys(container.data))
    end

    @test ucmc.extra_measurements(ucmc.ClockModel) == (:clock_order_parameter, :helicity_modulus)
    @test isempty(ucmc.extra_measurements(ucmc.IsingModel))
    @test_throws ArgumentError ucmc.binder_denominator(PairingTestModel())

    # models without the hooks fail loudly at construction, not mid-run
    @test_throws ArgumentError ucmc.observables(PairingTestModel(), geometry)
    @test_throws ArgumentError ucmc.derived_observables(PairingTestModel(), geometry)
    @test_throws ArgumentError ucmc.optional_observables(PairingTestModel(), geometry, Val(:abs_magnetization))
    @test_throws ArgumentError ucmc.optional_observables(PairingTestModel(), geometry, Val(:magnetization_fourth))
end
