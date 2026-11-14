@testset "Site-resolved Ising fields" begin
    geometry = two_orbital_geometry(2)
    model = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry)

    @test length(model.h) == lu.norbits(geometry.unit_cell)
    @test state.energy ≈ spin_energy(model, geometry, state.spins)

    for _ in 1:20
        ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end

    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        cluster_state = ucmc.initialize_state(model, geometry)
        for _ in 1:20
            ucmc.step!(algorithm, model, geometry, cluster_state, 2.0)
            @test cluster_state.magnetization == sum(cluster_state.spins)
            @test cluster_state.energy ≈ spin_energy(model, geometry, cluster_state.spins)
        end
    end

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.IsingModel([1.0, 2.0], [0.5]), geometry
    )
    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.IsingModel([1.0, 2.0], [0.5, -0.25, 0.0]), geometry
    )
end

@testset "Heisenberg matches the exact Langevin response" begin
    # J = 0: independent spins in a field along x, so the per-site response is
    # the Langevin function ⟨m_x⟩ = coth(h / T) - T / h, and ⟨m_y⟩ = ⟨m_z⟩ = 0.
    # The Wolff run exercises the ghost-only dynamics (no bonds to activate).
    geometry = ucmc.Geometry(
        lu.UnitCell(lattice_vecs = [[1.0, 0.0], [0.0, 1.0]], basis_vecs = [[0.0, 0.0]]),
        lu.Lattice(L = [4, 4], periodic = [true, true]),
        lu.Bond{2}[],
    )
    T = 1.0
    h = 0.5
    model = ucmc.HeisenbergModel(Float64[], [h])
    exact_mx = coth(h / T) - T / h

    for (algorithm, n_measurements) in (
        (ucmc.MetropolisAlgorithm(), 50_000), (ucmc.WolffAlgorithm(), 200_000)
    )
        rng = Xoshiro(19)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, n_measurements, 25)
        parameters = ucmc.SimulationParameters(T, 500, n_measurements, 0, 25)
        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, T, geometry.n_sites)
        mx, mx_err = result.primary[:magnetization_x]
        my, my_err = result.primary[:magnetization_y]
        mz, mz_err = result.primary[:magnetization_z]
        @test abs(mx - exact_mx) < 4 * mx_err
        @test abs(my) < 4 * my_err
        @test abs(mz) < 4 * mz_err
    end
end

@testset "Model/geometry validation" begin
    geometry = square_geometry(2)
    valid_model = ucmc.IsingModel([1.0, 1.0], [0.0])

    @test_throws ArgumentError ucmc.initialize_state(ucmc.IsingModel([1.0], [0.0]), geometry)

    bad_geometry = ucmc.Geometry(
        geometry.unit_cell,
        geometry.lattice,
        [lu.Bond(orbitals = (1, 1), displacement = [1, 0])],
    )
    @test_throws ArgumentError ucmc.validate_model_geometry(valid_model, bad_geometry)

    @test_throws ArgumentError ucmc.bond_strength(valid_model, geometry, 100)
end

struct PairingTestModel <: ucmc.AbstractModel{Float64} end
struct PairingTestState <: ucmc.AbstractState end

@testset "Model/state pairing" begin
    geometry = square_geometry(2)
    ising = ucmc.IsingModel([1.0, 1.0], [0.0])
    xy = ucmc.XYModel([1.0, 1.0], [0.0])
    clock = ucmc.ClockModel(4, [1.0, 1.0], [0.0])
    heisenberg = ucmc.HeisenbergModel([1.0, 1.0], [0.0])

    @test ucmc.state_type(ising) == ucmc.IsingState
    @test ucmc.state_type(xy) == ucmc.XYState
    @test ucmc.state_type(clock) == ucmc.ClockState
    @test ucmc.state_type(heisenberg) == ucmc.HeisenbergState
    @test_throws ArgumentError ucmc.state_type(PairingTestModel())

    # Models without their own methods fail loudly instead of falling back
    # to another model's physics.
    @test_throws ArgumentError ucmc.initialize_state(PairingTestModel(), geometry)

    ising_state = ucmc.initialize_state(ising, geometry)
    @test_throws ArgumentError ucmc.heatbath_draw(Xoshiro(1), PairingTestModel(),
                                                  geometry, ising_state, 1, 1.0)
    @test_throws ArgumentError ucmc.overrelax(PairingTestModel(), geometry, ising_state, 1)

    xy_state = ucmc.initialize_state(xy, geometry)
    heisenberg_state = ucmc.initialize_state(heisenberg, geometry)
    @test ising_state isa ucmc.state_type(ising)
    @test xy_state isa ucmc.state_type(xy)
    @test ucmc.initialize_state(clock, geometry) isa ucmc.state_type(clock)
    @test heisenberg_state isa ucmc.state_type(heisenberg)

    @test isnothing(ucmc.validate_model_state(ising, ising_state))
    @test_throws ArgumentError ucmc.validate_model_state(ising, xy_state)
    @test_throws ArgumentError ucmc.validate_model_state(xy, ising_state)
    @test_throws ArgumentError ucmc.validate_model_state(heisenberg, xy_state)
    @test_throws ArgumentError ucmc.validate_model_state(PairingTestModel(), PairingTestState())

    container = ucmc.MeasurementContainer(ising, geometry, 4, 2)
    parameters = ucmc.SimulationParameters(2.0, 0, 4, 0, 2)
    @test_throws ArgumentError ucmc.run_simulation(
        ucmc.MetropolisAlgorithm(), ising, geometry, xy_state, container, parameters
    )
    @test_throws ArgumentError ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), ising, geometry, xy_state, 0, 4, 0, 2, [2.0]
    )

    @test ucmc.magnetization_value(ising_state) == ising_state.magnetization
    @test ucmc.magnetization_value(xy_state) ==
        hypot(xy_state.magnetization_x, xy_state.magnetization_y)
    @test ucmc.magnetization_value(heisenberg_state) ==
        hypot(heisenberg_state.magnetization_x, heisenberg_state.magnetization_y,
              heisenberg_state.magnetization_z)
    @test ucmc.magnetization_value(ucmc.initialize_state(clock, geometry)) isa Real
    @test_throws ArgumentError ucmc.magnetization_value(PairingTestState())
end

@testset "Coupling and field accessors" begin
    geometry = two_orbital_geometry(2)
    for model in (ucmc.IsingModel([1.0, 2.0], [0.5, -0.25]),
                  ucmc.XYModel([1.0, 2.0], [0.5, -0.25]),
                  ucmc.ClockModel(4, [1.0, 2.0], [0.5, -0.25]),
                  ucmc.HeisenbergModel([1.0, 2.0], [0.5, -0.25]))
        @test ucmc.couplings(model) === model.J
        @test ucmc.fields(model) === model.h
        for site in 1:geometry.n_sites
            orbital = lu.site_to_orbital(site, geometry.unit_cell)
            @test ucmc.field_strength(model, geometry, site) === model.h[orbital]
        end
    end

    # cluster algorithms reject antiferromagnetic couplings for every model
    for model in (ucmc.IsingModel([1.0, -1.0], [0.0, 0.0]),
                  ucmc.XYModel([1.0, -1.0], [0.0, 0.0]),
                  ucmc.ClockModel(3, [1.0, -1.0], [0.0, 0.0]),
                  ucmc.HeisenbergModel([1.0, -1.0], [0.0, 0.0]))
        state = ucmc.initialize_state(model, geometry)
        for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
            @test_throws ArgumentError ucmc.step!(algorithm, model, geometry, state, 1.0)
        end
    end
end
