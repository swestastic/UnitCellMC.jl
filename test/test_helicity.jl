@testset "Helicity modulus measurement" begin
    geometry = square_geometry(3)
    N = geometry.n_sites

    # Fully aligned states: every bond is satisfied (dot product 1) and the
    # twist current vanishes, so the stiffness is (1/N) Σ_bonds J (Δr_d)^2,
    # which for unit nearest-neighbor bonds with J = 1 is 1 per direction.
    xy = ucmc.XYModel([1.0, 1.0], [0.0])
    xy_state = ucmc.initialize_state(xy, geometry)
    fill!(xy_state.cosines, 1.0)
    fill!(xy_state.sines, 0.0)
    obs, _ = ucmc.optional_observables(xy, geometry, Val(:helicity_modulus))
    samples = Dict(o.name => o.value(xy_state) for o in obs)
    @test samples[:helicity_stiffness] ≈ [1.0, 1.0]
    @test samples[:helicity_current] ≈ [0.0, 0.0] atol = 1e-14
    @test samples[:helicity_current_squared] ≈ [0.0, 0.0] atol = 1e-14

    # Anisotropic couplings map to the matching directions
    xy_aniso = ucmc.XYModel([1.0, 0.5], [0.0])
    obs_aniso, _ = ucmc.optional_observables(xy_aniso, geometry, Val(:helicity_modulus))
    samples_aniso = Dict(o.name => o.value(xy_state) for o in obs_aniso)
    @test samples_aniso[:helicity_stiffness] ≈ [1.0, 0.5]

    clock = ucmc.ClockModel(6, [1.0, 1.0], [0.0])
    clock_state = ucmc.initialize_state(clock, geometry)
    fill!(clock_state.spins, 0)
    obs_c, _ = ucmc.optional_observables(clock, geometry, Val(:helicity_modulus))
    @test Dict(o.name => o.value(clock_state) for o in obs_c)[:helicity_stiffness] ≈ [1.0, 1.0]

    heisenberg = ucmc.HeisenbergModel([1.0, 1.0], [0.0])
    heisenberg_state = ucmc.initialize_state(heisenberg, geometry)
    fill!(heisenberg_state.xs, 1.0)
    fill!(heisenberg_state.ys, 0.0)
    fill!(heisenberg_state.zs, 0.0)
    obs_h, _ = ucmc.optional_observables(heisenberg, geometry, Val(:helicity_modulus))
    @test Dict(o.name => o.value(heisenberg_state) for o in obs_h)[:helicity_stiffness] ≈ [1.0, 1.0]

    # Heisenberg stiffness factor is u_i · u_j - (u_i · ẑ)(u_j · ẑ): spins
    # aligned with the twist axis contribute nothing (a twist about z leaves
    # them invariant), and a state tilted by θ off the z axis carries sin²θ.
    fill!(heisenberg_state.xs, 0.0)
    fill!(heisenberg_state.zs, 1.0)
    samples_z = Dict(o.name => o.value(heisenberg_state) for o in obs_h)
    @test samples_z[:helicity_stiffness] ≈ [0.0, 0.0] atol = 1e-14
    @test samples_z[:helicity_current] ≈ [0.0, 0.0] atol = 1e-14
    s, c = 0.8, 0.6
    fill!(heisenberg_state.xs, s)
    fill!(heisenberg_state.zs, c)
    samples_tilted = Dict(o.name => o.value(heisenberg_state) for o in obs_h)
    @test samples_tilted[:helicity_stiffness] ≈ [s^2, s^2]
    @test samples_tilted[:helicity_current] ≈ [0.0, 0.0] atol = 1e-14

    # The derived quantity of an aligned state is exactly the stiffness
    container = ucmc.MeasurementContainer(xy, geometry, 4, 2; measurements = [:helicity_modulus])
    for _ in 1:4
        ucmc.measure!(container, xy_state)
    end
    derived = ucmc.analyze(container, 1.0, N).derived
    @test derived.helicity_modulus ≈ [1.0, 1.0]
    @test derived.helicity_modulus_err ≈ [0.0, 0.0] atol = 1e-14

    # Registry wiring: available for the vector-spin models, not Ising
    @test :helicity_modulus in ucmc.available_measurements(xy)
    @test :helicity_modulus in ucmc.available_measurements(clock)
    @test :helicity_modulus in ucmc.available_measurements(heisenberg)
    ising = ucmc.IsingModel([1.0, 1.0], [0.0])
    @test !(:helicity_modulus in ucmc.available_measurements(ising))
    @test_throws ArgumentError ucmc.MeasurementContainer(
        ising, geometry, 4, 2; measurements = [:helicity_modulus])
    @test_throws ArgumentError ucmc.optional_observables(ising, geometry, Val(:helicity_modulus))
end

@testset "Helicity modulus across the XY transition" begin
    # Deep in the quasi-ordered phase the stiffness is close to its T = 0
    # value of 1; deep in the disordered phase it vanishes.
    geometry = square_geometry(8)
    N = geometry.n_sites
    model = ucmc.XYModel([1.0, 1.0], [0.0])
    algorithm = ucmc.WolffAlgorithm()

    derived = map([0.4, 2.0]) do T
        rng = Xoshiro(7)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(
            model, geometry, 10_000, 20; measurements = [:helicity_modulus])
        ucmc.run_simulation(algorithm, model, geometry, state, container,
            ucmc.SimulationParameters(T, 1_000, 10_000, 0, 20); rng = rng)
        ucmc.analyze(container, T, N).derived
    end
    low, high = derived
    @test low.helicity_modulus[1] > 0.5
    @test low.helicity_modulus[2] > 0.5
    @test abs(high.helicity_modulus[1]) < 5 * high.helicity_modulus_err[1]
    @test abs(high.helicity_modulus[2]) < 5 * high.helicity_modulus_err[2]
end
