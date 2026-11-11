@testset "Ising state initialization" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    @test length(state.spins) == geometry.n_sites
    @test all(s in (-1, 1) for s in state.spins)
    @test state.magnetization == sum(state.spins)
    @test state.energy ≈ spin_energy(model, geometry, state.spins)
end

@testset "XY state and updates" begin
    geometry = two_orbital_geometry(2)
    model = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry)

    @test length(state.cosines) == geometry.n_sites
    @test length(state.sines) == geometry.n_sites
    @test state.cosines .^ 2 .+ state.sines .^ 2 ≈ ones(geometry.n_sites)
    @test state.energy ≈ xy_energy(model, geometry, state)
    @test state.magnetization_x ≈ sum(state.cosines)
    @test state.magnetization_y ≈ sum(state.sines)

    for _ in 1:50
        ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        @test state.energy ≈ xy_energy(model, geometry, state)
        @test state.magnetization_x ≈ sum(state.cosines)
        @test state.magnetization_y ≈ sum(state.sines)
        @test state.cosines .^ 2 .+ state.sines .^ 2 ≈ ones(geometry.n_sites)
    end

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.XYModel([1.0, 2.0], [0.5]), geometry
    )
end

@testset "Heisenberg state and updates" begin
    geometry = two_orbital_geometry(2)
    model = ucmc.HeisenbergModel([1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry)

    @test length(state.xs) == length(state.ys) == length(state.zs) == geometry.n_sites
    @test state.xs .^ 2 .+ state.ys .^ 2 .+ state.zs .^ 2 ≈ ones(geometry.n_sites)
    @test state.energy ≈ heisenberg_energy(model, geometry, state)
    @test state.magnetization_x ≈ sum(state.xs)
    @test state.magnetization_y ≈ sum(state.ys)
    @test state.magnetization_z ≈ sum(state.zs)

    for _ in 1:50
        ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        @test state.energy ≈ heisenberg_energy(model, geometry, state)
        @test state.magnetization_x ≈ sum(state.xs)
        @test state.magnetization_y ≈ sum(state.ys)
        @test state.magnetization_z ≈ sum(state.zs)
        @test state.xs .^ 2 .+ state.ys .^ 2 .+ state.zs .^ 2 ≈ ones(geometry.n_sites)
    end

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.HeisenbergModel([1.0, 2.0], [0.5]), geometry
    )
end
