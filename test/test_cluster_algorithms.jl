@testset "XY cluster updates and measurements" begin
    geometry = square_geometry(2)
    model = ucmc.XYModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        for _ in 1:10
            ucmc.step!(algorithm, model, geometry, state, 2.0)
            @test state.energy ≈ xy_energy(model, geometry, state)
        end
    end

    field_geometry = two_orbital_geometry(2)
    field_model = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        field_state = ucmc.initialize_state(field_model, field_geometry)
        for _ in 1:20
            ucmc.step!(algorithm, field_model, field_geometry, field_state, 2.0)
            @test field_state.energy ≈ xy_energy(field_model, field_geometry, field_state)
            @test field_state.magnetization_x ≈ sum(field_state.cosines)
            @test field_state.magnetization_y ≈ sum(field_state.sines)
        end
    end

    container = ucmc.MeasurementContainer(model, geometry, 4, 2)
    @test :magnetization_x in keys(container.data)
    @test :magnetization_y in keys(container.data)
    values = Dict(observable.name => observable.value(state) for observable in container.observables)
    @test values[:magnetization] ≈ hypot(state.magnetization_x, state.magnetization_y) / geometry.n_sites
    @test values[:magnetization_squared] ≈ values[:magnetization]^2
    susceptibility = only(filter(derived -> derived.names == (:susceptibility, :susceptibility_err), container.derived))
    @test susceptibility.depends_on == (
        :magnetization_x, :magnetization_x_squared,
        :magnetization_y, :magnetization_y_squared,
    )
    for _ in 1:4
        ucmc.measure!(container, state)
    end
    @test :susceptibility_x in keys(ucmc.analyze(container, 2.0, geometry.n_sites).derived)
end

@testset "XY cluster reflection" begin
    geometry = square_geometry(2)
    model = ucmc.XYModel([1.0, 1.0], [0.0])
    cosines = [1.0, 0.0, 1.0, 0.0]
    sines = [0.0, 1.0, 0.0, 1.0]
    state = ucmc.XYState(
        cosines, sines, xy_energy(model, geometry, cosines, sines), sum(cosines), sum(sines)
    )
    ucmc.apply_cluster!(model, geometry, state, [1], 0.0)

    @test state.cosines[1] == -1.0
    @test state.sines[1] == 0.0
    @test state.magnetization_x ≈ sum(state.cosines) atol = 1e-12
    @test state.magnetization_y ≈ sum(state.sines) atol = 1e-12
end

@testset "Heisenberg cluster updates" begin
    geometry = square_geometry(2)
    model = ucmc.HeisenbergModel([1.0, 1.0], [0.0])

    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        state = ucmc.initialize_state(model, geometry)
        for _ in 1:10
            ucmc.step!(algorithm, model, geometry, state, 2.0)
            @test state.energy ≈ heisenberg_energy(model, geometry, state)
        end
    end

    field_geometry = two_orbital_geometry(2)
    field_model = ucmc.HeisenbergModel([1.0, 2.0], [0.5, -0.25])
    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        field_state = ucmc.initialize_state(field_model, field_geometry)
        for _ in 1:20
            ucmc.step!(algorithm, field_model, field_geometry, field_state, 2.0)
            @test field_state.energy ≈ heisenberg_energy(field_model, field_geometry, field_state)
            @test field_state.magnetization_x ≈ sum(field_state.xs)
            @test field_state.magnetization_y ≈ sum(field_state.ys)
            @test field_state.magnetization_z ≈ sum(field_state.zs)
            @test field_state.xs .^ 2 .+ field_state.ys .^ 2 .+ field_state.zs .^ 2 ≈
                ones(field_geometry.n_sites)
        end
    end
end

@testset "Heisenberg cluster reflection" begin
    geometry = square_geometry(2)
    model = ucmc.HeisenbergModel([1.0, 1.0], [0.0])
    xs = [1.0, 0.0, 0.0, -1.0]
    ys = [0.0, 1.0, 0.0, 0.0]
    zs = [0.0, 0.0, 1.0, 0.0]
    state = ucmc.HeisenbergState(
        xs, ys, zs, heisenberg_energy(model, geometry, xs, ys, zs), sum(xs), sum(ys), sum(zs)
    )

    # reflecting x̂ about r̂ = x̂ flips it; a non-unit axis is normalized on entry
    ucmc.apply_cluster!(model, geometry, state, [1], (2.0, 0.0, 0.0))
    @test state.xs[1] == -1.0
    @test state.ys[1] == 0.0
    @test state.zs[1] == 0.0
    @test state.energy ≈ heisenberg_energy(model, geometry, state) atol = 1e-12
    @test state.magnetization_x ≈ sum(state.xs) atol = 1e-12
    @test state.magnetization_y ≈ sum(state.ys) atol = 1e-12
    @test state.magnetization_z ≈ sum(state.zs) atol = 1e-12

    # a spin perpendicular to the axis has p = 0 and is unchanged exactly
    ucmc.apply_cluster!(model, geometry, state, [3], (1.0, 0.0, 0.0))
    @test (state.xs[3], state.ys[3], state.zs[3]) == (0.0, 0.0, 1.0)

    # about r̂ = (x̂ + ŷ)/√2 the reflection maps -x̂ to +ŷ
    ucmc.apply_cluster!(model, geometry, state, [1], (1.0, 1.0, 0.0))
    @test state.xs[1] ≈ 0.0 atol = 1e-12
    @test state.ys[1] ≈ 1.0 atol = 1e-12
    @test state.zs[1] ≈ 0.0 atol = 1e-12
    @test state.energy ≈ heisenberg_energy(model, geometry, state) atol = 1e-12

    @test_throws ArgumentError ucmc.apply_cluster!(model, geometry, state, [1], (0.0, 0.0, 0.0))
end

@testset "Wolff updates keep invariants" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    algorithm = ucmc.WolffAlgorithm()

    for _ in 1:50
        result = ucmc.step!(algorithm, model, geometry, state, 2.0)
        @test result.cluster_size in 1:geometry.n_sites
        @test ucmc.diagnostic_values(algorithm, result) == (cluster_size = result.cluster_size,)
        @test length(state.spins) == geometry.n_sites
        @test all(s in (-1, 1) for s in state.spins)
        @test state.magnetization == sum(state.spins)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end

    field_model = ucmc.IsingModel([1.0, 1.0], [0.5])
    field_state = ucmc.initialize_state(field_model, geometry)
    for _ in 1:20
        result = ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
        @test result.cluster_size in 0:geometry.n_sites
        @test field_state.magnetization == sum(field_state.spins)
        @test field_state.energy ≈ spin_energy(field_model, geometry, field_state.spins)
    end
end

@testset "Wolff cluster_size counts only flipped clusters" begin
    # 4x4, not 2x2: on the smallest lattice every cluster spans the lattice
    # and always reaches a ghost, so nothing ever flips
    geometry = square_geometry(4)
    algorithm = ucmc.WolffAlgorithm()

    # with a strong field, ghost-attached clusters are never flipped and the
    # diagnostic reports zero exactly for those frozen steps
    model = ucmc.IsingModel([1.0, 1.0], [1.5])
    rng = Xoshiro(11)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    n_frozen = 0
    for _ in 1:2000
        before = copy(state.spins)
        result = ucmc.step!(algorithm, model, geometry, state, 1.0; rng = rng)
        if state.spins == before
            n_frozen += 1
            @test result.cluster_size == 0
        else
            @test result.cluster_size > 0
        end
    end
    @test 0 < n_frozen < 2000  # the field regime freezes most but not all steps

    # without a field there are no ghosts to freeze a cluster: every step
    # flips its (non-empty) cluster
    model0 = ucmc.IsingModel([1.0, 1.0], [0.0])
    state0 = ucmc.initialize_state(model0, geometry; rng = rng)
    for _ in 1:100
        @test ucmc.step!(algorithm, model0, geometry, state0, 1.0; rng = rng).cluster_size > 0
    end
end

@testset "Swendsen-Wang updates keep invariants" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    algorithm = ucmc.SwendsenWangAlgorithm()

    for _ in 1:50
        result = ucmc.step!(algorithm, model, geometry, state, 2.0)
        @test result.cluster_count in 1:geometry.n_sites
        @test result.updated_sites == geometry.n_sites
        @test 0.0 < result.largest_cluster_fraction <= 1.0
        diagnostics = ucmc.diagnostic_values(algorithm, result)
        @test propertynames(diagnostics) == (:cluster_count, :largest_cluster_fraction)
        @test diagnostics.cluster_count == result.cluster_count
        @test 0.0 < diagnostics.largest_cluster_fraction <= 1.0
        @test length(state.spins) == geometry.n_sites
        @test all(s in (-1, 1) for s in state.spins)
        @test state.magnetization == sum(state.spins)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end

    field_model = ucmc.IsingModel([1.0, 1.0], [0.5])
    field_state = ucmc.initialize_state(field_model, geometry)
    for _ in 1:20
        result = ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
        @test result.cluster_count in 1:geometry.n_sites
        @test field_state.magnetization == sum(field_state.spins)
        @test field_state.energy ≈ spin_energy(field_model, geometry, field_state.spins)
    end
end
