using Test
import UnitCellMC as ucmc
import LatticeUtilities as lu

function square_geometry(L::Int)
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0]],
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    bond_x = lu.Bond(orbitals = (1, 1), displacement = [1, 0])
    bond_y = lu.Bond(orbitals = (1, 1), displacement = [0, 1])
    return ucmc.Geometry(unit_cell, lattice, [bond_x, bond_y])
end

function two_orbital_geometry(L::Int)
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0], [0.5, 0.5]],
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1, 0])
    bond_2 = lu.Bond(orbitals = (2, 2), displacement = [1, 0])
    return ucmc.Geometry(unit_cell, lattice, [bond_1, bond_2])
end

function spin_energy(model, geometry, spins)
    energy = zero(promote_type(eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * spins[i] * spins[j]
        end
        energy += -ucmc.field_strength(model, geometry, i) * spins[i]
    end
    return energy
end

function xy_energy(model, geometry, angles)
    energy = zero(promote_type(eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * cos(angles[i] - angles[j])
        end
        energy += -ucmc.field_strength(model, geometry, i) * cos(angles[i])
    end
    return energy
end

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

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.IsingModel([1.0, 2.0], [0.5]), geometry
    )
    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.IsingModel([1.0, 2.0], [0.5, -0.25, 0.0]), geometry
    )
end

@testset "SimulationParameters validation" begin
    @test_throws ArgumentError ucmc.SimulationParameters(0.0, 0, 0, 0, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, -1, 0, 0, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, -1, 0, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, 0, -1, 1)
    @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, 0, 0, 0)
end

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

    @test length(state.angles) == geometry.n_sites
    @test state.energy ≈ xy_energy(model, geometry, state.angles)
    @test state.magnetization_x ≈ sum(cos, state.angles)
    @test state.magnetization_y ≈ sum(sin, state.angles)

    for _ in 1:50
        ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        @test state.energy ≈ xy_energy(model, geometry, state.angles)
        @test state.magnetization_x ≈ sum(cos, state.angles)
        @test state.magnetization_y ≈ sum(sin, state.angles)
    end

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.XYModel([1.0, 2.0], [0.5]), geometry
    )
end

@testset "XY cluster updates and measurements" begin
    geometry = square_geometry(2)
    model = ucmc.XYModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        for _ in 1:10
            ucmc.step!(algorithm, model, geometry, state, 2.0)
            @test state.energy ≈ xy_energy(model, geometry, state.angles)
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
    state = ucmc.XYState([0.0, π / 2, 0.0, π / 2], 0.0, 0.0, 0.0)
    ucmc.apply_cluster!(model, geometry, state, [1], 0.0)

    @test state.angles[1] ≈ π
    @test cos(state.angles[1]) == -1.0
    @test state.magnetization_x ≈ sum(cos, state.angles)
    @test state.magnetization_y ≈ sum(sin, state.angles)
end

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
    @test_throws ArgumentError ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
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
    @test_throws ArgumentError ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
end

@testset "MeasurementContainer and analysis" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(model, geometry, 4, 2; diagnostics = [:acceptance_ratio])

    for _ in 1:4
        result = ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        ucmc.measure!(container, state; diagnostics = (acceptance_ratio = result.accepted ? 1.0 : 0.0,))
    end

    @test length(container.data.energy) == 2
    @test container.bin_count == 0
    result = ucmc.analyze(container, 2.0, geometry.n_sites)
    @test haskey(result.primary, :energy)
    @test haskey(result.primary, :magnetization)

    sw_container = ucmc.MeasurementContainer(
        model,
        geometry,
        4,
        2;
        diagnostics = [:cluster_count, :largest_cluster_fraction],
    )
    for _ in 1:4
        result = ucmc.step!(ucmc.SwendsenWangAlgorithm(), model, geometry, state, 2.0)
        ucmc.measure!(sw_container, state; diagnostics = (
            cluster_count = result.cluster_count,
            largest_cluster_fraction = result.largest_cluster_fraction,
        ))
    end
    @test haskey(sw_container.data, :cluster_count)
    @test haskey(sw_container.data, :largest_cluster_fraction)
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

@testset "Sweep save edge cases" begin
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()
    L = [2, 2]
    root = mktempdir()

    empty_dir = ucmc.save_sweep_results([], Float64[], model, algorithm, L, 0, 0, 0, 1; root = root)
    @test isdir(empty_dir)
    @test isfile(joinpath(empty_dir, "metadata.txt"))
    @test !isfile(joinpath(empty_dir, "measurement_results.csv"))
    @test !isfile(joinpath(empty_dir, "correlation_results.csv"))

    @test_throws ArgumentError ucmc.save_sweep_results([Dict()], [1.0, 2.0], model, algorithm, L, 0, 0, 0, 1; root = root)
end
