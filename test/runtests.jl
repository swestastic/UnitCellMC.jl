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

function spin_energy(model, geometry, spins)
    energy = zero(promote_type(eltype(model.J), typeof(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * spins[i] * spins[j]
        end
        energy += -model.h * spins[i]
    end
    return energy
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
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
    state = ucmc.initialize_state(model, geometry)

    @test length(state.spins) == geometry.n_sites
    @test all(s in (-1, 1) for s in state.spins)
    @test state.magnetization == sum(state.spins)
    @test state.energy ≈ spin_energy(model, geometry, state.spins)
end

@testset "Metropolis updates keep invariants" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
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
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
    state = ucmc.initialize_state(model, geometry)
    algorithm = ucmc.WolffAlgorithm()

    for _ in 1:50
        result = ucmc.step!(algorithm, model, geometry, state, 2.0)
        @test result.cluster_size in 1:geometry.n_sites
        @test ucmc.diagnostic_name(algorithm) == :cluster_size
        @test ucmc.diagnostic_value(algorithm, result) == result.cluster_size
        @test length(state.spins) == geometry.n_sites
        @test all(s in (-1, 1) for s in state.spins)
        @test state.magnetization == sum(state.spins)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end

    field_model = ucmc.IsingModel([1.0, 1.0], 0.5)
    field_state = ucmc.initialize_state(field_model, geometry)
    @test_throws ArgumentError ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
end

@testset "Swendsen-Wang updates keep invariants" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
    state = ucmc.initialize_state(model, geometry)
    algorithm = ucmc.SwendsenWangAlgorithm()

    for _ in 1:50
        result = ucmc.step!(algorithm, model, geometry, state, 2.0)
        @test result.cluster_count in 1:geometry.n_sites
        @test result.updated_sites == geometry.n_sites
        @test ucmc.diagnostic_name(algorithm) == :cluster_count
        @test ucmc.diagnostic_value(algorithm, result) == result.cluster_count
        @test length(state.spins) == geometry.n_sites
        @test all(s in (-1, 1) for s in state.spins)
        @test state.magnetization == sum(state.spins)
        @test state.energy ≈ spin_energy(model, geometry, state.spins)
    end

    field_model = ucmc.IsingModel([1.0, 1.0], 0.5)
    field_state = ucmc.initialize_state(field_model, geometry)
    @test_throws ArgumentError ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
end

@testset "MeasurementContainer and analysis" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
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
end

@testset "Model/geometry validation" begin
    geometry = square_geometry(2)
    valid_model = ucmc.IsingModel([1.0, 1.0], 0.0)

    @test_throws ArgumentError ucmc.initialize_state(ucmc.IsingModel([1.0], 0.0), geometry)

    bad_geometry = ucmc.Geometry(
        geometry.unit_cell,
        geometry.lattice,
        [lu.Bond(orbitals = (1, 1), displacement = [1, 0])],
    )
    @test_throws ArgumentError ucmc.validate_model_geometry(valid_model, bad_geometry)

    @test_throws ArgumentError ucmc.bond_strength(valid_model, geometry, 100)
end

@testset "Optional correlation observable" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(
        model,
        geometry,
        4,
        2;
        measurements = [:correlation],
        diagnostics = [:acceptance_ratio],
    )

    for _ in 1:4
        result = ucmc.step!(ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0)
        ucmc.measure!(container, state; diagnostics = (acceptance_ratio = result.accepted ? 1.0 : 0.0,))
    end

    @test :correlation in keys(container.data)
    result = ucmc.analyze(container, 2.0, geometry.n_sites)
    @test haskey(result.primary, :correlation)
    @test haskey(result.derived, :correlation_connected)
end

@testset "Sweep save edge cases" begin
    model = ucmc.IsingModel([1.0, 1.0], 0.0)
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
