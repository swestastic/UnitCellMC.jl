using Test
using Random
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

    field_geometry = two_orbital_geometry(2)
    field_model = ucmc.XYModel([1.0, 2.0], [0.5, -0.25])
    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        field_state = ucmc.initialize_state(field_model, field_geometry)
        for _ in 1:20
            ucmc.step!(algorithm, field_model, field_geometry, field_state, 2.0)
            @test field_state.energy ≈ xy_energy(field_model, field_geometry, field_state.angles)
            @test field_state.magnetization_x ≈ sum(cos, field_state.angles)
            @test field_state.magnetization_y ≈ sum(sin, field_state.angles)
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
    angles = [0.0, π / 2, 0.0, π / 2]
    state = ucmc.XYState(
        angles, xy_energy(model, geometry, angles), sum(cos, angles), sum(sin, angles)
    )
    ucmc.apply_cluster!(model, geometry, state, [1], 0.0)

    @test state.angles[1] ≈ π
    @test cos(state.angles[1]) == -1.0
    @test state.magnetization_x ≈ sum(cos, state.angles) atol = 1e-12
    @test state.magnetization_y ≈ sum(sin, state.angles) atol = 1e-12
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
    for _ in 1:20
        result = ucmc.step!(algorithm, field_model, geometry, field_state, 2.0)
        @test result.cluster_size in 1:geometry.n_sites
        @test field_state.magnetization == sum(field_state.spins)
        @test field_state.energy ≈ spin_energy(field_model, geometry, field_state.spins)
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

@testset "steps_per_sweep" begin
    geometry = square_geometry(4)
    @test ucmc.steps_per_sweep(ucmc.MetropolisAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.WolffAlgorithm(), geometry) == 1
    @test ucmc.steps_per_sweep(ucmc.SwendsenWangAlgorithm(), geometry) == 1
end

@testset "MC_Sweep! update counts" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    n_sites = geometry.n_sites

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:acceptance_ratio])
    stats = [ucmc.UpdateStatistic(:acceptance_ratio)]
    ucmc.MC_Sweep!(ucmc.MetropolisAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == n_sites
    @test 0.0 <= ucmc.value(stats[1]) <= 1.0

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:cluster_size])
    stats = [ucmc.UpdateStatistic(:cluster_size)]
    ucmc.MC_Sweep!(ucmc.WolffAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == 1
    @test 1 <= ucmc.value(stats[1]) <= n_sites

    container = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:cluster_count, :largest_cluster_fraction])
    stats = [ucmc.UpdateStatistic(:cluster_count), ucmc.UpdateStatistic(:largest_cluster_fraction)]
    ucmc.MC_Sweep!(ucmc.SwendsenWangAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == 1
    @test stats[2].count == 1
    @test ucmc.value(stats[1]) >= 1
    @test 0.0 < ucmc.value(stats[2]) <= 1.0
end

@testset "Algorithms reject non-positive temperature" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    for alg in (ucmc.MetropolisAlgorithm(), ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        for T in (0.0, -1.0, -Inf)
            @test_throws ArgumentError ucmc.step!(alg, model, geometry, state, T)
        end
    end
end

@testset "run_simulation fills bins" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    parameters = ucmc.SimulationParameters(2.0, 5, 20, 2, 4)

    container = ucmc.MeasurementContainer(model, geometry, 20, 4; diagnostics = [:acceptance_ratio])
    result = ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    @test result === container
    @test length(container.data.energy) == 4
    @test length(container.data.acceptance_ratio) == 4
    @test all(v -> 0.0 <= v <= 1.0, container.data.acceptance_ratio)

    # empty diagnostics path
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(model, geometry, 20, 4)
    ucmc.run_simulation(ucmc.WolffAlgorithm(), model, geometry, state, container, parameters)
    @test length(container.data.energy) == 4
    @test !haskey(pairs(container.data), :acceptance_ratio)
end

@testset "sweep_Ts end-to-end" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    Ts = [3.0, 1.0]
    root = mktempdir()

    sweep_results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, state,
        3, 6, 1, 3, Ts;
        simulated_annealing = true,
        diagnostics = [:cluster_size],
        root = root
    )

    @test length(sweep_results) == length(Ts)
    @test all(r -> haskey(r.primary, :energy), sweep_results)
    @test all(r -> haskey(r.primary, :cluster_size), sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry.lattice.L; root = root)
    @test isdir(dir)
    @test isfile(joinpath(dir, "measurement_results.csv"))
end

@testset "merge_containers pools run bins" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    parameters = ucmc.SimulationParameters(2.0, 2, 8, 1, 4)

    containers = map(1:2) do _
        state = ucmc.initialize_state(model, geometry)
        container = ucmc.MeasurementContainer(model, geometry, 8, 4; diagnostics = [:acceptance_ratio])
        ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    end

    merged = ucmc.merge_containers(containers)
    @test length(merged.data.energy) == 8 # 2 runs * 4 bins per run
    @test merged.data.energy == vcat(containers[1].data.energy, containers[2].data.energy)
    @test length(merged.data.acceptance_ratio) == 8

    # analysis works on the pooled bins
    results = ucmc.analyze(merged, 2.0, geometry.n_sites)
    @test haskey(results.primary, :energy)

    # mismatched containers and empty input are rejected
    mismatched = ucmc.MeasurementContainer(model, geometry, 8, 4; diagnostics = [:cluster_size])
    @test_throws ArgumentError ucmc.merge_containers([first(containers), mismatched])
    @test_throws ArgumentError ucmc.merge_containers(ucmc.MeasurementContainer[])
end

@testset "sweep_Ts ensemble averaging" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    Ts = [3.0, 1.0]
    root = mktempdir()

    sweep_results = ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, state,
        3, 6, 1, 3, Ts;
        simulated_annealing = false,
        n_runs = 2,
        diagnostics = [:acceptance_ratio],
        root = root
    )

    @test length(sweep_results) == length(Ts)
    @test all(r -> haskey(r.primary, :energy), sweep_results)
    @test all(r -> haskey(r.primary, :acceptance_ratio), sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry.lattice.L; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))

    @test_throws ArgumentError ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, state,
        3, 6, 1, 3, Ts; n_runs = 0, root = root
    )
end

# AR(1) process x[i] = φ x[i-1] + √(1-φ²) ξ: ρ(k) = φ^k and
# τ_int = 1/2 + Σ_{k≥1} φ^k = (1 + φ) / (2(1 - φ)).
function ar1_series(phi, n)
    x = zeros(n)
    x[1] = randn()
    for i in 2:n
        x[i] = phi * x[i - 1] + sqrt(1 - phi^2) * randn()
    end
    return x
end
ar1_tau(phi) = (1 + phi) / (2 * (1 - phi))

@testset "autocorrelation estimators" begin
    Random.seed!(42)

    # white noise: ρ(0) = 1, ρ(k > 0) ≈ 0, τ_int = 1/2
    x = randn(20_000)
    lags, acf = ucmc.autocorrelation_function(x; max_lag = 50)
    @test lags == 0:50
    @test acf[1] == 1.0
    @test all(abs.(acf[2:end]) .< 5 / sqrt(length(x)))
    r = ucmc.autocorrelation(x)
    @test r isa ucmc.AutocorrelationResult
    @test r.status === :resolved
    @test r.n == length(x)
    @test r.tau ≈ 0.5 atol = 0.06   # empirical spread at n = 20_000 is ≈ 0.013
    @test r.window >= 1
    @test length(r.acf) == length(r.lags) == r.window + 1
    @test ucmc.integrated_autocorrelation_time(x) == r.tau

    # AR(1): recovers the exact ρ(k) and τ_int within the Madras–Sokal error
    for phi in (0.5, 0.8, 0.95)
        y = ar1_series(phi, 200_000)
        _, acf = ucmc.autocorrelation_function(y; max_lag = 5)
        @test acf[2:end] ≈ phi .^ (1:5) atol = 0.02
        r = ucmc.autocorrelation(y)
        @test r.status === :resolved
        @test abs(r.tau - ar1_tau(phi)) < 4 * r.tau_err
        @test r.tau_err < 0.1 * r.tau
        @test r.window >= 6 * r.tau   # self-consistent window condition
    end

    # anticorrelated series (φ < 0): τ_int = (1 + φ)/(2(1 - φ)) < 1/2
    r = ucmc.autocorrelation(ar1_series(-0.6, 50_000))
    @test r.status === :anticorrelated
    @test ucmc.tau_lower_bound(r) == 0.5

    # too short to resolve: a slow drift (ρ(k) ≈ 1 for every lag reachable)
    # never satisfies K ≥ 6 τ(K); the result is NaN with a lower bound
    ramp = collect(1.0:1_000.0)
    r = ucmc.autocorrelation(ramp)
    @test r.status === :unresolved
    @test isnan(r.tau) && isnan(r.tau_err)
    @test r.window == 0
    @test length(r.acf) == 1_000 ÷ 10 + 1
    @test ucmc.tau_lower_bound(r) ≈ 0.5 + sum(r.acf[2:end])
    @test ucmc.tau_lower_bound(r) > 50
    @test ucmc.autocorrelation(ramp; max_lag = 999, window = 1.0).status === :resolved

    # degenerate series never throw
    r = ucmc.autocorrelation(fill(1.0, 100))
    @test r.status === :constant
    @test isnan(r.tau) && isempty(r.acf) && r.window == 0
    @test ucmc.autocorrelation([1.0]).status === :insufficient_data
    @test ucmc.autocorrelation(Any[:a, :b, :c]).status === :unsupported
    @test_throws ArgumentError ucmc.autocorrelation_function(fill(2.0, 10))
    @test_throws ArgumentError ucmc.autocorrelation(x; window = 0)

    # vector-valued series: the slowest non-constant component is reported
    slow = ar1_series(0.8, 50_000)
    fast = randn(50_000)
    vs = [[fast[i], 3.0, slow[i]] for i in eachindex(slow)]
    rv = ucmc.autocorrelation(vs)
    @test rv.n == length(vs)
    @test rv.tau ≈ ucmc.autocorrelation(slow).tau
    @test ucmc.autocorrelation([[1.0, 2.0] for _ in 1:10]).status === :constant
    @test_throws ArgumentError ucmc.autocorrelation([[1.0], [1.0, 2.0]])

    # independent chains: lagged products never cross chain boundaries.
    # Two chains with different constant offsets look strongly correlated
    # when naively concatenated.
    a = randn(5_000)
    b = randn(5_000) .+ 3.0
    merged = vcat(a, b)
    @test ucmc.autocorrelation(merged).status === :unresolved
    rc = ucmc.autocorrelation(merged .- vcat(fill(0.0, 5_000), fill(3.0, 5_000));
        chain_lengths = [5_000, 5_000])
    @test rc.tau ≈ 0.5 atol = 4 * rc.tau_err
    @test_throws ArgumentError ucmc.autocorrelation(merged; chain_lengths = [10, 10])
    # with chains, the sum excludes the cross-boundary pairs exactly
    z = randn(200)
    _, acf_chained = ucmc.autocorrelation_function(z; max_lag = 3, chain_lengths = [100, 100])
    c = z .- sum(z) / 200
    expected = sum(c[i] * c[i + 1] for i in [1:99; 101:199]) / sum(abs2, c)
    @test acf_chained[2] ≈ expected
end

@testset "recommended_binning" begin
    # τ = 5 sweeps → measure every 5 sweeps (n_unmeasured = 4); thinned
    # τ_meas ≤ 1/2 + 4.5/5 = 1.4 → bin_size = ceil(10 * 1.4) = 14 measurements
    @test ucmc.recommended_binning(5.0) == (bin_size = 14, n_unmeasured = 4)
    # measuring every sweep: bins span bin_factor × τ
    @test ucmc.recommended_binning(20.0; n_unmeasured = 0) == (bin_size = 200, n_unmeasured = 0)
    # uncorrelated data needs no decorrelation sweeps
    @test ucmc.recommended_binning(0.5) == (bin_size = 5, n_unmeasured = 0)
    @test ucmc.recommended_binning(0.2).n_unmeasured == 0
    @test ucmc.recommended_binning(5.0; bin_factor = 20).bin_size == 28
    for bad in (NaN, Inf, 0.0, -1.0)
        @test_throws ArgumentError ucmc.recommended_binning(bad)
    end
    @test_throws ArgumentError ucmc.recommended_binning(5.0; n_unmeasured = -1)

    # the thinned-τ bound holds for a real (thinned) AR(1) series
    Random.seed!(7)
    y = ar1_series(0.9, 400_000)
    tau = ucmc.autocorrelation(y).tau
    for s in (2, 5, 10)
        @test ucmc.autocorrelation(y[1:s:end]).tau <= 0.5 + (tau - 0.5) / s + 0.1
    end
end

@testset "time series recording and analysis" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    parameters = ucmc.SimulationParameters(2.0, 5, 40, 1, 4)
    container = ucmc.MeasurementContainer(model, geometry, 40, 4;
        measurements = [:raw_correlation], diagnostics = [:acceptance_ratio], time_series = :all)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)

    @test keys(container.time_series) == keys(container.data)
    @test all(length(series) == 40 for series in container.time_series)
    @test container.time_series.correlation isa Vector{Vector{Float64}}
    # the binned data are exactly the bin means of the recorded time series
    @test container.data.energy ≈ [sum(container.time_series.energy[(b - 1) * 10 + 1:b * 10]) / 10 for b in 1:4]
    @test all(container.time_series.energy_squared .≈ container.time_series.energy .^ 2)

    results = ucmc.analyze(container, 2.0, geometry.n_sites)
    @test Set(keys(results.autocorrelation)) == Set(keys(container.data))
    @test all(r -> r isa ucmc.AutocorrelationResult, values(results.autocorrelation))
    @test all(r -> r.n == 40, values(results.autocorrelation))

    # merged runs keep their chain boundaries
    other = ucmc.MeasurementContainer(model, geometry, 40, 4;
        measurements = [:raw_correlation], diagnostics = [:acceptance_ratio], time_series = :all)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, other, parameters)
    merged = ucmc.merge_containers([container, other])
    @test merged.chain_lengths == [40, 40]
    @test merged.time_series.energy == vcat(container.time_series.energy, other.time_series.energy)
    @test ucmc.merge_containers([merged, other]).chain_lengths == [40, 40, 40]
    @test all(r -> r.n == 80, values(ucmc.analyze(merged, 2.0, geometry.n_sites).autocorrelation))
end

@testset "time_series recording options" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    parameters = ucmc.SimulationParameters(2.0, 2, 20, 0, 4)
    build(ts) = ucmc.MeasurementContainer(model, geometry, 20, 4;
        measurements = [:raw_correlation], diagnostics = [:acceptance_ratio], time_series = ts)
    run!(c) = ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry,
        ucmc.initialize_state(model, geometry), c, parameters)

    scalar_names = (:energy, :energy_squared, :magnetization, :magnetization_squared, :acceptance_ratio)

    # default: scalars only, the correlation vector is not stored
    default = run!(ucmc.MeasurementContainer(model, geometry, 20, 4;
        measurements = [:raw_correlation], diagnostics = [:acceptance_ratio]))
    @test Set(keys(default.time_series)) == Set(scalar_names)
    @test !haskey(default.time_series, :correlation)
    @test haskey(default.data, :correlation)   # binned data are unaffected
    @test Set(keys(build(:scalars).time_series)) == Set(scalar_names)

    @test keys(build(:all).time_series) == keys(build(:all).data)
    @test keys(build(true).time_series) == keys(build(:all).data)
    @test isempty(keys(build(:none).time_series))
    @test isempty(keys(build(false).time_series))
    @test keys(build([:energy, :correlation]).time_series) == (:energy, :correlation)

    @test_throws ArgumentError build(:everything)
    @test_throws ArgumentError build([:not_measured])
    @test_throws ArgumentError build("energy")

    # disabling time series still gives binned results, just no autocorrelations
    off = run!(build(:none))
    results = ucmc.analyze(off, 2.0, geometry.n_sites)
    @test haskey(results.primary, :energy)
    @test haskey(results.primary, :correlation)
    @test isempty(results.autocorrelation)
    @test ucmc.autocorrelation_rows(results) === nothing

    # only the selected names are analyzed
    selected = run!(build([:energy]))
    @test keys(ucmc.analyze(selected, 2.0, geometry.n_sites).autocorrelation) == Set([:energy])

    # merging respects the selection and rejects mismatched selections
    merged_off = ucmc.merge_containers([off, run!(build(:none))])
    @test isempty(keys(merged_off.time_series))
    @test isempty(merged_off.chain_lengths)
    @test length(merged_off.data.energy) == 8
    @test_throws ArgumentError ucmc.merge_containers([off, run!(build(:all))])

    # sweep_Ts passes the option through
    root = mktempdir()
    sweep = ucmc.sweep_Ts(ucmc.WolffAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        2, 8, 0, 2, [2.0]; n_runs = 2, measurements = [:raw_correlation], time_series = :none, root = root)
    @test isempty(sweep[1].autocorrelation)
    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry.lattice.L; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))
    @test !isfile(joinpath(dir, "autocorrelation_results.csv"))
    sweep = ucmc.sweep_Ts(ucmc.WolffAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        2, 8, 0, 2, [2.0]; n_runs = 2, measurements = [:raw_correlation], root = root)
    @test haskey(sweep[1].autocorrelation, :energy)
    @test !haskey(sweep[1].autocorrelation, :correlation)
end

@testset "autotune_binning" begin
    Random.seed!(11)
    geometry = square_geometry(6)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    Ts = [5.0, 2.3]

    tuning = ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, Ts;
        n_thermalization = 200, n_pilot = 4_000,
        measurements = [:abs_magnetization],
        observables = [:energy, :abs_magnetization],
    )
    @test length(tuning.per_temperature) == length(Ts)
    @test all(d -> Set(keys(d)) == Set([:energy, :abs_magnetization]), tuning.per_temperature)
    @test tuning.T in Ts
    @test tuning.observable in (:energy, :abs_magnetization)
    # the recommendation is driven by the slowest observable/temperature
    @test tuning.tau == maximum(ucmc.tau_lower_bound(r) for d in tuning.per_temperature for r in values(d))
    @test (bin_size = tuning.bin_size, n_unmeasured = tuning.n_unmeasured) ==
        ucmc.recommended_binning(tuning.tau)
    # Metropolis dynamics are correlated over more than one sweep, and much
    # slower close to T_c ≈ 2.269 than deep in the paramagnet (critical slowing down)
    @test tuning.tau > 1.0
    @test tuning.per_temperature[2][:abs_magnetization].tau > tuning.per_temperature[1][:abs_magnetization].tau

    # cluster updates decorrelate the order parameter much faster than Metropolis at T_c
    wolff = ucmc.autotune_binning(
        ucmc.SwendsenWangAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry), 2.3;
        n_thermalization = 200, n_pilot = 4_000,
        measurements = [:abs_magnetization], observables = [:abs_magnetization],
    )
    @test wolff.tau < tuning.per_temperature[2][:abs_magnetization].tau

    # with a fixed n_unmeasured only the bin size is tuned
    fixed = ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, 3.0;
        n_thermalization = 50, n_pilot = 2_000, n_unmeasured = 0,
    )
    @test fixed.n_unmeasured == 0
    @test fixed.bin_size == ceil(Int, 10 * fixed.tau)

    @test_throws ArgumentError ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0;
        n_pilot = 100, observables = [:not_measured],
    )
    @test_throws ArgumentError ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, Float64[]
    )
    @test_throws ArgumentError ucmc.autotune_binning(
        ucmc.MetropolisAlgorithm(), model, geometry, state, 2.0; observables = Symbol[]
    )

    # by default only scalar observables are tracked in the pilot; vectors on request
    pilot = ucmc.autotune_binning(
        ucmc.WolffAlgorithm(), model, geometry, state, 2.5;
        n_thermalization = 10, n_pilot = 501, measurements = [:raw_correlation],
    )
    @test !haskey(pilot.per_temperature[1], :correlation)
    @test haskey(pilot.per_temperature[1], :energy)
    pilot = ucmc.autotune_binning(
        ucmc.WolffAlgorithm(), model, geometry, state, 2.5;
        n_thermalization = 10, n_pilot = 501, measurements = [:raw_correlation],
        observables = [:correlation],
    )
    @test collect(keys(pilot.per_temperature[1])) == [:correlation]
end

@testset "autocorrelation results are saved" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    root = mktempdir()
    Ts = [3.0, 2.0]

    sweep_results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, state,
        10, 40, 0, 4, Ts;
        n_runs = 2, diagnostics = [:cluster_size], root = root
    )
    @test all(r -> haskey(r.autocorrelation, :energy), sweep_results)
    @test all(r -> r.autocorrelation[:energy].n == 80, sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry.lattice.L; root = root)
    path = joinpath(dir, "autocorrelation_results.csv")
    @test isfile(path)
    lines = readlines(path)
    @test lines[1] == "temperature,observable,tau,tau_err,window,n_samples,status"
    n_names = length(sweep_results[1].autocorrelation)
    @test length(lines) == 1 + length(Ts) * n_names

    rows = ucmc.autocorrelation_rows(sweep_results[1])
    @test rows.observable == sort(rows.observable)
    @test ucmc.autocorrelation_rows((primary = Dict(), derived = NamedTuple())) === nothing

    single = ucmc.save_results(model, ucmc.WolffAlgorithm(), geometry.lattice.L, 3.0,
        sweep_results[1], ucmc.SimulationParameters(3.0, 10, 40, 0, 4); root = root)
    @test isfile(joinpath(single, "autocorrelation_results.csv"))
end
