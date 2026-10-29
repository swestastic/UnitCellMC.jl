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

function clock_energy(model, geometry, spins)
    q = model.q
    energy = zero(promote_type(Float64, eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * cos(2π * (spins[i] - spins[j]) / q)
        end
        energy += -ucmc.field_strength(model, geometry, i) * cos(2π * spins[i] / q)
    end
    return energy
end

clock_mx(q, spins) = sum(n -> cos(2π * n / q), spins; init = 0.0)
clock_my(q, spins) = sum(n -> sin(2π * n / q), spins; init = 0.0)

function clock_invariants_hold(model, geometry, state)
    q = model.q
    return all(in(0:(q - 1)), state.spins) &&
        isapprox(state.energy, clock_energy(model, geometry, state.spins); atol = 1e-10) &&
        isapprox(state.magnetization_x, clock_mx(q, state.spins); atol = 1e-10) &&
        isapprox(state.magnetization_y, clock_my(q, state.spins); atol = 1e-10)
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

@testset "SimulationParameters and MeasurementContainer agree on binning" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])

    # (n_measurements, n_bins) pairs: both constructors accept or both reject
    valid = [(2, 2), (8, 4), (20, 4), (9, 3)]
    invalid = [(0, 2), (-4, 2), (4, 1), (4, 0), (1, 2), (10, 4), (95, 10)]
    for (n_measurements, n_bins) in valid
        @test ucmc.SimulationParameters(1.0, 0, n_measurements, 0, n_bins) isa ucmc.SimulationParameters
        @test ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins).bin_size ==
            n_measurements ÷ n_bins
    end
    for (n_measurements, n_bins) in invalid
        @test_throws ArgumentError ucmc.SimulationParameters(1.0, 0, n_measurements, 0, n_bins)
        @test_throws ArgumentError ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins)
    end
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
        @test state.energy ≈ xy_energy(model, geometry, state.angles)
        @test state.magnetization_x ≈ sum(cos, state.angles)
        @test state.magnetization_y ≈ sum(sin, state.angles)
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
end

@testset "Glauber Ising and XY match exact results" begin
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

    parameters = ucmc.SimulationParameters(T, 200, 20_000, 0, 20)
    rng = Xoshiro(2024)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    container = ucmc.MeasurementContainer(model, geometry, 20_000, 20)
    ucmc.run_simulation(ucmc.GlauberAlgorithm(), model, geometry, state, container, parameters; rng = rng)
    result = ucmc.analyze(container, T, N)
    e, e_err = result.primary[:energy]
    m, m_err = result.primary[:magnetization]
    @test abs(e - exact_e) < 4 * e_err
    @test abs(m - exact_m) < 4 * m_err

    # XY on a periodic two-site chain: exact moments by quadrature (the
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
        e = xy_energy(xy, chain, [a, b])
        w = exp(-e / T)
        Z += w
        E += w * e
        Mx += w * (cos(a) + cos(b))
    end
    exact_e, exact_mx = E / (2Z), Mx / (2Z)

    parameters = ucmc.SimulationParameters(T, 200, 20_000, 0, 20)
    rng = Xoshiro(7)
    state = ucmc.initialize_state(xy, chain; rng = rng)
    container = ucmc.MeasurementContainer(xy, chain, 20_000, 20)
    ucmc.run_simulation(ucmc.GlauberAlgorithm(), xy, chain, state, container, parameters; rng = rng)
    result = ucmc.analyze(container, T, chain.n_sites)
    e, e_err = result.primary[:energy]
    mx, mx_err = result.primary[:magnetization_x]
    @test abs(e - exact_e) < 4 * e_err
    @test abs(mx - exact_mx) < 4 * mx_err
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

struct PairingTestModel <: ucmc.AbstractModel{Float64} end
struct PairingTestState <: ucmc.AbstractState end

@testset "Model/state pairing" begin
    geometry = square_geometry(2)
    ising = ucmc.IsingModel([1.0, 1.0], [0.0])
    xy = ucmc.XYModel([1.0, 1.0], [0.0])
    clock = ucmc.ClockModel(4, [1.0, 1.0], [0.0])

    @test ucmc.state_type(ising) == ucmc.IsingState
    @test ucmc.state_type(xy) == ucmc.XYState
    @test ucmc.state_type(clock) == ucmc.ClockState
    @test_throws ArgumentError ucmc.state_type(PairingTestModel())

    # Models without their own methods fail loudly instead of falling back
    # to another model's physics.
    @test_throws ArgumentError ucmc.initialize_state(PairingTestModel(), geometry)

    ising_state = ucmc.initialize_state(ising, geometry)
    xy_state = ucmc.initialize_state(xy, geometry)
    @test ising_state isa ucmc.state_type(ising)
    @test xy_state isa ucmc.state_type(xy)
    @test ucmc.initialize_state(clock, geometry) isa ucmc.state_type(clock)

    @test isnothing(ucmc.validate_model_state(ising, ising_state))
    @test_throws ArgumentError ucmc.validate_model_state(ising, xy_state)
    @test_throws ArgumentError ucmc.validate_model_state(xy, ising_state)
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
    @test ucmc.magnetization_value(ucmc.initialize_state(clock, geometry)) isa Real
    @test_throws ArgumentError ucmc.magnetization_value(PairingTestState())
end

@testset "Single-site update contract" begin
    geometry = square_geometry(2)
    models_and_energy = (
        (ucmc.IsingModel([1.0, 1.0], [0.5]), spin_energy),
        (ucmc.XYModel([1.0, 1.0], [0.5]), xy_energy),
        (ucmc.ClockModel(4, [1.0, 1.0], [0.5]), clock_energy),
    )
    mag_totals(s::ucmc.IsingState) = (s.magnetization,)
    mag_totals(s) = (s.magnetization_x, s.magnetization_y)

    for (model, energy) in models_and_energy, algorithm in
            (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
             ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()),
             ucmc.GlauberAlgorithm(ucmc.SequentialProposal()))
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
            values = state isa ucmc.XYState ? state.angles : state.spins
            @test state.energy ≈ energy(model, geometry, values) atol = 1e-10
        end
    end
end

@testset "Proposal schemes" begin
    geometry = square_geometry(3)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    rng = Xoshiro(3)

    @test ucmc.MetropolisAlgorithm().proposal isa ucmc.RandomProposal
    @test ucmc.GlauberAlgorithm().proposal isa ucmc.RandomProposal
    @test ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()).proposal isa
        ucmc.SequentialProposal
    @test ucmc.GlauberAlgorithm(ucmc.SequentialProposal()).proposal isa
        ucmc.SequentialProposal

    # the default scheme is stateless
    state = ucmc.initialize_state(model, geometry; rng = rng)
    @test isnothing(ucmc.algorithm_workspace(ucmc.MetropolisAlgorithm(), model, geometry, state))

    sequential_algorithms = (ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()),
                             ucmc.GlauberAlgorithm(ucmc.SequentialProposal()))

    for algorithm in sequential_algorithms
        workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
        @test workspace isa ucmc.SequentialProposalWorkspace

        # sites are visited in site-index order, wrapping after a full sweep
        sites = [ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace).site
                 for _ in 1:geometry.n_sites]
        @test sites == collect(1:geometry.n_sites)
        @test ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace).site == 1

        # a stateful scheme without its workspace is an explicit error
        @test_throws ArgumentError ucmc.propose(rng, algorithm, model, geometry, state)
        @test_throws ArgumentError ucmc.step!(algorithm, model, geometry, state, 2.0; rng = rng)

        # running totals stay exact through sequential sweeps
        seq_state = ucmc.initialize_state(model, geometry; rng = rng)
        seq_workspace = ucmc.algorithm_workspace(algorithm, model, geometry, seq_state)
        for _ in 1:(3 * geometry.n_sites)
            ucmc.step!(algorithm, model, geometry, seq_state, 2.0;
                       rng = rng, workspace = seq_workspace)
            @test seq_state.energy ≈ spin_energy(model, geometry, seq_state.spins)
            @test seq_state.magnetization == sum(seq_state.spins)
        end
    end

    # the drivers build and thread the workspace automatically
    for algorithm in sequential_algorithms
        container = ucmc.MeasurementContainer(model, geometry, 4, 2;
                                              diagnostics = [:acceptance_ratio])
        parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
        ucmc.run_simulation(algorithm, model, geometry,
                            ucmc.initialize_state(model, geometry; rng = rng),
                            container, parameters; rng = rng)
        @test all(isfinite, container.data[:energy])
        @test all(isfinite, container.data[:acceptance_ratio])
    end

    # sweeps with different schemes write to different results directories
    @test ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry.lattice.L) !=
        ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()),
                           geometry.lattice.L)
end

@testset "Coupling and field accessors" begin
    geometry = two_orbital_geometry(2)
    for model in (ucmc.IsingModel([1.0, 2.0], [0.5, -0.25]),
                  ucmc.XYModel([1.0, 2.0], [0.5, -0.25]),
                  ucmc.ClockModel(4, [1.0, 2.0], [0.5, -0.25]))
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
                  ucmc.ClockModel(3, [1.0, -1.0], [0.0, 0.0]))
        state = ucmc.initialize_state(model, geometry)
        for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
            @test_throws ArgumentError ucmc.step!(algorithm, model, geometry, state, 1.0)
        end
    end
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
    @test haskey(connected_container.data, :magnetization_per_orbital)
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

@testset "Sweep save edge cases" begin
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()
    geometry = square_geometry(2)
    root = mktempdir()

    empty_dir = ucmc.save_sweep_results([], Float64[], model, algorithm, geometry, 0, 0, 0, 1; root = root)
    @test isdir(empty_dir)
    @test isfile(joinpath(empty_dir, "metadata.txt"))
    @test !isfile(joinpath(empty_dir, "measurement_results.csv"))
    @test !isfile(joinpath(empty_dir, "correlation_results.csv"))

    @test_throws ArgumentError ucmc.save_sweep_results([Dict()], [1.0, 2.0], model, algorithm, geometry, 0, 0, 0, 1; root = root)

    # CSV rows are deterministic: the primary (Dict) entries come out sorted
    fake = (primary = Dict(:z => (1.0, 0.1), :a => (2.0, 0.2), :m => (3.0, 0.3), :e => (4.0, 0.4)),
            derived = NamedTuple())
    names, values, errors = ucmc.measurement_rows(fake)
    @test names == ["a", "e", "m", "z"]
    @test values == [2.0, 4.0, 3.0, 1.0]
    @test errors == [0.2, 0.4, 0.3, 0.1]
end

@testset "sweep_Ts save keyword" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    root = mktempdir()

    # save = false: pure computation, nothing written
    results = ucmc.sweep_Ts(ucmc.MetropolisAlgorithm(), model, geometry, state,
        1, 4, 0, 2, [2.0]; root = root, save = false)
    @test length(results) == 1
    @test haskey(results[1].primary, :energy)
    @test isempty(readdir(root))

    # default (save = true) writes the sweep directory
    ucmc.sweep_Ts(ucmc.MetropolisAlgorithm(), model, geometry, state,
        1, 4, 0, 2, [2.0]; root = root)
    dir = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry.lattice.L; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))
end

@testset "steps_per_sweep" begin
    geometry = square_geometry(4)
    @test ucmc.steps_per_sweep(ucmc.MetropolisAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.GlauberAlgorithm(), geometry) == geometry.n_sites
    @test ucmc.steps_per_sweep(ucmc.WolffAlgorithm(), geometry) == 1
    @test ucmc.steps_per_sweep(ucmc.SwendsenWangAlgorithm(), geometry) == 1
end

@testset "MC_Sweep! update counts" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    n_sites = geometry.n_sites

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:acceptance_ratio])
    stats = (ucmc.UpdateStatistic(:acceptance_ratio),)
    ucmc.MC_Sweep!(ucmc.MetropolisAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == n_sites
    @test 0.0 <= ucmc.value(stats[1]) <= 1.0

    container = ucmc.MeasurementContainer(model, geometry, 10, 2; diagnostics = [:cluster_size])
    stats = (ucmc.UpdateStatistic(:cluster_size),)
    ucmc.MC_Sweep!(ucmc.WolffAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == 1
    @test 1 <= ucmc.value(stats[1]) <= n_sites

    container = ucmc.MeasurementContainer(model, geometry, 10, 2;
        diagnostics = [:cluster_count, :largest_cluster_fraction])
    stats = (ucmc.UpdateStatistic(:cluster_count), ucmc.UpdateStatistic(:largest_cluster_fraction))
    ucmc.MC_Sweep!(ucmc.SwendsenWangAlgorithm(), model, geometry, state, container, 2.0, stats)
    @test stats[1].count == 1
    @test stats[2].count == 1
    @test ucmc.value(stats[1]) >= 1
    @test 0.0 < ucmc.value(stats[2])
end

@testset "Algorithms reject non-positive temperature" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    for alg in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(), ucmc.WolffAlgorithm(),
                ucmc.SwendsenWangAlgorithm())
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

@testset "run_simulation rejects a container that does not match the parameters" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()

    # container bins 100 measurements into 10 bins of 10
    for parameters in (
        ucmc.SimulationParameters(2.0, 5, 90, 0, 10),   # bin_size 9
        ucmc.SimulationParameters(2.0, 5, 100, 0, 5),   # bin_size 20
        ucmc.SimulationParameters(2.0, 5, 50, 0, 10),   # bin_size 5
    )
        state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
        spins = copy(state.spins)
        container = ucmc.MeasurementContainer(model, geometry, 100, 10)
        @test_throws ArgumentError ucmc.run_simulation(
            algorithm, model, geometry, state, container, parameters; rng = Xoshiro(2)
        )
        # rejected before any sweep: neither the state nor the container changed
        @test state.spins == spins
        @test isempty(container.data.energy)
        @test container.bin_count == 0
    end

    # matching counts fill exactly n_bins complete bins
    parameters = ucmc.SimulationParameters(2.0, 5, 100, 0, 10)
    container = ucmc.MeasurementContainer(model, geometry, 100, 10)
    ucmc.run_simulation(algorithm, model, geometry, ucmc.initialize_state(model, geometry),
        container, parameters)
    @test length(container.data.energy) == 10
    @test container.bin_count == 0

    # diagnostics the algorithm does not provide are rejected up front
    state = ucmc.initialize_state(model, geometry)
    container = ucmc.MeasurementContainer(model, geometry, 100, 10; diagnostics = [:cluster_size])
    err = try
        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("cluster_size", err.msg)
    @test isempty(container.data.energy)
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

@testset "sweep_Ts validates inputs before the threaded loop" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    algorithm = ucmc.MetropolisAlgorithm()

    # Each call must throw the ArgumentError itself, not a CompositeException
    # from inside Threads.@threads, and must fail before any simulation runs:
    # the starting state is untouched and no results directory is written.
    function sweep_error(args...; kwargs...)
        state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
        spins = copy(state.spins)
        root = mktempdir()
        err = try
            ucmc.sweep_Ts(algorithm, model, geometry, state, args...;
                n_runs = 2, root = root, kwargs...)
            nothing
        catch e
            e
        end
        untouched = state.spins == spins && isempty(readdir(root))
        return err, untouched
    end

    cases = [
        ((3, 6, 1, 1, [2.0]), (;)),                                  # n_bins < 2
        ((3, 0, 1, 2, [2.0]), (;)),                                  # n_measurements = 0
        ((3, 7, 1, 2, [2.0]), (;)),                                  # not divisible
        ((3, 6, 1, 3, [2.0, 1.0, -1.0]), (;)),                       # bad last temperature
        ((3, 6, 1, 3, [2.0]), (; diagnostics = [:cluster_size])),    # wrong algorithm diagnostic
        ((3, 6, 1, 3, [2.0]), (; time_series = [:not_measured])),    # container rejects
    ]
    for (args, kwargs) in cases, annealing in (true, false)
        err, untouched = sweep_error(args...; simulated_annealing = annealing, kwargs...)
        @test err isa ArgumentError
        @test untouched
    end
end

@testset "RNG reproducibility" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    xy_model = ucmc.XYModel([1.0, 1.0], [0.0])
    clock_model = ucmc.ClockModel(5, [1.0, 1.0], [0.3])
    parameters = ucmc.SimulationParameters(2.0, 5, 20, 1, 4)

    # explicit rng: identical streams give identical states and bins
    for m in (model, xy_model, clock_model),
        algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(), ucmc.WolffAlgorithm(),
                      ucmc.SwendsenWangAlgorithm())
        containers = map(1:2) do _
            rng = Xoshiro(2024)
            state = ucmc.initialize_state(m, geometry; rng = rng)
            container = ucmc.MeasurementContainer(m, geometry, 20, 4)
            ucmc.run_simulation(algorithm, m, geometry, state, container, parameters; rng = rng)
        end
        @test isequal(containers[1].data, containers[2].data)
    end

    # the global RNG is the default, so Random.seed! reproduces a run
    runs = map(1:2) do _
        Random.seed!(99)
        state = ucmc.initialize_state(model, geometry)
        container = ucmc.MeasurementContainer(model, geometry, 20, 4)
        ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    end
    @test isequal(runs[1].data, runs[2].data)

    # threaded sweep_Ts runs are reproducible from the global seed
    root = mktempdir()
    primaries(results) = [r.primary for r in results]
    sweep(state; kwargs...) = ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, state,
        5, 20, 1, 4, [3.0, 2.0]; n_runs = 4, time_series = :none, root = root, kwargs...
    )
    seeded_sweep(seed; kwargs...) = (Random.seed!(seed);
        sweep(ucmc.initialize_state(model, geometry); kwargs...))
    for annealing in (true, false)
        @test isequal(
            primaries(seeded_sweep(7; simulated_annealing = annealing)),
            primaries(seeded_sweep(7; simulated_annealing = annealing))
        )
    end
    @test !isequal(primaries(seeded_sweep(7)), primaries(seeded_sweep(8)))

    # an explicit rng behaves like the seeded global RNG
    explicit_sweep(seed) = (rng = Xoshiro(seed);
        sweep(ucmc.initialize_state(model, geometry; rng = rng); rng = rng))
    @test isequal(primaries(explicit_sweep(5)), primaries(explicit_sweep(5)))
end

@testset "seed recorded in metadata" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    metadata(dir) = readlines(joinpath(dir, "metadata.txt"))
    sweep_dir(root) = ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry.lattice.L; root = root)
    sweep(root; kwargs...) = ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        2, 4, 0, 2, [2.0]; n_runs = 2, time_series = :none, root = root, kwargs...
    )

    # set_global_seed! seeds the global RNG exactly like Random.seed!
    @test ucmc.set_global_seed!(42) == 42
    @test ucmc.global_seed() == 42
    a = rand(3)
    Random.seed!(42)
    @test rand(3) == a

    # sweep_Ts records the global seed and Julia version
    root = mktempdir()
    ucmc.set_global_seed!(42)
    sweep(root)
    lines = metadata(sweep_dir(root))
    @test "seed = 42" in lines
    @test "julia_version = $VERSION" in lines

    # the recorded seed recreates the saved results
    first_csv = read(joinpath(sweep_dir(root), "measurement_results.csv"), String)
    ucmc.set_global_seed!(42)
    sweep(root)
    @test read(joinpath(sweep_dir(root), "measurement_results.csv"), String) == first_csv

    # without an argument a random seed is drawn, returned, and recorded
    seed = ucmc.set_global_seed!()
    @test ucmc.global_seed() == seed
    sweep(root)
    @test "seed = $seed" in metadata(sweep_dir(root))

    # an explicit rng has no recoverable seed unless one is passed alongside it
    sweep(root; rng = Xoshiro(5))
    @test "seed = unknown" in metadata(sweep_dir(root))
    sweep(root; rng = Xoshiro(5), seed = 5)
    @test "seed = 5" in metadata(sweep_dir(root))

    # single-temperature saves record the seed too
    ucmc.set_global_seed!(7)
    state = ucmc.initialize_state(model, geometry)
    parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
    container = ucmc.MeasurementContainer(model, geometry, 4, 2)
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, container, parameters)
    results = ucmc.analyze(container, 2.0, geometry.n_sites)
    single = ucmc.save_results(model, ucmc.MetropolisAlgorithm(), geometry, 2.0,
        results, parameters; root = root)
    @test "seed = 7" in metadata(single)
    single = ucmc.save_results(model, ucmc.MetropolisAlgorithm(), geometry, 2.0,
        results, parameters; root = root, seed = nothing)
    @test "seed = unknown" in metadata(single)
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

    single = ucmc.save_results(model, ucmc.WolffAlgorithm(), geometry, 3.0,
        sweep_results[1], ucmc.SimulationParameters(3.0, 10, 40, 0, 4); root = root)
    @test isfile(joinpath(single, "autocorrelation_results.csv"))
end

# Two orbitals on a non-square, periodic lattice with an intra-cell bond, so
# the off-diagonal structure factor is genuinely complex.
function rectangular_two_orbital_geometry(L1::Int, L2::Int; periodic = [true, true])
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0], [0.5, 0.5]],
    )
    lattice = lu.Lattice(L = [L1, L2], periodic = periodic)
    bond_1 = lu.Bond(orbitals = (1, 2), displacement = [0, 0])
    bond_2 = lu.Bond(orbitals = (2, 1), displacement = [1, 0])
    return ucmc.Geometry(unit_cell, lattice, [bond_1, bond_2])
end

struct NoComponentsModel <: ucmc.AbstractModel{Float64} end

@testset "operator_components reproduce operator_product" begin
    for s in (-1, 1), t in (-1, 1)
        a = ucmc.operator_components(ucmc.IsingModel([1.0], [0.0]), s)
        b = ucmc.operator_components(ucmc.IsingModel([1.0], [0.0]), t)
        @test length(a) == ucmc.n_operator_components(ucmc.IsingModel([1.0], [0.0]))
        @test sum(a .* b) == ucmc.operator_product(ucmc.IsingModel([1.0], [0.0]), s, t)
    end
    xy = ucmc.XYModel([1.0], [0.0])
    rng = Xoshiro(11)
    for _ in 1:20
        θ, φ = 2π * rand(rng), 2π * rand(rng)
        a, b = ucmc.operator_components(xy, θ), ucmc.operator_components(xy, φ)
        @test length(a) == ucmc.n_operator_components(xy)
        @test sum(a .* b) ≈ ucmc.operator_product(xy, θ, φ)
    end
    for q in (2, 3, 4, 6)
        clock = ucmc.ClockModel(q, [1.0], [0.0])
        @test ucmc.n_operator_components(clock) == 2
        for n in 0:(q - 1), m in 0:(q - 1)
            a, b = ucmc.operator_components(clock, n), ucmc.operator_components(clock, m)
            @test sum(a .* b) ≈ ucmc.operator_product(clock, n, m) atol = 1e-14
            @test ucmc.operator_product(clock, n, m) ≈ cos(2π * (n - m) / q) atol = 1e-14
        end
    end

    # Models without the hook are rejected when the measurement is built.
    geometry = square_geometry(2)
    @test_throws ArgumentError ucmc.n_operator_components(NoComponentsModel())
    @test_throws ArgumentError ucmc.operator_components(NoComponentsModel(), 1.0)
    @test_throws ArgumentError ucmc.structure_factor_entry(NoComponentsModel(), geometry)
end

@testset "momentum grid" begin
    geometry = rectangular_two_orbital_geometry(3, 4)
    momenta = ucmc.build_momentum_classes(geometry)
    @test momenta.n_cells == 12
    @test momenta.n_momenta == 12
    @test length(momenta.classes) == 2^2 * 12
    @test [c.orbitals for c in momenta.classes[1:12]] == fill((1, 1), 12)
    @test momenta.classes[13].orbitals == (1, 2)
    @test momenta.classes[1].momentum_index == (0, 0)
    @test momenta.classes[2].momentum_index == (1, 0)   # first dimension fastest
    @test momenta.classes[4].momentum_index == (0, 1)
    @test collect(momenta.classes[2].momentum) ≈ [2π / 3, 0.0]
    @test collect(momenta.classes[4].momentum) ≈ [0.0, 2π / 4]
    for o in 1:2, I in CartesianIndices((3, 4))
        site = momenta.sites[o][I]
        loc, orbital = lu.site_to_loc(site, geometry.unit_cell, geometry.lattice)
        @test orbital == o
        @test collect(loc) == collect(Tuple(I) .- 1)
    end

    # Open dimensions only carry k = 0, as in LatticeUtilities.calc_k_points.
    open_geometry = rectangular_two_orbital_geometry(3, 4; periodic = [true, false])
    open_momenta = ucmc.build_momentum_classes(open_geometry)
    @test open_momenta.n_momenta == 3
    @test open_momenta.n_cells == 12
    @test all(c -> c.momentum_index[2] == 0, open_momenta.classes)
end

@testset "structure factor is the Fourier transform of the correlation" begin
    geometry = rectangular_two_orbital_geometry(3, 4)
    L = geometry.lattice.L
    momenta = ucmc.build_momentum_classes(geometry)
    disp = ucmc.build_displacement_classes(geometry)
    rng = Xoshiro(3)

    for (model, algorithm) in (
        (ucmc.IsingModel([1.0, 0.5], [0.0, 0.0]), ucmc.MetropolisAlgorithm()),
        (ucmc.XYModel([1.0, 0.5], [0.0, 0.0]), ucmc.WolffAlgorithm()),
        (ucmc.ClockModel(6, [1.0, 0.5], [0.0, 0.0]), ucmc.SwendsenWangAlgorithm()),
    )
        state = ucmc.initialize_state(model, geometry; rng = rng)
        for _ in 1:5
            for _ in 1:ucmc.steps_per_sweep(algorithm, geometry)
                ucmc.step!(algorithm, model, geometry, state, 1.5; rng = rng)
            end
            S = ucmc.structure_factor_instantaneous(model, state, momenta)
            C = ucmc.correlation_instantaneous(model, state, disp)
            expected = map(momenta.classes) do c
                sum(
                    C[r] * cis(-2π * sum(c.momentum_index .* disp.displacements[r] ./ Tuple(L)))
                    for r in eachindex(C) if disp.orbitals[r] == c.orbitals
                )
            end
            @test S ≈ expected atol = 1e-10

            # Hermitian in the orbital indices; diagonal entries real and >= 0.
            n_k = momenta.n_momenta
            block(a, b) = S[((a - 1) * 2 + (b - 1)) * n_k .+ (1:n_k)]
            @test block(2, 1) ≈ conj.(block(1, 2))
            @test all(iszero ∘ imag, block(1, 1))
            @test all(>=(0) ∘ real, block(2, 2))
            # Sum rule for unit-length spins: mean over k of S_aa(k) is 1.
            @test sum(block(1, 1)) / n_k ≈ 1.0
            @test sum(block(2, 2)) / n_k ≈ 1.0
        end
    end
end

@testset "correlation via inverse DFT" begin
    for (model, algorithm) in (
        (ucmc.IsingModel([1.0, 0.5], [0.0, 0.0]), ucmc.MetropolisAlgorithm()),
        (ucmc.XYModel([1.0, 0.5], [0.0, 0.0]), ucmc.WolffAlgorithm()),
        (ucmc.ClockModel(4, [1.0, 0.5], [0.0, 0.0]), ucmc.SwendsenWangAlgorithm()),
    )
        rng = Xoshiro(9)
        for geometry in (rectangular_two_orbital_geometry(3, 4),                  # periodic: DFT path
                         rectangular_two_orbital_geometry(3, 4; periodic = [true, false]))  # open: direct fallback
            disp = ucmc.build_displacement_classes(geometry)
            state = ucmc.initialize_state(model, geometry; rng = rng)
            for _ in 1:3
                for _ in 1:ucmc.steps_per_sweep(algorithm, geometry)
                    ucmc.step!(algorithm, model, geometry, state, 1.5; rng = rng)
                end
            end
            container = ucmc.MeasurementContainer(model, geometry, 4, 2;
                measurements = [:raw_correlation])
            observable = only(o for o in container.observables if o.name == :correlation)
            direct = ucmc.correlation_instantaneous(model, state, disp)
            @test observable.value(state) ≈ direct atol = 1e-10
        end
    end
end

@testset "structure factor of ordered states" begin
    L = 4
    geometry = square_geometry(L)
    N = geometry.n_sites
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    momenta = ucmc.build_momentum_classes(geometry)
    index_of(n) = findfirst(c -> c.momentum_index == n, momenta.classes)

    ferro = ucmc.IsingState(ones(Int, N), 0.0, N)
    S = ucmc.structure_factor_instantaneous(model, ferro, momenta)
    @test S[index_of((0, 0))] ≈ N
    @test sum(abs, S) ≈ N

    neel_spins = [
        (-1)^sum(lu.site_to_loc(i, geometry.unit_cell, geometry.lattice)[1])
        for i in 1:N
    ]
    neel = ucmc.IsingState(neel_spins, 0.0, sum(neel_spins))
    S = ucmc.structure_factor_instantaneous(model, neel, momenta)
    @test S[index_of((L ÷ 2, L ÷ 2))] ≈ N
    @test sum(abs, S) ≈ N

    # S(0) of a single-orbital lattice is M² / N for any configuration, and the
    # XY model sums both spin components.
    xy = ucmc.XYModel([1.0, 1.0], [0.0])
    xy_state = ucmc.initialize_state(xy, geometry; rng = Xoshiro(5))
    S = ucmc.structure_factor_instantaneous(xy, xy_state, momenta)
    @test S[index_of((0, 0))] ≈ (xy_state.magnetization_x^2 + xy_state.magnetization_y^2) / N
end

@testset "Optional structure factor measurement" begin
    @test :structure_factor in ucmc.available_measurements(ucmc.IsingModel)
    @test :structure_factor in ucmc.available_measurements(ucmc.XYModel([1.0], [0.0]))
    @test :structure_factor in ucmc.available_measurements(ucmc.ClockModel(3, [1.0], [0.0]))

    geometry = rectangular_two_orbital_geometry(3, 2)
    n_classes = 2^2 * 6
    for model in (ucmc.IsingModel([1.0, 1.0], [0.0, 0.0]), ucmc.XYModel([1.0, 1.0], [0.0, 0.0]),
                  ucmc.ClockModel(4, [1.0, 1.0], [0.0, 0.0]))
        rng = Xoshiro(8)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, 8, 2;
            measurements = [:structure_factor])
        @test haskey(container.data, :structure_factor)
        @test container.data.structure_factor isa Vector{Vector{ComplexF64}}
        @test !haskey(container.time_series, :structure_factor)   # vector: not a scalar

        samples = Vector{ComplexF64}[]
        momenta = ucmc.build_momentum_classes(geometry)
        for _ in 1:8
            ucmc.step!(ucmc.SwendsenWangAlgorithm(), model, geometry, state, 2.0; rng = rng)
            push!(samples, ucmc.structure_factor_instantaneous(model, state, momenta))
            ucmc.measure!(container, state)
        end
        @test length(container.data.structure_factor) == 2
        @test container.data.structure_factor[1] ≈ sum(samples[1:4]) / 4

        result = ucmc.analyze(container, 2.0, geometry.n_sites)
        avg, err = result.primary[:structure_factor]
        @test avg ≈ sum(samples) / 8
        @test length(err) == n_classes
        @test all(e -> real(e) >= 0 && imag(e) >= 0, err)
        @test result.derived.structure_factor_classes == momenta.classes

        rows = ucmc.structure_factor_rows(result)
        @test size(rows, 1) == n_classes
        @test propertynames(rows) == [:k_index_1, :k_index_2, :k_1, :k_2, :orbital_i,
            :orbital_j, :real_value, :real_error, :imag_value, :imag_error]
        @test rows.real_value ≈ real.(avg)
        @test rows.imag_error ≈ imag.(err)
        # complex vectors are left out of the scalar measurement table
        @test "structure_factor" ∉ first(ucmc.measurement_rows(result))
    end

    # Requesting the full time series gives an autocorrelation estimate.
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    square = square_geometry(4)
    state = ucmc.initialize_state(model, square; rng = Xoshiro(1))
    container = ucmc.MeasurementContainer(model, square, 40, 2;
        measurements = [:structure_factor], time_series = :all)
    for _ in 1:40
        ucmc.step!(ucmc.WolffAlgorithm(), model, square, state, 2.5)
        ucmc.measure!(container, state)
    end
    result = ucmc.analyze(container, 2.5, square.n_sites)
    @test result.autocorrelation[:structure_factor].status !== :unsupported
    @test result.autocorrelation[:structure_factor].n == 40

    # requesting no structure factor yields no rows
    plain = ucmc.MeasurementContainer(model, square, 4, 2)
    @test !haskey(plain.data, :structure_factor)
    for _ in 1:4
        ucmc.measure!(plain, state)
    end
    @test ucmc.structure_factor_rows(ucmc.analyze(plain, 2.5, square.n_sites)) === nothing
end

@testset "structure factor results are saved" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)
    root = mktempdir()
    Ts = [3.0, 2.0]

    sweep_results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, state,
        4, 8, 0, 2, Ts;
        n_runs = 2, measurements = [:structure_factor], root = root
    )
    @test all(r -> haskey(r.primary, :structure_factor), sweep_results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry.lattice.L; root = root)
    path = joinpath(dir, "structure_factor_results.csv")
    @test isfile(path)
    lines = readlines(path)
    @test lines[1] == "temperature,k_index_1,k_index_2,k_1,k_2,orbital_i,orbital_j," *
        "real_value,real_error,imag_value,imag_error"
    @test length(lines) == 1 + length(Ts) * 4

    single = ucmc.save_results(model, ucmc.WolffAlgorithm(), geometry, 3.0,
        sweep_results[1], ucmc.SimulationParameters(3.0, 4, 8, 0, 2); root = root)
    @test isfile(joinpath(single, "structure_factor_results.csv"))
    @test length(readlines(joinpath(single, "structure_factor_results.csv"))) == 1 + 4

    # without the measurement, no structure factor file is written
    plain = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, state,
        2, 4, 0, 2, [3.0]; root = mktempdir()
    )
    @test ucmc.structure_factor_rows(plain[1]) === nothing
end

@testset "Clock model construction and state" begin
    @test_throws ArgumentError ucmc.ClockModel(1, [1.0], [0.0])
    @test_throws ArgumentError ucmc.ClockModel(0, [1.0], [0.0])

    model = ucmc.ClockModel(4, [1, 2], [0.5])
    @test model isa ucmc.ClockModel{Float64}
    @test model.q == 4
    # lookup tables are exact at the axes
    @test model.cosines == [1.0, 0.0, -1.0, 0.0]
    @test model.sines == [0.0, 1.0, 0.0, -1.0]
    @test ucmc.clock_cos(model, -1) == 0.0
    @test ucmc.clock_sin(model, 7) == -1.0
    @test repr(model) == "ClockModel{Float64}(4, [1.0, 2.0], [0.5])"

    geometry = two_orbital_geometry(2)
    model = ucmc.ClockModel(6, [1.0, 2.0], [0.5, -0.25])
    state = ucmc.initialize_state(model, geometry; rng = Xoshiro(1))
    @test state isa ucmc.ClockState
    @test length(state.spins) == geometry.n_sites
    @test clock_invariants_hold(model, geometry, state)

    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.ClockModel(6, [1.0, 2.0], [0.5]), geometry
    )
    @test_throws ArgumentError ucmc.initialize_state(
        ucmc.ClockModel(6, [1.0], [0.5, -0.25]), geometry
    )
end

@testset "Clock updates keep invariants" begin
    geometry = two_orbital_geometry(2)
    n_sites = geometry.n_sites
    for q in (2, 3, 4, 6), h in ([0.0, 0.0], [0.5, -0.25])
        model = ucmc.ClockModel(q, [1.0, 2.0], h)
        for algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
                          ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
            rng = Xoshiro(q)
            state = ucmc.initialize_state(model, geometry; rng = rng)
            invariants = true
            diagnostics_ok = true
            for _ in 1:30
                result = ucmc.step!(algorithm, model, geometry, state, 1.5; rng = rng)
                invariants &= clock_invariants_hold(model, geometry, state)
                values = ucmc.diagnostic_values(algorithm, result)
                diagnostics_ok &= collect(propertynames(values)) ==
                    ucmc.available_diagnostics(algorithm)
            end
            @test invariants
            @test diagnostics_ok
        end
    end

    # Metropolis always proposes a different clock state; for q = 2 it is a flip.
    rng = Xoshiro(3)
    for q in (2, 3, 5)
        model = ucmc.ClockModel(q, [1.0, 1.0], [0.0, 0.0])
        state = ucmc.initialize_state(model, geometry; rng = rng)
        proposals = [ucmc.propose(rng, ucmc.MetropolisAlgorithm(), model, geometry, state) for _ in 1:200]
        @test all(p -> p.new_value != state.spins[p.site] && p.new_value in 0:(q - 1), proposals)
        q == 2 && @test all(p -> p.new_value == 1 - state.spins[p.site], proposals)
        q > 2 && @test length(unique(p.new_value - state.spins[p.site] for p in proposals)) > 1
    end

    # cluster algorithms are ferromagnetic only; Metropolis accepts any sign
    antiferro = ucmc.ClockModel(3, [1.0, -1.0], [0.0, 0.0])
    state = ucmc.initialize_state(antiferro, geometry)
    for algorithm in (ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        @test_throws ArgumentError ucmc.step!(algorithm, antiferro, geometry, state, 1.0)
    end
    for _ in 1:20
        ucmc.step!(ucmc.MetropolisAlgorithm(), antiferro, geometry, state, 1.0)
        ucmc.step!(ucmc.GlauberAlgorithm(), antiferro, geometry, state, 1.0)
    end
    @test clock_invariants_hold(antiferro, geometry, state)
end

@testset "Clock cluster reflection" begin
    geometry = square_geometry(2)
    model = ucmc.ClockModel(4, [1.0, 1.0], [0.3])
    spins = [0, 1, 2, 3]
    make_state(spins) = ucmc.ClockState(
        copy(spins), clock_energy(model, geometry, spins), clock_mx(4, spins), clock_my(4, spins)
    )

    state = make_state(spins)
    ucmc.apply_cluster!(model, geometry, state, [1, 2], 1)   # n ↦ 1 - n
    @test state.spins == [1, 0, 2, 3]
    @test clock_invariants_hold(model, geometry, state)

    # Swendsen-Wang reflects each cluster as a whole, or not at all.
    reflect(n) = mod(2 - n, 4)
    for seed in 1:8
        state = make_state(spins)
        ucmc.apply_clusters!(Xoshiro(seed), model, geometry, state, [[1, 2], [3, 4]], 2)
        for cluster in ([1, 2], [3, 4])
            @test state.spins[cluster] == spins[cluster] ||
                state.spins[cluster] == reflect.(spins[cluster])
        end
        @test clock_invariants_hold(model, geometry, state)
    end

    # The mirror n ↦ m - n is the XY reflection θ ↦ 2φ + π - θ about the axis
    # φ = πm/q - π/2, and the context stores each state's component along φ.
    for q in (3, 4, 5), m in 0:(q - 1)
        context = ucmc.ClockClusterContext(ucmc.ClockModel(q, [1.0], [0.0]), m)
        φ = π * m / q - π / 2
        @test context.projections ≈ [cos(2π * n / q - φ) for n in 0:(q - 1)] atol = 1e-12
        @test context.ghost_projection ≈ cos(φ) atol = 1e-12
        @test all(cis(2π * mod(m - n, q) / q) ≈ cis(2φ + π - 2π * n / q) for n in 0:(q - 1))
    end
end

@testset "Clock q = 2 reduces to Ising" begin
    geometry = two_orbital_geometry(2)
    clock = ucmc.ClockModel(2, [1.0, 2.0], [0.5, -0.25])
    ising = ucmc.IsingModel([1.0, 2.0], [0.5, -0.25])
    rng = Xoshiro(4)
    for _ in 1:10
        spins = rand(rng, 0:1, geometry.n_sites)
        @test clock_energy(clock, geometry, spins) ≈ spin_energy(ising, geometry, 1 .- 2 .* spins)
    end

    # cluster moves use the spin-flip mirror with Ising projections ±1
    state = ucmc.initialize_state(clock, geometry; rng = rng)
    context = ucmc.cluster_context(rng, clock, geometry, state, 1.0)
    @test context.mirror == 1
    @test context.projections == [1.0, -1.0]

    @test ucmc.binder_denominator(clock) == 3.0
    @test ucmc.binder_denominator(ucmc.ClockModel(3, [1.0], [0.0])) == 2.0
end

@testset "Clock model matches exact enumeration" begin
    geometry = square_geometry(2)
    N = geometry.n_sites
    q, T = 3, 1.1
    model = ucmc.ClockModel(q, [1.0, 0.8], [0.35])

    Z = E = Mx = 0.0
    for c in 0:(q^N - 1)
        spins = digits(c; base = q, pad = N)
        e = clock_energy(model, geometry, spins)
        w = exp(-e / T)
        Z += w
        E += w * e
        Mx += w * clock_mx(q, spins)
    end
    exact_e, exact_mx = E / (Z * N), Mx / (Z * N)

    parameters = ucmc.SimulationParameters(T, 200, 20_000, 0, 20)
    for algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
                      ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        rng = Xoshiro(2024)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, 20_000, 20)
        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, T, N)
        e, e_err = result.primary[:energy]
        mx, mx_err = result.primary[:magnetization_x]
        @test abs(e - exact_e) < 4 * e_err
        @test abs(mx - exact_mx) < 4 * mx_err
    end
end

@testset "Clock measurements" begin
    geometry = square_geometry(3)
    N = geometry.n_sites
    model = ucmc.ClockModel(5, [1.0, 1.0], [0.2])

    parameters = ucmc.SimulationParameters(1.5, 2, 8, 0, 2)
    for algorithm in (ucmc.MetropolisAlgorithm(), ucmc.GlauberAlgorithm(),
                      ucmc.WolffAlgorithm(), ucmc.SwendsenWangAlgorithm())
        rng = Xoshiro(6)
        state = ucmc.initialize_state(model, geometry; rng = rng)
        container = ucmc.MeasurementContainer(model, geometry, 8, 2;
            measurements = ucmc.available_measurements(model),
            diagnostics = ucmc.available_diagnostics(algorithm))

        connected = only(filter(d -> :correlation_connected in d.names, container.derived))
        @test connected.depends_on == (:correlation, :magnetization_per_orbital)

        values = Dict(o.name => o.value(state) for o in container.observables)
        @test values[:magnetization] ≈ hypot(state.magnetization_x, state.magnetization_y) / N
        @test values[:abs_magnetization] ≈ values[:magnetization]
        @test values[:magnetization_fourth] ≈ values[:magnetization]^4
        @test values[:correlation][1] ≈ 1.0   # on-site class: cos(0)

        ucmc.run_simulation(algorithm, model, geometry, state, container, parameters; rng = rng)
        result = ucmc.analyze(container, 1.5, N)
        for name in (:energy, :magnetization_x, :magnetization_y, :magnetization,
                     :abs_magnetization, :magnetization_fourth, :correlation,
                     :structure_factor, :clock_order_parameter,
                     ucmc.available_diagnostics(algorithm)...)
            @test haskey(result.primary, name)
        end
        for name in (:specific_heat, :susceptibility, :susceptibility_x, :susceptibility_y,
                     :binder_cumulant, :correlation_connected, :structure_factor_classes)
            @test haskey(result.derived, name)
        end
        jk_x = ucmc.jackknife_samples(container.data.magnetization_x)
        jk_y = ucmc.jackknife_samples(container.data.magnetization_y)
        n = length(jk_x)
        mx_full = sum(container.data.magnetization_x) / n
        my_full = sum(container.data.magnetization_y) / n
        f_full = 1.0 - mx_full^2 - my_full^2
        v̄ = sum(1.0 - mx^2 - my^2 for (mx, my) in zip(jk_x, jk_y)) / n
        expected = n * f_full - (n - 1) * v̄
        @test result.derived.correlation_connected[1] ≈ expected
    end
end

@testset "Clock sweep_Ts, autotune, and saved results" begin
    geometry = square_geometry(2)
    model = ucmc.ClockModel(4, [1.0, 1.0], [0.0])
    root = mktempdir()
    Ts = [2.0, 1.0]

    results = ucmc.sweep_Ts(
        ucmc.WolffAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        3, 6, 1, 3, Ts;
        n_runs = 2,
        measurements = [:connected_correlation, :binder_cumulant, :structure_factor],
        diagnostics = [:cluster_size],
        root = root,
    )
    @test length(results) == length(Ts)
    @test all(r -> haskey(r.primary, :cluster_size), results)

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry.lattice.L; root = root)
    @test occursin("ClockModel_WolffAlgorithm", dir)
    for file in ("measurement_results.csv", "correlation_results.csv",
                 "structure_factor_results.csv")
        @test isfile(joinpath(dir, file))
    end
    @test "model = ClockModel{Float64}(4, [1.0, 1.0], [0.0])" in
        readlines(joinpath(dir, "metadata.txt"))

    tuning = ucmc.autotune_binning(
        ucmc.SwendsenWangAlgorithm(), model, geometry,
        ucmc.initialize_state(model, geometry), Ts; n_pilot = 200, rng = Xoshiro(2)
    )
    @test tuning.n_unmeasured >= 0
    @test tuning.bin_size >= 1
end

@testset "Clock order parameter measurement" begin
    @test :clock_order_parameter in ucmc.available_measurements(ucmc.ClockModel)
    @test :clock_order_parameter in ucmc.available_measurements(ucmc.ClockModel(6, [1.0], [0.0]))
    @test :clock_order_parameter ∉ ucmc.available_measurements(ucmc.IsingModel)
    @test :clock_order_parameter ∉ ucmc.available_measurements(ucmc.XYModel)

    geometry = square_geometry(2)
    N = geometry.n_sites
    for model in (ucmc.IsingModel([1.0, 1.0], [0.0]), ucmc.XYModel([1.0, 1.0], [0.0]))
        @test_throws ArgumentError ucmc.MeasurementContainer(model, geometry, 4, 2;
            measurements = [:clock_order_parameter])
    end

    # cos(q φ_M) for hand-built configurations
    model = ucmc.ClockModel(4, [1.0, 1.0], [0.0])
    make_state(spins) = ucmc.ClockState(
        spins, clock_energy(model, geometry, spins), clock_mx(4, spins), clock_my(4, spins)
    )
    order(spins) = ucmc.clock_angle_order(model, make_state(spins))
    @test order([0, 0, 0, 0]) ≈ 1.0    # locked on a clock axis
    @test order([1, 1, 1, 1]) ≈ 1.0
    @test order([0, 1, 0, 1]) ≈ -1.0   # halfway between axes, φ_M = π/4
    @test order([0, 1, 2, 3]) == 0.0   # M = 0: direction undefined
    @test order([0, 2, 1, 3]) == 0.0
    # rounding residue in the running totals does not leak into the angle
    @test ucmc.clock_angle_order(model, ucmc.ClockState([0, 1, 2, 3], 0.0, 1e-13, -2e-13)) == 0.0
    # odd q: M = 0 only when all states are equally occupied
    q5 = ucmc.ClockModel(5, [1.0], [0.0])
    @test ucmc.clock_angle_order(q5, ucmc.ClockState(collect(0:4), 0.0, 0.0, 0.0)) == 0.0
    # q = 2: the magnetization always lies on the clock axis
    q2 = ucmc.ClockModel(2, [1.0], [0.0])
    @test ucmc.clock_angle_order(q2, ucmc.ClockState([0, 1, 1, 1], 0.0, -2.0, 0.0)) ≈ 1.0

    # the measurement adds one scalar observable and no derived quantity
    model = ucmc.ClockModel(6, [1.0, 1.0], [0.0])
    plain = ucmc.MeasurementContainer(model, geometry, 8, 2)
    container = ucmc.MeasurementContainer(model, geometry, 8, 2;
        measurements = [:clock_order_parameter])
    @test setdiff(keys(container.data), keys(plain.data)) == [:clock_order_parameter]
    @test length(container.derived) == length(plain.derived)
    @test haskey(container.time_series, :clock_order_parameter)

    rng = Xoshiro(12)
    state = ucmc.initialize_state(model, geometry; rng = rng)
    samples = Float64[]
    for _ in 1:8
        ucmc.step!(ucmc.WolffAlgorithm(), model, geometry, state, 1.0; rng = rng)
        mx, my = clock_mx(6, state.spins), clock_my(6, state.spins)
        push!(samples, hypot(mx, my) < 1e-8 ? 0.0 : cos(6 * atan(my, mx)))
        ucmc.measure!(container, state)
    end
    @test container.data.clock_order_parameter ≈ [sum(samples[1:4]) / 4, sum(samples[5:8]) / 4]
    result = ucmc.analyze(container, 1.0, N)
    @test result.primary[:clock_order_parameter][1] ≈ sum(samples) / 8

    # deep in the ordered phase the magnetization is locked to a clock axis
    ordered = ucmc.ClockModel(6, [1.0, 1.0], [0.0])
    square = square_geometry(4)
    rng = Xoshiro(5)
    state = ucmc.initialize_state(ordered, square; rng = rng)
    container = ucmc.MeasurementContainer(ordered, square, 200, 4;
        measurements = [:clock_order_parameter])
    ucmc.run_simulation(ucmc.WolffAlgorithm(), ordered, square, state, container,
        ucmc.SimulationParameters(0.2, 200, 200, 0, 4); rng = rng)
    @test ucmc.analyze(container, 0.2, square.n_sites).primary[:clock_order_parameter][1] > 0.99

    # saved alongside the other scalar results
    root = mktempdir()
    ucmc.sweep_Ts(ucmc.SwendsenWangAlgorithm(), model, geometry,
        ucmc.initialize_state(model, geometry), 2, 4, 0, 2, [1.0];
        measurements = [:clock_order_parameter], root = root)
    dir = ucmc.sweep_dirname(model, ucmc.SwendsenWangAlgorithm(), geometry.lattice.L; root = root)
    saved = [split(line, ",")[2] for line in readlines(joinpath(dir, "measurement_results.csv"))[2:end]]
    @test "clock_order_parameter" in saved
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

    @test ucmc.extra_measurements(ucmc.ClockModel) == (:clock_order_parameter,)
    @test isempty(ucmc.extra_measurements(ucmc.IsingModel))
    @test_throws ArgumentError ucmc.binder_denominator(PairingTestModel())
end