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
        v, w = (cos(θ), sin(θ)), (cos(φ), sin(φ))
        a, b = ucmc.operator_components(xy, v), ucmc.operator_components(xy, w)
        @test length(a) == ucmc.n_operator_components(xy)
        @test sum(a .* b) ≈ ucmc.operator_product(xy, v, w)
        @test sum(a .* b) ≈ cos(θ - φ)
    end
    heisenberg = ucmc.HeisenbergModel([1.0], [0.0])
    for _ in 1:20
        v = ucmc.random_unit_vector(rng)
        w = ucmc.random_unit_vector(rng)
        a, b = ucmc.operator_components(heisenberg, v), ucmc.operator_components(heisenberg, w)
        @test length(a) == ucmc.n_operator_components(heisenberg)
        @test sum(a .* b) ≈ ucmc.operator_product(heisenberg, v, w)
        @test sum(a .* b) ≈ sum(v .* w)
        @test sum(a .* a) ≈ 1.0
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
        (ucmc.HeisenbergModel([1.0, 0.5], [0.0, 0.0]), ucmc.WolffAlgorithm()),
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
        (ucmc.HeisenbergModel([1.0, 0.5], [0.0, 0.0]), ucmc.SwendsenWangAlgorithm()),
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
    @test :structure_factor in ucmc.available_measurements(ucmc.HeisenbergModel([1.0], [0.0]))

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

    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry; root = root)
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
