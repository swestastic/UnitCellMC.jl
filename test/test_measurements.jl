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

@testset "average_and_error element types" begin
    # scalar bins of any real element type (a custom observable's bins may be
    # Float32, not Float64); the loose tolerance is Float32 precision
    avg, err = ucmc.average_and_error(Float32[1, 2, 3, 4])
    @test avg ≈ 2.5
    @test err ≈ sqrt(5 / 12) atol = 1e-6  # Bessel variance 5/3, divided by n_bins = 4

    # vector-valued bins of other real element types
    bins32 = [Float32[1, 2], Float32[3, 0], Float32[2, 1], Float32[0, 3]]
    avg32, err32 = ucmc.average_and_error(bins32)
    @test avg32 ≈ [1.5, 1.5]
    @test err32 ≈ [sqrt(5 / 12), sqrt(5 / 12)] atol = 1e-6
    @test eltype(avg32) == Float32

    # complex bins of any precision keep the per-component error convention
    bins_c32 = [ComplexF32[1 + 1im, 2], ComplexF32[3 - 1im, 0],
                ComplexF32[2, 1], ComplexF32[0 + 2im, 3]]
    avgc, errc = ucmc.average_and_error(bins_c32)
    @test real(avgc) ≈ Float32[1.5, 1.5]
    @test imag(avgc) ≈ Float32[0.5, 0.0]
    @test eltype(avgc) == ComplexF32

    # other array eltypes (here Float32 matrices) reach the generic fallback
    bins_m32 = [Float32[1 2; 3 4], Float32[3 0; 1 2], Float32[2 1; 2 3]]
    avgm, errm = ucmc.average_and_error(bins_m32)
    @test avgm ≈ Float32[2 1; 2 3]
    @test all(>(0), errm)

    # the built-in Float64 paths are unchanged
    bins64 = [Float64[1, 2], Float64[3, 0], Float64[2, 1], Float64[0, 3]]
    avg64, err64 = ucmc.average_and_error(bins64)
    @test avg64 ≈ [1.5, 1.5]
    @test err64 ≈ [sqrt(5 / 12), sqrt(5 / 12)]
    @test_throws ArgumentError ucmc.average_and_error(Float32[1.0])
    @test_throws ArgumentError ucmc.average_and_error([Float32[1.0]])
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

    # max_lag < 1 is an error (it used to be silently clamped to 1), while an
    # oversized max_lag is clamped to the longest chain minus one — the same
    # behavior as the windowed autocorrelation's _resolve_max_lag
    @test_throws ArgumentError ucmc.autocorrelation_function(x; max_lag = 0)
    @test_throws ArgumentError ucmc.autocorrelation_function(x; max_lag = -2)
    @test first(ucmc.autocorrelation_function(x; max_lag = 10^6)) == 0:(length(x) - 1)
    @test first(ucmc.autocorrelation_function(randn(200); max_lag = 10^6,
                                              chain_lengths = [100, 100])) == 0:99

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

    # measuring further into a merged container keeps analyze working: the
    # unrecorded tail counts as one more chain (it used to throw, since
    # chain_lengths summed to less than the series length)
    grown = ucmc.merge_containers([container, other])
    ucmc.measure!(grown, state; diagnostics = (acceptance_ratio = 1.0,))
    @test ucmc.chain_lengths_with_tail(grown.chain_lengths, 81) == [40, 40, 1]
    results_grown = ucmc.analyze(grown, 2.0, geometry.n_sites)
    @test all(r -> r.n == 81, values(results_grown.autocorrelation))

    # a partially filled bin must be completed before another run_simulation
    # picks the container up (a stale partial bin from an aborted run would
    # otherwise mix into the new run's first bin)
    @test_throws ArgumentError ucmc.run_simulation(
        ucmc.MetropolisAlgorithm(), model, geometry, state, grown, parameters)
    for _ in 1:9
        ucmc.measure!(grown, state; diagnostics = (acceptance_ratio = 1.0,))
    end
    @test grown.bin_count == 0

    # the same holds through a full run_simulation into the merged container
    ucmc.run_simulation(ucmc.MetropolisAlgorithm(), model, geometry, state, grown, parameters)
    results_rerun = ucmc.analyze(grown, 2.0, geometry.n_sites)
    @test all(r -> r.n == 81 + 9 + 40, values(results_rerun.autocorrelation))

    # the helper itself: exact, tail, empty, and inconsistent cases
    @test ucmc.chain_lengths_with_tail(Int[], 5) === nothing
    @test ucmc.chain_lengths_with_tail([8, 8], 16) == [8, 8]
    @test ucmc.chain_lengths_with_tail([8, 8], 20) == [8, 8, 4]
    @test_throws ArgumentError ucmc.chain_lengths_with_tail([8, 8], 10)
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
    dir = ucmc.sweep_dirname(model, ucmc.WolffAlgorithm(), geometry; root = root)
    @test isfile(joinpath(dir, "measurement_results.csv"))
    @test isfile(joinpath(dir, "correlation_results.csv"))  # raw correlation data must reach disk
    @test !isfile(joinpath(dir, "autocorrelation_results.csv"))
    sweep = ucmc.sweep_Ts(ucmc.WolffAlgorithm(), model, geometry, ucmc.initialize_state(model, geometry),
        2, 8, 0, 2, [2.0]; n_runs = 2, measurements = [:raw_correlation], root = root)
    @test haskey(sweep[1].autocorrelation, :energy)
    @test !haskey(sweep[1].autocorrelation, :correlation)
end
