"""
    ReweightingResult

Result of Ferrenberg–Swendsen single-histogram reweighting of one run's
measurements to nearby temperatures (see [`reweight`](@ref)).

All conventions match [`analyze`](@ref): `energy` and `magnetization` are
per-site values, while `specific_heat` and `susceptibility` are the
fluctuation formulas `N(⟨e²⟩ − ⟨e⟩²)/T²` and `N(⟨m²⟩ − ⟨m⟩²)/T` evaluated
on the reweighted per-site moments (i.e. per-site heat capacity and
susceptibility). Every value has a paired `_err` field holding its
jackknife standard error.

# Fields

- `T_sim::Float64`: Temperature the simulation was run at.
- `temperatures::Vector{Float64}`: The reweighted target temperatures.
- `n_sites::Int`: The system size `N` used in the Boltzmann weights.
- `energy::Vector{Float64}`: Reweighted per-site energy `⟨e⟩(T)`.
- `energy_err::Vector{Float64}`: Jackknife error of `energy`.
- `specific_heat::Vector{Float64}`: Reweighted `N(⟨e²⟩ − ⟨e⟩²)/T²`.
- `specific_heat_err::Vector{Float64}`: Jackknife error of `specific_heat`.
- `magnetization::Union{Nothing, Vector{Float64}}`: Reweighted per-site
  magnetization `⟨m⟩(T)` of the chosen series; `nothing` when reweighting
  was called with `magnetization = nothing`.
- `magnetization_err::Union{Nothing, Vector{Float64}}`: Jackknife error of
  `magnetization`.
- `susceptibility::Union{Nothing, Vector{Float64}}`: Reweighted
  `N(⟨m²⟩ − ⟨m⟩²)/T` of the chosen series; `nothing` when `magnetization`
  is `nothing`.
- `susceptibility_err::Union{Nothing, Vector{Float64}}`: Jackknife error of
  `susceptibility`.
- `n_eff::Vector{Float64}`: Effective number of samples,
  `(Σ w)² / Σ w²`, per target temperature. Outside the histogram-overlap
  window this collapses toward a single sample; distrust targets whose
  `n_eff` is much smaller than the number of measurements (see
  [`reweight`](@ref)).
"""
struct ReweightingResult
    T_sim::Float64
    temperatures::Vector{Float64}
    n_sites::Int
    energy::Vector{Float64}
    energy_err::Vector{Float64}
    specific_heat::Vector{Float64}
    specific_heat_err::Vector{Float64}
    magnetization::Union{Nothing, Vector{Float64}}
    magnetization_err::Union{Nothing, Vector{Float64}}
    susceptibility::Union{Nothing, Vector{Float64}}
    susceptibility_err::Union{Nothing, Vector{Float64}}
    n_eff::Vector{Float64}
end

function Base.show(io::IO, result::ReweightingResult)
    return print(io, "ReweightingResult(",
        length(result.temperatures), " temperatures reweighted from T_sim = ",
        result.T_sim,
        result.magnetization === nothing ? ", energy only" : ", with magnetization",
        ")")
end

# Mean of each block of bin_size consecutive samples — the reweighting
# analogue of the container's bins, used for the jackknife error estimates.
function _bin_means(x::Vector{Float64}, n_bins::Int, bin_size::Int)
    return [sum(@view x[((b - 1) * bin_size + 1):(b * bin_size)]) / bin_size
            for b in 1:n_bins]
end

# Fetch a recorded scalar time series as Float64, validating its length
# against the completed bins (a partially filled container, or one measured
# into after merging, fails the length check).
function _checked_series(container::MeasurementContainer, name::Symbol, n_expected::Int)
    haskey(container.time_series, name) || throw(ArgumentError(
        "the :$name time series was not recorded; available: " *
        "$(collect(keys(container.time_series))). Record it via the " *
        "time_series keyword of MeasurementContainer (the default :scalars " *
        "records all scalar observables)."))
    x = Float64.(container.time_series[name])
    length(x) == n_expected || throw(ArgumentError(
        "the :$name time series has $(length(x)) samples but $n_expected are " *
        "expected from the $(n_expected ÷ container.bin_size) completed bins " *
        "of $(container.bin_size); is the container fully filled?"))
    return x
end

"""
    reweight(container, T_sim, temperatures, N; magnetization = :magnetization, energy = :energy)

Reweight the measurements of a simulation run at temperature `T_sim` to the
nearby temperatures in `temperatures`, using Ferrenberg–Swendsen
single-histogram reweighting [1].

Each recorded measurement `t`, taken at inverse temperature
`β_sim = 1 / T_sim`, is assigned the Boltzmann weight ratio

```math
w_t(\\beta) = \\exp\\!\\big((\\beta_{sim} - \\beta)\\, E_t\\big),
```

where `E_t = N · e_t` is the total energy of the sample and `e_t` the
recorded per-site value. Thermal averages at `β = 1 / T` follow as the
weighted ratios `⟨A⟩(T) = Σ_t w_t A_t / Σ_t w_t`. From the reweighted
moments of the per-site energy (and magnetization, when requested) the
specific heat `N(⟨e²⟩ − ⟨e⟩²)/T²` and susceptibility
`N(⟨m²⟩ − ⟨m⟩²)/T` are computed with the same conventions as
[`analyze`](@ref), so reweighting to `T_sim` itself reproduces the direct
analysis up to roundoff.

Error bars are jackknife estimates over the container's bins: consecutive
blocks of `bin_size` measurements play the role of the bins, and every
weighted ratio is recomputed leaving one block out (via
[`jackknife_stats`](@ref)), which propagates the correlations between the
weight sums and the observable moments. As with [`analyze`](@ref), the
blocks must span many autocorrelation times for the errors to be reliable.

Reweighting only interpolates within the sampled energy histogram: it is
reliable while `|β − β_sim| σ_E ≲ 1` (with `σ_E` the total-energy standard
deviation at `T_sim`), i.e. roughly `|T − T_sim| ≲ T_sim / √(N c_v)`. The
returned `n_eff` — the effective number of samples, `(Σ w)² / Σ w²` —
collapses toward a single histogram entry outside that window; distrust
targets whose `n_eff` is much smaller than the number of measurements.

# Arguments

- `container`: A filled [`MeasurementContainer`](@ref) whose `:energy` time
    series was recorded (the default `time_series = :scalars` records it).
- `T_sim`: The temperature the run was performed at.
- `temperatures`: The target temperatures to reweight to.
- `N`: The number of sites, used to convert the recorded per-site energies
    to total energies in the weights (same role as in [`analyze`](@ref)).
- `energy`: Name of the recorded per-site energy time series to reweight
    (default `:energy`). A model recording its energy under a different name
    passes it here; the values must be per-site (the weights use
    `E_t = N · e_t`).
- `magnetization`: Name of the recorded magnetization time series to
    reweight, or `nothing` to skip magnetic quantities. The default
    `:magnetization` is the signed magnetization for [`IsingModel`](@ref)
    and the magnetization magnitude for [`XYModel`](@ref),
    [`ClockModel`](@ref), and [`HeisenbergModel`](@ref); the reweighted
    susceptibility is then `N(⟨m²⟩ − ⟨m⟩²)/T` of that series, which for the
    planar models differs from the component susceptibility of
    [`analyze`](@ref) (pass e.g. `magnetization = :magnetization_x` for one
    Cartesian component). For the Ising model below `T_c`,
    `:abs_magnetization` is the more useful choice (record it via
    `measurements = [:abs_magnetization]`).

# Throws

`ArgumentError` if `T_sim` or any target temperature is not positive, if
`N` is not positive, if the container holds fewer than two completed bins,
if the energy (or requested magnetization) time series was not recorded,
or if a time series length is inconsistent with the completed bins (e.g. a
partially filled container).

# Returns

A [`ReweightingResult`](@ref) with one entry per target temperature, in the
order of `temperatures`.

[1] A. M. Ferrenberg and R. H. Swendsen, "New Monte Carlo technique for
studying phase transitions", Phys. Rev. Lett. 61, 2635 (1988); "Optimized
Monte Carlo data analysis", Phys. Rev. Lett. 63, 1195 (1989).
"""
function reweight(
    container::MeasurementContainer, T_sim, temperatures, N;
    magnetization::Union{Symbol, Nothing} = :magnetization,
    energy::Symbol = :energy,
)
    T_sim > 0 || throw(ArgumentError("T_sim must be positive, got $T_sim"))
    N >= 1 || throw(ArgumentError("N must be positive, got $N"))
    Ts = Float64.(collect(temperatures))
    isempty(Ts) && throw(ArgumentError("temperatures must not be empty"))
    all(>(0), Ts) || throw(ArgumentError(
        "target temperatures must all be positive, got $Ts"))

    haskey(container.data, energy) || throw(ArgumentError(
        "the container has no :$energy observable to reweight"))
    series = container.time_series
    haskey(series, energy) || throw(ArgumentError(
        "the :$energy time series was not recorded; reweighting needs the " *
        "unbinned energy history (the default time_series = :scalars records it)"))

    n_bins = length(container.data[energy])
    n_bins >= 2 || throw(ArgumentError(
        "the container has $n_bins completed bin(s); at least two are " *
        "required. Was it filled by a run (see `run_simulation`)?"))
    bin_size = container.bin_size
    n_expected = n_bins * bin_size

    e = _checked_series(container, energy, n_expected)
    m = magnetization === nothing ? nothing :
        _checked_series(container, magnetization, n_expected)

    β_sim = 1 / T_sim
    e_shift = Statistics.mean(e)   # cancels in the ratios; keeps exponents small

    energies = Float64[]
    energy_err = Float64[]
    specific_heat = Float64[]
    specific_heat_err = Float64[]
    mag = Float64[]
    mag_err = Float64[]
    chi = Float64[]
    chi_err = Float64[]
    n_eff = Float64[]

    for T in Ts
        # log weight up to an additive constant; shifting by the maximum
        # avoids overflow for targets far from T_sim.
        logw = (β_sim - 1 / T) .* N .* (e .- e_shift)
        logw .-= maximum(logw)
        w = exp.(logw)
        push!(n_eff, sum(w)^2 / sum(abs2, w))

        jkW = jackknife_samples(_bin_means(w, n_bins, bin_size))
        jkE = jackknife_samples(_bin_means(w .* e, n_bins, bin_size))
        jkE2 = jackknife_samples(_bin_means(w .* abs2.(e), n_bins, bin_size))

        value, err = jackknife_stats((we, wj) -> we / wj, jkE, jkW)
        push!(energies, value)
        push!(energy_err, err)
        value, err = jackknife_stats(
            (we, we2, wj) -> N * (we2 / wj - (we / wj)^2) / T^2, jkE, jkE2, jkW)
        push!(specific_heat, value)
        push!(specific_heat_err, err)

        if m !== nothing
            jkM = jackknife_samples(_bin_means(w .* m, n_bins, bin_size))
            jkM2 = jackknife_samples(_bin_means(w .* abs2.(m), n_bins, bin_size))
            value, err = jackknife_stats((wm, wj) -> wm / wj, jkM, jkW)
            push!(mag, value)
            push!(mag_err, err)
            value, err = jackknife_stats(
                (wm, wm2, wj) -> N * (wm2 / wj - (wm / wj)^2) / T, jkM, jkM2, jkW)
            push!(chi, value)
            push!(chi_err, err)
        end
    end

    return ReweightingResult(
        Float64(T_sim), Ts, N, energies, energy_err, specific_heat, specific_heat_err,
        m === nothing ? nothing : mag, m === nothing ? nothing : mag_err,
        m === nothing ? nothing : chi, m === nothing ? nothing : chi_err,
        n_eff,
    )
end
