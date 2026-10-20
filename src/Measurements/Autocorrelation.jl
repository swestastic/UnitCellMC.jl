"""
    AutocorrelationResult

Result of an autocorrelation analysis of one observable's Monte Carlo time
series.

All times are in units of the *sampling interval* of the series. For the
time series recorded by [`measure!`](@ref) that is one measurement, i.e.
`n_unmeasured + 1` sweeps (see [`SimulationParameters`](@ref)); for a pilot
run with `n_unmeasured = 0` (as in [`autotune_binning`](@ref)) it is one
sweep.

# Fields

- `tau::Float64`: Integrated autocorrelation time
  `τ_int = 1/2 + Σ_{k=1}^{K} ρ(k)`, with `K` chosen by self-consistent
  windowing. With this convention uncorrelated samples have `τ_int = 1/2`
  and the variance of the sample mean is `2 τ_int σ² / n`. Finite when
  `status` is `:resolved` or `:anticorrelated`, `NaN` otherwise.
- `tau_err::Float64`: Statistical error of `tau` from the Madras–Sokal
  formula `max(τ_int, 1/2) √(2(2K + 1)/n)`.
- `acf::Vector{Float64}`: Normalized autocorrelation function `ρ(k)` for
  `k = lags`, with `ρ(0) = 1`. It is computed only up to the window `K`
  (or up to the maximum lag when unresolved), not over the whole series.
- `lags::Vector{Int}`: Lags `0, 1, …` corresponding to `acf`.
- `n::Int`: Number of samples in the series (summed over all chains).
- `window::Int`: The self-consistent window `K` (`0` unless resolved).
- `status::Symbol`: One of
  - `:resolved`: `tau` is a valid estimate;
  - `:anticorrelated`: `tau` is significantly below `1/2`
    (`tau + 2 tau_err < 1/2`), i.e. successive samples are anticorrelated.
    This happens e.g. for the signed magnetization under cluster updates,
    which flip the sign of a large cluster every step. The true value lies
    in `(0, 1/2)` but the windowed estimate is unreliable (it may even be
    negative); naive or binned error bars are then conservative, so for
    binning purposes such an observable behaves like uncorrelated data;
  - `:unresolved`: the window condition was never met within the maximum
    lag, i.e. the series is too short compared to its autocorrelation time.
    `acf` then holds the computed lags, and `0.5 + sum(acf[2:end])` is a
    lower-bound estimate of `tau`;
  - `:constant`: the series never changes (e.g. the energy of a frozen
    configuration), so there are no fluctuations to correlate;
  - `:insufficient_data`: fewer than two samples;
  - `:unsupported`: the element type cannot be analyzed.

For vector-valued observables (e.g. a correlation function), every
non-constant component is analyzed separately and the result for the
*slowest* component is returned: an unresolved component takes precedence,
otherwise the component with the largest `tau`.
"""
struct AutocorrelationResult
    tau::Float64
    tau_err::Float64
    acf::Vector{Float64}
    lags::Vector{Int}
    n::Int
    window::Int
    status::Symbol
end

AutocorrelationResult(n::Integer, status::Symbol) =
    AutocorrelationResult(NaN, NaN, Float64[], Int[], n, 0, status)

"""
    tau_lower_bound(result::AutocorrelationResult)

Return the integrated autocorrelation time accumulated up to the largest lag
that was computed. For a `:resolved` result this is `result.tau`; for an
`:unresolved` result it is a lower-bound estimate (the true `tau` is larger
than what the series could resolve); an `:anticorrelated` result returns
`1/2`, the value for uncorrelated samples, which is an upper bound for it.
Returns `NaN` for any other status.
"""
function tau_lower_bound(result::AutocorrelationResult)
    result.status === :resolved && return result.tau
    result.status === :anticorrelated && return 0.5
    result.status === :unresolved || return NaN
    return 0.5 + sum(@view result.acf[2:end])
end

# --- internal helpers -------------------------------------------------------

# Index ranges of the independent Markov chains that were concatenated into
# one series. `nothing` means a single chain.
_chain_ranges(n::Integer, ::Nothing) = [1:n]
function _chain_ranges(n::Integer, chain_lengths::AbstractVector{<:Integer})
    isempty(chain_lengths) && return [1:n]
    all(>(0), chain_lengths) ||
        throw(ArgumentError("chain_lengths must all be positive, got $chain_lengths"))
    sum(chain_lengths) == n || throw(ArgumentError(
        "chain_lengths sum to $(sum(chain_lengths)) but the series has $n samples"
    ))
    stops = cumsum(chain_lengths)
    return [(stop - len + 1):stop for (stop, len) in zip(stops, chain_lengths)]
end

# Default maximum lag: one tenth of the shortest chain. Resolving tau needs a
# window K ≈ window·tau ≤ max_lag, so this resolves tau up to ~n/60. Beyond
# that the Madras-Sokal error √(2(2K+1)/n) exceeds ~60% and an estimate is not
# meaningful anyway; the cap also keeps the direct O(n·K) sum cheap.
function _resolve_max_lag(max_lag, chains)
    longest = maximum(length, chains)
    longest >= 2 || throw(ArgumentError("every chain has fewer than two samples"))
    lag = something(max_lag, max(1, minimum(length, chains) ÷ 10))
    lag >= 1 || throw(ArgumentError("max_lag must be at least 1, got $lag"))
    return min(lag, longest - 1)
end

# Σ over chains of Σ_i c[i] c[i+k], never pairing samples from different chains.
function _lag_sum(c::Vector{Float64}, chains, k::Int)
    s = 0.0
    @inbounds for r in chains
        for i in first(r):(last(r) - k)
            s += c[i] * c[i + k]
        end
    end
    return s
end

# Centered copy of `x` and its sum of squares, or `nothing` if `x` is constant.
# Exact equality is used for the constant check so that floating-point
# rounding in the mean of a constant series is not mistaken for fluctuations.
function _centered(x::AbstractVector{<:Real})
    x1 = first(x)
    all(==(x1), x) && return nothing
    c = Float64.(x)
    c .-= Statistics.mean(c)   # one global mean over all chains
    return c, sum(abs2, c)
end

function _autocorrelation_scalar(x::AbstractVector{<:Real}, chains, max_lag::Int, window::Real)
    n = length(x)
    centered = _centered(x)
    centered === nothing && return AutocorrelationResult(n, :constant)
    c, c0 = centered

    acf = [1.0]
    tau = 0.5
    K = 0
    for k in 1:max_lag
        rho = _lag_sum(c, chains, k) / c0
        push!(acf, rho)
        tau += rho
        if k >= window * tau
            K = k
            break
        end
    end
    lags = collect(0:(length(acf) - 1))
    K == 0 && return AutocorrelationResult(NaN, NaN, acf, lags, n, 0, :unresolved)

    tau_err = max(tau, 0.5) * sqrt(2 * (2K + 1) / n)
    status = tau + 2 * tau_err < 0.5 ? :anticorrelated : :resolved
    return AutocorrelationResult(tau, tau_err, acf, lags, n, K, status)
end

# Pick the more slowly decorrelating of two results (unresolved beats resolved).
_slower(::Nothing, b::AutocorrelationResult) = b
function _slower(a::AutocorrelationResult, b::AutocorrelationResult)
    a_unres = a.status === :unresolved
    b_unres = b.status === :unresolved
    a_unres != b_unres && return a_unres ? a : b
    return tau_lower_bound(b) > tau_lower_bound(a) ? b : a
end

function _check_window(window)
    window > 0 || throw(ArgumentError("window must be positive, got $window"))
    return nothing
end

# --- public API ----------------------------------------------------------------

"""
    autocorrelation_function(x; max_lag = length(x) ÷ 2, chain_lengths = nothing)

Compute the normalized autocorrelation function of a scalar time series,

    ρ(k) = Σ_i (x[i] - x̄)(x[i + k] - x̄) / Σ_i (x[i] - x̄)²,   ρ(0) = 1,

using the standard (biased, `1/n`-normalized) estimator, which has lower
variance than the `1/(n - k)` form and is the usual choice for windowed
estimates of the integrated autocorrelation time.

If `x` is the concatenation of several independent Markov chains, pass their
lengths in `chain_lengths`: lagged products are then only formed within each
chain (the mean `x̄` is still the global mean).

# Returns

`(lags, acf)` with `lags = 0:max_lag` and `acf[k + 1] = ρ(k)`.

# Throws

`ArgumentError` if `x` has fewer than two samples or is constant, or if
`chain_lengths` is inconsistent with `x`.
"""
function autocorrelation_function(
    x::AbstractVector{<:Real};
    max_lag::Union{Integer, Nothing} = nothing,
    chain_lengths::Union{AbstractVector{<:Integer}, Nothing} = nothing,
)
    n = length(x)
    n >= 2 || throw(ArgumentError("autocorrelation_function needs at least two samples, got $n"))
    chains = _chain_ranges(n, chain_lengths)
    longest = maximum(length, chains)
    longest >= 2 || throw(ArgumentError("every chain has fewer than two samples"))
    max_lag = clamp(something(max_lag, n ÷ 2), 1, longest - 1)

    centered = _centered(x)
    centered === nothing && throw(ArgumentError("time series has zero variance"))
    c, c0 = centered

    acf = Vector{Float64}(undef, max_lag + 1)
    acf[1] = 1.0
    for k in 1:max_lag
        acf[k + 1] = _lag_sum(c, chains, k) / c0
    end
    return (0:max_lag, acf)
end

"""
    autocorrelation(series; max_lag = nothing, window = 6.0, chain_lengths = nothing)

Estimate the integrated autocorrelation time of one observable's time series
with Sokal's self-consistent window: `τ_int(K) = 1/2 + Σ_{k=1}^{K} ρ(k)` is
accumulated lag by lag and the sum stops at the first `K` with
`K ≥ window × τ_int(K)`. The default `window = 6` is appropriate for roughly
exponential decay; use a larger value for slowly decaying tails.

`series` may be scalar (`AbstractVector{<:Real}`) or vector-valued
(`AbstractVector{<:AbstractVector{<:Real}}`, e.g. a correlation function
sampled at several displacements). Vector-valued series are analyzed per
component and the slowest non-constant component is reported (see
[`AutocorrelationResult`](@ref)).

# Keywords

- `max_lag`: Largest lag considered. Defaults to one tenth of the shortest
  chain; if the window condition is not met by then the result is
  `:unresolved`.
- `window`: Window constant `c` in `K ≥ c τ_int(K)`.
- `chain_lengths`: Lengths of independent chains concatenated in `series`
  (e.g. several runs merged by `merge_containers`); lagged products never
  straddle two chains.

A constant series returns a `:constant` result rather than throwing.

# Returns

An [`AutocorrelationResult`](@ref).
"""
function autocorrelation(
    series::AbstractVector{<:Real};
    max_lag::Union{Integer, Nothing} = nothing,
    window::Real = 6.0,
    chain_lengths::Union{AbstractVector{<:Integer}, Nothing} = nothing,
)
    _check_window(window)
    n = length(series)
    n >= 2 || return AutocorrelationResult(n, :insufficient_data)
    chains = _chain_ranges(n, chain_lengths)
    lag = _resolve_max_lag(max_lag, chains)
    return _autocorrelation_scalar(series, chains, lag, window)
end

function autocorrelation(
    series::AbstractVector{<:AbstractVector{<:Real}};
    max_lag::Union{Integer, Nothing} = nothing,
    window::Real = 6.0,
    chain_lengths::Union{AbstractVector{<:Integer}, Nothing} = nothing,
)
    _check_window(window)
    n = length(series)
    n >= 2 || return AutocorrelationResult(n, :insufficient_data)
    n_components = length(first(series))
    all(s -> length(s) == n_components, series) ||
        throw(ArgumentError("all vector-valued samples must have the same length"))
    chains = _chain_ranges(n, chain_lengths)
    lag = _resolve_max_lag(max_lag, chains)

    slowest = nothing
    component = Vector{Float64}(undef, n)
    for comp in 1:n_components
        for i in 1:n
            component[i] = series[i][comp]
        end
        result = _autocorrelation_scalar(component, chains, lag, window)
        result.status === :constant && continue
        slowest = _slower(slowest, result)
    end
    return something(slowest, AutocorrelationResult(n, :constant))
end

# Fallback for element types that cannot be analyzed (e.g. custom observables).
autocorrelation(series::AbstractVector; kwargs...) =
    AutocorrelationResult(length(series), :unsupported)

"""
    integrated_autocorrelation_time(series; max_lag = nothing, window = 6.0,
                                    chain_lengths = nothing)

Shorthand for `autocorrelation(series; ...).tau`; see [`autocorrelation`](@ref).
Returns `NaN` when the autocorrelation time cannot be resolved or the series
is constant.
"""
integrated_autocorrelation_time(series::AbstractVector; kwargs...) =
    autocorrelation(series; kwargs...).tau

# Upper bound on τ_int of a series thinned to every `s`-th sample, given τ_int
# of the full series. For a non-increasing ρ (the generic case for MC chains),
# ρ(s·k) ≤ (1/s) Σ_{j=s(k-1)+1}^{s·k} ρ(j), so summing over k gives
# τ_s - 1/2 ≤ (τ - 1/2)/s. This is exact in both limits (s = 1 and s ≫ τ) and
# does not assume a single exponential mode.
_thinned_tau_bound(tau::Real, s::Integer) = s == 1 ? Float64(tau) : 0.5 + (tau - 0.5) / s

"""
    recommended_binning(tau; bin_factor = 10, n_unmeasured = nothing)

Turn an integrated autocorrelation time `tau`, measured in **sweeps** (i.e.
from a series sampled every sweep, as produced by [`autotune_binning`](@ref)),
into a suggested `(bin_size, n_unmeasured)`.

- `n_unmeasured`: if not given, measurements are spaced
  `s = max(1, round(tau))` sweeps apart, i.e. `n_unmeasured = s - 1`. Closer
  spacing would add little information per measurement, wider spacing would
  discard it. Pass `n_unmeasured` explicitly to keep a spacing of your choice
  (e.g. `0` when measurements are cheap).
- `bin_size`: the number of *measurements* per bin, chosen so that each bin
  spans `bin_factor` integrated autocorrelation times in units of
  measurements: `bin_size = ceil(bin_factor × τ_meas)`, where
  `τ_meas = 1/2 + (tau - 1/2)/s` is an upper bound on the integrated time of
  the thinned series (valid whenever the autocorrelation function is
  non-increasing; it tends to `1/2` once measurements are effectively
  independent). Bins spanning many `τ_meas` are nearly independent, so the
  binned/jackknife errors are reliable; the residual relative underestimate
  of the variance is of order `τ_meas / bin_size`.

`tau` below `1/2` (anticorrelation) is treated as `1/2`.

# Returns

A named tuple `(bin_size = ..., n_unmeasured = ...)`. Choose the production
`n_measurements = bin_size × n_bins` for the desired number of bins.

# Throws

`ArgumentError` if `tau` is not finite and positive (an unresolved `NaN`
autocorrelation time means the pilot run was too short; there is no safe
default), or if `bin_factor` is not positive.
"""
function recommended_binning(
    tau::Real;
    bin_factor::Real = 10,
    n_unmeasured::Union{Integer, Nothing} = nothing,
)
    isfinite(tau) && tau > 0 || throw(ArgumentError(
        "tau must be finite and positive, got $tau; a NaN autocorrelation time " *
        "means the series was too short to resolve it, so run a longer pilot"
    ))
    bin_factor > 0 || throw(ArgumentError("bin_factor must be positive, got $bin_factor"))
    τ = max(Float64(tau), 0.5)
    spacing = if n_unmeasured === nothing
        max(1, round(Int, τ))
    else
        n_unmeasured >= 0 || throw(ArgumentError("n_unmeasured must be >= 0, got $n_unmeasured"))
        n_unmeasured + 1
    end
    tau_measurements = _thinned_tau_bound(τ, spacing)
    bin_size = max(1, ceil(Int, bin_factor * tau_measurements))
    return (bin_size = bin_size, n_unmeasured = spacing - 1)
end
