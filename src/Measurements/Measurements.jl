include("Observables.jl")
include("Container.jl")
include("Values.jl")
include("Correlations.jl")
include("StructureFactor.jl")
include("Optional.jl")
include("BinProcessing.jl")
include("Autocorrelation.jl")

"""
    measure!(measurements::MeasurementContainer, state; diagnostics)

Take one measurement of `state` and accumulate it into `measurements`,
flushing a completed bin into `data` once `bin_size` measurements have been
accumulated.

Each observable in `measurements.observables` is evaluated on `state` via
its `value` function, and the supplied algorithm diagnostic values are added
to the same named tuple. The results are added elementwise into
`measurements.bin_sums`. Once `measurements.bin_count` reaches
`measurements.bin_size`, the accumulated sums are averaged (dividing by
`bin_size`) and pushed onto the corresponding vector in `measurements.data`,
after which `bin_sums` is reset to zero (array accumulators are zeroed in
place and reused for the next bin) and `bin_count` returns to `0` to begin
accumulating the next bin.

# Arguments

- `measurements`: The `MeasurementContainer` to update in place.
- `state`: The current configuration to measure, passed to each observable's
    `value` function.
- `diagnostics`: A named tuple of per-sweep algorithm statistics. Its names
    must match the `diagnostics` requested when the container was constructed.

# Returns

`nothing`. `measurements` is mutated in place.
"""
function measure!(
    measurements::MeasurementContainer{D, N}, state; diagnostics = NamedTuple()
) where {D, N}
    given_names = propertynames(diagnostics)
    given_names == D ||
        throw(ArgumentError("Expected diagnostics $D, got $given_names"))

    # N carries the observable names at the type level and the observables are
    # a concretely typed tuple, so this NamedTuple is fully inferred and each
    # value function is called without dynamic dispatch.
    values = merge(
        NamedTuple{N}(map(o -> o.value(state), measurements.observables)),
        diagnostics
    )

    measurements.bin_count += 1
    measurements.bin_sums = map(_accumulate, measurements.bin_sums, values)

    # Record the per-sweep raw values for autocorrelation analysis. Restricting
    # `values` to the recorded names keeps this fully inferred and
    # allocation-free (a `for` loop over `keys(series)` would not constant-fold
    # the field accesses).
    series = measurements.time_series
    recorded = NamedTuple{keys(series)}(values)
    map(push!, Base.values(series), Base.values(recorded))

    if measurements.bin_count == measurements.bin_size
        bin_size = measurements.bin_size
        measurements.data = map(measurements.data, measurements.bin_sums) do bins, s
            push!(bins, s ./ bin_size)
            bins
        end
        measurements.bin_sums = map(_reset_sum, measurements.bin_sums)
        measurements.bin_count = 0
    end
    return nothing
end

# Add one sample into a running bin sum. Array accumulators are updated in
# place, so vector-valued observables do not allocate per measurement.
_accumulate(sum::Number, value::Number) = sum + value
_accumulate(sum::AbstractArray, value::AbstractArray) = (sum .+= value; sum)

# Reset a flushed bin sum to zero. Array accumulators are zeroed in place and
# reused for the next bin instead of being reallocated.
_reset_sum(sum::Number) = zero(sum)
_reset_sum(sum::AbstractArray) = fill!(sum, zero(eltype(sum)))