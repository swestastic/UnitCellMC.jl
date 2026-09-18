include("Observables.jl")
include("Container.jl")
include("Values.jl")
include("Correlations.jl")
include("Optional.jl")
include("BinProcessing.jl")

"""
    measure!(measurements::MeasurementContainer, state)

Take one measurement of `state` and accumulate it into `measurements`,
flushing a completed bin into `data` once `bin_size` measurements have been
accumulated.

Each observable in `measurements.observables` is evaluated on `state` via
its `value` function, and the results are added elementwise into
`measurements.bin_sums`. Once `measurements.bin_count` reaches
`measurements.bin_size`, the accumulated sums are averaged (dividing by
`bin_size`) and pushed onto the corresponding vector in `measurements.data`,
after which `bin_sums` is reset to zero (via each observable's `zero()`) and
`bin_count` returns to `0` to begin accumulating the next bin.

# Arguments

- `measurements`: The `MeasurementContainer` to update in place.
- `state`: The current configuration to measure, passed to each observable's
  `value` function.

# Returns

`nothing`. `measurements` is mutated in place.
"""
function measure!(measurements::MeasurementContainer, state)
    values = NamedTuple(o.name => o.value(state) for o in measurements.observables)

    measurements.bin_count += 1
    measurements.bin_sums = map(+, measurements.bin_sums, values)

    if measurements.bin_count == measurements.bin_size
        measurements.data = map(measurements.data, measurements.bin_sums) do bins, s
            push!(bins, s / measurements.bin_size)
            bins
        end
        measurements.bin_sums = NamedTuple(o.name => o.zero() for o in measurements.observables)
        measurements.bin_count = 0
    end
    return nothing
end