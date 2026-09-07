include("Observables.jl")
include("Container.jl")
include("Values.jl")
include("Correlations.jl")
include("Optional.jl")
include("BinProcessing.jl")

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