include("Container.jl")
include("Values.jl")
include("BinProcessing.jl")

function measure!(
    measurements::MeasurementContainer,
    model,
    state
)
    values = measurement_values(model, state)

    measurements.bin_count += 1

    # Add measurement to current bin
    measurements.bin_sums = map(
        +,
        measurements.bin_sums,
        values
    )

    # Bin is complete
    if measurements.bin_count == measurements.bin_size

        measurements.data = map(
            measurements.data,
            measurements.bin_sums
        ) do bins, sum
            push!(bins, sum / measurements.bin_size)
            bins
        end

        # Reset current bin
        measurements.bin_sums = map(
            _ -> 0.0,
            measurements.bin_sums
        )

        measurements.bin_count = 0
    end

    return nothing
end