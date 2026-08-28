mutable struct MeasurementContainer{T, S}
    data::T
    bin_sums::S
    bin_count::Int
    bin_size::Int
end

function MeasurementContainer(
    model, 
    n_steps, 
    n_bins
)
    n_steps % n_bins == 0 ||
        throw(ArgumentError("n_steps must be divisible by n_bins"))

    data = measurement_template(model)
    bin_sums = map(_ -> 0.0, data)

    return MeasurementContainer(
        data,
        bin_sums,
        0,
        n_steps ÷ n_bins
    )
end