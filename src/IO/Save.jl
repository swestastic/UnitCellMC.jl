using CSV
using DataFrames

"""
    results_dirname(model, algorithm, L, T; root = "results")

Construct the output directory path for a single-temperature run, encoding
the model type, algorithm type, system size `L`, and temperature `T` into
the directory name.

# Arguments

- `model`: The model instance being simulated (only its type name is used).
- `algorithm`: The algorithm instance being used (only its type name is used).
- `L`: System size, joined component-wise with `"x"` (e.g. `(4, 4)` → `"4x4"`).
- `T`: Temperature, rounded to 6 digits for the directory name.
- `root`: Base directory under which the run directory is created.

# Returns

The path `root/ModelName_AlgorithmName_LWxH_tempT`, without creating it.
"""
function results_dirname(model, algorithm, L, T; root::AbstractString = "results")
    model_name = string(nameof(typeof(model)))
    alg_name   = string(nameof(typeof(algorithm)))
    L_str      = join(L, "x")
    T_str      = string(round(T, digits = 6))
    return joinpath(root, "$(model_name)_$(alg_name)_L$(L_str)_temp$(T_str)")
end

"""
    save_metadata(dir, model, algorithm, L, T, parameters)

Write a `metadata.txt` file in `dir` recording the model, algorithm, system
size, temperature, and simulation `parameters` used for a single run.

# Arguments

- `dir`: Directory to write `metadata.txt` into (must already exist).
- `model`: The model instance, written via its `show` representation.
- `algorithm`: The algorithm instance, written via its `show` representation.
- `L`: System size.
- `T`: Temperature.
- `parameters`: Simulation parameters (e.g. a `SimulationParameters`), written
  via its `show` representation.
"""
function save_metadata(dir, model, algorithm, L, T, parameters)
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "model = ", model)
        println(io, "algorithm = ", algorithm)
        println(io, "L = ", L)
        println(io, "temperature = ", T)
        println(io, "parameters = ", parameters)
    end
    return nothing
end

"""
    measurement_rows(results)

Flatten the scalar-valued entries of `results` into parallel vectors of
observable names, values, and errors, suitable for building a `DataFrame`.

Both `results.primary` (a collection of `(average, error)` pairs keyed by
name) and `results.derived` (a `NamedTuple`-like collection of derived
quantities, some paired with a `*_err` sibling) are scanned. Only scalar
(`Real`) entries are included; vector- or struct-valued entries (e.g.
`:correlation`, `correlation_classes`) are skipped, since those are handled
separately by [`correlation_rows`](@ref). A derived value with no matching
`*_err` property gets `missing` for its error.

# Returns

A `(names, values, errors)` tuple of equal-length vectors:
- `names::Vector{String}`: Observable names.
- `values::Vector{Float64}`: Observable values.
- `errors::Vector{Union{Float64,Missing}}`: Corresponding errors, or `missing`.
"""
function measurement_rows(results)
    names  = String[]
    values = Float64[]
    errors = Union{Float64,Missing}[]

    for (name, (avg, err)) in pairs(results.primary)
        avg isa Real || continue   # vector-valued (e.g. :correlation) handled by correlation_rows
        push!(names, string(name))
        push!(values, avg)
        push!(errors, err)
    end

    derived = results.derived
    for name in propertynames(derived)
        value = getproperty(derived, name)
        value isa Real || continue   # skips vectors (correlation_connected) and structs (correlation_classes)

        name_str = string(name)
        endswith(name_str, "_err") && continue   # picked up below alongside its paired value

        err_name = Symbol(name_str * "_err")
        err = hasproperty(derived, err_name) ? getproperty(derived, err_name) : missing

        push!(names, name_str)
        push!(values, value)
        push!(errors, err)
    end

    return names, values, errors
end

"""
    correlation_rows(results)

Build a `DataFrame` of per-displacement-class correlation data from
`results`, or return `nothing` if `results` has no `correlation_classes`.

Each row corresponds to one displacement class (see
[`build_displacement_classes`](@ref)) and records its displacement
components, orbital pair, raw correlation value/error, and connected
correlation value/error.

# Arguments

- `results`: A results object with `results.derived.correlation_classes`
  (a vector of class descriptors with `displacement` and `orbitals` fields),
  `results.primary[:correlation]` (raw correlation `(values, errors)`), and
  `results.derived.correlation_connected` / `correlation_connected_err`.

# Returns

A `DataFrame` with columns `disp_1, ..., disp_D, orbital_i, orbital_j,
raw_value, raw_error, connected_value, connected_error`, or `nothing` if
`results` has no correlation data.
"""
function correlation_rows(results)
    hasproperty(results.derived, :correlation_classes) || return nothing

    classes = results.derived.correlation_classes
    isempty(classes) && return nothing
    raw_avg, raw_err = results.primary[:correlation]
    conn_avg = results.derived.correlation_connected
    conn_err = results.derived.correlation_connected_err

    D = length(first(classes).displacement)

    df = DataFrame()
    for k in 1:D
        df[!, Symbol("disp_$k")] = [c.displacement[k] for c in classes]
    end
    df.orbital_i       = [c.orbitals[1] for c in classes]
    df.orbital_j       = [c.orbitals[2] for c in classes]
    df.raw_value       = raw_avg
    df.raw_error       = raw_err
    df.connected_value = conn_avg
    df.connected_error = conn_err

    return df
end

"""
    save_measurements(dir, results)

Write `results`'s scalar observables (via [`measurement_rows`](@ref)) to
`dir/measurement_results.csv`, with columns `observable`, `value`, `error`.
"""
function save_measurements(dir, results)
    names, values, errors = measurement_rows(results)
    df = DataFrame(observable = names, value = values, error = errors)
    CSV.write(joinpath(dir, "measurement_results.csv"), df)
    return nothing
end

"""
    save_correlations(dir, results)

Write `results`'s per-displacement-class correlation data (via
[`correlation_rows`](@ref)) to `dir/correlation_results.csv`. Does nothing
if `results` has no correlation data.
"""
function save_correlations(dir, results)
    df = correlation_rows(results)
    df === nothing && return nothing
    CSV.write(joinpath(dir, "correlation_results.csv"), df)
    return nothing
end

"""
    save_results(model, algorithm, L, T, results, parameters; root = "results")

Save all output for a single-temperature run: create the run directory,
write metadata, measurements, and correlations (if any).

Combines [`results_dirname`](@ref), [`save_metadata`](@ref),
[`save_measurements`](@ref), and [`save_correlations`](@ref) into one call.

# Arguments

- `model`, `algorithm`, `L`, `T`: Identify the run, used to name the directory
  and recorded in its metadata.
- `results`: The measurement results to save.
- `parameters`: Simulation parameters recorded in the metadata.
- `root`: Base directory under which the run directory is created.

# Returns

The path to the created run directory.
"""
function save_results(model, algorithm, L, T, results, parameters; root::AbstractString = "results")
    dir = results_dirname(model, algorithm, L, T; root = root)
    mkpath(dir)
    save_metadata(dir, model, algorithm, L, T, parameters)
    save_measurements(dir, results)
    save_correlations(dir, results)
    return dir
end

"""
    sweep_dirname(model, algorithm, L; root = "results")

Construct the output directory path for a temperature sweep (multiple
temperatures saved together), encoding the model type, algorithm type, and
system size `L` into the directory name.

# Arguments

- `model`: The model instance being simulated (only its type name is used).
- `algorithm`: The algorithm instance being used (only its type name is used).
- `L`: System size, joined component-wise with `"x"`.
- `root`: Base directory under which the sweep directory is created.

# Returns

The path `root/ModelName_AlgorithmName_LWxH_sweep`, without creating it.
"""
function sweep_dirname(model, algorithm, L; root::AbstractString = "results")
    model_name = string(nameof(typeof(model)))
    alg_name   = string(nameof(typeof(algorithm)))
    L_str      = join(L, "x")
    return joinpath(root, "$(model_name)_$(alg_name)_L$(L_str)_sweep")
end

"""
    save_sweep_metadata(dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins)

Write a `metadata.txt` file in `dir` recording the model, algorithm, system
size, the full list of temperatures `Ts`, and the sweep's simulation
parameters.

# Arguments

- `dir`: Directory to write `metadata.txt` into (must already exist).
- `model`: The model instance, written via its `show` representation.
- `algorithm`: The algorithm instance, written via its `show` representation.
- `L`: System size.
- `Ts`: The temperatures included in the sweep.
- `n_thermalization`, `n_measurements`, `n_unmeasured`, `n_bins`: Sweep
  simulation parameters, recorded as-is.
"""
function save_sweep_metadata(dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins)
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "model = ", model)
        println(io, "algorithm = ", algorithm)
        println(io, "L = ", L)
        println(io, "temperatures = ", Ts)
        println(io, "n_thermalization = ", n_thermalization)
        println(io, "n_measurements = ", n_measurements)
        println(io, "n_unmeasured = ", n_unmeasured)
        println(io, "n_bins = ", n_bins)
    end
    return nothing
end

"""
    save_sweep_results(sweep_results, Ts, model, algorithm, L,
                        n_thermalization, n_measurements, n_unmeasured, n_bins;
                        root = "results")

Save the combined output of a temperature sweep into a single directory: one
`measurement_results.csv` and one `correlation_results.csv` spanning all
temperatures, each with a leading `temperature` column, plus sweep metadata.

`sweep_results[i]` must correspond to `Ts[i]`; per-temperature rows are built
with [`measurement_rows`](@ref) and [`correlation_rows`](@ref) and
concatenated across all temperatures. Temperatures with no correlation data
are simply omitted from `correlation_results.csv`; if none have correlation
data, that file is not written at all.

# Arguments

- `sweep_results`: Results for each temperature, same length and order as `Ts`.
- `Ts`: Temperatures included in the sweep.
- `model`, `algorithm`, `L`: Identify the run, used to name the directory
  and recorded in its metadata.
- `n_thermalization`, `n_measurements`, `n_unmeasured`, `n_bins`: Sweep
  simulation parameters, recorded in the metadata.
- `root`: Base directory under which the sweep directory is created.

# Throws

`ArgumentError` if `sweep_results` and `Ts` have different lengths.

# Returns

The path to the created sweep directory.
"""
function save_sweep_results(
    sweep_results, Ts, model, algorithm, L,
    n_thermalization, n_measurements, n_unmeasured, n_bins;
    root::AbstractString = "results"
)
    length(sweep_results) == length(Ts) ||
        throw(ArgumentError("sweep_results ($(length(sweep_results))) and Ts ($(length(Ts))) must have the same length"))

    dir = sweep_dirname(model, algorithm, L; root = root)
    mkpath(dir)
    save_sweep_metadata(dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins)

    measurement_dfs = map(zip(Ts, sweep_results)) do (T, results)
        names, values, errors = measurement_rows(results)
        DataFrame(temperature = T, observable = names, value = values, error = errors)
    end
    CSV.write(joinpath(dir, "measurement_results.csv"), reduce(vcat, measurement_dfs))

    correlation_dfs = map(zip(Ts, sweep_results)) do (T, results)
        df = correlation_rows(results)
        df === nothing && return nothing
        insertcols!(df, 1, :temperature => T)
        df
    end
    filter!(!isnothing, correlation_dfs)
    correlation_path = joinpath(dir, "correlation_results.csv")
    if isempty(correlation_dfs)
        isfile(correlation_path) && rm(correlation_path)
    else
        CSV.write(correlation_path, reduce(vcat, correlation_dfs))
    end

    return dir
end