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
    save_metadata(dir, model, algorithm, L, T, parameters; seed = global_seed())

Write a `metadata.txt` file in `dir` recording the model, algorithm, system
size, temperature, and simulation `parameters` used for a single run, plus
the RNG `seed` and Julia version needed to recreate it.

# Arguments

- `dir`: Directory to write `metadata.txt` into (must already exist).
- `model`: The model instance, written via its `show` representation.
- `algorithm`: The algorithm instance, written via its `show` representation.
- `L`: System size.
- `T`: Temperature.
- `parameters`: Simulation parameters (e.g. a `SimulationParameters`), written
  via its `show` representation.
- `seed`: RNG seed of the run; defaults to the seed set with
  [`set_global_seed!`](@ref). Written as `unknown` when `nothing`.
"""
function save_metadata(dir, model, algorithm, L, T, parameters; seed = global_seed())
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "model = ", model)
        println(io, "algorithm = ", algorithm)
        println(io, "L = ", L)
        println(io, "temperature = ", T)
        println(io, "parameters = ", parameters)
        write_reproducibility_metadata(io, seed)
    end
    return nothing
end

"""
    write_reproducibility_metadata(io, seed)

Write the `seed = ...` (or `seed = unknown`) and `julia_version = ...` lines
of `metadata.txt`. The Julia version is recorded because the random streams
produced from a seed may change between Julia releases.
"""
function write_reproducibility_metadata(io, seed)
    println(io, "seed = ", something(seed, "unknown"))
    println(io, "julia_version = ", VERSION)
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
`*_err` property gets `missing` for its error. Primary entries are emitted in
sorted name order (`primary` is a `Dict`, whose iteration order varies
between sessions), so the CSV output is deterministic.

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

    for name in sort!(collect(keys(results.primary)))
        avg, err = results.primary[name]
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
    structure_factor_rows(results)

Build a `DataFrame` of structure-factor data from `results`, or return
`nothing` if `results` has no `structure_factor_classes` (i.e. the
`:structure_factor` measurement was not requested).

Each row corresponds to one [`StructureFactorClass`](@ref): an orbital pair
and a momentum on the finite-lattice grid (see
[`build_momentum_classes`](@ref)).

# Arguments

- `results`: A results object with `results.derived.structure_factor_classes`
  and `results.primary[:structure_factor]` (complex `(values, errors)`, as
  returned by [`analyze`](@ref)).

# Returns

A `DataFrame` with columns `k_index_1, ..., k_index_D` (integer grid
location), `k_1, ..., k_D` (Cartesian momentum), `orbital_i`, `orbital_j`,
`real_value`, `real_error`, `imag_value`, `imag_error`, or `nothing` if
`results` has no structure-factor data.
"""
function structure_factor_rows(results)
    hasproperty(results.derived, :structure_factor_classes) || return nothing

    classes = results.derived.structure_factor_classes
    isempty(classes) && return nothing
    avg, err = results.primary[:structure_factor]

    D = length(first(classes).momentum_index)

    df = DataFrame()
    for d in 1:D
        df[!, Symbol("k_index_$d")] = [c.momentum_index[d] for c in classes]
    end
    for d in 1:D
        df[!, Symbol("k_$d")] = [c.momentum[d] for c in classes]
    end
    df.orbital_i  = [c.orbitals[1] for c in classes]
    df.orbital_j  = [c.orbitals[2] for c in classes]
    df.real_value = real.(avg)
    df.real_error = real.(err)
    df.imag_value = imag.(avg)
    df.imag_error = imag.(err)

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
    save_structure_factor(dir, results)

Write `results`'s structure-factor data (via [`structure_factor_rows`](@ref))
to `dir/structure_factor_results.csv`. Does nothing if `results` has no
structure-factor data.
"""
function save_structure_factor(dir, results)
    df = structure_factor_rows(results)
    df === nothing && return nothing
    CSV.write(joinpath(dir, "structure_factor_results.csv"), df)
    return nothing
end

"""
    autocorrelation_rows(results)

Build a `DataFrame` of integrated autocorrelation times from
`results.autocorrelation` (as produced by [`analyze`](@ref)), one row per
observable sorted by name, or return `nothing` if `results` has no
autocorrelation data.

Columns: `observable`, `tau`, `tau_err`, `window`, `n_samples`, `status`.
`tau` is in units of measurements (`n_unmeasured + 1` sweeps) and is `NaN`
unless `status == "resolved"`; see [`AutocorrelationResult`](@ref).
"""
function autocorrelation_rows(results)
    hasproperty(results, :autocorrelation) || return nothing
    ac = results.autocorrelation
    isempty(ac) && return nothing
    names = sort!(collect(keys(ac)))
    return DataFrame(
        observable = string.(names),
        tau        = [ac[name].tau for name in names],
        tau_err    = [ac[name].tau_err for name in names],
        window     = [ac[name].window for name in names],
        n_samples  = [ac[name].n for name in names],
        status     = [string(ac[name].status) for name in names],
    )
end

"""
    save_autocorrelation(dir, results)

Write `results`'s integrated autocorrelation times (via
[`autocorrelation_rows`](@ref)) to `dir/autocorrelation_results.csv`. Does
nothing if `results` has no autocorrelation data.
"""
function save_autocorrelation(dir, results)
    df = autocorrelation_rows(results)
    df === nothing && return nothing
    CSV.write(joinpath(dir, "autocorrelation_results.csv"), df)
    return nothing
end

"""
    save_results(model, algorithm, geometry, T, results, parameters;
        root = "results", seed = global_seed())

Save all output for a single-temperature run: create the run directory,
write metadata, measurements, correlations (if any), the structure factor
(if any), and autocorrelation times (if any).

Combines [`results_dirname`](@ref), [`save_metadata`](@ref),
[`save_measurements`](@ref), [`save_correlations`](@ref),
[`save_structure_factor`](@ref), and [`save_autocorrelation`](@ref) into
one call.

# Arguments

- `model`, `algorithm`, `geometry`, `T`: Identify the run; the system size is
  read from `geometry.lattice.L` to name the directory and record the
  metadata.
- `results`: The measurement results to save.
- `parameters`: Simulation parameters recorded in the metadata.
- `root`: Base directory under which the run directory is created.
- `seed`: RNG seed recorded in the metadata (see [`save_metadata`](@ref)).

# Returns

The path to the created run directory.
"""
function save_results(
    model, algorithm, geometry, T, results, parameters;
    root::AbstractString = "results", seed = global_seed()
)
    L = geometry.lattice.L
    dir = results_dirname(model, algorithm, L, T; root = root)
    mkpath(dir)
    save_metadata(dir, model, algorithm, L, T, parameters; seed = seed)
    save_measurements(dir, results)
    save_correlations(dir, results)
    save_structure_factor(dir, results)
    save_autocorrelation(dir, results)
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
    save_sweep_metadata(dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins;
        seed = global_seed())

Write a `metadata.txt` file in `dir` recording the model, algorithm, system
size, the full list of temperatures `Ts`, the sweep's simulation
parameters, and the RNG `seed` and Julia version needed to recreate it.

# Arguments

- `dir`: Directory to write `metadata.txt` into (must already exist).
- `model`: The model instance, written via its `show` representation.
- `algorithm`: The algorithm instance, written via its `show` representation.
- `L`: System size.
- `Ts`: The temperatures included in the sweep.
- `n_thermalization`, `n_measurements`, `n_unmeasured`, `n_bins`: Sweep
  simulation parameters, recorded as-is.
- `seed`: RNG seed of the sweep; defaults to the seed set with
  [`set_global_seed!`](@ref). Written as `unknown` when `nothing`.
"""
function save_sweep_metadata(
    dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins;
    seed = global_seed()
)
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "model = ", model)
        println(io, "algorithm = ", algorithm)
        println(io, "L = ", L)
        println(io, "temperatures = ", Ts)
        println(io, "n_thermalization = ", n_thermalization)
        println(io, "n_measurements = ", n_measurements)
        println(io, "n_unmeasured = ", n_unmeasured)
        println(io, "n_bins = ", n_bins)
        write_reproducibility_metadata(io, seed)
    end
    return nothing
end

"""
    _write_sweep_csv(dir, filename, rows_fn, Ts, sweep_results)

Build and write one per-temperature output file of a temperature sweep:
`rows_fn(results)` produces each temperature's `DataFrame` (or `nothing` when
that temperature has no such data), a leading `temperature` column is added,
and the concatenated frame is written to `dir/filename`. If no temperature
produced rows, the file is not written — and a stale file from an earlier run
is removed, so the directory never mixes data from different `measurements`
sets.

# Returns

`nothing`.
"""
function _write_sweep_csv(dir, filename, rows_fn, Ts, sweep_results)
    dfs = map(zip(Ts, sweep_results)) do (T, results)
        results === nothing && return nothing
        df = rows_fn(results)
        df === nothing && return nothing
        insertcols!(df, 1, :temperature => T)
        return df
    end
    filter!(!isnothing, dfs)
    path = joinpath(dir, filename)
    if isempty(dfs)
        isfile(path) && rm(path)
    else
        CSV.write(path, reduce(vcat, dfs))
    end
    return nothing
end

# Adapts `measurement_rows` (a tuple of vectors) to the `_write_sweep_csv`
# `rows_fn` shape (a `DataFrame` or `nothing`).
function _measurement_frame(results)
    names, values, errors = measurement_rows(results)
    isempty(names) && return nothing
    return DataFrame(observable = names, value = values, error = errors)
end

"""
    save_sweep_results(sweep_results, Ts, model, algorithm, geometry,
                        n_thermalization, n_measurements, n_unmeasured, n_bins;
                        root = "results", seed = global_seed())

Save the combined output of a temperature sweep into a single directory:
`measurement_results.csv`, `correlation_results.csv`,
`structure_factor_results.csv`, and `autocorrelation_results.csv`, each
spanning all temperatures with a leading `temperature` column, plus sweep
metadata.

`sweep_results[i]` must correspond to `Ts[i]`; per-temperature rows are built
with [`measurement_rows`](@ref), [`correlation_rows`](@ref),
[`structure_factor_rows`](@ref), and [`autocorrelation_rows`](@ref) and
concatenated across all temperatures.
Temperatures with no correlation (structure-factor, autocorrelation) data are simply
omitted from the corresponding file; if none have such data, that file is not
written at all. Autocorrelation times are in units of measurements, i.e.
`n_unmeasured + 1` sweeps.

# Arguments

- `sweep_results`: Results for each temperature, same length and order as `Ts`.
- `Ts`: Temperatures included in the sweep.
- `model`, `algorithm`, `geometry`: Identify the run; the system size is read
  from `geometry.lattice.L` to name the directory and record the metadata.
- `n_thermalization`, `n_measurements`, `n_unmeasured`, `n_bins`: Sweep
  simulation parameters, recorded in the metadata.
- `root`: Base directory under which the sweep directory is created.
- `seed`: RNG seed recorded in the metadata (see [`save_sweep_metadata`](@ref)).

# Throws

`ArgumentError` if `sweep_results` and `Ts` have different lengths.

# Returns

The path to the created sweep directory.
"""
function save_sweep_results(
    sweep_results, Ts, model, algorithm, geometry,
    n_thermalization, n_measurements, n_unmeasured, n_bins;
    root::AbstractString = "results",
    seed = global_seed()
)
    L = geometry.lattice.L
    if isempty(sweep_results) && isempty(Ts)
        dir = sweep_dirname(model, algorithm, L; root = root)
        mkpath(dir)
        save_sweep_metadata(dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins;
            seed = seed)
        return dir
    end

    length(sweep_results) == length(Ts) ||
        throw(ArgumentError("sweep_results ($(length(sweep_results))) and Ts ($(length(Ts))) must have the same length"))

    dir = sweep_dirname(model, algorithm, L; root = root)
    mkpath(dir)
    save_sweep_metadata(dir, model, algorithm, L, Ts, n_thermalization, n_measurements, n_unmeasured, n_bins;
        seed = seed)

    _write_sweep_csv(dir, "measurement_results.csv", _measurement_frame, Ts, sweep_results)
    _write_sweep_csv(dir, "correlation_results.csv", correlation_rows, Ts, sweep_results)
    _write_sweep_csv(dir, "structure_factor_results.csv", structure_factor_rows, Ts, sweep_results)
    _write_sweep_csv(dir, "autocorrelation_results.csv", autocorrelation_rows, Ts, sweep_results)

    return dir
end