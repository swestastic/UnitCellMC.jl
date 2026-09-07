using CSV
using DataFrames

function results_dirname(model, algorithm, L, β; root::AbstractString = "results")
    model_name = string(nameof(typeof(model)))
    alg_name   = string(nameof(typeof(algorithm)))
    L_str      = join(L, "x")
    β_str      = string(round(β, digits = 6))
    return joinpath(root, "$(model_name)_$(alg_name)_L$(L_str)_beta$(β_str)")
end

function save_metadata(dir, model, algorithm, L, β, parameters)
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "model = ", model)
        println(io, "algorithm = ", algorithm)
        println(io, "L = ", L)
        println(io, "beta = ", β)
        println(io, "parameters = ", parameters)
    end
    return nothing
end

function save_measurements(dir, results)
    names  = String[]
    values = Float64[]
    errors = Union{Float64,Missing}[]

    for (name, (avg, err)) in pairs(results.primary)
        avg isa Real || continue   # vector-valued (e.g. :correlation) handled by save_correlations
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

    df = DataFrame(observable = names, value = values, error = errors)
    CSV.write(joinpath(dir, "measurement_results.csv"), df)
    return nothing
end

function save_correlations(dir, results)
    hasproperty(results.derived, :correlation_classes) || return nothing

    classes = results.derived.correlation_classes
    raw_avg, raw_err = results.primary[:correlation]
    conn_avg = results.derived.correlation_connected
    conn_err = results.derived.correlation_connected_err

    D = length(first(classes).displacement)

    df = DataFrame()
    for k in 1:D
        df[!, Symbol("disp_$k")] = [c.displacement[k] for c in classes]
    end
    df.orbital_i         = [c.orbitals[1] for c in classes]
    df.orbital_j         = [c.orbitals[2] for c in classes]
    df.raw_value         = raw_avg
    df.raw_error         = raw_err
    df.connected_value   = conn_avg
    df.connected_error   = conn_err

    CSV.write(joinpath(dir, "correlation_results.csv"), df)
    return nothing
end

function save_results(model, algorithm, L, β, results, parameters; root::AbstractString = "results")
    dir = results_dirname(model, algorithm, L, β; root = root)
    mkpath(dir)
    save_metadata(dir, model, algorithm, L, β, parameters)
    save_measurements(dir, results)
    save_correlations(dir, results)
    return dir
end