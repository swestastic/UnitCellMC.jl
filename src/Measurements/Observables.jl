struct Observable
    name::Symbol
    value::Function
    zero::Function
    template::Function
end

Observable(name::Symbol, value::Function; zero::Function = () -> 0.0, template::Function = () -> Float64[]) =
    Observable(name, value, zero, template)

struct DerivedObservable
    names::Tuple{Vararg{Symbol}}
    compute::Function
    depends_on::Tuple{Vararg{Symbol}}
end

DerivedObservable(names::Tuple{Vararg{Symbol}}, compute::Function; depends_on::Tuple{Vararg{Symbol}} = ()) =
    DerivedObservable(names, compute, depends_on)

function DerivedObservable(value_name::Symbol, err_name::Symbol, depends_on::Tuple{Vararg{Symbol}}, f::Function)
    names = (value_name, err_name)
    compute = (jk, T, N) -> begin
        v, err = jackknife_stats((args...) -> f(args...; T = T, N = N), (jk[k] for k in depends_on)...)
        NamedTuple{names}((v, err))
    end
    return DerivedObservable(names, compute, depends_on)
end