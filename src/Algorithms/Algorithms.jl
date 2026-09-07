include("Metropolis.jl")

struct SpinUpdate{T}
    site::Int
    new_value::T
end