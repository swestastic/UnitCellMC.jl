include("Metropolis.jl")
include("Diagnostics.jl")

"""
    SpinUpdate{T}

A proposed single-site update: change the spin at `site` to `new_value`.

# Fields

- `site::Int`: Index of the site to update.
- `new_value::T`: The proposed value for that site.
"""
struct SpinUpdate{T}
    site::Int
    new_value::T
end