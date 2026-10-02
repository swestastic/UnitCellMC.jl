"""
    XYState{S,T<:Real} <: AbstractState

The mutable configuration of an XY model simulation. Each site stores an angle
in radians, while energy and the two Cartesian magnetization components are
cached for incremental updates.
"""
mutable struct XYState{S,T<:Real} <: AbstractState
    angles::S
    energy::T
    magnetization_x::T
    magnetization_y::T
end

function _initialize_xy_state(model, geometry::Geometry)
    validate_model_geometry(model, geometry)

    angles = 2π .* rand(geometry.n_sites)
    T = promote_type(Float64, eltype(model.J), eltype(model.h))
    energy = zero(T)
    magnetization_x = zero(T)
    magnetization_y = zero(T)

    for site in 1:geometry.n_sites
        angle = angles[site]
        magnetization_x += cos(angle)
        magnetization_y += sin(angle)
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            energy += -0.5 * bond_strength(model, geometry, bond_id) *
                cos(angle - angles[neighbor])
        end
        energy += -field_strength(model, geometry, site) * cos(angle)
    end

    return XYState(angles, energy, magnetization_x, magnetization_y)
end
