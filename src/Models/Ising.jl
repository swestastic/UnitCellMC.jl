abstract type AbstractModel end

struct IsingModel <: AbstractModel
    J::Vector{Float64}   # one entry per bond template, same order as geometry.bonds
    h::Float64
end

function bond_strength(model::IsingModel, geometry::Geometry, bond_id::Int)
    template_id = geometry.bond_id_to_template[bond_id]
    return model.J[template_id]
end

function update!(
    ::IsingModel,
    state,
    lattice,
    site,
    ΔE
)
    s = state.configuration[site]

    state.configuration[site] = -s

    state.energy += ΔE
    state.magnetization -= 2s

    return nothing
end