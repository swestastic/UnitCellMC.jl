function get_bond_id(
    geometry::Geometry{D,T,N},
    bond::Bond{D}
) where {D,T,N}

    idx = findfirst(==(bond), geometry.bonds)

    isnothing(idx) ? 0 : idx
end


function add_bond!(
    geometry::Geometry{D,T,N},
    bond::Bond{D}
) where {D,T,N}

    bond_id = get_bond_id(geometry, bond)

    if iszero(bond_id)
        push!(geometry.bonds, bond)
        bond_id = length(geometry.bonds)
    end

    return bond_id
end