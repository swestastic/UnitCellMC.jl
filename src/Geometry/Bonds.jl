function get_bond_id(
    geometry::Geometry{D,T,N},
    bond::Bond{D}
) where {D,T,N}

    idx = findfirst(==(bond), geometry.bonds)

    isnothing(idx) ? 0 : idx
end