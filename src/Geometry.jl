import LatticeUtilities as lu
import LatticeUtilities.UnitCell
import LatticeUtilities.Bond
import LatticeUtilities.Lattice

struct ModelGeometry{D, T<:AbstractFloat, N}
	unit_cell::UnitCell{D,T,N}
	lattice::Lattice{D}
	bonds::Vector{Bond{D}}
end

function ModelGeometry(unit_cell::UnitCell{D}, lattice::Lattice{D}) where {D}
	@assert all(i -> i, lattice.periodic) "All spatial dimensions must be periodic."
	n     = unit_cell.n
	bonds = Bond{D}[]
	for i in 1:n
		push!(bonds, Bond((i, i), zeros(Int, D)))
	end
	return ModelGeometry(unit_cell, lattice, bonds)
end

function get_bond_id(model_geometry::ModelGeometry{D,T}, bond::Bond{D}) where {D,T}
	(; bonds) = model_geometry
	bond in bonds ? findfirst(b -> b == bond, bonds) : 0
end

function add_bond!(model_geometry::ModelGeometry{D,T}, bond::Bond{D}) where {D,T}
	bond_id = get_bond_id(model_geometry, bond)
	if iszero(bond_id)
		push!(model_geometry.bonds, bond)
		bond_id = length(model_geometry.bonds)
	end
	return bond_id
end
