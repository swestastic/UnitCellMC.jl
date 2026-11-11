function square_geometry(L::Int)
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0]],
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    bond_x = lu.Bond(orbitals = (1, 1), displacement = [1, 0])
    bond_y = lu.Bond(orbitals = (1, 1), displacement = [0, 1])
    return ucmc.Geometry(unit_cell, lattice, [bond_x, bond_y])
end

function two_orbital_geometry(L::Int)
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0], [0.5, 0.5]],
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1, 0])
    bond_2 = lu.Bond(orbitals = (2, 2), displacement = [1, 0])
    return ucmc.Geometry(unit_cell, lattice, [bond_1, bond_2])
end

function triangular_geometry(L::Int)
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.5, sqrt(3) / 2]],
        basis_vecs = [[0.0, 0.0]],
    )
    lattice = lu.Lattice(L = [L, L], periodic = [true, true])
    bonds = [lu.Bond(orbitals = (1, 1), displacement = [1, 0]),
             lu.Bond(orbitals = (1, 1), displacement = [0, 1]),
             lu.Bond(orbitals = (1, 1), displacement = [1, -1])]
    return ucmc.Geometry(unit_cell, lattice, bonds)
end

function spin_energy(model, geometry, spins)
    energy = zero(promote_type(eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * spins[i] * spins[j]
        end
        energy += -ucmc.field_strength(model, geometry, i) * spins[i]
    end
    return energy
end

function xy_energy(model, geometry, cosines, sines)
    energy = zero(promote_type(eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * (cosines[i] * cosines[j] + sines[i] * sines[j])
        end
        energy += -ucmc.field_strength(model, geometry, i) * cosines[i]
    end
    return energy
end
xy_energy(model, geometry, state::ucmc.XYState) =
    xy_energy(model, geometry, state.cosines, state.sines)

function heisenberg_energy(model, geometry, xs, ys, zs)
    energy = zero(promote_type(eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * (xs[i] * xs[j] + ys[i] * ys[j] + zs[i] * zs[j])
        end
        energy += -ucmc.field_strength(model, geometry, i) * xs[i]
    end
    return energy
end
heisenberg_energy(model, geometry, state::ucmc.HeisenbergState) =
    heisenberg_energy(model, geometry, state.xs, state.ys, state.zs)

function clock_energy(model, geometry, spins)
    q = model.q
    energy = zero(promote_type(Float64, eltype(model.J), eltype(model.h)))
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for (bond_id, j) in zip(info.bonds, info.neighbors)
            J = ucmc.bond_strength(model, geometry, bond_id)
            energy += -0.5 * J * cos(2π * (spins[i] - spins[j]) / q)
        end
        energy += -ucmc.field_strength(model, geometry, i) * cos(2π * spins[i] / q)
    end
    return energy
end

clock_mx(q, spins) = sum(n -> cos(2π * n / q), spins; init = 0.0)
clock_my(q, spins) = sum(n -> sin(2π * n / q), spins; init = 0.0)

function clock_invariants_hold(model, geometry, state)
    q = model.q
    return all(in(0:(q - 1)), state.spins) &&
        isapprox(state.energy, clock_energy(model, geometry, state.spins); atol = 1e-10) &&
        isapprox(state.magnetization_x, clock_mx(q, state.spins); atol = 1e-10) &&
        isapprox(state.magnetization_y, clock_my(q, state.spins); atol = 1e-10)
end
