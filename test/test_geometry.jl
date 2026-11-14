@testset "Geometry bond sources" begin
    geometry = square_geometry(3)
    @test length(geometry.bond_sources) == length(geometry.bond_id_to_template)

    # Every concrete bond's source is one of its endpoints, and the other
    # endpoint is reached from the source by the template's displacement.
    for i in 1:geometry.n_sites
        info = geometry.neighbor_table[i]
        for p in eachindex(info.bonds)
            b = info.bonds[p]
            j = info.neighbors[p]
            s = geometry.bond_sources[b]
            @test s == i || s == j
            if s == i
                template = geometry.bonds[geometry.bond_id_to_template[b]]
                @test j == lu.site_to_site(s, template.displacement, template.orbitals[2],
                                           geometry.unit_cell, geometry.lattice)
            end
        end
    end

    # The displacement accessor returns the template's real-space vector.
    @test ucmc.bond_displacement(geometry, 1) ≈ [1.0, 0.0]
    @test ucmc.bond_displacement(geometry, 2) ≈ [0.0, 1.0]

    # Degenerate bond sets leave the source list empty.
    empty_geometry = ucmc.Geometry(geometry.unit_cell, geometry.lattice, lu.Bond{2}[])
    @test isempty(empty_geometry.bond_sources)

    # Self-bond templates (same orbital, zero displacement) are rejected at
    # construction; a zero-displacement bond between different orbitals is a
    # legitimate on-site coupling.
    self_bond = lu.Bond(orbitals = (1, 1), displacement = [0, 0])
    @test_throws ArgumentError ucmc.Geometry(geometry.unit_cell, geometry.lattice, [self_bond])
    uc2 = two_orbital_geometry(2).unit_cell
    onsite = ucmc.Geometry(uc2, geometry.lattice, [lu.Bond(orbitals = (1, 2), displacement = [0, 0])])
    @test !isempty(onsite.bond_id_to_template)
end

@testset "Geometry tag distinguishes same-size lattices" begin
    square = square_geometry(3)
    two_orbital = two_orbital_geometry(3)
    @test ucmc.geometry_tag(square) != ucmc.geometry_tag(two_orbital)
    @test ucmc.geometry_tag(square) == ucmc.geometry_tag(square_geometry(4))

    # same model type, algorithm, and size, different unit cell: the results
    # directories still differ
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    @test ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), square) !=
          ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), two_orbital)
end
