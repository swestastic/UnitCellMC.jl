function build_displacement_classes(geometry::Geometry{D}) where D
    (; unit_cell, lattice) = geometry
    (; L, periodic) = lattice
    n_orb = LatticeUtilities.norbits(unit_cell)

    displacements = NTuple{D,Int}[]
    orbitals      = NTuple{2,Int}[]
    valid_ranges  = NTuple{D,UnitRange{Int}}[]
    counts        = Int[]

    Δranges = ntuple(d -> 0:(L[d]-1), D)

    for oi in 1:n_orb, oj in 1:n_orb
        for Δl in Iterators.product(Δranges...)
            ranges = ntuple(D) do d
                if periodic[d]
                    0:(L[d]-1)
                else
                    lo = max(0, -Δl[d])
                    hi = min(L[d]-1, L[d]-1-Δl[d])
                    lo:hi
                end
            end
            count = prod(length(r) for r in ranges)
            count > 0 || continue   # only possible when an open dim has no valid pairs

            push!(displacements, Δl)
            push!(orbitals, (oi, oj))
            push!(valid_ranges, ranges)
            push!(counts, count)
        end
    end

    return (displacements = displacements, orbitals = orbitals,
            valid_ranges = valid_ranges, counts = counts,
            unit_cell = unit_cell, lattice = lattice)
end