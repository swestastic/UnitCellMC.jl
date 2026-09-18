"""
    build_displacement_classes(geometry::Geometry{D}) where D

Enumerate the distinct (orbital pair, displacement) classes for `geometry`,
along with the range of valid starting sites for each class under the
lattice's boundary conditions.

For every pair of orbitals `(oi, oj)` and every displacement `Δl` (in unit
cell coordinates, one component per dimension), this computes the set of
starting-site coordinates for which the site at `Δl` offset from the start
remains within the lattice. Along periodic dimensions the full range
`0:(L[d]-1)` is always valid; along open (non-periodic) dimensions the range
is clipped so that both the starting site and its displaced partner lie
in-bounds. Classes with zero valid starting sites (only possible along an
open dimension) are skipped.

This is typically used to precompute, once per `geometry`, the bookkeeping
needed to average a correlation function over all equivalent site pairs at a
given orbital pair and displacement.

# Arguments

- `geometry`: A `Geometry{D}` providing the unit cell (for orbital count)
  and lattice (for dimensions `L` and per-dimension periodicity `periodic`).

# Returns

A named tuple with matching-length vectors describing each class, plus the
source `unit_cell` and `lattice`:

- `displacements`: The displacement `Δl` (an `NTuple{D,Int}`) for each class.
- `orbitals`: The `(oi, oj)` orbital index pair for each class.
- `valid_ranges`: Per-dimension `UnitRange{Int}` of valid starting-site
  coordinates for each class.
- `counts`: Number of valid starting sites for each class (the product of
  `valid_ranges` lengths).
- `unit_cell`: The unit cell from `geometry`, passed through unchanged.
- `lattice`: The lattice from `geometry`, passed through unchanged.
"""
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