"""
    build_momentum_grid(geometry::Geometry{D}) where D

Precompute, once per `geometry`, the momentum grid and lookup tables needed
for momentum-space measurements such as the structure factor (which wraps
this in [`build_momentum_classes`](@ref) to attach per-entry labels).

The momentum grid has `L[d]` points along each periodic dimension and a
single point (`k_d = 0`) along each open dimension, matching
`LatticeUtilities.calc_k_points`. Momenta are ordered in column-major order
(first dimension fastest), the same ordering as
[`build_displacement_classes`](@ref).

# Arguments

- `geometry`: A `Geometry{D}` providing the unit cell and lattice.

# Returns

A named tuple with fields

- `momentum_indices::Vector{NTuple{D,Int}}`: Integer location `n` of each
  momentum on the grid, `k = Σ_d (n[d] / L[d]) b_d` with `b_d` the reciprocal
  lattice vectors.
- `momenta::Vector{NTuple{D,T}}`: The Cartesian momentum `k` of each grid
  point.
- `sites::Vector{Array{Int,D}}`: For each orbital, the site index of that
  orbital in the unit cell at location `l` stored at `sites[o][(l .+ 1)...]`.
- `phases::NTuple{D,Matrix{ComplexF64}}`: Per-dimension Fourier matrices,
  `phases[d][n + 1, l + 1] = exp(-2πi n l / L[d])` (a single row of ones
  along open dimensions).
- `n_cells::Int`: Number of unit cells, used for normalization.
- `n_orbitals::Int`: Number of orbitals per unit cell.
- `n_momenta::Int`: Number of momenta on the grid.
"""
function build_momentum_grid(geometry::Geometry{D}) where D
    (; unit_cell, lattice) = geometry
    (; L, periodic) = lattice
    n_orb = LatticeUtilities.norbits(unit_cell)
    dims = ntuple(d -> L[d], D)
    k_dims = ntuple(d -> periodic[d] ? L[d] : 1, D)

    sites = [Array{Int,D}(undef, dims) for _ in 1:n_orb]
    for o in 1:n_orb, I in CartesianIndices(dims)
        sites[o][I] = site_at(geometry, Tuple(I) .- 1, o)
    end

    phases = ntuple(D) do d
        [cis(-2π * n * l / L[d]) for n in 0:(k_dims[d] - 1), l in 0:(L[d] - 1)]
    end

    momentum_indices = [Tuple(I) .- 1 for I in CartesianIndices(k_dims)]
    momenta = map(momentum_indices) do n
        Tuple(LatticeUtilities.calc_k_point(collect(n), unit_cell, lattice))
    end

    return (momentum_indices = momentum_indices, momenta = momenta,
            sites = sites, phases = phases,
            n_cells = prod(dims), n_orbitals = n_orb, n_momenta = prod(k_dims))
end
