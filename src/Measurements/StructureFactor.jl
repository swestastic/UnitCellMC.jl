"""
    n_operator_components(model::AbstractModel)

Return the number of real components of the local operator of `model`, i.e.
the length of the tuple returned by [`operator_components`](@ref).

This is the model hook that enables the `:structure_factor` measurement (see
[`structure_factor_entry`](@ref)). A new model supports the structure factor
by implementing this method together with [`operator_components`](@ref);
nothing else in the measurement code needs to change. For example, a
Heisenberg model would return `3`, a `q`-state clock model `2`.

# Throws

`ArgumentError` for models that do not implement the hook.
"""
n_operator_components(model::AbstractModel) = throw(ArgumentError(
    "$(nameof(typeof(model))) does not support the structure factor; implement " *
    "`n_operator_components(::$(nameof(typeof(model))))` and " *
    "`operator_components(::$(nameof(typeof(model))), value)`"
))

"""
    operator_components(model::AbstractModel, value)

Return the real components of the local operator `value` (as returned by
[`local_operator`](@ref)) as a tuple of length
[`n_operator_components`](@ref)`(model)`.

The components must reproduce the model's two-body product,

    operator_product(model, a, b) ≈ sum(operator_components(model, a) .* operator_components(model, b))

so that the structure factor is exactly the Fourier transform of the
correlation function measured by [`correlation_instantaneous`](@ref).
For an Ising spin `s` this is `(s,)`; for an XY angle `θ` it is
`(cos θ, sin θ)`.

# Throws

`ArgumentError` for models that do not implement the hook.
"""
operator_components(model::AbstractModel, value) = throw(ArgumentError(
    "$(nameof(typeof(model))) does not implement " *
    "`operator_components(::$(nameof(typeof(model))), ::$(typeof(value)))`, " *
    "required by the structure factor"
))

"""
    StructureFactorClass{D,T}

Labels one entry of the structure factor: the momentum `k` and the orbital
pair `(orbital_i, orbital_j)` of ``S_{ab}(\\mathbf{k})``.

# Fields

- `momentum_index::NTuple{D,Int}`: Integer location `n` of the momentum on the
  finite-lattice grid, `k = Σ_d (n[d] / L[d]) b_d` with `b_d` the reciprocal
  lattice vectors. Along open (non-periodic) dimensions only `n[d] = 0` is
  used, matching `LatticeUtilities.calc_k_points`.
- `momentum::NTuple{D,T}`: The Cartesian momentum `k`.
- `orbitals::Tuple{Int,Int}`: The `(orbital_i, orbital_j)` pair.

See also [`build_momentum_classes`](@ref) and [`CorrelationClass`](@ref),
its real-space counterpart.
"""
struct StructureFactorClass{D,T}
    momentum_index::NTuple{D,Int}
    momentum::NTuple{D,T}
    orbitals::Tuple{Int,Int}
end

"""
    build_momentum_classes(geometry::Geometry{D}) where D

Precompute, once per `geometry`, the momentum grid and lookup tables needed
by [`structure_factor_instantaneous`](@ref).

The momentum grid has `L[d]` points along each periodic dimension and a
single point (`k_d = 0`) along each open dimension. Entries are ordered by
orbital pair `(oi, oj)` (with `oj` varying fastest) and, within each pair,
by momentum in column-major order (first dimension fastest), the same
ordering as [`build_displacement_classes`](@ref).

# Arguments

- `geometry`: A `Geometry{D}` providing the unit cell and lattice.

# Returns

A named tuple with fields

- `classes::Vector{StructureFactorClass}`: Label of each structure-factor
  entry, in output order.
- `sites::Vector{Array{Int,D}}`: For each orbital, the site index of that
  orbital in the unit cell at location `l` stored at `sites[o][(l .+ 1)...]`.
- `phases::NTuple{D,Matrix{ComplexF64}}`: Per-dimension Fourier matrices,
  `phases[d][n + 1, l + 1] = exp(-2πi n l / L[d])` (a single row of ones
  along open dimensions).
- `n_cells::Int`: Number of unit cells, used for normalization.
- `n_orbitals::Int`: Number of orbitals per unit cell.
- `n_momenta::Int`: Number of momenta on the grid.
"""
function build_momentum_classes(geometry::Geometry{D}) where D
    (; unit_cell, lattice) = geometry
    (; L, periodic) = lattice
    n_orb = LatticeUtilities.norbits(unit_cell)
    dims = ntuple(d -> L[d], D)
    k_dims = ntuple(d -> periodic[d] ? L[d] : 1, D)

    sites = [Array{Int,D}(undef, dims) for _ in 1:n_orb]
    for o in 1:n_orb, I in CartesianIndices(dims)
        sites[o][I] = LatticeUtilities.loc_to_site(collect(Tuple(I) .- 1), o, unit_cell, lattice)
    end

    phases = ntuple(D) do d
        [cis(-2π * n * l / L[d]) for n in 0:(k_dims[d] - 1), l in 0:(L[d] - 1)]
    end

    T = eltype(unit_cell.reciprocal_vecs)
    classes = StructureFactorClass{D,T}[]
    for oi in 1:n_orb, oj in 1:n_orb, I in CartesianIndices(k_dims)
        n = Tuple(I) .- 1
        k = LatticeUtilities.calc_k_point(collect(n), unit_cell, lattice)
        push!(classes, StructureFactorClass{D,T}(n, Tuple(k), (oi, oj)))
    end

    return (classes = classes, sites = sites, phases = phases,
            n_cells = prod(dims), n_orbitals = n_orb, n_momenta = prod(k_dims))
end

"""
    fourier_transform_dim(A, W, d)

Contract dimension `d` of the array `A` with the matrix `W`,
`B[…, n, …] = Σ_l W[n, l] A[…, l, …]`. Applying this along every dimension
with the matrices from [`build_momentum_classes`](@ref) gives a
`D`-dimensional discrete Fourier transform in `O(N Σ_d L[d])` operations
without an FFT dependency.
"""
function fourier_transform_dim(A::Array{T}, W::AbstractMatrix, d::Int) where T
    sz = size(A)
    n_out = size(W, 1)
    pre = prod(sz[1:(d - 1)]; init = 1)
    post = prod(sz[(d + 1):end]; init = 1)
    Ar = reshape(A, pre, sz[d], post)
    B = zeros(promote_type(T, eltype(W)), Base.setindex(sz, n_out, d))
    Br = reshape(B, pre, n_out, post)
    @inbounds for p in 1:post, l in 1:sz[d], n in 1:n_out
        w = W[n, l]
        for i in 1:pre
            Br[i, n, p] += w * Ar[i, l, p]
        end
    end
    return B
end

"""
    structure_factor_instantaneous(model, state, momenta)

Compute one instantaneous sample of the orbital-resolved structure factor

```math
S_{ab}(\\mathbf{k}) = \\frac{1}{N_c} \\sum_\\alpha
    \\overline{\\rho^\\alpha_a(\\mathbf{k})}\\, \\rho^\\alpha_b(\\mathbf{k}),
\\qquad
\\rho^\\alpha_a(\\mathbf{k}) = \\sum_{\\mathbf{R}} S^\\alpha_{\\mathbf{R},a}\\,
    e^{-i\\mathbf{k}\\cdot\\mathbf{R}},
```

where `R` runs over the `N_c` unit cells, `a`, `b` are orbitals, and
`S^α_{R,a}` are the [`operator_components`](@ref) of the local operator on
that site. On a periodic lattice this equals
`Σ_Δ C_ab(Δ) exp(-i k·Δ)`, the Fourier transform of the correlation function
from [`correlation_instantaneous`](@ref). Phases use unit-cell positions
only; the orbital basis offsets are not included.

# Arguments

- `model`: The model providing [`operator_components`](@ref).
- `state`: The current configuration, read via [`local_operator`](@ref).
- `momenta`: Momentum data from [`build_momentum_classes`](@ref).

# Returns

A `Vector{ComplexF64}` with one entry per class in `momenta.classes`, in the
same order. Diagonal orbital entries (`a == b`) are real and non-negative;
`S_ba(k) == conj(S_ab(k))`.
"""
function structure_factor_instantaneous(model, state, momenta)
    (; sites, phases, n_cells, n_orbitals, n_momenta) = momenta
    n_comp = n_operator_components(model)
    D = length(phases)

    ρ = Matrix{Array{ComplexF64,D}}(undef, n_comp, n_orbitals)
    for o in 1:n_orbitals
        site_grid = sites[o]
        for α in 1:n_comp
            ρ[α, o] = Array{ComplexF64,D}(undef, size(site_grid))
        end
        for I in eachindex(site_grid)
            components = operator_components(model, local_operator(state, site_grid[I]))
            for α in 1:n_comp
                ρ[α, o][I] = components[α]
            end
        end
        for α in 1:n_comp, d in 1:D
            ρ[α, o] = fourier_transform_dim(ρ[α, o], phases[d], d)
        end
    end

    S = Vector{ComplexF64}(undef, n_orbitals^2 * n_momenta)
    index = 0
    for oi in 1:n_orbitals, oj in 1:n_orbitals, k in 1:n_momenta
        acc = zero(ComplexF64)
        for α in 1:n_comp
            acc += conj(ρ[α, oi][k]) * ρ[α, oj][k]
        end
        S[index += 1] = acc / n_cells
    end
    return S
end
