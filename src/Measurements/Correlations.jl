"""
    CorrelationClass{D}

Identifies a single equivalence class of site pairs contributing to a
correlation function: all pairs at relative displacement `displacement`
between orbitals `orbitals`, averaged together by translational symmetry.

# Fields

- `displacement::NTuple{D,Int}`: Relative displacement (in unit cell
  coordinates) between the two sites, one component per dimension.
- `orbitals::Tuple{Int,Int}`: The `(orbital_i, orbital_j)` pair for the two sites.

See also [`build_displacement_classes`](@ref), which enumerates the classes
this type labels.
"""
struct CorrelationClass{D}
    displacement::NTuple{D,Int}
    orbitals::Tuple{Int,Int}
end

"""
    correlation_instantaneous(model, state, disp)

Compute one instantaneous sample of the correlation function for `state`,
one value per displacement class in `disp`.

For each displacement class `r`, averages `operator_product(model, ...)`
over every valid starting site (site `i`, orbital `oi`) paired with its
partner at the class's displacement (site `j`, orbital `oj`), wrapping
coordinates with periodic boundary conditions where applicable, and dividing
by the class's site count for a proper average.

# Arguments

- `model`: The model providing [`operator_product`](@ref).
- `state`: The current configuration, read via [`local_operator`](@ref).
- `disp`: Displacement class data as returned by
  [`build_displacement_classes`](@ref), giving `geometry`, `displacements`,
  `orbitals`, `valid_ranges`, and `counts`.

# Returns

A `Vector{Float64}` of length `length(disp.displacements)`, one correlation
value per displacement class, in the same order as `disp`.
"""
function correlation_instantaneous(model, state, disp)
    (; geometry) = disp
    (; L, periodic) = geometry.lattice
    n_disp = length(disp.displacements)
    C = zeros(n_disp)

    for r in 1:n_disp
        Δl = disp.displacements[r]
        oi, oj = disp.orbitals[r]
        acc = 0.0
        for loc_i in Iterators.product(disp.valid_ranges[r]...)
            i = site_at(geometry, loc_i, oi)
            loc_j = ntuple(d -> periodic[d] ? mod(loc_i[d] + Δl[d], L[d]) : loc_i[d] + Δl[d], length(L))
            j = site_at(geometry, loc_j, oj)
            acc += operator_product(model, local_operator(state, i), local_operator(state, j))
        end
        C[r] = acc / disp.counts[r]
    end
    return C
end

"""
    correlation_entry(model, geometry)

Build the raw and derived observable entries needed to measure the
correlation function of `model` on `geometry`.

Precomputes the displacement classes once (via
[`build_displacement_classes`](@ref)) and wraps them in an
[`Observable`](@ref) named `:correlation` (sampled per measurement with
[`correlation_instantaneous`](@ref)) and a [`DerivedObservable`](@ref) that
computes the connected correlation via jackknife resampling (see
[`jackknife_stats`](@ref)), subtracting the per-orbital magnetization product
`⟨S_oi⟩·⟨S_oj⟩` of each class's orbital pair (summed over operator
components) from the `:magnetization_per_orbital` observable (see
[`magnetization_per_orbital_entry`](@ref)) — not the lattice-averaged
`⟨m⟩²`, which is only correct when the sublattices are equivalent. The
result also includes the [`CorrelationClass`](@ref) list labeling each entry.

# Arguments

- `model`: The model to measure correlations for.
- `geometry`: The lattice geometry.

# Returns

An `(observable, derived)` tuple:
- `observable`: An `Observable` for `:correlation`, sampling a
  `Vector{Float64}` (one value per displacement class) each measurement.
- `derived`: A `DerivedObservable` computing `:correlation_connected`,
  `:correlation_connected_err`, and `:correlation_classes` from `:correlation`
    and `:magnetization_per_orbital` (its declared `depends_on`).
"""
function correlation_entry(model, geometry)
    disp = build_displacement_classes(geometry)
    n_disp = length(disp.displacements)

    classes = [
        CorrelationClass(Tuple(disp.displacements[r]), disp.orbitals[r])
        for r in 1:n_disp
    ]

    observable = Observable(
        :correlation,
        state -> correlation_instantaneous(model, state, disp);
        zero     = () -> zeros(n_disp),
        template = () -> Vector{Float64}[],
    )

    derived = DerivedObservable(
        (:correlation_connected, :correlation_connected_err, :correlation_classes),
        (jk, T, N) -> begin
            mpo = jk[:magnetization_per_orbital]
            n_comp = size(first(mpo), 1)
            # The product ⟨S_oi⟩·⟨S_oj⟩ is evaluated *inside* the jackknifed
            # function so its O(1/n_bins) bias is corrected along with the rest.
            # The series order is [oi components..., oj components...] (in a
            # `for`-chain the last iterator is the innermost loop).
            connected = [
                jackknife_stats(
                    (c, comps...) -> c - sum(comps[α] * comps[n_comp + α] for α in 1:n_comp),
                    [s[r] for s in jk[:correlation]],
                    ([s[α, o] for s in mpo] for o in (oi, oj) for α in 1:n_comp)...,
                )
                for (r, (oi, oj)) in enumerate(disp.orbitals)
            ]
            (
                correlation_connected     = first.(connected),
                correlation_connected_err = last.(connected),
                correlation_classes       = classes,
            )
        end;
        depends_on = (:correlation, :magnetization_per_orbital),
    )

    return observable, derived
end

# `:raw_correlation` measures only the correlation function;
# `:connected_correlation` additionally derives the connected form.
function optional_observables(model, geometry, ::Val{:raw_correlation})
    observable, _ = correlation_entry(model, geometry)
    return [observable], DerivedObservable[]
end

function optional_observables(model, geometry, ::Val{:connected_correlation})
    observable, derived = correlation_entry(model, geometry)
    orbital_obs, _ = magnetization_per_orbital_entry(model, geometry)
    return [observable; orbital_obs], [derived]
end