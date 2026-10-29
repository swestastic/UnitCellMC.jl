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
  [`build_displacement_classes`](@ref), giving `unit_cell`, `lattice`,
  `displacements`, `orbitals`, `valid_ranges`, and `counts`.

# Returns

A `Vector{Float64}` of length `length(disp.displacements)`, one correlation
value per displacement class, in the same order as `disp`.
"""
function correlation_instantaneous(model, state, disp)
    (; unit_cell, lattice) = disp
    (; L, periodic) = lattice
    n_disp = length(disp.displacements)
    C = zeros(n_disp)

    for r in 1:n_disp
        Δl = disp.displacements[r]
        oi, oj = disp.orbitals[r]
        acc = 0.0
        for loc_i in Iterators.product(disp.valid_ranges[r]...)
            i = LatticeUtilities.loc_to_site(collect(loc_i), oi, unit_cell, lattice)
            loc_j = ntuple(d -> periodic[d] ? mod(loc_i[d] + Δl[d], L[d]) : loc_i[d] + Δl[d], length(L))
            j = LatticeUtilities.loc_to_site(collect(loc_j), oj, unit_cell, lattice)
            acc += operator_product(model, local_operator(state, i), local_operator(state, j))
        end
        C[r] = acc / disp.counts[r]
    end
    return C
end

"""
    magnetization_components(model)

Return the names of the raw observables holding the Cartesian components of
the per-site magnetization of `model`, as measured by
[`observables`](@ref). The connected correlation built by
[`correlation_entry`](@ref) subtracts the sum of their squared averages.

Defaults to `(:magnetization,)` (a single-component order parameter such as
the Ising magnetization). Planar models ([`XYModel`](@ref),
[`ClockModel`](@ref)) return `(:magnetization_x, :magnetization_y)`. A new
model with a multi-component order parameter extends this method.
"""
magnetization_components(model) = (:magnetization,)
magnetization_components(::Union{XYModel,ClockModel}) = (:magnetization_x, :magnetization_y)

"""
    correlation_entry(model, geometry)

Build the raw and derived observable entries needed to measure the
correlation function of `model` on `geometry`.

Precomputes the displacement classes once (via
[`build_displacement_classes`](@ref)) and wraps them in an
[`Observable`](@ref) named `:correlation` (sampled per measurement with
[`correlation_instantaneous`](@ref)) and a [`DerivedObservable`](@ref) that
computes the connected correlation via jackknife resampling (see
[`jackknife_stats`](@ref)), subtracting the squared average of every
magnetization component named by [`magnetization_components`](@ref):
`⟨m_x⟩² + ⟨m_y⟩²` for the XY and clock models, `⟨m⟩²` for Ising. The
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
    and the Cartesian magnetization components (their declared `depends_on`).
"""
function correlation_entry(model, geometry)
    components = magnetization_components(model)
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
            connected = [
                jackknife_stats(
                    (c, ms...) -> foldl((acc, m) -> acc - m^2, ms; init = c),
                    [s[r] for s in jk[:correlation]],
                    (jk[name] for name in components)...,
                )
                for r in 1:n_disp
            ]
            (
                correlation_connected     = first.(connected),
                correlation_connected_err = last.(connected),
                correlation_classes       = classes,
            )
        end;
        depends_on = (:correlation, components...),
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
    return [observable], [derived]
end