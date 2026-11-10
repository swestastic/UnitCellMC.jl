"""
    helicity_stiffness_product(model, a, b)

Return the per-bond spin factor of the helicity-modulus stiffness sum — the
second derivative of the bond energy under a uniform twist about the
spin-space ``z`` axis, divided by the coupling — given the local operator
values `a` and `b` of the bond's endpoints (see
[`helicity_modulus_entry`](@ref)). For a vector-spin model with bond energy
``-J_{ij} \\, \\mathbf{u}_i \\cdot \\mathbf{u}_j``, rotating one endpoint of
a bond by an infinitesimal angle about ``\\hat{z}`` gives

```math
\\mathbf{u}_i \\cdot \\mathbf{u}_j -
(\\mathbf{u}_i \\cdot \\hat{z}) (\\mathbf{u}_j \\cdot \\hat{z}).
```

The default returns [`operator_product`](@ref) (the plain dot product),
which is exact for models whose spins have no component along the twist
axis, such as the planar [`XYModel`](@ref) and [`ClockModel`](@ref). Models
whose spins leave the ``xy`` plane must specialize it, as
[`HeisenbergModel`](@ref) does.
"""
helicity_stiffness_product(model, a, b) = operator_product(model, a, b)

"""
    helicity_modulus_entry(model, geometry)

Build the optional helicity-modulus (spin-stiffness) measurement for a
vector-spin model ([`XYModel`](@ref), [`ClockModel`](@ref),
[`HeisenbergModel`](@ref)).

The helicity modulus is the response of the free energy to an infinitesimal
uniform twist of the spins along spatial direction `d`,

```math
\\Upsilon_d = \\frac{1}{N} \\left\\langle \\sum_{\\langle ij \\rangle}
J_{ij} (\\Delta r_{ij})_d^2 \\big[ \\mathbf{u}_i \\cdot \\mathbf{u}_j -
(\\mathbf{u}_i \\cdot \\hat{z}) (\\mathbf{u}_j \\cdot \\hat{z}) \\big]
\\right\\rangle - \\frac{\\beta}{N} \\left( \\langle I_d^2 \\rangle -
\\langle I_d \\rangle^2 \\right), \\quad
I_d = \\sum_{\\langle ij \\rangle} J_{ij} (\\Delta r_{ij})_d
\\big( \\mathbf{u}_i \\times \\mathbf{u}_j \\big)_z ,
```

where `u_i` is the unit spin vector of site `i`, the sums run over unique
bonds, and `Δr_ij` is the real-space displacement vector of a bond (see
[`bond_displacement`](@ref)). The twist is taken about the spin-space z
axis; for the two-component models (XY, clock) this is the only choice, and
for the Heisenberg model every axis is equivalent at zero field. The
stiffness factor in brackets is the second twist derivative of the bond
energy, [`helicity_stiffness_product`](@ref); its `(u · ẑ)(u · ẑ)` term
vanishes identically for the planar XY/clock spins and matters only for the
Heisenberg model. In the XY/clock models a nonzero `Υ` signals the
quasi-ordered phase, and the Berezinskii-Kosterlitz-Thouless transition is
located by the universal jump `Υ(T_BKT) = 2 T_BKT / π`.

The estimator is intended for zero external field: a uniform twist also
rotates the spins relative to the field axis, a contribution that is not
included here.

# Raw observables

- `:helicity_stiffness`: The first ("kinetic") sum, per site — a length-`D`
  vector, one entry per spatial direction.
- `:helicity_current`: The twist current `I_d / N`, per site.
- `:helicity_current_squared`: `(I_d / N)^2`, sampled per measurement so
  that `⟨I_d²⟩` is available for the fluctuation term.

# Arguments

- `model`: The model being measured; must implement
    [`n_operator_components`](@ref) (used for the spin dot/cross products)
    with at least two spin components.
- `geometry`: The lattice geometry, supplying the neighbor table and the
    bond displacement vectors.

# Throws

`ArgumentError` if `model` has fewer than two spin components (e.g.
[`IsingModel`](@ref), for which the helicity modulus is not defined), or if
`geometry` has no bonds.

# Returns

An `(obs, der)` tuple: `obs` holds the three raw observables above and `der`
the derived `:helicity_modulus`/`:helicity_modulus_err` (length-`D` vectors,
per site, with jackknife errors from [`jackknife_stats`](@ref)).
"""
function helicity_modulus_entry(model, geometry::Geometry{D}) where {D}
    n_comp = n_operator_components(model)   # fail at container construction
    n_comp >= 2 || throw(ArgumentError(
        "The :helicity_modulus measurement requires a vector-spin model " *
        "(XYModel, ClockModel, HeisenbergModel); got $(nameof(typeof(model)))"))
    isempty(geometry.bond_id_to_template) && throw(ArgumentError(
        "The :helicity_modulus measurement requires a geometry with bonds"))

    n_sites = geometry.n_sites

    # Real-space displacement vector of each bond template.
    template_vecs = [bond_displacement(geometry, template)
                     for template in eachindex(geometry.bonds)]

    # Per directed neighbor-table entry, the coefficients of the stiffness
    # (C2) and twist-current (C1) sums.
    C1 = [zeros(Float64, length(info.neighbors), D) for info in geometry.neighbor_table]
    C2 = [zeros(Float64, length(info.neighbors), D) for info in geometry.neighbor_table]
    for i in 1:n_sites
        info = geometry.neighbor_table[i]
        for p in eachindex(info.neighbors)
            bid = info.bonds[p]
            J = bond_strength(model, geometry, bid)
            sgn = geometry.bond_sources[bid] == i ? 1 : -1
            v = template_vecs[info.templates[p]]
            for d in 1:D
                C1[i][p, d] = J * sgn * v[d]
                C2[i][p, d] = J * v[d]^2
            end
        end
    end

    # One pass over the neighbor table per sample. Every bond appears from
    # both endpoints (hence the 1/2); the current coefficient C1 carries the
    # orientation sign, so the two directed contributions to a bond's current
    # are equal and the same 1/2 applies. Both sums are stored per site.
    function sample_stiffness(state)
        A = zeros(Float64, D)
        for i in 1:n_sites
            oi = local_operator(state, i)
            info = geometry.neighbor_table[i]
            c2 = C2[i]
            for p in eachindex(info.neighbors)
                dot = helicity_stiffness_product(model, oi, local_operator(state, info.neighbors[p]))
                for d in 1:D
                    A[d] += c2[p, d] * dot
                end
            end
        end
        A .*= 0.5 / n_sites
        return A
    end

    function sample_current(state)
        I = zeros(Float64, D)
        for i in 1:n_sites
            ui = operator_components(model, local_operator(state, i))
            info = geometry.neighbor_table[i]
            c1 = C1[i]
            for p in eachindex(info.neighbors)
                uj = operator_components(model, local_operator(state, info.neighbors[p]))
                cross = ui[1] * uj[2] - ui[2] * uj[1]
                for d in 1:D
                    I[d] += c1[p, d] * cross
                end
            end
        end
        I .*= 0.5 / n_sites
        return I
    end

    obs = Observable[
        Observable(:helicity_stiffness, sample_stiffness;
            zero = () -> zeros(Float64, D), template = () -> Vector{Float64}[]),
        Observable(:helicity_current, sample_current;
            zero = () -> zeros(Float64, D), template = () -> Vector{Float64}[]),
        # The squared current is needed as its own raw series so that ⟨I²⟩
        # reaches the fluctuation term; it costs a second current pass per
        # measurement (the measurement framework evaluates each observable
        # independently).
        Observable(:helicity_current_squared, state -> sample_current(state) .^ 2;
            zero = () -> zeros(Float64, D), template = () -> Vector{Float64}[]),
    ]

    compute = function (jk, T, N)
        β = 1 / T
        values = zeros(Float64, D)
        errors = zeros(Float64, D)
        for d in 1:D
            a = [s[d] for s in jk[:helicity_stiffness]]
            i1 = [s[d] for s in jk[:helicity_current]]
            i2 = [s[d] for s in jk[:helicity_current_squared]]
            v, e = jackknife_stats(
                (a_d, i_d, i2_d) -> a_d - β * N * (i2_d - i_d^2), a, i1, i2)
            values[d] = v
            errors[d] = e
        end
        return (helicity_modulus = values, helicity_modulus_err = errors)
    end
    der = DerivedObservable[
        DerivedObservable(
            (:helicity_modulus, :helicity_modulus_err),
            compute;
            depends_on = (:helicity_stiffness, :helicity_current,
                :helicity_current_squared),
        ),
    ]
    return obs, der
end

optional_observables(model, geometry, ::Val{:helicity_modulus}) =
    helicity_modulus_entry(model, geometry)
