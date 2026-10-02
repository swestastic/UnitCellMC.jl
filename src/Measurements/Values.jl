"""
    observables(::IsingModel, geometry)

Return the raw per-measurement observables tracked for every [`IsingModel`](@ref)
simulation: energy and magnetization (and their squares), each normalized by
the number of sites.

These are the always-on observables included by [`MeasurementContainer`](@ref)
for any `IsingModel` run, independent of the optional `measurements` list
handled by [`optional_observables`](@ref).

# Arguments

- `model`: The `IsingModel` being measured (unused beyond dispatch).
- `geometry`: The lattice geometry, used for `n_sites` to normalize per-site
  quantities.

# Returns

A `Vector{Observable}` with four entries: `:energy`, `:energy_squared`,
`:magnetization`, `:magnetization_squared`, each sampling
`state.energy`/`state.magnetization` divided (or divided-and-squared) by
`n_sites`.
"""
observables(::IsingModel, geometry) = begin
    n_sites = geometry.n_sites
    [
        Observable(:energy, state -> state.energy / n_sites),
        Observable(:energy_squared, state -> (state.energy / n_sites)^2),
        Observable(:magnetization, state -> state.magnetization / n_sites),
        Observable(:magnetization_squared, state -> (state.magnetization / n_sites)^2),
    ]
end

"""
    derived_observables(::IsingModel, geometry)

Return the always-on derived quantities computed for every [`IsingModel`](@ref)
simulation: specific heat and magnetic susceptibility, obtained via
fluctuation formulas from the raw energy/magnetization observables.

Specific heat is computed as `N * (⟨e²⟩ - ⟨e⟩²) / T²` and susceptibility as
`N * (⟨m²⟩ - ⟨m⟩²) / T`, using jackknife samples of the per-site energy and
magnetization observables from [`observables`](@ref) (each `DerivedObservable`
declares its dependency on the corresponding `:energy`/`:magnetization` pair,
so [`validate_dependencies`](@ref) can confirm they're being measured, and
errors are obtained automatically via [`jackknife_stats`](@ref)).

# Arguments

- `model`: The `IsingModel` being measured (unused beyond dispatch).
- `geometry`: The lattice geometry (unused directly; included for a
  consistent `(model, geometry)` dispatch signature).

# Returns

A `Vector{DerivedObservable}` with two entries: `(:specific_heat,
:specific_heat_err)` depending on `(:energy, :energy_squared)`, and
`(:susceptibility, :susceptibility_err)` depending on `(:magnetization,
:magnetization_squared)`.
"""
derived_observables(::IsingModel, geometry) = [
    DerivedObservable(:specific_heat, :specific_heat_err, (:energy, :energy_squared),
        (e, e2; T, N) -> N * (e2 - e^2) / T^2),
    DerivedObservable(:susceptibility, :susceptibility_err, (:magnetization, :magnetization_squared),
        (m, m2; T, N) -> N * (m2 - m^2) / T),
]

observables(::XYModel, geometry) = begin
    n_sites = geometry.n_sites
    [
        Observable(:energy, state -> state.energy / n_sites),
        Observable(:energy_squared, state -> (state.energy / n_sites)^2),
        Observable(:magnetization_x, state -> state.magnetization_x / n_sites),
        Observable(:magnetization_x_squared, state -> (state.magnetization_x / n_sites)^2),
        Observable(:magnetization_y, state -> state.magnetization_y / n_sites),
        Observable(:magnetization_y_squared, state -> (state.magnetization_y / n_sites)^2),
        Observable(:magnetization, state -> hypot(state.magnetization_x, state.magnetization_y) / n_sites),
        Observable(:magnetization_squared, state -> (
            (state.magnetization_x^2 + state.magnetization_y^2) / n_sites^2
        )),
    ]
end

derived_observables(::XYModel, geometry) = [
    DerivedObservable(:specific_heat, :specific_heat_err, (:energy, :energy_squared),
        (e, e2; T, N) -> N * (e2 - e^2) / T^2),
    DerivedObservable(:susceptibility_x, :susceptibility_x_err,
        (:magnetization_x, :magnetization_x_squared),
        (mx, mx2; T, N) -> N * (mx2 - mx^2) / T),
    DerivedObservable(:susceptibility_y, :susceptibility_y_err,
        (:magnetization_y, :magnetization_y_squared),
        (my, my2; T, N) -> N * (my2 - my^2) / T),
    DerivedObservable(:susceptibility, :susceptibility_err,
        (:magnetization_x, :magnetization_x_squared,
            :magnetization_y, :magnetization_y_squared),
        (mx, mx2, my, my2; T, N) -> N * (mx2 - mx^2 + my2 - my^2) / T),
]