observables(::IsingModel, geometry) = [
    Observable(:energy, state -> state.energy),
    Observable(:energy_squared, state -> state.energy^2),
    Observable(:magnetization, state -> state.magnetization),
    Observable(:magnetization_squared, state -> state.magnetization^2),
]

derived_observables(::IsingModel, geometry) = [
    DerivedObservable(:specific_heat, :specific_heat_err, (:energy, :energy_squared),
        (e, e2; β, N) -> β^2 / N * (e2 - e^2)),
    DerivedObservable(:susceptibility, :susceptibility_err, (:magnetization, :magnetization_squared),
        (m, m2; β, N) -> β / N * (m2 - m^2)),
]