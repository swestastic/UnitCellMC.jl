measurement_template(::IsingModel) = (
    energy = Float64[],
    energy_squared = Float64[],
    magnetization = Float64[],
    magnetization_squared = Float64[],
)

measurement_values(::IsingModel, state) = (
    energy = state.energy,
    energy_squared = state.energy^2,
    magnetization = state.magnetization,
    magnetization_squared = state.magnetization^2,
)

function derived_observables(::IsingModel, β, N, jk)
    C, C_err = jackknife_stats((e, e2) -> β^2/N * (e2 - e^2), jk[:energy], jk[:energy_squared])
    χ, χ_err = jackknife_stats((m, m2) -> β/N * (m2 - m^2), jk[:magnetization], jk[:magnetization_squared])
    return (specific_heat = C, specific_heat_err = C_err,
            susceptibility = χ, susceptibility_err = χ_err)
end