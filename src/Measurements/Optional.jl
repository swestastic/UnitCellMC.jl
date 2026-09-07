function optional_observables(model, geometry, measurements)
    obs = Observable[]
    der = DerivedObservable[]

    if :correlation in measurements
        o, d = correlation_entry(model, geometry)
        push!(obs, o)
        push!(der, d)
    end
    # future: :structure_factor in measurements && (push! obs/der similarly)

    return obs, der
end