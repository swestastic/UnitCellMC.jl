struct CorrelationClass{D}
    displacement::NTuple{D,Int}
    orbitals::Tuple{Int,Int}
end


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
            connected = [
                jackknife_stats((c, m) -> c - m^2, [s[r] for s in jk[:correlation]], jk[:magnetization])
                for r in 1:n_disp
            ]
            (
                correlation_connected     = first.(connected),
                correlation_connected_err = last.(connected),
                correlation_classes       = classes,
            )
        end;
        depends_on = (:correlation, :magnetization),
    )

    return observable, derived
end