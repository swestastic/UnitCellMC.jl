# Verifications

How do we know that we can trust results from this package?

Here are some of the steps we've taken to ensure that results are accurate.

## Exact enumeration for small systems

We implement an exact enumeration system for small Ising systems, which reuses the existing lattice code. The number of enumerations scales by $2^N$, so we are limited to quite modest sizes before we run out of memory or hit increasingly long compute times.

The Hamiltonian of the Ising model:

$$ H(\{S\}) = -J \sum_{\langle i,j \rangle} S_i S_j - \sum_{i=1}^{N} h_i S_i, $$

Partition function:

$$ Z(\beta) = \sum_{\{S\}} e^{-\beta H(\{S\})}. $$

When we perform exact enumeration, we calculate the energy $H(\{S\})$ for every state and accumulate its Boltzmann weight $e^{-\beta H( \{ S \})}$ into Z.

Since UnitCellMC.jl stores the spins in a 1D array and then uses LatticeUtilities.jl to look up neighbors, regardless of lattice geometry, it is easy to generate these configurations using bitwise operators.

```julia
function generate_states(n_sites)

    states = Vector{Vector{Int8}}(undef, 2^n_sites)

    for config_int in 0:(2^n_sites - 1)
        spins = Int8[
            2 * ((config_int >> i) & 1) - 1
            for i in 0:n_sites-1
        ]

        states[config_int + 1] = spins
    end

    return states
end
```

### Observable calculation

We can calculate the expectation value of an observable as follows

$$ \langle \mathcal{O} \rangle = \frac{1}{Z} \sum_{\{S\}} \mathcal{O}(\{S\}) e^{-\beta H(\{S\})}. $$

Some examples:

**Energy:**

$$ \langle E \rangle = \frac{1}{Z} \sum_{\{S\}} H(\{S\}) , e^{-\beta H(\{S\})} = -\frac{\partial \ln Z}{\partial \beta}. $$

**Specific Heat:**

$$ C_v = \frac{\partial \langle E \rangle}{\partial T} = \beta^2 \left( \langle E^2 \rangle - \langle E \rangle^2 \right). $$

**Magnetization:**

$$ \langle M \rangle = \frac{1}{Z} \sum_{\{S\}} M(\{S\})  e^{-\beta H(\{S\})}, $$
where $M(\{S\}) = \sum_i S_i$ is the instantaneous magnetization of a configuration.

Note: at $h=0$, every configuration $\{S\}$ has a degenerate partner $\{-S\}$ with the same energy, for any temperature. Exact enumeration sums over both of these with equal Boltzmann weights, giving $\langle M \rangle = 0$.

Typically a single MC run will have $\langle M \rangle = 0$ for $T>T_c$, and $\langle M \rangle = \pm1$ for $T < T_c$. averaging over many runs will recover the result where $\langle M \rangle = 0$ for all $T$.

**Susceptibility**
$$ \chi = \beta \left( \langle M^2 \rangle - \langle M \rangle^2 \right), $$
which, at $h=0$, reduces to $\chi = \beta \langle M^2 \rangle$ due to the symmetry above.
