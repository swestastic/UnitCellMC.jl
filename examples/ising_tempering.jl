"""
This example demonstrates parallel tempering (replica exchange) with the
UnitCellMC.jl package, simulating the 2D Ising model on a square lattice.

Instead of annealing one configuration through a temperature sweep (see
examples/ising_square.jl), parallel tempering simulates one replica per
temperature simultaneously and periodically attempts to exchange the
configurations of neighboring temperatures. Low-temperature replicas
equilibrate through excursions to high temperature, which is what makes the
method valuable where plain Metropolis struggles: first-order transitions,
frustrated couplings, and low temperatures. (Here the 2D Ising transition is
continuous, so the example serves to show the workflow: build a ladder, run
`tempering_sweep`, and check the swap acceptance rates.)

Run with multiple threads to simulate the replicas in parallel:

    julia --project --threads auto examples/ising_tempering.jl
"""

#### Import the necessary modules

using UnitCellMC
import LatticeUtilities as lu

#### Set the global RNG seed

# Every random number in this script (the per-replica initial states, all
# replica updates, and the swap decisions) derives from the global RNG, so
# rerunning with the same seed reproduces the results exactly, for any number
# of threads. The seed is saved to the results folder's metadata.txt. Change
# it for a statistically independent run, or call set_global_seed!() for a
# random one.
seed = 1234
set_global_seed!(seed)

#### Initialize Geometry

L = [8, 8] # In this case this is [Lx, Ly] because the lattice_vectors are [x̂, ŷ]

lattice_vectors = [[1.0, 0.0], [0.0, 1.0]] # [x̂, ŷ] in 2D
basis_vectors = [[0.0, 0.0]] # our unit cell is a single site at the origin

unit_cell = lu.UnitCell(
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true, true] # periodic boundary conditions along x̂ and ŷ
)

bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1, 0]) # bond along x̂
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [0, 1]) # bond along ŷ
bonds = [bond_1, bond_2]

geometry = Geometry(unit_cell, lattice, bonds)

#### Initialize the model and algorithm

J1 = 1.0 # Coupling strength along x̂
J2 = 1.0 # Coupling strength along ŷ
J = [J1, J2]
h = [0.0] # External magnetic field strength per unit-cell site

model = IsingModel(J, h)
algorithm = MetropolisAlgorithm() # the per-replica update; WolffAlgorithm() works too

#### Build the temperature ladder

# Temperatures must be sorted ascending (Ts[1] is the coldest replica).
# Geometric spacing is the standard starting point; the ladder spans the
# critical temperature T_c = 2/ln(1+√2) ≈ 2.269 (J = 1).
Ts = geometric_ladder(1.0, 3.0, 12)
println("Ladder: ", round.(Ts, digits = 3))

#### Set simulation parameters

# Counts are in *rounds*: one round is one sweep per replica plus, every
# swap_every rounds, a swap pass over adjacent temperature pairs. Swaps mix
# configurations across the ladder, so fewer unmeasured rounds are needed
# than the single-temperature autocorrelation time suggests.
n_thermalization = 5_000 # warmup rounds before measurements
n_measurements = 10_000 # measurement rounds
n_unmeasured = 4 # decorrelation rounds between measurements
n_bins = 100 # bins per replica for jackknife error estimation
swap_every = 1 # rounds between swap passes

diagnostics = available_diagnostics(algorithm)
measurements = available_measurements(model)

#### Perform the parallel tempering simulation

result = tempering_sweep(
    algorithm,
    model,
    geometry,
    Ts,
    n_thermalization,
    n_measurements,
    n_unmeasured,
    n_bins;
    swap_every = swap_every,
    measurements = measurements,
    diagnostics = diagnostics,
    time_series = :scalars # unbinned history for autocorrelations: :scalars (skips the correlation vector), :all, or :none
)

#### Check the swap acceptance rates

# Swap acceptance per adjacent temperature pair, written to
# swap_acceptance.csv in the results folder. Rates near 0.2-0.5 indicate a
# well-spaced ladder; much lower means the pair is too far apart (add
# temperatures there), much higher means the replicas are redundant.
println("\nSwap acceptance per adjacent pair:")
for i in eachindex(result.swap_acceptance)
    T_low, T_high = result.temperatures[i], result.temperatures[i + 1]
    acc = result.swap_acceptance[i]
    verdict = 0.2 <= acc <= 0.5 ? "ok" : (acc < 0.2 ? "add temperatures here" : "replicas may be redundant")
    println("  T = $(round(T_low, digits = 3)) <-> $(round(T_high, digits = 3)): $(round(acc, digits = 3)) ($verdict)")
end

#### Report the observables across the ladder

println("\nObservables per temperature (per site):")
for (T, results) in zip(result.temperatures, result.results)
    e, e_err = results.primary[:energy]
    m, m_err = results.primary[:abs_magnetization]
    println("  T = $(round(T, digits = 3)): E = $(round(e, digits = 4)) ± $(round(e_err, digits = 4)), |m| = $(round(m, digits = 4)) ± $(round(m_err, digits = 4))")
end
