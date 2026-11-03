"""
This example implements the 2D Ising model on the triangular lattice with
the UnitCellMC.jl package, and demonstrates the lattice-coloring proposal
scheme `ColoredProposal`.

The triangular lattice is tripartite: its sites can be 3-colored so that no
bond connects same-color sites. `ColoredProposal(3)` builds that coloring
once per sweep (a saturation-first greedy over the geometry's bonds) and
then visits the sites color class by color class. Same-color sites share no
bond, so the updates within each color class are mutually independent — the
scheme combines the fixed-order determinism of `SequentialProposal` with a
visiting order that respects the lattice structure. (On bipartite lattices
the 2-color special case is `CheckerboardProposal`; the triangular lattice
is not bipartite, so the checkerboard scheme would throw an `ArgumentError`
here.)

Run with multiple threads to parallelize the independent runs:

    julia --project --threads auto examples/ising_triangular_colored.jl

Results are saved under `results/`; the directory name carries the scheme
(`..._MetropolisAlgorithm-ColoredProposal3_...`).
"""

#### Import the necessary modules

using UnitCellMC
import LatticeUtilities as lu

#### Set the global RNG seed

# Every random number in this script (initial states, the lattice coloring,
# and all n_runs parallel runs in sweep_Ts) derives from the global RNG, so
# rerunning with the same seed reproduces the results exactly, for any number
# of threads. The seed is saved to the results folder's metadata.txt. Change
# it for a statistically independent run, or call set_global_seed!() for a
# random one.
seed = 1234
set_global_seed!(seed)

#### Initialize Geometry

# The triangular lattice as a Bravais lattice with a single-site unit cell:
# lattice vectors at 60°, and three nearest-neighbor bond directions.
L = [12, 12] # [Lx, Ly] in units of the lattice vectors

lattice_vectors = [[1.0, 0.0], [0.5, sqrt(3) / 2]] # 60° between them
basis_vectors = [[0.0, 0.0]] # one site per unit cell

unit_cell = lu.UnitCell(
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true, true] # periodic boundary conditions
)

bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1, 0])   # along the first lattice vector
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [0, 1])   # along the second lattice vector
bond_3 = lu.Bond(orbitals = (1, 1), displacement = [1, -1])  # along the diagonal
bonds = [bond_1, bond_2, bond_3]

geometry = Geometry(unit_cell, lattice, bonds)

n_sites = geometry.n_sites # 144 sites, each with 6 nearest neighbors

#### Initialize Model

J = [1.0, 1.0, 1.0] # ferromagnetic coupling along the three bond directions
h = [0.0] # no external field

model = IsingModel(J, h)

#### Choose the colored proposal scheme

# The triangular lattice is 3-colorable: no bond connects same-color sites.
# Each sweep updates all sites of color 1, then color 2, then color 3.
algorithm = MetropolisAlgorithm(ColoredProposal(3))

#### Set simulation parameters

# The triangular-lattice Ising model orders at T_c = 4 / log(3) ≈ 3.641
# (J = 1). Sample densely around it, ordered high to low so simulated
# annealing carries each run's state sensibly.
Tc = 4 / log(3)
Ts = [4.5, 4.0, 3.8, Tc, 3.4, 3.0, 2.5]

n_thermalization = 2_000  # warmup sweeps before measurements
n_measurements = 5_000    # measurement sweeps per run; scale up for tighter errors
n_unmeasured = 2          # decorrelation sweeps between measurements
n_bins = 50               # bins for jackknife error estimation
n_runs = 4                # independent runs averaged at each temperature

#### Perform the temperature sweep

sweep_results = sweep_Ts(
    algorithm,
    model,
    geometry,
    initialize_state(model, geometry),
    n_thermalization,
    n_measurements,
    n_unmeasured,
    n_bins,
    Ts;
    simulated_annealing = true,
    n_runs = n_runs,
    measurements = [:abs_magnetization, :abs_magnetization_squared],
    diagnostics = [:acceptance_ratio],
    time_series = :scalars
)

#### Report the observables across the sweep

# The magnetization rises from zero as the sweep cools through
# T_c = 4 / log(3) ≈ 3.641.
println("\n2D triangular Ising, L = $(L[1]), ColoredProposal(3):")
for (T, results) in zip(Ts, sweep_results)
    e, e_err = results.primary[:energy]
    m, m_err = results.primary[:abs_magnetization]
    marker = isapprox(T, Tc) ? " <- T_c" : ""
    println("  T = $(rpad(round(T, digits = 3), 5)): E = $(round(e, digits = 4)) ± $(round(e_err, digits = 4)), |m| = $(round(m, digits = 4)) ± $(round(m_err, digits = 4))$marker")
end
