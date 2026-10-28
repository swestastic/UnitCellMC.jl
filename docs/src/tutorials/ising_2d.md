# Ising Model 2D Square Lattice

Download this example as a complete [Julia script](https://github.com/swestastic/UnitCellMC.jl/blob/main/examples/ising_square.jl).

In this example we will simulate the Ising model with nearest-neighbor interactions on a square lattice. The Hamiltonian for the Ising model is given by

```math
H = -\sum_{\langle i, j \rangle} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i,
```

where $J_{ij}$ is the coupling strength between sites $i$ and $j$. $\sigma_i$ is the value of site $i$ ($\pm1$). $h_i$ is the magnetic field strength at site $i$.

## Import Packages

We can start with importing [UnitCellMC](https://github.com/swestastic/UnitCellMC.jl) and [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl).

```julia
import UnitCellMC as ucmc
import LatticeUtilities as lu
```

## Set the Global RNG Seed

All random numbers in a run (the initial spins, every Monte Carlo update, and each of the parallel `n_runs` in `sweep_Ts`) are drawn from Julia's global RNG unless an explicit `rng` keyword is passed. Seeding it once at the top of the script makes the whole simulation reproducible:

```julia
seed = 1234
ucmc.set_global_seed!(seed)
```

Rerunning with the same seed and inputs gives identical results, for any number of threads. Use a different seed for a statistically independent run, or call `ucmc.set_global_seed!()` without an argument to draw a random seed. Either way the seed is written to the `metadata.txt` of every results folder, together with the Julia version (seeded random streams can change between Julia releases), so each saved run records how to recreate it. Seeding with `Random.seed!` directly also makes runs reproducible, but the seed is then recorded as `unknown`.

## Initialize Geometry

Define the lattice grid to be 10 unit cells along each vector

```julia
Lx = 10
Ly = 10
L = [Lx, Ly]
```

Then define the lattice vectors and basis vectors. The basis vectors are the locations of sites in our unit cell. In this case, it's a single site at $(0,0)$. The lattice vectors extend the unit cell to create the lattice.

```julia
lattice_vectors = [[1.0, 0.0], [0.0, 1.0]]
basis_vectors = [[0.0, 0.0]]
```

Next we initialize the `unit_cell` and `lattice` items. Here we can decide if we want periodic boundaries along each lattice vector.

```julia
unit_cell = lu.UnitCell(
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true, true]
)
```

Now we can create our bonds between sites, and define the coupling strength for each bond.

```julia
bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1, 0]) # bond along x̂
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [0, 1]) # bond along ŷ

bonds = [bond_1, bond_2]

geometry = ucmc.Geometry(unit_cell, lattice, bonds)

n_sites = lu.nsites(unit_cell, lattice)
```

## Initialize State

Here we'll initialize our Ising model state. We can define a coupling strength $J$ for each bond in `bonds` by supplying it to the same index in the array `J` as it is in `bonds`. We will also define the external magnetic field strength $h$ which is applied to all sites on the lattice.

```julia
J1 = 1.0 # Coupling strength along ̂x
J2 = 1.0 # Coupling strength along ̂y
J = [J1, J2]
h = [0.0] # External magnetic field strength per unit-cell site
```

The `IsingModel` object holds these values for reference, and `initialize_state` creates an `IsingState` with a random `state.spins` array, along with the energy $E$ and magnetization $M$ of the initial configuration.

```julia
model = ucmc.IsingModel(J, h)
state = ucmc.initialize_state(model, geometry) # Generate random spin configuration, calculate the initial energy and magnetization
```

## Choosing the update algorithm

Next we'll choose which algorithm we want to use. This example will implement the Metropolis-Hastings algorithm.

```julia
algorithm = ucmc.MetropolisAlgorithm()
```

## Setting Simulation Parameters

This section defines the number of sweeps and bins that will be used.

```julia
n_thermalization = 10_000 # warmup / thermalization sweeps before measurements
n_measurements = 10_000 # number of sweeps (L**2 Metropolis steps per sweep) to perform measurements
n_unmeasured = 5 # number of unmeasured sweeps between each measurement to reduce autocorrelation
n_bins = 100 # number of bins for jackknife error estimation
T = 2.27 # Temperature to evaluate
```

These are stored in an immutable struct of type `SimulationParameters`. Note that if you want to sweep over multiple $T$ values, our T value will be overwritten later when we supply the $Ts$ array, as we create a new `SimulationParameters` for every temperature.

```julia
parameters = ucmc.SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)
```

## The Simulation Loop

The package provides the simulation drivers used in this tutorial. The first, `MC_Sweep!`, performs one Monte Carlo sweep: `n_sites` single-site `step!` updates for a local-update algorithm like Metropolis, or a single cluster update for the Wolff and Swendsen-Wang cluster algorithms, since one cluster update already moves a macroscopic fraction of the lattice. This convention is defined by `steps_per_sweep`, which you can extend for custom algorithms. During the sweep, per-update diagnostics (e.g. the acceptance ratio for Metropolis) are accumulated into `UpdateStatistic`s.

```julia
ucmc.MC_Sweep!(algorithm, model, geometry, state, container, parameters.T, stats)
```

The second, `run_simulation`, handles the different kinds of sweeps (warmup, measurement, decorrelation) and the measurements themselves. It returns the filled measurement container.

```julia
ucmc.run_simulation(algorithm, model, geometry, state, container, parameters)
```

## Running the Simulation

### Single Temperature Runs

If your only interest is in gathering data for a single temperature, you'll need to create a container for storing the measurement results. Then we can run the simulation.

```julia
container = ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins;
    measurements = Symbol[], # Alternatively, set measurements = [:raw_correlation] or [:connected_correlation]
    diagnostics = [:acceptance_ratio] # to calculate 2-site correlation functions as well.
)
container = ucmc.run_simulation(algorithm, model, geometry, state, container, parameters)
```

Next we perform the bin processing and errorbar calculation with `analyze`

```julia
processed_results = ucmc.analyze(container, T, n_sites)
```

Finally, save them to a CSV file with `save_results`

```julia
ucmc.save_results(model, algorithm, L, T, processed_results, parameters)
```

### Multiple Temperature Run

For multiple temperature runs, the package provides `sweep_Ts`, which initializes a new `SimulationParameters` and measurement container for every temperature, runs `run_simulation` and `analyze` at each one, and finally saves the combined results to a condensed CSV output folder with `save_sweep_results` (the output directory can be changed with the `root` keyword).

The `simulated_annealing` flag determines whether the lattice configuration is preserved between temperature changes. If `simulated_annealing = false` is set, then a new random lattice configuration will be initialized for every new temperature.

Then we can create an array `Ts` to iterate over:

```julia
Ts = collect(3.0:-0.1:1.0) # Evenly spaced temperatures from T = 3.0 to T = 1.0
```

Note that with `simulated_annealing = true`, it's best practice to start at the highest temperature, and decrease it. This is not strictly enforced in the code, but it will ensure best results.

Finally, call `sweep_Ts` to run the full simulation:

```julia
sweep_results = ucmc.sweep_Ts(
    algorithm, model, geometry, state,
    n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
    simulated_annealing = true,
    measurements = [:connected_correlation], # enable connected correlation measurements
    # measurements = Symbol[],
    diagnostics = [:acceptance_ratio]
)
```

To average over several independent simulation runs at each temperature, set the `n_runs` keyword (e.g. `n_runs = 20`). The bins of all runs are merged before the jackknife analysis, so the reported error bars reflect the variance between independent runs as well as within them. Each run anneals its own configuration across the temperature sweep when `simulated_annealing = true`, and is re-initialized randomly at every temperature otherwise. The simulations execute in parallel over Julia threads; start Julia with `--threads auto` to benefit. With `simulated_annealing = true` the temperatures run one after another (each run carries its configuration forward), so the `n_runs` runs at each temperature are the parallel unit; with `simulated_annealing = false` every (temperature, run) pair is independent, so all `length(Ts) * n_runs` simulations form a single parallel pool. Before the threads start, `sweep_Ts` draws one seed per simulation from the global RNG (or from its `rng` keyword) and gives every simulation its own generator, so each one consumes the same random stream no matter which thread executes it. The results are therefore reproducible from the seed set above, independent of the thread count.

Now you should see an output folder with results saved in a CSV format, which can easily be used for plotting.
