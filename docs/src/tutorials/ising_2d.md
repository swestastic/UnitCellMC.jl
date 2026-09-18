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
bond_1 = lu.Bond(orbitals = (1, 1), displacement = [0, 1])
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [-1, 0])

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
h = 0.0 # External magnetic field strength
```

We then create an `IsingModel` object to hold these values for reference. Finally, we'll create an `IsingState` object which generates a random `state.spins` array, and the energy $E$ and magnetization $M$ of the initial configuration are calculated.

```julia
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

## Building the Simulation Loop

We'll create two functions for our simulation loop. The first, `MC_Sweep!` simply loops over the number of sites in the lattice, and performs that number of Metropolis updates.

```julia
function MC_Sweep!(
    algorithm,
    model,
    geometry,
    state,
    container::ucmc.MeasurementContainer,
    T
)
    for _ in 1:n_sites
        ucmc.step!(algorithm, model, geometry, state, T)
    end
end
```

Next we'll define a function `run_simulation` which handles the different kinds of sweeps (warmup, measurement, decorrelation) and measurements.

```julia
function run_simulation(algorithm, model, geometry, state, container::ucmc.MeasurementContainer, parameters)
    T = parameters.T

    for _ in 1:parameters.n_thermalization
        MC_Sweep!(algorithm, model, geometry, state, container, T)
    end

    for _ in 1:parameters.n_measurements
        MC_Sweep!(algorithm, model, geometry, state, container, T)
        ucmc.measure!(container, state)

        for __ in 1:parameters.n_unmeasured           
            MC_Sweep!(algorithm, model, geometry, state, container, T)
        end
    end

    return container
end
```

## Running the Simulation

### Single Temperature Runs

If your only interest is in gathering data for a single temperature, you'll need to create a container for storing the measurement results. Then we can run the simulation.

```julia
container = ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins; measurements = Symbol[]) # Alternatively, set measurements = [:correlation] to calculate 2 site correlation functions as well.
container = run_simulation(algorithm, model, geometry, state, container, parameters)
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

For multiple temperature runs, we construct a function `sweep_Ts` which will handle the initialization of a new measurement container for every temperature.

```julia
function sweep_Ts(
    algorithm,
    model,
    geometry,
    state,
    n_thermalization,
    n_measurements,
    n_unmeasured,
    n_bins,
    Ts;
    simulated_annealing = true,
    measurements = Symbol[]
)

    sweep_results = Vector{Any}(undef, length(Ts))
    n_sites = geometry.n_sites
    L = geometry.lattice.L

    for (i, T) in enumerate(Ts)

        parameters = ucmc.SimulationParameters(T, n_thermalization, n_measurements, n_unmeasured, n_bins)

        if !simulated_annealing
            state = ucmc.initialize_state(model, geometry)
        end

        container = ucmc.MeasurementContainer(model, geometry, n_measurements, n_bins; measurements = measurements)
        container = run_simulation(algorithm, model, geometry, state, container, parameters)

        processed_results = ucmc.analyze(container, T, n_sites)
        sweep_results[i] = processed_results
        # ucmc.save_results(model, algorithm, L, T, processed_results, parameters) # Use this to save results for each T separately, if desired
    end

    return sweep_results
end
```

The `simulated_annealing` flag determines whether the lattice configuration is preserved between temperature changes. If `simulated_annealing = false` is set, then a new random lattice configuration will be initialized for every new temperature.

Then we can create an array `Ts` to iterate over:

```julia
Ts = collect(3.0:-0.1:1.0) # Evenly spaced temperatures from T = 3.0 to T = 1.0
```

Note that with `simulated_annealing = true`, it's best practice to start at the highest temperature, and decrease it. This is not strictly enforced in the code, but it will ensure best results.

Finally, call `Sweep_Ts` to run the simulation

```julia
sweep_results = sweep_Ts(
    algorithm, model, geometry, state,
    n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
    simulated_annealing = true,
    measurements = [:correlation] # enable correlation measurements
    # measurements = []
)
```

And we can save the results to a condensed CSV file as follows:

```julia
ucmc.save_sweep_results(
    sweep_results, Ts, model, algorithm, L,
    n_thermalization, n_measurements, n_unmeasured, n_bins
)
```

Now you should see an output folder with results saved in a CSV format, which can easily be used for plotting.
