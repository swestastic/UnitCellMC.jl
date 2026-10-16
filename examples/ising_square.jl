"""
This example code implements the 2D Ising model using the UnitCellMC.jl package. 
It demonstrates how to set up the geometry, initialize the state, and perform 
Metropolis Monte Carlo steps to simulate the system's evolution.
"""

#### Import the necessary modules

using UnitCellMc
import LatticeUtilities as lu

#### Initialize Geometry

L = [16, 16] # In this case this is [Lx, Ly] becuase the lattice_vectors are [̂x, ̂y]

lattice_vectors = [[1.0, 0.0], [0.0, 1.0]] # [x̂, ŷ] in 2D
basis_vectors = [[0.0, 0.0]] # our unit cell is a single site at the origin

unit_cell = lu.UnitCell( 
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true, true] # periodic boundary conditions along ̂x and ŷ
)

bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1, 0]) # bond along ̂x
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [0, 1]) # bond along ̂y
bonds = [bond_1, bond_2]

geometry = Geometry(unit_cell, lattice, bonds)

n_sites = geometry.n_sites

#### Initialize State
J1 = 1.0 # Coupling strength along ̂x
J2 = 1.0 # Coupling strength along ̂y
J = [J1, J2]
h = [0.0] # External magnetic field strength per unit-cell site

model = IsingModel(J, h)
algorithm = MetropolisAlgorithm()

state = initialize_state(model, geometry) # Generate random spin configuration, calculate the initial energy and magnetization

#### Set simulation parameters

n_thermalization = 10_000 # warmup / thermalization sweeps before measurements
n_measurements = 10_000 # number of measurement sweeps per temperature (one cluster update per sweep for Swendsen-Wang)
n_unmeasured = 5 # number of unmeasured sweeps between each measurement to reduce autocorrelation
n_bins = 100 # number of bins for jackknife error estimation
n_runs = 20 # number of independent runs to average over

diagnostics = available_diagnostics(algorithm)
measurements = available_measurements(model)

#### Perform simulation

Ts = collect(3.0:-0.1:1.0) # temperatures from T = 3.0 down to T = 1.0

sweep_results = sweep_Ts(
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
    n_runs = n_runs,
    measurements = measurements,
    diagnostics = diagnostics
)
