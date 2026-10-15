"""
This example code implements the 1D Ising chain using the UnitCellMC.jl package.
It demonstrates how to set up the geometry, initialize the state, and perform
Metropolis Monte Carlo steps to simulate the system's evolution.
"""

#### Import the necessary modules

import UnitCellMC as ucmc
import LatticeUtilities as lu

#### Initialize Geometry

L = [8] # the unit cell is repeated 8 times along x̂

lattice_vectors = [[1.0]]
basis_vectors = [[0.0]] # our unit cell is a single site at the origin

unit_cell = lu.UnitCell( 
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true] # periodic boundary conditions along ̂x
)

bond_1 = lu.Bond(orbitals = (1, 1), displacement = [1]) # bond along ̂x
bonds = [bond_1]

geometry = ucmc.Geometry(unit_cell, lattice, bonds)

n_sites = geometry.n_sites

#### Initialize State
J1 = 1.0 # Coupling strength along ̂x
J = [J1]
h = [0.0] # External magnetic field strength per unit-cell site

model = ucmc.IsingModel(J, h)
algorithm = ucmc.MetropolisAlgorithm()

state = ucmc.initialize_state(model, geometry) # Generate random spin configuration, calculate the initial energy and magnetization

#### Set simulation parameters

n_thermalization = 10_000 # warmup / thermalization sweeps before measurements
n_measurements = 10_000 # number of measurement sweeps per temperature (L Metropolis steps per sweep)
n_unmeasured = 5 # number of unmeasured sweeps between each measurement to reduce autocorrelation
n_bins = 100 # number of bins for jackknife error estimation

#### Perform simulation

Ts = collect(0.1:0.1:1.0) # 10 temperatures from T = 0.1 to T = 1.0

sweep_results = ucmc.sweep_Ts(
    algorithm, model, geometry, state,
    n_thermalization, n_measurements, n_unmeasured, n_bins, Ts;
    simulated_annealing = true,
    measurements = [:connected_correlation], # enable connected correlation measurements
    # measurements = Symbol[],
    diagnostics = [:acceptance_ratio]
)
