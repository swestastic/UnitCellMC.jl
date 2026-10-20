"""
This example code implements the 2D XY model using the UnitCellMC.jl package. 
It demonstrates how to set up the geometry, initialize the state, and perform 
Metropolis Monte Carlo steps to simulate the system's evolution.
"""

#### Import the necessary modules

using UnitCellMC
import LatticeUtilities as lu

#### Initialize Geometry

L = [8, 8] # In this case this is [Lx, Ly] becuase the lattice_vectors are [̂x, ̂y]

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

model = XYModel(J, h)
algorithm = WolffAlgorithm() 

diagnostics = available_diagnostics(algorithm)
measurements = available_measurements(model)

state = initialize_state(model, geometry) # Generate random spin configuration, calculate the initial energy and magnetization

#### Set simulation parameters

n_thermalization = 10_000 # warmup / thermalization sweeps before measurements
n_bins = 100 # number of bins for jackknife error estimation
n_runs = 20 # number of independent runs to average over

Ts = collect(3.0:-0.1:1.0) # temperatures from T = 3.0 down to T = 1.0

#### Choose n_unmeasured and the bin size from the autocorrelation time

# A short pilot run at every temperature measures after every sweep and
# estimates the integrated autocorrelation time τ (in sweeps) of each
# observable; the slowest one sets the measurement spacing and bin size.
# For Wolff a sweep is a single cluster update. At high temperature the
# clusters are only a few sites, so τ in these units is largest there
# (roughly N / ⟨cluster size⟩), not near the BKT transition (T_BKT ≈ 0.89).
tuning = autotune_binning(
    algorithm, model, geometry, initialize_state(model, geometry), Ts;
    n_pilot = 5_000,
)
tuned_observables = collect(keys(first(tuning.per_temperature))) # all measured observables
println("Slowest τ = $(round(tuning.tau, digits = 2)) sweeps ($(tuning.observable) at T = $(tuning.T))")

n_unmeasured = tuning.n_unmeasured # sweeps between measurements, ≈ τ - 1
n_measurements = tuning.bin_size * n_bins # measurements per run; scale up for smaller error bars
println("Using n_unmeasured = $n_unmeasured, bin_size = $(tuning.bin_size), n_measurements = $n_measurements")

#### Perform simulation

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
    measurements = measurements,
    diagnostics = diagnostics,
    time_series = :scalars # unbinned history for autocorrelations: :scalars (skips the correlation vector), :all, or :none
)

#### Check the binning against the production autocorrelation times

# `analyze` estimates the integrated autocorrelation time of every observable
# from the production time series. These times are in units of measurements
# (n_unmeasured + 1 sweeps); bins spanning >= 10 of them give reliable error bars.
bin_size = n_measurements ÷ n_bins
for (T, results) in zip(Ts, sweep_results)
    taus = [tau_lower_bound(results.autocorrelation[name]) for name in tuned_observables]
    taus = filter(isfinite, taus)
    if isempty(taus)
        println("T = $T: no fluctuations recorded")
        continue
    end
    tau = maximum(taus)
    verdict = bin_size >= 10 * tau ? "ok" : "bin_size < 10 tau: increase n_pilot or bin_factor"
    println("T = $T: tau = $(round(tau, digits = 2)) measurements, bin_size / tau = $(round(bin_size / tau, digits = 1)) ($verdict)")
end
