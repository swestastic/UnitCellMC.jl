# UnitCellMC.jl

[![CI (main)](https://github.com/swestastic/UnitCellMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/swestastic/UnitCellMC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://swestastic.github.io/UnitCellMC.jl/dev/)

Classical Markov Chain Monte Carlo simulations of spin models with user-defined lattice geometries.

## Features

- **Models:** Ising, XY, q-state clock, and Heisenberg with per-bond couplings and per-site external fields
- **Algorithms:** Metropolis, Glauber, Wolff (single-cluster), and Swendsen-Wang, with random, sequential, checkerboard, and graph-colored site-selection schemes
- **Geometry:** arbitrary lattices built from [LatticeUtilities.jl](https://github.com/SmoQySuite/LatticeUtilities.jl) unit cells, lattices, and bonds. Built-in functionality for coloring lattices.
- **Measurements:** energy, specific heat, magnetization, susceptibility, Binder cumulant, correlation functions, and structure factors, with binning, jackknife error estimation, and autocorrelation analysis
- **Drivers:** single-temperature runs, temperature sweeps with optional simulated annealing, and parallel tempering (replica exchange); independent runs and replicas parallelize over threads, with autocorrelation-based autotuning of the measurement spacing and bin size
- **Reproducibility:** every run derives from a single recorded seed and is bit-identical for any thread count; results saved as CSV

## Installation

Install using Julia's package manager:

```julia
import Pkg
Pkg.add(url="https://github.com/swestastic/UnitCellMC.jl")
```

## Usage

The main user-facing API is exported — `using UnitCellMC` brings `Geometry`, the models and algorithms, `initialize_state`, `SimulationParameters`, `MeasurementContainer`, the simulation drivers (`run_simulation`, `sweep_Ts`, `tempering_sweep`, ...), `analyze`, and the save functions into scope. Lattice types (`UnitCell`, `Lattice`, `Bond`) come from [LatticeUtilities.jl](https://github.com/SmoQySuite/LatticeUtilities.jl) and must be loaded separately.

See `examples/` for runnable 1D Ising chain and 2D square-lattice Ising/XY/clock simulations, including temperature sweeps, proposal-scheme comparisons, and parallel tempering.
