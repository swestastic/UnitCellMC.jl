# UnitCellMC.jl

[![CI](https://github.com/swestastic/UnitCellMC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/swestastic/UnitCellMC.jl/actions/workflows/CI.yml)

Classical Markov Chain Monte Carlo simulations of spin models with user-defined lattice geometries.

## Features

- **Models:** Ising and XY, with per-bond couplings and per-site external fields
- **Algorithms:** Metropolis, Wolff (single-cluster), and Swendsen-Wang
- **Geometry:** arbitrary lattices built from [LatticeUtilities.jl](https://github.com/SmoQySuite/LatticeUtilities.jl) unit cells, lattices, and bonds
- **Measurements:** energy, specific heat, magnetization, susceptibility, Binder cumulant, and correlation functions, with binning and jackknife error estimation
- **Sweeps:** temperature sweeps with optional simulated annealing and parallel independent runs over threads; results saved as CSV

## Installation

Install using Julia's package manager:

```julia
import Pkg
Pkg.add(url="github.com/swestastic/UnitCellMC.jl")
```

## Usage

See `examples/` for runnable 1D Ising chain and 2D square-lattice Ising/XY simulations. All functionality is accessed through qualified names, e.g. `UnitCellMC.Geometry(...)`, `UnitCellMC.IsingModel(...)`, `UnitCellMC.sweep_Ts(...)`.
