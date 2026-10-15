# Getting Started

## Installation

Install UnitCellMC using Julia's package manager:

```julia
import Pkg
Pkg.add(url="github.com/swestastic/UnitCellMC.jl")
```

Nothing is exported; access the API through qualified names. The
examples alias the module for brevity:

```julia
import UnitCellMC as ucmc
import LatticeUtilities as lu
```

## Basic Workflow

A typical simulation consists of several components:

1. Define a lattice geometry.
2. Define a model.
3. Define the simulation parameters.
4. Select a Monte Carlo algorithm.
5. Run the simulation.
6. Analyze the measurements.

The [`UnitCellMC.sweep_Ts`](@ref) driver automates this workflow over
an array of temperatures, including binning, jackknife error
estimation, and CSV output. See the
[Ising Model 2D Square Lattice](@ref) tutorial or the `examples/`
directory for complete scripts.
