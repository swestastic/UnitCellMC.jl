# Getting Started

## Installation

Install UnitCellMC using Julia's package manager:

```julia
import Pkg
Pkg.add(url="github.com/swestastic/UnitCellMC.jl")
```

The main user-facing types and functions are exported, so
`using UnitCellMC` is enough for a standard run. Lattice types
(`UnitCell`, `Lattice`, `Bond`) are provided by LatticeUtilities, which
must be loaded separately. The examples alias the modules for brevity
and use qualified names:

```julia
import UnitCellMC as ucmc
import LatticeUtilities as lu
```

Equivalently:

```julia
using UnitCellMC
using LatticeUtilities
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
