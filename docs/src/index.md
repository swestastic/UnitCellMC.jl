# UnitCellMC.jl

Monte Carlo simulations of classical lattice models on arbitrary
lattice geometries.

## Overview

UnitCellMC.jl provides a flexible framework for simulating classical
lattice models. It currently supports the Ising, XY, and q-state clock
models with Metropolis, Glauber, Wolff, and Swendsen-Wang update algorithms, with
further extensions planned.

The geometry of the system is separated from the physical model,
allowing the same simulation framework to be used with different
lattice geometries.

For a guided walkthrough, see the [Getting Started](@ref) page or one
of the tutorials; to add your own model, algorithm, or measurement, see
[Extending UnitCellMC](@ref); standalone scripts live in the
[`examples/`](https://github.com/swestastic/UnitCellMC.jl/tree/main/examples) directory.

## Acknowledgements

This package makes use of [LatticeUtilities.jl](https://github.com/SmoQySuite/LatticeUtilities.jl), and takes some design inspiration from [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl).
