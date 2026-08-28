# UnitCellMC.jl

Monte Carlo simulations of classical lattice models on arbitrary
lattice geometries.

## Overview

UnitCellMC.jl is designed to provide a flexible framework for
simulating classical lattice models. Currently, it supports the Ising model with the Metropolis algorithm, with further extensions to come in the future.

The geometry of the system is separated from the physical model,
allowing the same simulation framework to be used with different
lattice geometries.

## Acknowledgements

This package makes use of [LatticeUtilities.jl](https://github.com/SmoQySuite/LatticeUtilities.jl), and takes some design inspiration from [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl).
