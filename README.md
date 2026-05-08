# UnitCellMC.jl

This package allows for classical Markov Chain Monte Carlo simulations of spin models with user-defined lattice geometries. This approach is implemented using [LatticeUtilities.jl](https://github.com/SmoQySuite/LatticeUtilities.jl)].

## Usage

install using 
```julia
import Pkg
Pkg.add(url="github.com/swestastic/UnitCellMC.jl")
```

An example is provided in `examples/ising_squarelattice.jl`. This should be enough to hopefully get you started!

## Notes

This is pretty barebones for now. It currently only supports the Ising model using the Metropolis-Hastings algorithm. Ultimately it will receive data binning and averaging, file outputs, and more models, algorithms, and observables. My hope is that this will be a good tool for easily simulating different classical models (Ising, XY, Potts, etc.) on different geometries. It would also be nice to add support for next-nearest-neighbor or other bond formats, with adjustable interaction strength.  

## Acknowledgements

Some of the bond code is borrowed from [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl)]. 