# To Do / In Progress

In the future we'd like to add some further functionality to this package.

## Models

- XY Model
- Heisenberg Model
- Clock Model
- Potts Model

## Algorithms

- Glauber
- Heatbath

## Functionality

- Define $h$ by site in unit cell rather than for the entire lattice
- Checkpointing (JLD2 or HDF5)

## Optimizations

- Calculate energy difference across bonds (wolff, swendsen-wang). This would prevent having to recalculate the energy of the entire lattice after every cluster move.
