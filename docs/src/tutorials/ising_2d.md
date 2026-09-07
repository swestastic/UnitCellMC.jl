# Ising Model 2D Square Lattice

Download this example as a complete [Julia script](https://github.com/swestastic/UnitCellMC.jl/blob/main/examples/ising_square.jl).

In this example we will simulate the Ising model with nearest-neighbor interactions on a square lattice. The Hamiltonian for the Ising model is given by

$$
H = -\sum_{\langle i, j \rangle} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i,
$$
where $J_{ij}$ is the coupling strength between sites $i$ and $j$. $\sigma_i$ is the value of site $i$ ($\pm1$). $h_i$ is the magnetic field strength at site $i$.

## Import Packages

We can start with importing [UnitCellMC](https://github.com/swestastic/UnitCellMC.jl) and [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl).

```julia
import UnitCellMC as ucmc
import LatticeUtilities as lu
```

## Initialize Geometry

Define the lattice grid to be 10 unit cells along each vector

```julia
Lx = 10
Ly = 10
L = [Lx, Ly]
```

Then define the lattice vectors and basis vectors. The basis vectors are the locations of sites in our unit cell. In this case, it's a single site at $(0,0)$. The lattice vectors extend the unit cell to create the lattice.

```julia
lattice_vectors = [[1.0, 0.0], [0.0, 1.0]]
basis_vectors = [[0.0, 0.0]]
```

Next we initialize the `unit_cell` and `lattice` items. Here we can decide if we want periodic boundaries along each lattice vector.

```julia
unit_cell = lu.UnitCell(
    lattice_vecs = lattice_vectors,
    basis_vecs = basis_vectors
)

lattice = lu.Lattice(
    L = L,
    periodic = [true, true]
)
```

Now we can create our bonds between sites, and define the coupling strength for each bond.

```julia
bond_1 = lu.Bond(orbitals = (1, 1), displacement = [0, 1])
bond_2 = lu.Bond(orbitals = (1, 1), displacement = [-1, 0])

bonds = [bond_1, bond_2]
J1 = 1.0
J2 = 1.0
J = [J1, J2]
```

```julia

geometry = ucmc.Geometry(unit_cell, lattice, bonds)

n_sites = lu.nsites(unit_cell, lattice)
```
