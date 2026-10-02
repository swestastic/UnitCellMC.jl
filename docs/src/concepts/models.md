# Models

## Ising Model

The Ising model is described by the following Hamiltonian:
$$H=-\sum_{\langle i, j \rangle} J_{ij}s_i s_j - \sum_i h_i s_i$$

Where $J_{ij}$ is the interaction strength between site $i$ and site $j$, $s_i$ is the value of site $i$ ($\pm1$), and $h_i$ is the external field strength at site $i$.

Create an Ising model with one coupling per bond template and one field value
per site in the unit cell:

```julia
model = IsingModel([J₁, J₂], [h₁, h₂])
```

The length of `h` must equal the number of sites in the unit cell.
