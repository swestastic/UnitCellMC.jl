# Algorithms

Descriptions of the different algorithms that can be used in UnitCellMC.jl

## Metropolis-Hastings

The Metropolis-Hastings algorithm is a single-spin-flip algorithm, which sweeps through the lattice and updates sites one at a time.

A Monte-Carlo step is as follows.

- Select a site $i$ on the lattice.
- Pick a new value for the site
  - **Ising model**: $s_i = +1 \rightarrow s_i = -1$ or $s_i = -1 \rightarrow s_i = +1$
  - **XY model**: $s_i = \theta_i \in [0, 2\pi) \rightarrow s_i = \phi_i \in [0, 2\pi)$, where $\phi$ is a new random angle
- Calculate the change in energy $\Delta E$
  - **Ising model**: $\Delta E = 2 \sum_{j} J_{ij}s_i s_j + 2 h s_i$,  where $j$ is sites that are bonded with site $i$, and $J_{ij}$ is the interaction strength of that bond. $h$ is the magnetic field strength.
  - **XY model**: $\Delta E = -\sum_j J_{ij} [\cos(\phi_i-\theta_j) - \cos(\theta_i - \theta_j)] - h [\cos(\phi_i)-\cos(\theta_i)]$.
- Draw a random number $r\in[0,1]$
- Accept the update with probability $\text{min}(1,e^{-\beta\Delta E})$, where $\beta=\frac{1}{T}$.
