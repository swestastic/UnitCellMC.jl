# Algorithms

Descriptions of the different algorithms that can be used in UnitCellMC.jl

## Metropolis-Hastings

The Metropolis-Hastings algorithm is a single-spin-flip algorithm, which sweeps through the lattice and updates sites one at a time.

A Monte-Carlo step is as follows.

- Select a site $i$ on the lattice.
- Pick a new value for the site
  - **Ising model**: $s_i = +1 \rightarrow s_i = -1$ or $s_i = -1 \rightarrow s_i = +1$
  <!-- - **XY model**: $s_i = \theta_i \in [0, 2\pi) \rightarrow s_i = \phi_i \in [0, 2\pi)$, where $\phi$ is a new random angle -->
- Calculate the change in energy $\Delta E$
  - **Ising model**: $\Delta E = 2 \sum_{j} J_{ij}s_i s_j + 2 h s_i$,  where $j$ is sites that are bonded with site $i$, and $J_{ij}$ is the interaction strength of that bond. $h$ is the magnetic field strength.
  <!-- - **XY model**: $\Delta E = -\sum_j J_{ij} [\cos(\phi_i-\theta_j) - \cos(\theta_i - \theta_j)] - h [\cos(\phi_i)-\cos(\theta_i)]$. -->
- Draw a random number $r\in[0,1]$
- Accept the update with probability $\text{min}(1,e^{-\beta\Delta E})$, where $\beta=\frac{1}{T}$.

## Wolff Cluster

The Wolff algorithm grows and flips one connected cluster at a time. For the
zero-field ferromagnetic Ising model:

- Select a random seed site.
- Visit neighboring sites with the same spin.
- Add each such neighbor with probability $1-e^{-2J_{ij}/T}$.
- Flip every spin in the completed cluster.

The cluster traversal is independent of the model. A model supplies the bond
activation probability and the operation that applies a completed cluster, so
other discrete or continuous spin models can reuse the algorithm interface.

## Swendsen-Wang Cluster

The Swendsen-Wang algorithm builds all clusters before updating the state:

- Activate each compatible bond with probability $1-e^{-2J_{ij}/T}$.
- Find the connected components of the activated-bond graph.
- Assign each cluster a new allowed orientation.

As with Wolff, bond activation and cluster mutation are model hooks. The
algorithm itself only handles bond traversal and connected-component finding.
