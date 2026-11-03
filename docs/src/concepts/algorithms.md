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

### Proposal schemes

The first step — selecting the site — is independent of the acceptance rule
and is controlled by the algorithm's proposal scheme
(`MetropolisAlgorithm(RandomProposal())` is the default):

- `RandomProposal()`: every update draws a site uniformly at random.
- `SequentialProposal()`: updates visit the sites in site-index order, so one
  sweep attempts every site exactly once. Since `LatticeUtilities.jl` numbers
  sites in column-major unit-cell order, consecutive updates target
  neighboring sites — on a square lattice the sweep proceeds column by
  column (equivalently row by row, up to transposition).
- `CheckerboardProposal()`: for bipartite lattices, sites are 2-colored so
  that no bond connects same-color sites, and each sweep updates all sites of
  one color and then all sites of the other. Same-color sites share no bond,
  so the updates within each half-sweep do not interact directly. A
  non-bipartite lattice (e.g. triangular) is rejected with an error when the
  run starts.
- `ColoredProposal(k)`: the same idea with up to `k` colors, for lattices
  that need more than two — e.g. `ColoredProposal(3)` for the tripartite
  triangular or kagome lattices. The coloring is a saturation-first greedy
  (DSATUR) heuristic; if it cannot color the lattice with `k` colors the run
  fails with an error when it starts.

Each single-site update satisfies detailed balance in every scheme, so the
completed sweep leaves the Boltzmann distribution invariant; the schemes
differ only in visiting order, and hence in how successive measurements are
correlated. Comparing them is a simple way to study how site-selection
order affects autocorrelation times. The scheme is orthogonal to the
acceptance rule: `GlauberAlgorithm(SequentialProposal())` works the same way.

## Glauber

The Glauber algorithm is a single-spin-flip algorithm with the same trial moves
as Metropolis-Hastings (a spin flip for Ising, a uniformly random new angle for
XY, one of the other $q - 1$ states for the clock model). Only the acceptance
rule differs:

- Accept the update with probability $\frac{1}{1 + e^{\beta\Delta E}}$, where $\beta=\frac{1}{T}$.

Because the proposals are symmetric, the ratio of forward and reverse
acceptance probabilities is $e^{-\beta\Delta E}$ and detailed balance holds.
Unlike Metropolis, moves that lower the energy are not accepted with
certainty; a move with $\Delta E = 0$ is accepted with probability $1/2$. For
the Ising model a Glauber spin flip is equivalent to a heat-bath update of the
chosen spin.

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
