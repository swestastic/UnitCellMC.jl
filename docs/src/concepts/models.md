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

## XY Model

The XY model is described by the Hamiltonian:
$$H=-\sum_{\langle i, j \rangle} J_{ij}\cos(\theta_i - \theta_j) - \sum_i h_i \cos(\theta_i)$$

where $\theta_i \in [0, 2\pi)$ is the angle of the $O(2)$ spin on site $i$,
and the external field points along the x-axis. It is constructed with the
same argument layout as the Ising model:

```julia
model = XYModel([J₁, J₂], [h₁, h₂])
```

## Clock Model

The $q$-state clock model restricts the XY spin to $q$ equally spaced angles,
$\theta_i = 2\pi n_i / q$ with $n_i \in \{0, \dots, q-1\}$, and has the same
Hamiltonian:
$$H=-\sum_{\langle i, j \rangle} J_{ij}\cos(\theta_i - \theta_j) - \sum_i h_i \cos(\theta_i)$$

The state stores the integer clock index $n_i$ of every site. $q = 2$ is the
Ising model, $q = 4$ is two decoupled Ising models with coupling $J/2$, and
$q \to \infty$ approaches the XY model. It takes $q$ first, then the same
arguments as the XY model:

```julia
model = ClockModel(q, [J₁, J₂], [h₁, h₂])
```

Metropolis and Glauber propose one of the other $q - 1$ states uniformly. The Wolff and
Swendsen-Wang updates reflect clusters across one of the $q$ mirror lines of
the clock, $n \mapsto (m - n) \bmod q$, which maps clock states onto clock
states.

Besides the measurements shared with the XY model, the clock model accepts
the optional `:clock_order_parameter` measurement,
$\langle \cos(q\phi_M) \rangle$, where $\phi_M$ is the direction of the
total magnetization. It tends to 1 when the magnetization locks onto a clock
direction and to 0 when its direction is uniformly distributed, so for
$q \ge 5$ it distinguishes the ordered phase from the intermediate critical
phase.
