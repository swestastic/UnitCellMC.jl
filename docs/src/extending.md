# Extending UnitCellMC

UnitCellMC is built so that **geometry**, **models**, and **algorithms** are
independent of one another: any model can be simulated on any geometry with
any algorithm, and each of the three can be extended without touching the
other two. This page describes the contracts that make that work and gives
step-by-step recipes for adding a model, an algorithm, or a measurement.

If you follow the conventions below, your extension gets the whole driver
stack for free: [`run_simulation`](@ref UnitCellMC.run_simulation), [`sweep_Ts`](@ref UnitCellMC.sweep_Ts) (threaded runs,
annealing, reproducible per-run RNGs), binning and jackknife analysis, and
the IO layer.

## The layering rule

```text
Core          AbstractModel / AbstractAlgorithm / AbstractState + the layout contract
Geometry      unit cell + lattice + bond templates -> neighbor table (no model/algorithm knowledge)
State         configuration + running totals; the update carrier (SpinUpdate) and apply_update!
Algorithms    update traversal (step!), diagnostics, cluster drivers (call model hooks)
Models        model struct + all physics (proposals, energies, cluster probabilities)
Measurements  observables and analysis (call model/state hooks)
Simulation    drivers that tie everything together
IO            writing results
```

The dependency direction only ever points downward in this list. In practice
that means:

- **Algorithms** never read model fields. They call hook functions
  ([`propose`](@ref UnitCellMC.propose), [`energy_difference`](@ref UnitCellMC.energy_difference),
  [`cluster_bond_probability`](@ref UnitCellMC.cluster_bond_probability), ...) that models implement.
- **Models** never see algorithm internals. The cluster algorithms hand them
  a context object (which the model itself created in `cluster_context`) and,
  for Swendsen-Wang, a [`ClusterWorkspace`](@ref UnitCellMC.ClusterWorkspace) whose labels they read via
  the documented fields.
- **Geometry** exposes the neighbor table (`geometry.neighbor_table[site]`
  gives parallel `bonds`/`templates`/`neighbors` vectors) and coordinate
  helpers ([`site_at`](@ref UnitCellMC.site_at), [`build_displacement_classes`](@ref UnitCellMC.build_displacement_classes),
  [`build_momentum_grid`](@ref UnitCellMC.build_momentum_grid)). Nothing in the geometry layer knows models
  exist.
- **State** objects own their running totals. Only `apply_update!` and the
  model's cluster workers mutate them, and every mutation keeps them exactly
  consistent with the configuration (the test suite recomputes them from
  scratch after every update — your extension should keep that discipline).

### Source tree and include order

```text
src/
  Core.jl                     # abstract types + the coupling layout contract
  Geometry/                   # Geometry.jl, Displacements.jl, Momenta.jl
  State/                      # Updates.jl, IsingState.jl, ..., Initialization.jl
  Algorithms/                 # Algorithms.jl (marker includes), Metropolis.jl, ...
  Models/                     # Models.jl (includes + shared accessors), Ising.jl, ...
  State/Initialization.jl     # included *after* Models (see below)
  Measurements/
  Simulation/
  IO/
```

Include order in `src/UnitCellMC.jl` is significant: state structs come
before models (model method signatures name state types), while
`initialize_state` methods come after models (their signatures name model
types). Algorithms come before models because models add methods to
algorithm-dispatched hooks (`propose`, `cluster_bond_probability`, ...).
When you add files, include them in the matching directory's umbrella file
(`State/State.jl`, `Algorithms/Algorithms.jl`, `Models/Models.jl`), not in
`UnitCellMC.jl`.

## Conventions

- **Naming:** `CamelCase` types, `snake_case` functions, `!` on mutating
  functions, Unicode math names (`ΔE`) where they aid readability.
- **Filenames:** one type family per file, named after it
  (`Models/Heisenberg.jl`, `Algorithms/Heatbath.jl`, `State/HeisenbergState.jl`).
- **RNG:** every random draw goes through an explicit `rng`. Public functions
  take `rng = Random.default_rng()` as a keyword; internal hooks take `rng`
  as the first positional argument. Never call `rand()` without an `rng`, and
  never change the order of draws in an existing code path (simulation
  results are guaranteed bit-identical across thread counts).
- **Validation:** validate eagerly and throw `ArgumentError` with the
  offending value in the message.
- **Docstrings:** every function gets one with `# Arguments` / `# Returns` /
  `# Throws` sections as applicable, and `[`name`](@ref UnitCellMC.name)` cross-references.
  Every docstring must be listed in a `docs/src/api/*.md` page — the docs
  build fails otherwise.
- **Style:** 4-space indent; keep allocations out of per-update hot paths.

## Adding a model

A model is five pieces: a state type, an initializer, a model struct, the
update methods, and the measurement hooks. Worked example skeleton: a
Heisenberg model.

### 1. The state type — `src/State/HeisenbergState.jl`

```julia
mutable struct HeisenbergState{S,T<:Real} <: AbstractState
    spins::S              # per-site unit 3-vectors
    energy::T             # running totals: kept exactly consistent
    magnetization_x::T
    magnetization_y::T
    magnetization_z::T
end
```

Include it from `src/State/State.jl`. The running totals are the contract:
any function that mutates `spins` must update them in the same step.

### 2. The update application — in the same state file

`apply_update!` is model-free by construction: the [`SpinUpdate`](@ref UnitCellMC.SpinUpdate)
proposal carries everything needed, including the magnetization change (only
the model knows how its local values map to magnetization).

```julia
function apply_update!(state::HeisenbergState, proposal::SpinUpdate, ΔE::Real)
    state.spins[proposal.site] = proposal.new_value
    state.energy += ΔE
    Δm = proposal.magnetization_change
    state.magnetization_x += Δm[1]
    state.magnetization_y += Δm[2]
    state.magnetization_z += Δm[3]
    return nothing
end
```

### 3. The initializer — `src/State/Initialization.jl`

```julia
function initialize_state(model::HeisenbergModel, geometry::Geometry;
                          rng::AbstractRNG = Random.default_rng())
    validate_model_geometry(model, geometry)
    spins = [random_unit_vector(rng) for _ in 1:geometry.n_sites]
    # compute energy and magnetization from scratch so the running totals
    # start exactly consistent; use bond_strength/field_strength for lookups
    ...
    return HeisenbergState(spins, energy, mx, my, mz)
end
```

The fallback `initialize_state` throws an `ArgumentError` for models without
their own method, so forgetting this step fails loudly.

### 4. The model struct and physics — `src/Models/Heisenberg.jl`

```julia
struct HeisenbergModel{T<:Real} <: AbstractModel{T}
    J::Vector{T}   # one coupling per bond template of the Geometry
    h::Vector{T}   # one field per unit-cell orbital
end
```

That is the **coupling layout contract** (documented under
[`AbstractModel`](@ref UnitCellMC.AbstractModel)): one coupling per bond template, one field per
unit-cell orbital, read everywhere through the [`couplings`](@ref UnitCellMC.couplings) /
[`fields`](@ref UnitCellMC.fields) accessors. If your model stores them differently, override
those two accessors and [`validate_model_geometry`](@ref UnitCellMC.validate_model_geometry) — nothing else
needs to change.

Required methods:

```julia
state_type(::HeisenbergModel) = HeisenbergState

# state readout (used by measurements and the proposal)
local_operator(state::HeisenbergState, site) = state.spins[site]
magnetization_value(state::HeisenbergState) =
    sqrt(state.magnetization_x^2 + state.magnetization_y^2 + state.magnetization_z^2)

# the single-site update contract — the algorithm's proposal scheme picks the
# site (random, sequential, ...), the model proposes the new value there
function propose(rng::AbstractRNG, ::MetropolisAlgorithm, model::HeisenbergModel, geometry, state, site::Int)
    new = random_unit_vector(rng)
    return SpinUpdate(site, new, new .- state.spins[site])
end

function energy_difference(model::HeisenbergModel, geometry, state, proposal)
    site = proposal.site
    info = geometry.neighbor_table[site]
    J = couplings(model)
    ΔE = 0.0
    for (template, neighbor) in zip(info.templates, info.neighbors)
        neighbor == site && continue
        ΔE -= J[template] * sum(proposal.new_value .* state.spins[neighbor]
                                .- state.spins[site] .* state.spins[neighbor])
    end
    ΔE -= field_strength(model, geometry, site) * proposal.magnetization_change[3]  # field along z
    return ΔE
end
```

That is all the Metropolis algorithm needs; **Glauber works automatically**
(a generic fallback forwards [`propose`](@ref UnitCellMC.propose) to the Metropolis proposal —
override it if you want a different Glauber proposal). In the inner loop,
hoist `J = couplings(model)` once and index `J[template]`; the per-site
`templates` vector exists precisely for this.

Measurement hooks:

```julia
observables(model::HeisenbergModel, geometry) = [...]          # raw per-measurement Observables
derived_observables(model::HeisenbergModel, geometry) = [...]  # DerivedObservables from jackknife samples
```

Optional hooks enable optional features:

- [`n_operator_components`](@ref UnitCellMC.n_operator_components) + [`operator_components`](@ref UnitCellMC.operator_components) — enable
  `:structure_factor`, the (inverse-DFT) `:raw_correlation` /
  `:connected_correlation` on periodic lattices, and
  `:magnetization_per_orbital`. The components must satisfy
  `operator_product(model, a, b) ≈ sum(operator_components(model, a) .* operator_components(model, b))`.
- [`operator_product`](@ref UnitCellMC.operator_product) — the two-body product, needed by the direct
  correlation path (open boundaries).
- [`binder_denominator`](@ref UnitCellMC.binder_denominator) — enables `:binder_cumulant`.
- `extra_measurements(::Type{<:HeisenbergModel})` — advertise model-specific
  optional measurements (see "Adding a measurement" below).

### 5. Optional: cluster algorithms (Wolff / Swendsen-Wang)

Cluster updates are for **ferromagnetic** couplings only (the generic
[`validate_cluster_model`](@ref UnitCellMC.validate_cluster_model) enforces `J >= 0`; extend it if your model
has more constraints). The model supplies the physics through these hooks:

```julia
cluster_context(rng, model, geometry, state, T)          # per-update scratch (e.g. a reflection axis); `nothing` if unused
cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T, context)
cluster_ghost_count(model, geometry, state, context)     # 2 for a +/- external field, 0 for no field coupling
cluster_ghost_probability(model, geometry, state, site, ghost, T, context)
cluster_ghost_attached!(context, ghost)
apply_cluster!(model, geometry, state, cluster, context, in_cluster)           # Wolff: mutate one cluster
apply_clusters!(rng, model, geometry, state, workspace, context)               # Swendsen-Wang: mutate all clusters
```

The ghost-site contract: ghost `g` sits at index [`ghost_site`](@ref UnitCellMC.ghost_site)`(geometry, g)`
(`= n_sites + g`), and ghost-attached clusters are never flipped. In the
Swendsen-Wang form, call [`draw_cluster_flips!`](@ref UnitCellMC.draw_cluster_flips!)`(rng, workspace,
geometry)` first — it draws one coin flip per cluster and freezes the
ghost-attached ones — then mutate using `workspace.labels` (site → cluster
label). Look at `Models/XY.jl` for a complete example.

### 6. Registration checklist

- [ ] Export the model type in `src/UnitCellMC.jl`.
- [ ] Add the new docstrings to `docs/src/api/models.md` and `docs/src/api/state.md`.
- [ ] Tests in `test/runtests.jl` (`import UnitCellMC as ucmc`, qualify
      everything): running-total invariants after every kind of update, for
      every algorithm the model supports; RNG reproducibility; ideally an
      exact-enumeration check on a tiny lattice.
- [ ] Run `julia --project -e "using Pkg; Pkg.test()"` and
      `julia --project=docs docs/make.jl` — both must pass.

## Adding an algorithm

An algorithm is a marker struct plus `step!`, and it lives in
`src/Algorithms/MyAlgorithm.jl` (included from `src/Algorithms/Algorithms.jl`).
Worked skeleton: a heat-bath algorithm.

```julia
struct HeatbathAlgorithm <: AbstractAlgorithm end

struct HeatbathUpdateResult <: AbstractUpdateResult
    accepted::Bool     # whatever describes one update
end

function step!(::HeatbathAlgorithm, model::AbstractModel, geometry::Geometry,
               state::AbstractState, T::Real; rng::AbstractRNG = Random.default_rng())
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))
    ...
    return HeatbathUpdateResult(...)
end

diagnostic_values(::HeatbathAlgorithm, result) = (acceptance_ratio = ...,)
available_diagnostics(::Type{<:HeatbathAlgorithm}) = [:acceptance_ratio]
```

Rules of the road:

- **Reuse the single-site contract.** If your algorithm is a local-update
  scheme, build `step!` out of [`propose`](@ref UnitCellMC.propose) → [`energy_difference`](@ref UnitCellMC.energy_difference)
  → [`apply_update!`](@ref UnitCellMC.apply_update!) (as `Metropolis.jl`/`Glauber.jl` do): every model
  then works with it immediately.
- **Never read model fields or state fields directly in the driver** — go
  through the hooks. The algorithm should not know what a spin is.
- [`steps_per_sweep`](@ref UnitCellMC.steps_per_sweep) defaults to `geometry.n_sites` (one update per
  site per sweep). Define `steps_per_sweep(::MyAlgorithm, ::Geometry) = 1` in
  your algorithm's file if one `step!` already moves a macroscopic fraction
  of the lattice (as the cluster algorithms do).
- If a step needs lattice-sized scratch arrays, implement
  [`algorithm_workspace`](@ref UnitCellMC.algorithm_workspace)`(::MyAlgorithm, model, geometry, state)` to
  return a reusable workspace (see [`ClusterWorkspace`](@ref UnitCellMC.ClusterWorkspace)); `step!` must
  also accept `workspace = nothing` and allocate then.
- The diagnostic keys returned by [`diagnostic_values`](@ref UnitCellMC.diagnostic_values) must be exactly
  [`available_diagnostics`](@ref UnitCellMC.available_diagnostics)`(MyAlgorithm)`; the container validates
  this up front ([`validate_run`](@ref UnitCellMC.validate_run)).

Registration checklist:

- [ ] Export the algorithm type in `src/UnitCellMC.jl`.
- [ ] Add docstrings to `docs/src/api/algorithms.md`.
- [ ] Tests: invariants for every model the algorithm supports, and
      `diagnostic_values` keys matching `available_diagnostics`.

## Adding an optional measurement

Optional measurements are requested by name in a [`MeasurementContainer`](@ref UnitCellMC.MeasurementContainer)
and resolved by the registry in `src/Measurements/Optional.jl`. To add one
(e.g. `:helicity_modulus`):

1. Write an entry builder `helicity_modulus_entry(model, geometry)` returning
   `(obs::Vector{`[`Observable`](@ref UnitCellMC.Observable)`}, der::Vector{`[`DerivedObservable`](@ref UnitCellMC.DerivedObservable)`})`.
2. Register it with one method:
   `optional_observables(model, geometry, ::Val{:helicity_modulus}) = helicity_modulus_entry(model, geometry)`.
3. Advertise it: add the name to `_COMMON_MEASUREMENTS` (if every model
   supports it) or to `extra_measurements(::Type{<:MyModel})` (model-specific).
4. If requesting it implicitly pulls in other measurements, declare them in
   `measurement_implies(::Val{:helicity_modulus}) = (:other_name,)`.
5. If it is derived from other observables, set `depends_on` so
   [`validate_dependencies`](@ref UnitCellMC.validate_dependencies) checks them at construction.

Unknown or unsupported names throw an `ArgumentError` listing the
[`available_measurements`](@ref UnitCellMC.available_measurements) — so the registry is the single source of
truth. Model physics used by a measurement (like [`binder_denominator`](@ref UnitCellMC.binder_denominator)
or [`clock_angle_order`](@ref UnitCellMC.clock_angle_order)) belongs in the model's file, not in the
measurement code.

## A note on geometry

There is nothing to extend: a [`Geometry`](@ref UnitCellMC.Geometry) is built from any
LatticeUtilities `UnitCell`/`Lattice` plus a list of bond templates, and
multiple orbitals, long-range bonds, and mixed periodic/open boundaries all
work out of the box. If you need coordinate machinery in a new measurement,
use [`site_at`](@ref UnitCellMC.site_at), [`build_displacement_classes`](@ref UnitCellMC.build_displacement_classes), and
[`build_momentum_grid`](@ref UnitCellMC.build_momentum_grid) rather than calling LatticeUtilities conversion
routines directly.
