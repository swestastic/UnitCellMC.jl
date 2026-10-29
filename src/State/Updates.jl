"""
    SpinUpdate{V,M}

A proposed single-site update: change the local degree of freedom at `site`
to `new_value`.

The proposal carries the change the update makes to the total magnetization
in `magnetization_change` — a scalar for [`IsingModel`](@ref), a
`(ΔMx, ΔMy)` tuple for [`XYModel`](@ref) and [`ClockModel`](@ref) — because
only the model knows how its local values map to magnetization (clock
indices, for example, through their lookup tables). [`apply_update!`](@ref)
can therefore keep the state's running totals exact without access to the
model.

Every single-site-update model uses this one proposal type; see the update
contract under [`apply_update!`](@ref).

# Fields

- `site::Int`: Index of the site to update.
- `new_value::V`: The proposed value for that site.
- `magnetization_change::M`: The change the update makes to the total
  magnetization.
"""
struct SpinUpdate{V,M}
    site::Int
    new_value::V
    magnetization_change::M
end

"""
    apply_update!(state, proposal, ΔE)

Apply `proposal` to `state` in place, updating the site's degree of freedom
along with the running total energy and magnetization.

This is the final step of the single-site update contract shared by every
local-update algorithm and model:

1. `propose(rng, algorithm, model, geometry, state; workspace)` picks a site
   via the algorithm's [`AbstractProposalScheme`](@ref) (see
   [`propose_site`](@ref)) and calls the model's
   `propose(rng, algorithm, model, geometry, state, site)` method, which
   returns a [`SpinUpdate`](@ref) carrying everything needed to apply the
   move — including the magnetization change, which only the model can
   compute;
2. `energy_difference(model, geometry, state, proposal)` returns the energy
   change `ΔE`;
3. the algorithm decides acceptance and calls
   `apply_update!(state, proposal, ΔE)`, a state-layer function that is
   model-free by construction.

Each state type provides its own method in its `State/` file.

# Returns

`nothing`; `state` is mutated in place.
"""
function apply_update! end
