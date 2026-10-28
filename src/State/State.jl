include("IsingState.jl")
include("XYState.jl")
include("ClockState.jl")

# The per-model `initialize_state` methods live in `Initialization.jl`, which
# `src/UnitCellMC.jl` includes after `Models/`: the state structs must exist
# before the model files (their signatures name the state types), while the
# `initialize_state` signatures name the model types.

"""
    initialize_state(model, geometry; rng = Random.default_rng())

Build a random initial state for `model` on `geometry`, with the running
totals (energy, magnetization) computed from scratch to match the
configuration.

There is no model-agnostic initialization: each model provides its own
method (see `State/Initialization.jl`), so this fallback throws for any
model that does not.

# Arguments

- `model`: The model supplying the couplings/field and the state type.
- `geometry`: The lattice geometry, giving `n_sites` and the neighbor table.
- `rng`: Random number generator for the initial configuration.

# Throws

`ArgumentError` if `model` does not provide its own `initialize_state`
method. To support a custom model, define
`initialize_state(model::MyModel, geometry::Geometry; rng)` returning a
state whose cached totals are consistent with the configuration it builds.

# Returns

For models that implement it, a freshly randomized `AbstractState` whose
cached energy/magnetization match its configuration.
"""
function initialize_state(
    model::AbstractModel,
    geometry::Geometry;
    rng::AbstractRNG = Random.default_rng()
)
    throw(ArgumentError(
        "initialize_state is not implemented for $(nameof(typeof(model))); " *
        "define `initialize_state(model::$(nameof(typeof(model))), geometry::Geometry; rng)` " *
        "(see src/State/Initialization.jl)"
    ))
end
