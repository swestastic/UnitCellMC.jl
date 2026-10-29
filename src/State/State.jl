include("Updates.jl")
include("IsingState.jl")
include("XYState.jl")
include("ClockState.jl")

"""
    state_type(model)

Return the `AbstractState` subtype that `model` simulates: the type
that [`initialize_state`](@ref) constructs for it, and the type that
[`step!`](@ref), measurements, and the simulation drivers expect to receive
(see [`validate_model_state`](@ref)).

Each model defines its own method (e.g. `state_type(::IsingModel) =
IsingState`); this fallback throws so that a model without one fails loudly,
like the [`initialize_state`](@ref) fallback.

# Throws

`ArgumentError` if `model` does not define its own `state_type` method. To
support a custom model, define `state_type(::MyModel) = MyState`.
"""
function state_type(model::AbstractModel)
    throw(ArgumentError(
        "state_type is not implemented for $(nameof(typeof(model))); " *
        "define `state_type(::$(nameof(typeof(model))))` returning the " *
        "AbstractState subtype the model simulates"
    ))
end

"""
    validate_model_state(model, state)

Check that `state` is an instance of [`state_type`](@ref)`(model)`, so that a
mismatched model/state pair fails with a descriptive error before any update
runs instead of deep inside an update loop.

# Throws

`ArgumentError` if `state` is not a `state_type(model)`, or if `model` does
not define `state_type`.

# Returns

`nothing`.
"""
function validate_model_state(model::AbstractModel, state::AbstractState)
    S = state_type(model)
    state isa S || throw(ArgumentError(
        "$(nameof(typeof(model))) simulates a $S state; got $(typeof(state)). " *
        "Construct the state with `initialize_state(model, geometry)`"
    ))
    return nothing
end

"""
    magnetization_value(state)

Return the scalar magnetization represented by `state`: the stored total
magnetization for an [`IsingState`](@ref), and the magnitude
`hypot(magnetization_x, magnetization_y)` of the two components for an
[`XYState`](@ref) or [`ClockState`](@ref). Used by the optional
order-parameter measurements (`:abs_magnetization`, `:binder_cumulant`,
`:magnetization_fourth`).

Each state type provides its own method (in the corresponding model's file,
next to [`local_operator`](@ref)); this fallback throws for state types
without one.

# Throws

`ArgumentError` if the state type does not define `magnetization_value`.
"""
function magnetization_value(state::AbstractState)
    throw(ArgumentError(
        "magnetization_value is not implemented for $(nameof(typeof(state)))"
    ))
end

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
