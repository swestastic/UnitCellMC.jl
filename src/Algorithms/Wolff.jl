"""
    WolffAlgorithm

Marker type selecting a single-cluster Wolff update. The cluster traversal is
model-independent; models provide `cluster_seed`, `cluster_bond_probability`,
and `apply_cluster!` methods.
"""
struct WolffAlgorithm <: AbstractAlgorithm end

"""
    WolffUpdateResult

Result of one Wolff update, recording the number of sites in the flipped
cluster.
"""
struct WolffUpdateResult <: AbstractUpdateResult
    cluster_size::Int
end

cluster_seed(rng::AbstractRNG, ::AbstractModel, geometry, ::AbstractState) = rand(rng, 1:geometry.n_sites)
validate_cluster_model(::AbstractModel, ::Any, ::AbstractState, ::Real) = nothing
cluster_context(::AbstractRNG, ::AbstractModel, ::Any, ::AbstractState, ::Real) = nothing
cluster_ghost_count(::AbstractModel, ::Any, ::AbstractState, ::Any) = 0
cluster_ghost_probability(::AbstractModel, ::Any, ::AbstractState, ::Int, ::Int, ::Real, ::Any) = 0.0
cluster_ghost_attached!(::Any, ::Int) = nothing

function cluster_bond_probability(
    ::AbstractModel,
    ::Any,
    ::AbstractState,
    ::Int,
    ::Int,
    ::Int,
    ::Real
)
    throw(MethodError(cluster_bond_probability, ()))
end

cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T, ::Nothing) =
    cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T)
cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T, ::Any) =
    cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T)

function apply_cluster!(::AbstractModel, ::Any, ::AbstractState, ::Vector{Int})
    throw(MethodError(apply_cluster!, ()))
end

apply_cluster!(model, geometry, state, cluster, ::Nothing) =
    apply_cluster!(model, geometry, state, cluster)

"""
    step!(WolffAlgorithm(), model, geometry, state, T; rng = Random.default_rng())

Grow and flip one Wolff cluster. Model-specific behavior is supplied by
`cluster_bond_probability` and `apply_cluster!`, which keeps the cluster
traversal reusable for models with different local degrees of freedom. All
random numbers (cluster context, seed site, bond activation) come from `rng`.
"""
function step!(
    ::WolffAlgorithm,
    model::AbstractModel,
    geometry,
    state::AbstractState,
    T::Real;
    rng::AbstractRNG = Random.default_rng()
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))
    validate_cluster_model(model, geometry, state, T)
    context = cluster_context(rng, model, geometry, state, T)

    seed = cluster_seed(rng, model, geometry, state)
    1 <= seed <= geometry.n_sites || throw(ArgumentError(
        "cluster_seed returned site $seed outside 1:$(geometry.n_sites)"
    ))

    in_cluster = falses(geometry.n_sites)
    in_cluster[seed] = true
    cluster = Int[seed]
    frontier = Int[seed]
    ghost_count = cluster_ghost_count(model, geometry, state, context)
    any_ghost_attached = false

    while !isempty(frontier)
        any_ghost_attached && break
        site = pop!(frontier)
        info = geometry.neighbor_table[site]

        for (bond_template, neighbor) in zip(info.templates, info.neighbors)
            neighbor == site && continue
            in_cluster[neighbor] && continue

            probability = cluster_bond_probability(
                model, geometry, state, site, neighbor, bond_template, T, context
            )
            0 <= probability <= 1 || throw(ArgumentError(
                "cluster_bond_probability returned $probability; expected a value in [0, 1]"
            ))

            if rand(rng) < probability
                in_cluster[neighbor] = true
                push!(cluster, neighbor)
                push!(frontier, neighbor)
            end
        end

        for ghost in 1:ghost_count
            probability = cluster_ghost_probability(
                model, geometry, state, site, ghost, T, context
            )
            0 <= probability <= 1 || throw(ArgumentError(
                "cluster_ghost_probability returned $probability; expected a value in [0, 1]"
            ))

            if probability > 0 && rand(rng) < probability
                cluster_ghost_attached!(context, ghost)
                any_ghost_attached = true
                break
            end
        end
    end

    apply_cluster!(model, geometry, state, cluster, context)
    return WolffUpdateResult(length(cluster))
end