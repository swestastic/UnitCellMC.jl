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

cluster_seed(::AbstractModel, geometry, ::AbstractState) = rand(1:geometry.n_sites)
validate_cluster_model(::AbstractModel, ::Any, ::AbstractState, ::Real) = nothing

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

function apply_cluster!(::AbstractModel, ::Any, ::AbstractState, ::Vector{Int})
    throw(MethodError(apply_cluster!, ()))
end

"""
    step!(WolffAlgorithm(), model, geometry, state, T)

Grow and flip one Wolff cluster. Model-specific behavior is supplied by
`cluster_bond_probability` and `apply_cluster!`, which keeps the cluster
traversal reusable for models with different local degrees of freedom.
"""
function step!(
    ::WolffAlgorithm,
    model::AbstractModel,
    geometry,
    state::AbstractState,
    T::Real
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))
    validate_cluster_model(model, geometry, state, T)

    seed = cluster_seed(model, geometry, state)
    1 <= seed <= geometry.n_sites || throw(ArgumentError(
        "cluster_seed returned site $seed outside 1:$(geometry.n_sites)"
    ))

    in_cluster = falses(geometry.n_sites)
    in_cluster[seed] = true
    cluster = Int[seed]
    frontier = Int[seed]

    while !isempty(frontier)
        site = pop!(frontier)
        info = geometry.neighbor_table[site]

        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            neighbor == site && continue
            in_cluster[neighbor] && continue

            probability = cluster_bond_probability(
                model, geometry, state, site, neighbor, bond_id, T
            )
            0 <= probability <= 1 || throw(ArgumentError(
                "cluster_bond_probability returned $probability; expected a value in [0, 1]"
            ))

            if rand() < probability
                in_cluster[neighbor] = true
                push!(cluster, neighbor)
                push!(frontier, neighbor)
            end
        end
    end

    apply_cluster!(model, geometry, state, cluster)
    return WolffUpdateResult(length(cluster))
end