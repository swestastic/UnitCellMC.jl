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

steps_per_sweep(::WolffAlgorithm, ::Geometry) = 1

cluster_seed(rng::AbstractRNG, ::AbstractModel, geometry, ::AbstractState) = rand(rng, 1:geometry.n_sites)

"""
    validate_cluster_model(model, geometry, state, T)

Check that `model` can be updated by the cluster algorithms
([`WolffAlgorithm`](@ref), [`SwendsenWangAlgorithm`](@ref)), called once at
the start of every cluster [`step!`](@ref).

The default enforces the cluster updates' core requirement — non-negative
(ferromagnetic) couplings, so the bond-activation probabilities stay in
`[0, 1]` — for models with the default coupling layout (see
[`couplings`](@ref)); a model without a `J` field passes. Models with extra
structure extend this method with their own checks.

# Throws

`ArgumentError` if any coupling is negative.

# Returns

`nothing`.
"""
function validate_cluster_model(model::AbstractModel, geometry, state::AbstractState, ::Real)
    hasproperty(model, :J) || return nothing
    all(>=(0), couplings(model)) || throw(ArgumentError(
        "Cluster algorithms require non-negative couplings; got J = $(couplings(model))"
    ))
    return nothing
end
cluster_context(::AbstractRNG, ::AbstractModel, ::Any, ::AbstractState, ::Real) = nothing
cluster_ghost_count(::AbstractModel, ::Any, ::AbstractState, ::Any) = 0
cluster_ghost_probability(::AbstractModel, ::Any, ::AbstractState, ::Int, ::Int, ::Real, ::Any) = 0.0
cluster_ghost_attached!(::Any, ::Int) = nothing

cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T, ::Any) =
    cluster_bond_probability(model, geometry, state, site, neighbor, bond_template, T)

# Models without a cluster context ignore both it and the reusable mask.
apply_cluster!(model, geometry, state, cluster, ::Nothing, ::BitVector) =
    apply_cluster!(model, geometry, state, cluster)

algorithm_workspace(::WolffAlgorithm, model, geometry, state) = ClusterWorkspace()

"""
    step!(WolffAlgorithm(), model, geometry, state, T;
        rng = Random.default_rng(), workspace = nothing)

Grow and flip one Wolff cluster. Model-specific behavior is supplied by
`cluster_bond_probability` and `apply_cluster!`, which keeps the cluster
traversal reusable for models with different local degrees of freedom. All
random numbers (cluster context, seed site, bond activation) come from `rng`.

Growth stops as soon as the cluster attaches to a ghost site (see
[`ghost_site`](@ref)): ghost-attached clusters are never flipped, so growing
further would be wasted work.

The cluster mask and member/frontier lists are taken from `workspace` (a
[`ClusterWorkspace`](@ref)) when given and allocated fresh otherwise; drivers
allocate one workspace per run via [`algorithm_workspace`](@ref).
"""
function step!(
    ::WolffAlgorithm,
    model::AbstractModel,
    geometry,
    state::AbstractState,
    T::Real;
    rng::AbstractRNG = Random.default_rng(),
    workspace::Union{Nothing, ClusterWorkspace} = nothing
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))
    validate_cluster_model(model, geometry, state, T)
    context = cluster_context(rng, model, geometry, state, T)

    seed = cluster_seed(rng, model, geometry, state)
    1 <= seed <= geometry.n_sites || throw(ArgumentError(
        "cluster_seed returned site $seed outside 1:$(geometry.n_sites)"
    ))

    ws = workspace === nothing ? ClusterWorkspace() : workspace
    in_cluster = _resize_buffer!(ws.in_cluster, geometry.n_sites)
    fill!(in_cluster, false)
    cluster = empty!(ws.cluster)
    frontier = empty!(ws.frontier)
    in_cluster[seed] = true
    push!(cluster, seed)
    push!(frontier, seed)
    ghost_count = cluster_ghost_count(model, geometry, state, context)
    any_ghost_attached = false

    while !isempty(frontier)
        # Ghost-attached clusters are never flipped, so stop growing at the
        # first ghost attachment; the remaining work would be discarded.
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

    apply_cluster!(model, geometry, state, cluster, context, in_cluster)
    return WolffUpdateResult(length(cluster))
end