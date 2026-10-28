"""
    SwendsenWangAlgorithm

Marker type selecting a Swendsen-Wang cluster update. Bond activation is
delegated to `cluster_bond_probability`, and model-specific cluster mutations
are delegated to `apply_clusters!`.
"""
struct SwendsenWangAlgorithm <: AbstractAlgorithm end

"""
    SwendsenWangUpdateResult

Result of one Swendsen-Wang update, recording the number of clusters, the
total number of lattice sites processed (`updated_sites`; always equal to
`geometry.n_sites`, since every site belongs to exactly one cluster), and
the fraction of sites contained in the largest cluster.
"""
struct SwendsenWangUpdateResult <: AbstractUpdateResult
    cluster_count::Int
    updated_sites::Int
    largest_cluster_fraction::Float64
end

function apply_clusters!(::AbstractRNG, ::AbstractModel, ::Any, ::AbstractState, ::ClusterWorkspace)
    throw(MethodError(apply_clusters!, ()))
end

# Translate an explicit partition of sites into root-id labels (0 marks sites
# absent from the partition), for the `clusters::Vector{Vector{Int}}` forms of
# `apply_clusters!`.
function _cluster_labels(geometry, clusters::Vector{Vector{Int}})
    labels = zeros(Int, geometry.n_sites)
    for (c, cluster) in enumerate(clusters), site in cluster
        site <= geometry.n_sites && (labels[site] = c)
    end
    return labels
end

function _sw_find!(parents::Vector{Int}, site::Int)
    root = site
    while parents[root] != root
        root = parents[root]
    end
    while parents[site] != site
        next_site = parents[site]
        parents[site] = root
        site = next_site
    end
    return root
end

function _sw_union!(parents::Vector{Int}, ranks::Vector{UInt8}, first::Int, second::Int)
    first_root = _sw_find!(parents, first)
    second_root = _sw_find!(parents, second)
    first_root == second_root && return nothing

    if ranks[first_root] < ranks[second_root]
        parents[first_root] = second_root
    elseif ranks[first_root] > ranks[second_root]
        parents[second_root] = first_root
    else
        parents[second_root] = first_root
        ranks[first_root] += UInt8(1)
    end
    return nothing
end

"""
    step!(SwendsenWangAlgorithm(), model, geometry, state, T;
        rng = Random.default_rng(), workspace = nothing)

Activate each undirected bond once, form connected components with union-find,
and pass the completed clusters to `apply_clusters!`. The bond traversal is
model-independent and the model controls both bond activation and mutation.
All random numbers (cluster context, bond activation, cluster flips) are
drawn from `rng`.

Clusters are labelled by union-find root id: after the bond pass,
`workspace.labels[i]` is the label of site (or ghost) `i` and
`workspace.roots` lists the labels that contain at least one lattice site.
The union-find arrays and label buffers come from `workspace` (a
[`ClusterWorkspace`](@ref)) when given and are allocated fresh otherwise;
drivers allocate one workspace per run via [`algorithm_workspace`](@ref).
"""
function step!(
    ::SwendsenWangAlgorithm,
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

    ghost_count = cluster_ghost_count(model, geometry, state, context)
    total_sites = geometry.n_sites + ghost_count
    ws = workspace === nothing ? ClusterWorkspace() : workspace
    parents = _resize_buffer!(ws.parents, total_sites)
    ranks = _resize_buffer!(ws.ranks, total_sites)
    parents .= 1:total_sites
    fill!(ranks, zero(UInt8))

    for site in 1:geometry.n_sites
        info = geometry.neighbor_table[site]
        for (bond_template, neighbor) in zip(info.templates, info.neighbors)
            neighbor > site || continue

            probability = cluster_bond_probability(
                model, geometry, state, site, neighbor, bond_template, T, context
            )
            0 <= probability <= 1 || throw(ArgumentError(
                "cluster_bond_probability returned $probability; expected a value in [0, 1]"
            ))

            probability > 0 && rand(rng) < probability &&
                _sw_union!(parents, ranks, site, neighbor)
        end

        for ghost in 1:ghost_count
            probability = cluster_ghost_probability(
                model, geometry, state, site, ghost, T, context
            )
            0 <= probability <= 1 || throw(ArgumentError(
                "cluster_ghost_probability returned $probability; expected a value in [0, 1]"
            ))

            probability > 0 && rand(rng) < probability && _sw_union!(
                parents, ranks, site, geometry.n_sites + ghost
            )
        end
    end

    # Label every site by its union-find root and collect the roots that
    # contain at least one lattice site, in first-encounter order. Ghosts are
    # labelled too, so models can recognize ghost-attached clusters by a label
    # above n_sites.
    labels = _resize_buffer!(ws.labels, total_sites)
    sizes = _resize_buffer!(ws.sizes, total_sites)
    fill!(sizes, 0)
    seen = _resize_buffer!(ws.seen, total_sites)
    fill!(seen, false)
    _resize_buffer!(ws.flip, total_sites)
    roots = empty!(ws.roots)

    cluster_count = 0
    largest_cluster_size = 0
    for site in 1:geometry.n_sites
        root = _sw_find!(parents, site)
        labels[site] = root
        sizes[root] += 1
        largest_cluster_size = max(largest_cluster_size, sizes[root])
        seen[root] && continue
        seen[root] = true
        push!(roots, root)
        cluster_count += 1
    end
    for ghost in 1:ghost_count
        labels[geometry.n_sites + ghost] = _sw_find!(parents, geometry.n_sites + ghost)
    end
    largest_cluster_fraction = geometry.n_sites > 0 ? largest_cluster_size / geometry.n_sites : 0.0

    apply_clusters!(rng, model, geometry, state, ws, context)
    return SwendsenWangUpdateResult(cluster_count, geometry.n_sites, largest_cluster_fraction)
end

algorithm_workspace(::SwendsenWangAlgorithm, model, geometry, state) = ClusterWorkspace()