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
number of sites updated, and the fraction of sites contained in the largest
cluster.
"""
struct SwendsenWangUpdateResult <: AbstractUpdateResult
    cluster_count::Int
    updated_sites::Int
    largest_cluster_fraction::Float64
end

function apply_clusters!(::AbstractModel, ::Any, ::AbstractState, ::Vector{Vector{Int}})
    throw(MethodError(apply_clusters!, ()))
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
    step!(SwendsenWangAlgorithm(), model, geometry, state, T)

Activate each undirected bond once, form connected components with union-find,
and pass the completed clusters to `apply_clusters!`. The bond traversal is
model-independent and the model controls both bond activation and mutation.
"""
function step!(
    ::SwendsenWangAlgorithm,
    model::AbstractModel,
    geometry,
    state::AbstractState,
    T::Real
)
    T > 0 || throw(ArgumentError("Temperature must be positive, got $T"))
    validate_cluster_model(model, geometry, state, T)
    context = cluster_context(model, geometry, state, T)

    parents = collect(1:geometry.n_sites)
    ranks = zeros(UInt8, geometry.n_sites)

    for site in 1:geometry.n_sites
        info = geometry.neighbor_table[site]
        for (bond_id, neighbor) in zip(info.bonds, info.neighbors)
            neighbor > site || continue

            probability = cluster_bond_probability(
                model, geometry, state, site, neighbor, bond_id, T, context
            )
            0 <= probability <= 1 || throw(ArgumentError(
                "cluster_bond_probability returned $probability; expected a value in [0, 1]"
            ))

            probability > 0 && rand() < probability &&
                _sw_union!(parents, ranks, site, neighbor)
        end
    end

    clusters_by_root = Dict{Int,Vector{Int}}()
    for site in 1:geometry.n_sites
        root = _sw_find!(parents, site)
        push!(get!(clusters_by_root, root, Int[]), site)
    end
    clusters = collect(values(clusters_by_root))
    largest_cluster_size = isempty(clusters) ? 0 : maximum(length, clusters)
    largest_cluster_fraction = geometry.n_sites > 0 ? largest_cluster_size / geometry.n_sites : 0.0

    apply_clusters!(model, geometry, state, clusters, context)
    return SwendsenWangUpdateResult(length(clusters), geometry.n_sites, largest_cluster_fraction)
end