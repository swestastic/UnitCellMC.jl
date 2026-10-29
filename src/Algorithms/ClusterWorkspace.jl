"""
    ClusterWorkspace

Reusable scratch buffers for the cluster algorithms ([`WolffAlgorithm`](@ref)
and [`SwendsenWangAlgorithm`](@ref)).

Cluster updates need several lattice-sized arrays per step (cluster masks,
member lists, union-find arrays). A `ClusterWorkspace` is allocated once per
run by [`run_simulation`](@ref) and threaded through [`MC_Sweep!`](@ref) into
[`step!`](@ref), so repeated steps reuse the same buffers instead of
reallocating them. Calling [`step!`](@ref) without a workspace allocates one
per call.

The buffers are grown lazily on first use and reset at the start of every
step. A workspace carries mutable state between steps, so it must not be
shared between concurrent runs.

# Fields

- `in_cluster::BitVector`: Wolff cluster-membership mask.
- `cluster::Vector{Int}`: Wolff cluster member list.
- `frontier::Vector{Int}`: Wolff breadth-first stack.
- `parents::Vector{Int}`: Swendsen-Wang union-find parent of each site, then
  each ghost site.
- `ranks::Vector{UInt8}`: Swendsen-Wang union-find ranks.
- `labels::Vector{Int}`: Swendsen-Wang union-find root ("cluster label") of
  each site, then each ghost site, after the bond-activation pass.
- `roots::Vector{Int}`: The distinct labels containing at least one lattice
  site, in first-encounter order.
- `sizes::Vector{Int}`: Per-root lattice-site counts.
- `flip::Vector{Bool}`: Per-root cluster orientation decision, drawn by
  [`draw_cluster_flips!`](@ref).
- `seen::BitVector`: First-encounter markers used to build `roots`.
"""
mutable struct ClusterWorkspace
    in_cluster::BitVector
    cluster::Vector{Int}
    frontier::Vector{Int}
    parents::Vector{Int}
    ranks::Vector{UInt8}
    labels::Vector{Int}
    roots::Vector{Int}
    sizes::Vector{Int}
    flip::Vector{Bool}
    seen::BitVector
end

ClusterWorkspace() = ClusterWorkspace(
    BitVector(), Int[], Int[], Int[], UInt8[], Int[], Int[], Int[], Bool[], BitVector[]
)

"""
    algorithm_workspace(algorithm, model, geometry, state)

Return the per-run scratch space that [`step!`](@ref) reuses for `algorithm`:
a [`ClusterWorkspace`](@ref) for cluster algorithms, `nothing` otherwise.
[`run_simulation`](@ref) allocates it once and passes it through
[`MC_Sweep!`](@ref). Extend this for custom algorithms with per-step
allocations.
"""
algorithm_workspace(::AbstractAlgorithm, model, geometry, state) = nothing

"""
    ghost_site(geometry, ghost)

Return the index of ghost site `ghost` in the cluster algorithms'
site-indexed arrays. Ghosts (used to attach external fields to the cluster
updates) follow the lattice sites, so ghost `ghost` lives at index
`geometry.n_sites + ghost`. Clusters attached to a ghost are never flipped
(see [`draw_cluster_flips!`](@ref)).
"""
ghost_site(geometry, ghost::Int) = geometry.n_sites + ghost

"""
    draw_cluster_flips!(rng, workspace, geometry)

Draw the per-cluster orientation decisions of a Swendsen-Wang update — one
coin flip per cluster root, in the first-encounter order of
`workspace.roots` — recording them per label in `workspace.flip`, then
freeze every ghost-attached cluster by setting its `flip` entry to `false`
(ghost labels are those of the [`ghost_site`](@ref) indices), since
ghost-attached clusters are never flipped.

Each model's [`apply_clusters!`](@ref) calls this before mutating the state,
so the ghost convention lives in one place; models supply only the per-site
mutation.

# Returns

`workspace.flip`.
"""
function draw_cluster_flips!(rng::AbstractRNG, workspace::ClusterWorkspace, geometry)
    labels = workspace.labels
    flip = workspace.flip
    for root in workspace.roots
        flip[root] = rand(rng, Bool)
    end
    for ghost in 1:(length(labels) - geometry.n_sites)
        flip[labels[ghost_site(geometry, ghost)]] = false
    end
    return flip
end

# Resize a position-indexed buffer to length `n` (a no-op when it already has
# that length) and return it. List-like buffers are instead `empty!`ed, which
# keeps their capacity.
function _resize_buffer!(buffer::AbstractVector, n::Int)
    length(buffer) == n || resize!(buffer, n)
    return buffer
end
