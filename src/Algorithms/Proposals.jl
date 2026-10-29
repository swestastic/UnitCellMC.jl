"""
    RandomProposal

Site-selection scheme for local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)): every
[`step!`](@ref) update draws a site uniformly at random from `rng`. This is
the default scheme.
"""
struct RandomProposal <: AbstractProposalScheme end

"""
    SequentialProposal

Site-selection scheme for local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)): [`step!`](@ref)
updates visit the sites in site-index order, so one sweep
([`steps_per_sweep`](@ref) updates) attempts every site exactly once instead
of `n_sites` random draws.

`LatticeUtilities.jl` numbers sites in column-major unit-cell order (the
first lattice direction varies fastest), so consecutive updates target
neighboring sites — on a square lattice the sweep proceeds column by column
(equivalently row by row, up to transposition).

Each single-site update still satisfies detailed balance, so the completed
sweep leaves the Boltzmann distribution invariant; only the visiting order
differs from [`RandomProposal`](@ref).

The cursor is mutable per-run state held in a
[`SequentialProposalWorkspace`](@ref), built once per run by
[`algorithm_workspace`](@ref) and threaded into [`step!`](@ref) by the
drivers. Calling [`step!`](@ref) or [`propose`](@ref) directly without a
`workspace` throws an `ArgumentError`, since a fresh cursor would restart at
site 1 on every call.
"""
struct SequentialProposal <: AbstractProposalScheme end

"""
    CheckerboardProposal

Site-selection scheme for local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)) on bipartite
lattices: the sites are 2-colored so that no bond connects same-color sites
(see [`checkerboard_colors`](@ref)), and each sweep first updates all sites
of one color, then all sites of the other. Within a color, sites are visited
in site-index order. For lattices that need more than two colors see
[`ColoredProposal`](@ref).

Same-color sites share no bond, so the updates within each half-sweep do not
interact directly — on a square lattice this is the classic checkerboard
decomposition, where one color holds the even-parity sites
`mod(x + y, 2) == 0` and the other the odd-parity ones. Each single-site
update still satisfies detailed balance, so the completed sweep leaves the
Boltzmann distribution invariant.

The visiting order and cursor are per-run state held in a
[`ColoredProposalWorkspace`](@ref), built once per run by
[`algorithm_workspace`](@ref) and threaded into [`step!`](@ref) by the
drivers. Calling [`step!`](@ref) or [`propose`](@ref) directly without a
`workspace` throws an `ArgumentError`, as does any non-bipartite lattice
(e.g. triangular) when the workspace is built.
"""
struct CheckerboardProposal <: AbstractProposalScheme end

"""
    ColoredProposal(k)

Site-selection scheme for local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)): the sites are
colored with at most `k` colors so that no bond connects same-color sites
(see [`color_lattice`](@ref)), and each sweep updates all sites of one color
class, then the next — e.g. `ColoredProposal(3)` for the tripartite
triangular or kagome lattices. Within a color class, sites are visited in
site-index order.

Same-color sites share no bond, so the updates within each color block do not
interact directly, and each single-site update satisfies detailed balance, so
the completed sweep leaves the Boltzmann distribution invariant.

The coloring is a saturation-first greedy (DSATUR) with randomized
restarts; see [`color_lattice`](@ref). Exact coloring is NP-hard in general,
so if no attempt manages with `k` colors an `ArgumentError` is thrown when
the workspace is built — even where a `k`-coloring exists. DSATUR is exact
on bipartite graphs, but for `k = 2` prefer [`CheckerboardProposal`](@ref),
whose coloring is cheaper and needs no restarts.

The visiting order and cursor are per-run state held in a
[`ColoredProposalWorkspace`](@ref), built once per run by
[`algorithm_workspace`](@ref) and threaded into [`step!`](@ref) by the
drivers. Calling [`step!`](@ref) or [`propose`](@ref) directly without a
`workspace` throws an `ArgumentError`.

# Throws

`ArgumentError` if `k < 2`.
"""
struct ColoredProposal <: AbstractProposalScheme
    ncolors::Int

    function ColoredProposal(k::Integer)
        k >= 2 || throw(ArgumentError("ColoredProposal needs at least 2 colors, got $k"))
        return new(Int(k))
    end
end

"""
    SequentialProposalWorkspace

Per-run cursor for the [`SequentialProposal`](@ref) scheme: `next` is the
site the next [`step!`](@ref) update visits. Built by
[`proposal_workspace`](@ref) via [`algorithm_workspace`](@ref). Like
[`ClusterWorkspace`](@ref), it carries mutable state between steps and must
not be shared between concurrent runs.
"""
mutable struct SequentialProposalWorkspace
    next::Int
end

"""
    ColoredProposalWorkspace

Per-run state for the [`CheckerboardProposal`](@ref) and
[`ColoredProposal`](@ref) schemes. Built by [`proposal_workspace`](@ref) via
[`algorithm_workspace`](@ref). Like [`ClusterWorkspace`](@ref), it carries
mutable state between steps and must not be shared between concurrent runs.

# Fields

- `order::Vector{Int}`: The visiting order — each color class in turn, in
  site-index order within a class (see [`checkerboard_colors`](@ref) and
  [`color_lattice`](@ref)).
- `next::Int`: Cursor into `order`; the next [`step!`](@ref) update visits
  `order[next]`.
"""
mutable struct ColoredProposalWorkspace
    order::Vector{Int}
    next::Int
end

"""
    checkerboard_colors(geometry)

2-color the bond graph of `geometry`: return a `Vector{Int}` with one entry
per site, 0 or 1, such that the two endpoints of every bond have opposite
colors. On a square lattice this is the parity `mod(x + y, 2)` of the unit
cell coordinates. The neighbor table lists each bond from both endpoints, so
every bond constrains the coloring exactly once per direction.

# Throws

`ArgumentError` if the bond graph is not bipartite — a bond whose endpoints
the coloring forces to equal colors (e.g. the triangles of a triangular
lattice), or a bond from a site to itself.
"""
function checkerboard_colors(geometry::Geometry)
    colors = fill(-1, geometry.n_sites)
    stack = Int[]
    for root in 1:geometry.n_sites
        colors[root] >= 0 && continue
        colors[root] = 0
        empty!(stack)
        push!(stack, root)
        while !isempty(stack)
            site = pop!(stack)
            for neighbor in geometry.neighbor_table[site].neighbors
                if neighbor == site || colors[neighbor] == colors[site]
                    throw(ArgumentError(
                        "Geometry is not bipartite (bond $(site) ↔ $(neighbor) " *
                        "cannot be 2-colored); CheckerboardProposal requires a " *
                        "bipartite lattice"
                    ))
                elseif colors[neighbor] < 0
                    colors[neighbor] = 1 - colors[site]
                    push!(stack, neighbor)
                end
            end
        end
    end
    return colors
end

"""
    color_lattice(geometry, k)

Color the bond graph of `geometry` with at most `k` colors: return a
`Vector{Int}` with one entry per site in `1:k` such that the two endpoints of
every bond have different colors.

The coloring is a saturation-first greedy (DSATUR): each step colors the
uncolored site adjacent to the most distinct colors, breaking ties by degree.
Fewer than `k` colors may be used. DSATUR is exact on bipartite graphs and
handles the usual regular lattices (e.g. the tripartite triangular and kagome
lattices with `k = 3`), but exact coloring is NP-hard in general. The first
pass is the classical deterministic algorithm (ties by degree, then site
index; smallest available color); if a site gets stuck, up to 15 further
passes randomize both the tie-breaking and the choice among available colors.
The restarts draw from a fixed internal random stream, so the resulting
coloring is identical between runs and independent of the simulation `rng`.
Even so, success is not guaranteed where a `k`-coloring exists; for `k = 2`
use [`checkerboard_colors`](@ref), which is cheaper and needs no restarts.

# Throws

`ArgumentError` if `k < 2`, if a site is bonded to itself, or if no attempt
manages with `k` colors.
"""
function color_lattice(geometry::Geometry, k::Integer)
    k >= 2 || throw(ArgumentError("need at least 2 colors, got $k"))
    for site in 1:geometry.n_sites
        site in geometry.neighbor_table[site].neighbors && throw(ArgumentError(
            "Site $site is bonded to itself, so the geometry cannot be colored"))
    end
    degree = map(info -> length(info.neighbors), geometry.neighbor_table)
    rng = Xoshiro(0xD5A7)  # fixed stream: restarts are deterministic run-to-run
    for attempt in 1:16
        colors = _dsatur_colors(geometry, k, degree, rng; randomize = attempt > 1)
        colors === nothing || return colors
    end
    throw(ArgumentError(
        "color_lattice could not color the geometry with $k colors in 16 " *
        "attempts; the lattice may not be $k-colorable, or the heuristic missed it"))
end

# One DSATUR pass: per-site colors in 1:k, or `nothing` if a site ended up
# with neighbors in every color. With `randomize = false` this is the
# classical deterministic algorithm (ties by degree, then site index;
# smallest available color). With `randomize = true`, ties are broken
# uniformly (reservoir sampling) and the color is drawn uniformly from the
# available ones — a systematic smallest-color bias can otherwise starve
# rigid k-chromatic graphs (e.g. the 4-colorable 4×4 triangular torus).
function _dsatur_colors(
    geometry::Geometry, k::Int, degree::Vector{Int}, rng::AbstractRNG;
    randomize::Bool
)
    n = geometry.n_sites
    colors = zeros(Int, n)                      # 0 = uncolored
    adjacent = [falses(k) for _ in 1:n]         # colors adjacent to each site
    saturation = zeros(Int, n)                  # distinct adjacent colors

    for _ in 1:n
        # most constrained uncolored site
        site = 0
        best_key = (-1, -1)
        ties = 0
        for candidate in 1:n
            colors[candidate] == 0 || continue
            key = (saturation[candidate], degree[candidate])
            if key > best_key
                site, best_key, ties = candidate, key, 1
            elseif key == best_key
                if randomize
                    ties += 1
                    rand(rng) < inv(ties) && (site = candidate)
                else
                    candidate < site && (site = candidate)
                end
            end
        end

        available = findall(==(false), adjacent[site])
        isempty(available) && return nothing
        colors[site] = randomize ? rand(rng, available) : first(available)

        for neighbor in geometry.neighbor_table[site].neighbors
            if colors[neighbor] == 0 && !adjacent[neighbor][colors[site]]
                adjacent[neighbor][colors[site]] = true
                saturation[neighbor] += 1
            end
        end
    end
    return colors
end

# Visiting order shared by the colored schemes: each color class in turn,
# sites in site-index order within a class.
_color_blocked_order(colors, classes) =
    [site for class in classes for site in eachindex(colors) if colors[site] == class]

"""
    propose_site(rng, scheme, geometry, workspace)

Return the site the next local [`step!`](@ref) update visits under `scheme`.
[`RandomProposal`](@ref) draws uniformly from `rng` and ignores `workspace`;
[`SequentialProposal`](@ref), [`CheckerboardProposal`](@ref), and
[`ColoredProposal`](@ref) return the cursor of their workspace and advance
it, wrapping after `geometry.n_sites`.

# Throws

`ArgumentError` if a stateful scheme is called without its workspace.
"""
propose_site(rng::AbstractRNG, ::RandomProposal, geometry::Geometry, workspace) =
    rand(rng, 1:geometry.n_sites)

function propose_site(
    ::AbstractRNG,
    ::SequentialProposal,
    geometry::Geometry,
    workspace::SequentialProposalWorkspace
)
    site = workspace.next
    workspace.next = mod1(site + 1, geometry.n_sites)
    return site
end

function propose_site(
    ::AbstractRNG,
    ::Union{CheckerboardProposal, ColoredProposal},
    geometry::Geometry,
    workspace::ColoredProposalWorkspace
)
    site = workspace.order[workspace.next]
    workspace.next = mod1(workspace.next + 1, geometry.n_sites)
    return site
end

propose_site(::AbstractRNG, scheme::SequentialProposal, ::Geometry, ::Nothing) =
    _proposal_workspace_missing(scheme)
propose_site(::AbstractRNG, scheme::Union{CheckerboardProposal, ColoredProposal}, ::Geometry, ::Nothing) =
    _proposal_workspace_missing(scheme)

function _proposal_workspace_missing(scheme::AbstractProposalScheme)
    throw(ArgumentError(
        "$(nameof(typeof(scheme))) requires its workspace: build one with " *
        "`algorithm_workspace(algorithm, model, geometry, state)` and pass it via " *
        "the `workspace` keyword of `step!`/`propose` (the drivers " *
        "`run_simulation`/`MC_Sweep!` do this automatically)"
    ))
end

"""
    proposal_workspace(scheme, geometry)

Return the per-run site-selection state for `scheme`: `nothing` for
stateless schemes ([`RandomProposal`](@ref)), a
[`SequentialProposalWorkspace`](@ref) for [`SequentialProposal`](@ref), a
[`ColoredProposalWorkspace`](@ref) for [`CheckerboardProposal`](@ref) and
[`ColoredProposal`](@ref). [`algorithm_workspace`](@ref) forwards to this for
local-update algorithms, and [`run_simulation`](@ref) threads the result
through [`MC_Sweep!`](@ref) into [`step!`](@ref). Extend it for custom
schemes with per-run state.

# Throws

`ArgumentError` from [`CheckerboardProposal`](@ref) if `geometry` is not
bipartite (see [`checkerboard_colors`](@ref)), or from
[`ColoredProposal`](@ref) if [`color_lattice`](@ref) cannot color it with the
requested number of colors.
"""
proposal_workspace(::RandomProposal, geometry) = nothing
proposal_workspace(::SequentialProposal, geometry) = SequentialProposalWorkspace(1)
proposal_workspace(::CheckerboardProposal, geometry::Geometry) =
    ColoredProposalWorkspace(_color_blocked_order(checkerboard_colors(geometry), 0:1), 1)
proposal_workspace(scheme::ColoredProposal, geometry::Geometry) = ColoredProposalWorkspace(
    _color_blocked_order(color_lattice(geometry, scheme.ncolors), 1:scheme.ncolors), 1
)
