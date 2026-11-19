"""
    AlgorithmCycle(entries...)
    AlgorithmCycle(entries::Union{Tuple, AbstractVector})

A composite update algorithm running several algorithms in sequence: one
sweep of an `AlgorithmCycle` is one sweep of each entry, in order. Every
sweep driver — [`MC_Sweep!`](@ref), [`run_simulation`](@ref),
[`sweep_Ts`](@ref), [`tempering_sweep`](@ref), [`autotune_binning`](@ref) —
accepts an `AlgorithmCycle` anywhere a single algorithm goes, so updates can
be mixed: cluster sweeps interleaved with local ones, or a thermalizing
algorithm alternated with overrelaxation.

Each entry is an algorithm, or an `algorithm => n_sweeps` pair running
`n_sweeps` consecutive sweeps of that algorithm per cycle:

```julia
# One Wolff cluster update followed by one Metropolis sweep per cycle
AlgorithmCycle(WolffAlgorithm(), MetropolisAlgorithm())

# The classic overrelaxation recipe: one heatbath sweep, then three
# energy-conserving reflection sweeps
AlgorithmCycle(HeatbathAlgorithm(), OverrelaxationAlgorithm() => 3)
```

Every entry's sweep leaves the canonical distribution invariant (see
[`step!`](@ref)), so the composed cycle does too. Ergodicity is inherited
from the entries: include at least one ergodic, thermalizing algorithm —
[`OverrelaxationAlgorithm`](@ref) alone is not ergodic, so a cycle of only
overrelaxation entries samples a constant-energy shell.

Each entry keeps its own sweep semantics and its own per-run workspace: a
cycle sweep runs [`steps_per_sweep`](@ref)`(entry, geometry)` [`step!`](@ref)
updates of each entry (repeated by its count) through that entry's own
[`sweep_steps!`](@ref), so a fused fast path applies to its entry inside a
cycle exactly as it does standalone. The cycle's
[`shared_workspace`](@ref)/[`algorithm_workspace`](@ref) are tuples with one
entry per algorithm.

# Diagnostics

Diagnostic names are namespaced per entry to keep them unique:
[`available_diagnostics`](@ref)`(cycle)` prefixes each entry's diagnostics
with [`diagnostic_prefix`](@ref)`(entry)` and an underscore — e.g.
`[:wolff_cluster_size, :metropolis_acceptance_ratio]` for
`AlgorithmCycle(WolffAlgorithm(), MetropolisAlgorithm())`. Request the
prefixed names via the `diagnostics` argument of
[`MeasurementContainer`](@ref). Each reported value is the sweep average
over that entry's own updates only. Entries whose prefixed names would
collide — two entries of the same algorithm type — are rejected at
construction; combine repeats of one algorithm into a single
`algorithm => n_sweeps` entry instead.

# Fields

- `algorithms::Tuple{Vararg{AbstractAlgorithm}}`: The entries, in sweep
  order.
- `counts::Tuple{Vararg{Int}}`: Sweeps of each entry per cycle.

# Throws

`ArgumentError` if no entries are given, if an entry is neither an algorithm
nor an `algorithm => n_sweeps` pair, if a count is less than 1, or if two
entries produce the same prefixed diagnostic name.
"""
struct AlgorithmCycle{A<:Tuple{Vararg{AbstractAlgorithm}}, C<:Tuple{Vararg{Int}}} <: AbstractAlgorithm
    algorithms::A
    counts::C

    function AlgorithmCycle(algorithms::Tuple{Vararg{AbstractAlgorithm}},
                            counts::Tuple{Vararg{Int}})
        length(algorithms) == length(counts) || throw(ArgumentError(
            "algorithms ($(length(algorithms))) and counts ($(length(counts))) " *
            "must have the same length"))
        isempty(algorithms) && throw(ArgumentError(
            "AlgorithmCycle needs at least one algorithm"))
        all(>=(1), counts) || throw(ArgumentError(
            "Each algorithm must run at least one sweep per cycle, got counts = $counts"))
        _validate_cycle_diagnostics(algorithms)
        return new{typeof(algorithms), typeof(counts)}(algorithms, counts)
    end
end

# One entry of the varargs constructor, normalized to an
# `(algorithm, sweep count)` pair.
_cycle_entry(algorithm::AbstractAlgorithm) = (algorithm, 1)
_cycle_entry(entry::Pair{<:AbstractAlgorithm, <:Integer}) = (entry.first, Int(entry.second))
_cycle_entry(entry) = throw(ArgumentError(
    "AlgorithmCycle entries must be algorithms or `algorithm => n_sweeps` " *
    "pairs, got $(repr(entry))"))

function AlgorithmCycle(entries...)
    pairs = map(_cycle_entry, entries)
    return AlgorithmCycle(map(first, pairs), map(last, pairs))
end

AlgorithmCycle(entries::Union{Tuple, AbstractVector}) = AlgorithmCycle(entries...)

function Base.show(io::IO, cycle::AlgorithmCycle)
    entries = map(zip(cycle.algorithms, cycle.counts)) do (algorithm, count)
        count == 1 ? string(algorithm) : string(algorithm, " => ", count)
    end
    return print(io, "AlgorithmCycle(", join(entries, ", "), ")")
end

"""
    diagnostic_prefix(algorithm)

Return the `Symbol` prepended to each of `algorithm`'s diagnostic names when
the algorithm runs as an entry of an [`AlgorithmCycle`](@ref): the
algorithm's type name with a trailing `"Algorithm"` removed, converted to
`snake_case` — `:metropolis` for [`MetropolisAlgorithm`](@ref),
`:swendsen_wang` for [`SwendsenWangAlgorithm`](@ref). Override it for a
custom algorithm whose default prefix would collide with another entry's.
"""
function diagnostic_prefix(algorithm::AbstractAlgorithm)
    name = string(nameof(typeof(algorithm)))
    endswith(name, "Algorithm") && (name = chop(name; tail = length("Algorithm")))
    name = replace(name, r"([a-z0-9])([A-Z])" => s"\1_\2")
    return Symbol(lowercase(name))
end

# Construction-time uniqueness check of the prefixed diagnostic names: two
# entries of the same algorithm type share a `diagnostic_prefix`, so their
# diagnostics would collide. Catching this here keeps the per-sweep
# partitioning of `_cycle_diagnostic_groups` unambiguous.
function _validate_cycle_diagnostics(algorithms)
    seen = Set{Symbol}()
    for algorithm in algorithms
        prefix = diagnostic_prefix(algorithm)
        for name in available_diagnostics(algorithm)
            prefixed = Symbol(prefix, :_, name)
            prefixed in seen && throw(ArgumentError(
                "AlgorithmCycle entries produce the duplicate diagnostic name " *
                ":$prefixed — the diagnostic prefixes of two entries coincide " *
                "(:$prefix); combine repeats of one algorithm into a single " *
                "`algorithm => n_sweeps` entry, or give one entry's type a " *
                "distinct `diagnostic_prefix`"))
            push!(seen, prefixed)
        end
    end
    return nothing
end

"""
    available_diagnostics(cycle::AlgorithmCycle)

Return the diagnostic names `cycle` provides: each entry's
[`available_diagnostics`](@ref) prefixed by [`diagnostic_prefix`](@ref) and
an underscore, in entry order — e.g. `[:wolff_cluster_size,
:metropolis_acceptance_ratio]` for `AlgorithmCycle(WolffAlgorithm(),
MetropolisAlgorithm())`. See the type-level list for the bare names of each
algorithm.
"""
function available_diagnostics(cycle::AlgorithmCycle)
    return [Symbol(diagnostic_prefix(algorithm), :_, name)
            for algorithm in cycle.algorithms
            for name in available_diagnostics(algorithm)]
end

# A cycle sweep is each entry's sweep in turn, repeated by the entry's count.
steps_per_sweep(cycle::AlgorithmCycle, geometry::Geometry) =
    sum(count * steps_per_sweep(algorithm, geometry)
        for (algorithm, count) in zip(cycle.algorithms, cycle.counts))

# The shared/per-run workspaces of a cycle are tuples with one entry per
# algorithm, so every entry's scratch space is built — and shared across
# runs — exactly as it would be standalone.
shared_workspace(cycle::AlgorithmCycle, model, geometry) =
    map(algorithm -> shared_workspace(algorithm, model, geometry), cycle.algorithms)

function algorithm_workspace(cycle::AlgorithmCycle, model, geometry, state,
                             shared = shared_workspace(cycle, model, geometry))
    return map(cycle.algorithms, shared) do algorithm, entry_shared
        algorithm_workspace(algorithm, model, geometry, state, entry_shared)
    end
end

# Group the flat, prefixed diagnostic names `D` requested from a cycle into
# per-entry `(indices, bare names)` pairs: `indices` locates the entry's
# statistics within the run's flat statistic tuple, `bare` holds the
# unprefixed names its `diagnostic_values` returns. The prefixes are
# distinct (checked at construction), so every name belongs to exactly one
# entry.
function _cycle_diagnostic_groups(cycle::AlgorithmCycle, D)
    return map(cycle.algorithms) do algorithm
        prefix = diagnostic_prefix(algorithm)
        bare = Tuple(name for name in available_diagnostics(algorithm)
                     if Symbol(prefix, :_, name) in D)
        indices = Tuple(findfirst(==(Symbol(prefix, :_, name)), D) for name in bare)
        (indices, bare)
    end
end

"""
    sweep_steps!(cycle::AlgorithmCycle, model, geometry, state, T, stats, diagnostics;
        rng, workspace)

Run one sweep of an [`AlgorithmCycle`](@ref): for each entry in turn, its
own [`sweep_steps!`](@ref) — and therefore its own fused fast path, if it
has one — repeated by the entry's sweep count, drawing from the shared `rng`
in sweep order.

`stats` holds one [`UpdateStatistic`](@ref) per *prefixed* diagnostic name
(the cycle's [`available_diagnostics`](@ref)); each entry's sub-sweep
accumulates only its own diagnostics, under their bare names, so a
diagnostic's sweep average covers that entry's updates only. `workspace` is
the per-entry tuple built by [`algorithm_workspace`](@ref), or `nothing` to
let every entry's [`step!`](@ref) allocate its own.
"""
function sweep_steps!(
    cycle::AlgorithmCycle, model, geometry, state, T, stats::Tuple, ::Val{D};
    rng::AbstractRNG, workspace
) where {D}
    groups = _cycle_diagnostic_groups(cycle, D)
    for i in eachindex(cycle.algorithms)
        indices, bare = groups[i]
        sub_stats = ntuple(j -> stats[indices[j]], length(indices))
        sub_workspace = workspace === nothing ? nothing : workspace[i]
        for _ in 1:cycle.counts[i]
            sweep_steps!(cycle.algorithms[i], model, geometry, state, T,
                         sub_stats, Val(bare); rng = rng, workspace = sub_workspace)
        end
    end
    return stats
end
