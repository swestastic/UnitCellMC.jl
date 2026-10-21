"""
    GLOBAL_SEED

Seed most recently passed to [`set_global_seed!`](@ref), or `nothing` if the
global RNG has not been seeded through it in this session.
"""
const GLOBAL_SEED = Ref{Union{Nothing, Integer}}(nothing)

"""
    set_global_seed!(seed = rand(Random.RandomDevice(), UInt32))

Seed Julia's global (task-local) RNG with `seed` via `Random.seed!` and
record it, so that the result writers ([`save_results`](@ref),
[`save_sweep_results`](@ref), and therefore [`sweep_Ts`](@ref)) store it in
`metadata.txt`. Call it once near the start of a script: every random number
drawn afterwards (initial states, autotuning pilots, and all parallel runs of
`sweep_Ts`) then follows from `seed`, and rerunning the script with the same
seed, inputs, and Julia version reproduces the results exactly, for any
number of threads.

Without an argument a fresh seed is drawn from the operating system's
entropy source, so even an unplanned run can be recreated from its metadata.

Seeding with `Random.seed!` directly also makes runs reproducible, but the
seed is then not known to UnitCellMC and is recorded as `unknown` (or, if
`set_global_seed!` was called earlier in the session, as that earlier seed,
which no longer describes the run), so prefer this function.

# Returns

The seed that was set.
"""
function set_global_seed!(seed::Integer = rand(Random.RandomDevice(), UInt32))
    Random.seed!(seed)
    GLOBAL_SEED[] = seed
    return seed
end

"""
    global_seed()

Return the seed most recently set with [`set_global_seed!`](@ref), or
`nothing` if none has been set.
"""
global_seed() = GLOBAL_SEED[]

"""
    recorded_seed(rng)

The seed to record in metadata for a run driven by `rng`: the
[`global_seed`](@ref) when `rng` is the global RNG, otherwise `nothing`,
since the seed of an explicitly supplied generator cannot be recovered.
"""
recorded_seed(rng::AbstractRNG) = rng === Random.default_rng() ? global_seed() : nothing
