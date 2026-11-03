"""
    AbstractModel{T<:Real}

Abstract supertype for the spin models simulated by this package
([`IsingModel`](@ref), [`XYModel`](@ref), [`ClockModel`](@ref)).

# Coupling layout contract

A model provides one bond coupling per bond template of the
[`Geometry`](@ref) and one external-field value per unit-cell orbital,
exposed through the [`couplings`](@ref) and [`fields`](@ref) accessors (by
default the model's `J` and `h` fields). The checked
[`bond_strength`](@ref)/[`field_strength`](@ref) lookups, the from-scratch
energy sums, and every hot inner loop read couplings and fields through these
accessors, so a model with a different storage layout overrides exactly these
two functions (plus [`validate_model_geometry`](@ref) for its checks) instead
of hunting down every read site.

A model also declares its state type ([`state_type`](@ref)) and implements
the single-site update contract ([`propose`](@ref),
[`energy_difference`](@ref); [`apply_update!`](@ref) comes with the state),
its observables ([`observables`](@ref), [`derived_observables`](@ref)), and
optionally the cluster-update hooks and [`binder_denominator`](@ref).
"""
abstract type AbstractModel{T<:Real} end

abstract type AbstractAlgorithm end
abstract type AbstractState end

"""
    AbstractProposalScheme

Abstract supertype for the site-selection schemes of local-update algorithms
([`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref)).

A scheme decides which site each [`step!`](@ref) update visits —
[`RandomProposal`](@ref) draws uniformly at random,
[`SequentialProposal`](@ref) cycles through the lattice in site-index order,
[`CheckerboardProposal`](@ref) and [`ColoredProposal`](@ref) sweep one color
class of sites and then the next — while the model's [`propose`](@ref)
method decides the new value at that site. Stateful schemes carry their per-run state in a workspace built by
[`proposal_workspace`](@ref), which the drivers thread into [`step!`](@ref).
"""
abstract type AbstractProposalScheme end
