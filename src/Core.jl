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
