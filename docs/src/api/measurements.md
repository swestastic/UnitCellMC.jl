# Measurements

```@docs
UnitCellMC.average_and_error
UnitCellMC.jackknife_samples
UnitCellMC.jackknife_stats
UnitCellMC.analyze
```

## Autocorrelation

```@docs
UnitCellMC.AutocorrelationResult
UnitCellMC.autocorrelation
UnitCellMC.autocorrelation_function
UnitCellMC.integrated_autocorrelation_time
UnitCellMC.tau_lower_bound
UnitCellMC.recommended_binning
```

```@docs
UnitCellMC.MeasurementContainer
UnitCellMC.validate_binning
UnitCellMC.validate_dependencies
```

```@docs
UnitCellMC.CorrelationClass
UnitCellMC.correlation_instantaneous
UnitCellMC.correlation_instantaneous_dft
UnitCellMC.correlation_entry
UnitCellMC.magnetization_per_orbital_entry
```

## Structure factor

```@docs
UnitCellMC.StructureFactorClass
UnitCellMC.build_momentum_classes
UnitCellMC.fourier_transform_dim
UnitCellMC.structure_factor_instantaneous
UnitCellMC.structure_factor_entry
```

```@meta
CollapsedDocStrings = true
```

```@docs
UnitCellMC.measure!
```

```@docs
UnitCellMC.Observable
UnitCellMC.DerivedObservable
UnitCellMC.time_series_names
```

```@docs
UnitCellMC.optional_observables
UnitCellMC.abs_magnetization_entry
UnitCellMC.binder_cumulant_entry
UnitCellMC.clock_order_parameter_entry
```

```@docs
UnitCellMC.observables
UnitCellMC.derived_observables
UnitCellMC.available_measurements
UnitCellMC.extra_measurements
```
