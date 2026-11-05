# To Do / In Progress

In the future we'd like to add some further functionality to this package.

## Models

- Potts Model

## Algorithms

- Heatbath
- Overrelaxation (XY/Heisenberg)

## Functionality

- Checkpointing (JLD2 or HDF5)
- Bipartite gauge transformations (e.g. to use clustering algorithms on AFM square lattice).
- Raw spin config saving for data analysis later
- Run-level uncertainty quantification for `sweep_Ts`: per-run analysis as an
  opt-in diagnostic, and run-level bootstrap of pooled derived quantities.
  Motivation: under ergodicity breaking (e.g. zero-field susceptibility below
  T_c, where runs freeze into ± symmetry sectors) the pooled bin/jackknife
  error bars are silently overconfident — the dominant randomness is the
  sector count across runs, which per-bin resampling cannot see. Run-level
  bootstrap keeps the pooled estimand with honest error bars; per-run
  analysis makes the failure visible (note it changes the estimand of
  nonlinear derived quantities to the single-sector/connected one).
