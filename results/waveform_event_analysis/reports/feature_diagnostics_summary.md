# Feature Diagnostics Summary

## Main finding

The regenerated direct-automation SLG dataset fixed the previous raw-data corruption issue:
- `Normal` and `SLG_Fault` no longer share the same feature signature.
- `SLG_Fault` now shows clear event-trigger features distinct from `Normal`.

However, the current classifier still fails to separate:
- `SLG_Fault`
- `ThreePhase_Fault`

and instead swaps them almost completely.

## Evidence

- `Normal` feature hash group is isolated to the 3 Normal cases.
- `SLG_Fault` now has 30 distinct hashes rather than sharing the old Normal hash.
- Direct smoke v2 metrics show strong AG-fault behavior:
  - `Bus02`: `max_dV_energy_3ph ≈ 0.2935`, `max_dI_energy_3ph ≈ 2.6931`, `max_sag ≈ 0.8048`
  - `Bus10`: `max_dV_energy_3ph ≈ 0.2992`, `max_dI_energy_3ph ≈ 0.0194`, `max_sag ≈ 0.9192`
  - `Bus30`: `max_dV_energy_3ph ≈ 0.2932`, `max_dI_energy_3ph ≈ 0.0325`, `max_sag ≈ 0.9835`
- In spite of this, the best flat and hierarchical classifiers both give:
  - `Normal recall = 1.0`
  - `LoadSwitch recall = 1.0`
  - `SLG recall = 0.0`
  - `ThreePhase recall = 0.0`

## Interpretation

- The original pipeline bug / raw integrity problem has been resolved far enough to eliminate the false `Normal == SLG` equivalence.
- The next bottleneck is no longer no-event triggering.
- The next bottleneck is **fault subtype separability**, especially `SLG_Fault` vs `ThreePhase_Fault`.

## Practical takeaway

The current result is good enough to conclude:
1. direct exact-parameter SLG automation works,
2. the previous SLG raw dataset was bad,
3. regenerated SLG waveforms are fault-like and non-Normal,
4. but additional feature engineering or richer fault-condition variation is still required before claiming reliable SLG-vs-ThreePhase classification.
