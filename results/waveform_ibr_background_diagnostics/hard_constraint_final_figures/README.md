# Hard-constraint minimum WMU placement

## 1. Research objective

This analysis does not optimize simple event detection, a macro-F1 curve, or
exact fault-bus localization. It finds the smallest WMU set that detects every
fault while never classifying Normal or LoadSwitch as a fault.

## 2. Problem definition

`BinaryFaultLabel = 0` for `SSO_Normal` and `SSO_LoadSwitch`.
`BinaryFaultLabel = 1` for `SSO_SLG_Fault` and
`SSO_ThreePhase_Fault`.

A placement is feasible only when all four hard constraints hold:

1. Normal false positives = 0
2. LoadSwitch false positives = 0
3. SLG false negatives = 0
4. Three-phase-fault false negatives = 0

## 3. Dataset

- 84 cases: 3 Normal, 21 LoadSwitch, 30 SLG fault, 30 three-phase fault
- 30 observed WMU buses per case
- Existing feature tables only; Simulink and raw-data generation were not run

## 4. Feature strategy

The fixed physical score combines normalized fault evidence from voltage/current
disturbance energy, sag, RMS jump, sequence/unbalance, and apparent-impedance
features. LoadSwitch rejection uses the balanced 67–77 Hz voltage
cycle-difference ratio:

`FaultScore = max(fault evidence) - 0.5 * LoadSwitchEvidence`

Each feature is normalized independently per observed bus using label-independent
5th and 95th percentiles. `LoadSwitchEvidence` is the minimum of the three phase
67–77 Hz ratios, because the current deterministic LoadSwitch response is high
on all three phases. The coefficient 0.5 is fixed globally and is not optimized
per subset.

## 5. Exhaustive subset search

All combinations were evaluated through k=4:

- k=1: 6, k=2: 11, k=3: 6, k=4: 1

The first feasible level is k=1.
Feasible counts need not increase with k because the fixed set score takes the
maximum fault evidence and maximum LoadSwitch evidence across selected WMUs;
adding a WMU can increase the LoadSwitch penalty and reduce the separation
margin.

## 6. Final result

- Selected WMU set: Bus 5
- Normal FP: 0
- LoadSwitch FP: 0
- SLG FN: 0
- Three-phase FN: 0
- Fault recall: 1.000
- Non-fault specificity: 1.000
- LoadSwitch specificity: 1.000
- Separation margin: 0.170377
- Threshold: 0.109490

The threshold result is an in-dataset physical separability result. ML
leave-one-case-out, leave-target-location-out, and stratified three-fold results
are reported separately as supporting checks; they do not define the minimum.

## 7. Final figures

1. `Fig01_fault_nonfault_feature_separation.png`
2. `Fig02_minimum_wmu_hard_constraint.png`
3. `Fig03_selected_wmu_binary_confusion.png`
4. `Fig04_per_fault_bus_coverage.png`
5. `Fig05_margin_or_safety_map.png`

Previous localization figures are excluded from this main result. They remain
available only in the pre-existing diagnostics/archive directories.

## 8. Limitations

- One deterministic IBR-like SSO condition
- One 15% LoadSwitch condition
- Fixed fault resistance and inception conditions
- No measurement noise or missing channels
- Limited operating-point diversity
- Only three Normal cases
- The physical threshold and normalization were assessed on the current dataset;
  external-condition robustness is not yet established

## 9. Next steps

Repeat the complete hard-constraint subset search after varying LoadSwitch
magnitude, fault resistance, inception angle, SSO amplitude/frequency,
measurement noise, missing channels, and operating point. The key question is
whether Bus 5 and k=1 remain feasible.
