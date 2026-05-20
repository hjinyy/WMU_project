# Final Waveform Event Analysis Summary

- Input raw event files: 83
- Class counts: {'SLG_Fault': 30, 'ThreePhase_Fault': 30, 'LoadSwitch': 20, 'Normal': 3}
- LoadSwitch case count is 20 because only buses with existing Pd or Qd were regenerated under the 15% abrupt load-increase condition.
- Quality status counts: OK=83, WARNING=0, FAILED=0
- Generated bus-level feature count: 69
- Full-WMU best model: RandomForest | macro-F1=0.7381 | balanced_accuracy=0.7500
- Best feature ablation group: DV_energy_only (RandomForest) | macro-F1=0.7381
- All_features vs DV_energy_only macro-F1 delta: +0.0000
- Sensor-count saturation k (within 0.01 macro-F1 of best greedy result): 1
- Selected WMU combination up to saturation k: [1]

## LoadSwitch misclassified as Fault
- none

## Fault localization preliminary
```
   Evaluation        EventType  exact_bus_accuracy  top3_accuracy  one_hop_accuracy
self_matching        SLG_Fault            0.033333           0.10              0.10
self_matching ThreePhase_Fault            1.000000           1.00              1.00
self_matching          Overall            0.516667           0.55              0.55
   strict_loo        SLG_Fault            0.000000           0.00              0.10
   strict_loo ThreePhase_Fault            0.000000           0.00              0.50
   strict_loo          Overall            0.000000           0.00              0.30
```

## Current limitations
- Normal class has only 3 cases.
- LoadSwitch cases currently cover only one disturbance strength: 15% load increase.
- Fault resistance, clearing time, and inception angle are not varied.
- Fault localization remains preliminary because there is only one case per bus per fault type.

## Next steps
- Expand LoadSwitch intensity levels to 5%, 15%, and 25%.
- Vary fault resistance.
- Vary clearing time and inception angle.
- Build a larger dataset with train/test separation across repeated conditions.
- Extend to weak-grid or DER-integrated operating scenarios.
