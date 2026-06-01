# Final Waveform Event Analysis Summary

- Input raw event files: 83
- Class counts: {'SLG_Fault': 30, 'ThreePhase_Fault': 30, 'LoadSwitch': 20, 'Normal': 3}
- LoadSwitch case count is 20 because only buses with existing Pd or Qd were regenerated under the 15% abrupt load-increase condition.
- Quality status counts: OK=83, WARNING=0, FAILED=0
- Generated bus-level feature count: 69
- Best flat classifier: LogisticRegression | macro-F1=1.0000 | balanced_accuracy=1.0000 | Normal_recall=1.0000
- Best hierarchical classifier: rule_based + RandomForest | macro-F1=1.0000 | balanced_accuracy=1.0000 | Normal_recall=1.0000
- Sensor-count saturation k (within 0.01 macro-F1 of best greedy result): 1
- Selected WMU combination up to saturation k: [1]

## Best ablation rows
```
                      FeatureGroup     Strategy TriggerMethod   EventModel  macro_f1  Normal_recall  LoadSwitch_recall  Fault_precision
                      All_features         flat          none RandomForest       1.0            1.0                1.0              1.0
                    DV_energy_only         flat          none RandomForest       1.0            1.0                1.0              1.0
                   Impedance_added         flat          none RandomForest       1.0            1.0                1.0              1.0
                   Voltage_current         flat          none RandomForest       1.0            1.0                1.0              1.0
Voltage_current_unbalance_sequence         flat          none RandomForest       1.0            1.0                1.0              1.0
                 Voltage_time_freq         flat          none RandomForest       1.0            1.0                1.0              1.0
                 Voltage_time_only         flat          none RandomForest       1.0            1.0                1.0              1.0
                      All_features hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
                    DV_energy_only hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
                   Impedance_added hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
                   Voltage_current hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
Voltage_current_unbalance_sequence hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
                 Voltage_time_freq hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
                 Voltage_time_only hierarchical    rule_based RandomForest       1.0            1.0                1.0              1.0
```

## LoadSwitch misclassified as Fault (best hierarchical)
- none

## Fault localization preliminary
```
   Evaluation        EventType  exact_bus_accuracy  top3_accuracy  one_hop_accuracy
self_matching        SLG_Fault                 1.0            1.0          1.000000
self_matching ThreePhase_Fault                 1.0            1.0          1.000000
self_matching          Overall                 1.0            1.0          1.000000
   strict_loo        SLG_Fault                 0.0            0.0          0.633333
   strict_loo ThreePhase_Fault                 0.0            0.0          0.666667
   strict_loo          Overall                 0.0            0.0          0.650000
```

## Current limitations
- Normal class has only 3 cases.
- LoadSwitch cases currently cover only one disturbance strength: 15% load increase.
- Fault resistance, clearing time, and inception angle are not varied.
- Fault localization remains preliminary because there is only one case per bus per fault type.
- If raw event files are mislabeled or duplicated across classes, classifier metrics should be treated as data-integrity diagnostics rather than final performance claims.

## Next steps
- Audit SLG raw files against the Normal baseline when event-trigger features remain near zero.
- Expand LoadSwitch intensity levels to 5%, 15%, and 25%.
- Vary fault resistance.
- Vary clearing time and inception angle.
- Build a larger dataset with train/test separation across repeated conditions.
- Extend to weak-grid or DER-integrated operating scenarios.

## Representation study update

A dedicated subtype-debug pass tested whether the earlier SLG↔ThreePhase swap was caused by label order or by location-dominant representation.

Conclusion:
- label mapping was not the cause;
- the underlying issue was that the old wide representation could strongly preserve target-bus fingerprint;
- for subtype studies, event classification and fault localization should be treated as separate representation problems.

## Fault subtype representation study

- A dedicated representation study was added after direct SLG/ThreePhase regeneration.
- Distance analysis showed that the same-target-bus opposite fault type was **not** the nearest sample under the refreshed regenerated dataset.
- Grouped-by-TargetBus evaluation does not degrade the regenerated dataset: the best settings remain perfectly separable.
- The new `event_type_summary_features` and `hybrid_features` both achieve perfect 4-class and SLG-vs-ThreePhase binary performance on the current regenerated dataset.
