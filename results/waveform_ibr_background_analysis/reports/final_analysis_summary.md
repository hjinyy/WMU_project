# Final Waveform Event Analysis Summary

- Scenario: ibr-background
- Input raw event files: 84
- Class counts: {'SSO_SLG_Fault': 30, 'SSO_ThreePhase_Fault': 30, 'SSO_LoadSwitch': 21, 'SSO_Normal': 3}
- All cases include the IBR-like 25 Hz SSO background condition.
- SSO_LoadSwitch cases use 21 target buses with LoadAdd P/QL set to 15% of the existing Load bus P/QL.
- LoadSwitch event time is 0.1 s; fault start/clear times are 0.3/0.36 s.
- Quality status counts: OK=84, WARNING=0, FAILED=0
- Generated bus-level feature count: 81
- Best flat classifier: RandomForest | macro-F1=1.0000 | balanced_accuracy=1.0000 | Normal_recall=1.0000
- Best hierarchical classifier: rule_based + RandomForest | macro-F1=1.0000 | balanced_accuracy=1.0000 | Normal_recall=1.0000
- Sensor-count saturation k (within 0.01 macro-F1 of best greedy result): 1
- Selected WMU combination up to saturation k: [1]

## Best ablation rows
```
                      FeatureGroup     Strategy TriggerMethod   EventModel  macro_f1  Normal_recall  LoadSwitch_recall  Fault_precision
                      All_features         flat          none RandomForest   1.00000            1.0                1.0              1.0
                    DV_energy_only         flat          none RandomForest   0.94165            1.0                1.0              1.0
                   Impedance_added         flat          none RandomForest   1.00000            1.0                1.0              1.0
                   Voltage_current         flat          none RandomForest   1.00000            1.0                1.0              1.0
Voltage_current_unbalance_sequence         flat          none RandomForest   1.00000            1.0                1.0              1.0
                 Voltage_time_freq         flat          none RandomForest   1.00000            1.0                1.0              1.0
                 Voltage_time_only         flat          none RandomForest   1.00000            1.0                1.0              1.0
                      All_features hierarchical    rule_based RandomForest   1.00000            1.0                1.0              1.0
                    DV_energy_only hierarchical    rule_based RandomForest   0.94165            1.0                1.0              1.0
                   Impedance_added hierarchical    rule_based RandomForest   1.00000            1.0                1.0              1.0
                   Voltage_current hierarchical    rule_based RandomForest   1.00000            1.0                1.0              1.0
Voltage_current_unbalance_sequence hierarchical    rule_based RandomForest   1.00000            1.0                1.0              1.0
                 Voltage_time_freq hierarchical    rule_based RandomForest   1.00000            1.0                1.0              1.0
                 Voltage_time_only hierarchical    rule_based RandomForest   1.00000            1.0                1.0              1.0
```

## LoadSwitch misclassified as Fault (best hierarchical)
- none

## Fault localization preliminary
```
   Evaluation            EventType  exact_bus_accuracy  top3_accuracy  one_hop_accuracy
self_matching        SSO_SLG_Fault                 1.0            1.0          1.000000
self_matching SSO_ThreePhase_Fault                 1.0            1.0          1.000000
self_matching              Overall                 1.0            1.0          1.000000
   strict_loo        SSO_SLG_Fault                 0.0            0.0          0.600000
   strict_loo SSO_ThreePhase_Fault                 0.0            0.0          0.333333
   strict_loo              Overall                 0.0            0.0          0.466667
```

## Current limitations
- The IBR-like SSO background is intentionally present in every class, so event classification measures the incremental disturbance on top of the shared oscillatory background.
- The SSO frequency and envelope are fixed at one configured condition.
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
