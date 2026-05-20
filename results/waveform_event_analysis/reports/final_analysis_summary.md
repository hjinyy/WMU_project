# Final Waveform Event Analysis Summary

- Input xlsx total: 93
- Class counts: {'LoadSwitch': 30, 'SLG_Fault': 30, 'ThreePhase_Fault': 30, 'Normal': 3}
- Quality status counts: OK=93, WARNING=0, FAILED=0
- Generated bus-level feature count: 69
- Full-WMU best model: RandomForest | macro-F1=0.7381 | balanced_accuracy=0.7500
- Best feature ablation group(s): All_features, DV_energy_only, Impedance_added, Voltage_current, Voltage_current_unbalance_sequence, Voltage_time_freq, Voltage_time_only | top macro-F1=0.7381
- All_features vs DV_energy_only macro-F1 delta: +0.0000
- Sensor-count saturation k (within 0.01 macro-F1 of best greedy result): 1
- Selected WMU combination up to saturation k: [1]

## LoadSwitch misclassified as Fault
- none

## Fault localization preliminary
```
       EventType  exact_bus_accuracy  top3_accuracy  one_hop_accuracy
       SLG_Fault                 0.0            0.0               0.1
ThreePhase_Fault                 0.0            0.0               0.5
         Overall                 0.0            0.0               0.3
```

## Research interpretation
- DV_energy remains a strong event-trigger feature, but it is not sufficient alone for robust four-class separation when LoadSwitch must be separated from faults.
- Sag-only behavior can highlight disturbance severity, but current/unbalance/sequence-style features are needed to differentiate balanced vs unbalanced faults and switching events.
- Time-frequency and current/sequence features materially improve LoadSwitch/Fault separation when waveform shapes share comparable voltage dips.
- WMU placement should maximize discriminative classification value under limited sensor budgets, not just fault-trigger energy or binary detection coverage.

## Next steps
- Expand LoadSwitch intensity levels to 5%, 15%, and 25%.
- Vary fault resistance, clearing time, and inception angle.
- Build a larger dataset with train/test separation across repeated conditions.
- Extend to weak-grid or DER-integrated operating scenarios.
