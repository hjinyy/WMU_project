# Waveform Event Analysis

## 1. Research background

The original repository focused on DV-energy-driven SLG fault observability and preliminary minimum-WMU placement. This extension targets a different question: whether richer 3-phase WMU waveform features can preserve event classification and preliminary fault-localization performance even when only a limited number of WMUs are retained.

Target event classes:
- Normal
- LoadSwitch
- SLG_Fault
- ThreePhase_Fault

## 2. Why WMU waveform placement differs from PMU placement

Traditional PMU placement often optimizes topological observability or state-estimation redundancy. The present workflow instead optimizes **waveform discriminability**:
- event-trigger strength,
- disturbance morphology,
- class separability,
- and bus-fingerprint distinctiveness.

A WMU location is therefore valuable when its local waveform is informative for classifying switching vs. fault events, not only when it improves static network observability.

## 3. Dataset used in the current run

Input directory used:
- `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

Detected event files:
- Total raw event files: 83
- Normal: 3
- LoadSwitch: 20
- SLG_Fault: 30
- ThreePhase_Fault: 30

Format mix:
- xlsx: SLG, ThreePhase, Normal
- csv: LoadSwitch

LoadSwitch cases now represent **15% abrupt load increase** conditions only at buses with existing Pd/Qd:
`2, 3, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 29, 30`

Each file contains one event case. `EventType` and `TargetBus` are inferred from the filename. `Normal` cases are stored with `TargetBus = NaN`.

## 4. Quality-control stage

For every event file, the pipeline checks:
- file readability,
- `Time` column existence,
- monotonic time stamps,
- sampling-interval mean/std,
- simulation duration,
- observed bus count,
- presence of `Va/Vb/Vc/Ia/Ib/Ic` per bus,
- NaN and inf counts,
- zero-like buses.

Current run summary:
- OK: 83
- WARNING: 0
- FAILED: 0

## 5. Feature extraction

The pipeline extracts bus-level features for every case and observed bus, then aggregates them into:
- `feature_table_by_bus.csv`
- `feature_table_by_case.csv`
- `feature_table_by_case_wide.csv`

Feature families:
- voltage time-domain sag and cycle-difference energy,
- current cycle-difference energy and RMS jump,
- time-frequency ratios on cycle-difference waveforms,
- unbalance and sequence-like features,
- apparent-impedance change,
- voltage-current Lissajous features.

See `docs/waveform_feature_definitions.md` for definitions.

## 6. Classification method

Full-WMU case classification uses the case-wide feature vector and evaluates:
- Logistic Regression
- Decision Tree
- Random Forest
- LinearSVC

Validation:
- Leave-One-Out Cross Validation (primary)
- StratifiedKFold when class counts permit it

Important limitation:
- only 3 Normal cases are available, so Normal-class metrics are unstable and the report explicitly keeps LOO as the primary result.

## 7. Full-WMU classification results

Best LOO result:
- Model: RandomForest
- Accuracy: 0.9639
- Macro-F1: 0.7381
- Balanced accuracy: 0.7500
- LoadSwitch recall: 1.0000
- Fault precision: 0.9524
- Normal false alarm rate: 1.0000

Interpretation:
- LoadSwitch, SLG, and ThreePhase cases remain strongly separable.
- The main weakness is the tiny Normal set, which causes Normal recall to collapse to zero in the tested models.

## 8. Feature ablation

The pipeline now writes `feature_ablation_used_columns.txt` so each feature group's actual columns and counts can be inspected directly.

Observed result on this corrected 83-file dataset:
- the RandomForest top result is tied across all seven feature groups at macro-F1 = 0.7381,
- but the groups do **not** use the same matrices;
- column counts differ substantially across groups.

Interpretation:
- the previous concern about accidental column reuse is addressed by explicit column export and fingerprinting;
- the remaining tie appears to come from dataset separability, not from a feature-group wiring bug.

## 9. Sensor-count study and leakage control

The sensor-count workflow evaluates:
- feature-aware greedy addition,
- DV-energy ranking baseline,
- random placement baseline.

Leakage guard applied in the current version:
- k-WMU evaluation is built only from `feature_table_by_bus` rows for the selected buses,
- full-system case-level summary features such as `max_dV_energy`, `star_bus_dV`, and similar all-bus aggregates are excluded from k-WMU experiments,
- `sensor_selection_debug.csv` is written to the external output folder for auditability.

Current preliminary result:
- the nearest-neighbor sensor-count metric still saturates immediately (`k = 1`) on this dataset.

Interpretation:
- after removing the all-bus leakage path, the corrected result still indicates very easy class separability under the present dataset,
- so the current sensor-count curve should be treated as a baseline, not as a final placement conclusion.

## 10. Fault localization preliminary

Localization target set:
- SLG_Fault
- ThreePhase_Fault

Two evaluations are reported:
- `self_matching`: sanity-check separability with self included,
- `strict_loo`: leave-one-out preliminary localization with self excluded.

Strict LOO result:
- SLG exact bus accuracy: 0.0
- ThreePhase exact bus accuracy: 0.0
- Overall exact bus accuracy: 0.0
- Overall top-3 accuracy: 0.0
- Overall 1-hop accuracy: 0.3

Interpretation:
- with one case per bus per fault type, localization remains only a **preliminary** fingerprint test,
- self-matching is just a sanity check,
- strict LOO is the more relevant result and remains weak.

## 11. Main takeaways

- DV-energy is still a strong event-trigger feature.
- Sag alone is not a sufficient research endpoint; the broader feature set is still the right direction for future switching-vs-fault studies.
- The current LoadSwitch 15% dataset is highly separable from faults in full-WMU classification.
- WMU placement for waveform analytics should maximize discriminative information, not only detection coverage.

## 12. Current limitations

- Normal class count is only 3.
- LoadSwitch uses only one intensity level: 15% increase.
- Fault resistance, clearing time, and inception angle are not varied.
- Fault localization remains preliminary because there is only one case per bus per fault type.

## 13. Next research steps

- Add LoadSwitch levels at 5%, 15%, and 25%.
- Vary fault resistance.
- Vary clearing time and inception angle.
- Expand the dataset so train/test separation is meaningful.
- Extend the study toward weak-grid or DER-integrated operating conditions.
