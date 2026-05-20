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

## 3. Dataset used in this run

Input directory used:
- `/mnt/c/Users/user/Documents/MATLAB/WMU_test/WMU_batch_raw`

Detected files:
- Total xlsx: 93
- Normal: 3
- LoadSwitch: 30
- SLG_Fault: 30
- ThreePhase_Fault: 30

Naming rules parsed by the pipeline:
- `SLG_Fault_Bus01.xlsx`
- `ThreePhase_Fault_Bus01.xlsx`
- `LoadSwitch_Bus02.xlsx`
- `LoadSwitch15pct_Bus02.xlsx`
- `Normal_Case01.xlsx`

Each file contains one event case. `EventType` and `TargetBus` are inferred from the filename. `Normal` cases are stored with `TargetBus = NaN`.

## 4. Quality-control stage

For every xlsx file, the pipeline checks:
- workbook readability,
- sheet readability,
- `Time` column existence,
- monotonic time stamps,
- sampling-interval mean/std,
- simulation duration,
- observed bus count,
- presence of `Va/Vb/Vc/Ia/Ib/Ic` per bus,
- NaN and inf counts,
- zero-like buses.

Run summary:
- OK: 93
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
- only 3 Normal cases are available, so Normal-class metrics are unstable and the report explicitly keeps LOO as the primary validation result.

## 7. Full-WMU classification results

Best LOO result:
- Model: RandomForest
- Accuracy: 0.9677
- Macro-F1: 0.7381
- Balanced accuracy: 0.7500
- LoadSwitch recall: 1.0000
- Fault precision: 0.9524
- Normal false alarm rate: 1.0000

Interpretation:
- All disturbance classes except `Normal` separate very strongly under the available dataset.
- The main weakness is the tiny Normal set, which causes Normal recall to collapse to zero across the tested models.

## 8. Feature ablation

Observed result on this dataset:
- `DV_energy_only`, `Voltage_time_only`, `Voltage_time_freq`, `Voltage_current`, `Voltage_current_unbalance_sequence`, `Impedance_added`, and `All_features` all reached the same top macro-F1 of 0.7381 under at least one model.

Interpretation:
- On the current dataset, event separation is already dominated by strong disturbance signatures.
- Richer features remain useful for future dataset expansion because they should matter more once switching intensity, fault resistance, clearing time, and operating-point diversity are broadened.

## 9. Sensor-count study

The sensor-count workflow evaluates:
- feature-aware greedy addition,
- DV-energy ranking baseline,
- random placement baseline.

Current preliminary result:
- the nearest-neighbor separability metric saturates immediately (`k = 1`) for the present dataset.

Interpretation:
- this indicates the current dataset is extremely separable at the class level and is not yet stressing the limited-sensor placement problem.
- the placement workflow is therefore implemented and reproducible, but the present numerical result should be treated as a low-diversity baseline rather than a final placement conclusion.

## 10. Fault localization preliminary

Localization target set:
- SLG_Fault
- ThreePhase_Fault

Method:
- nearest-fingerprint matching on fault-only case vectors,
- IEEE 30-bus 1-hop scoring using `data/ieee30_edges.csv`.

Result:
- SLG exact bus accuracy: 0.0
- ThreePhase exact bus accuracy: 0.0
- Overall exact bus accuracy: 0.0
- Overall top-3 accuracy: 0.0
- Overall 1-hop accuracy: 0.3

Interpretation:
- with one case per bus per fault type, localization remains only a **preliminary separability analysis**.
- exact leave-one-out matching is too strict for this dataset shape and should not be over-interpreted.

## 11. Main takeaways

- DV-energy is still a strong event-trigger feature.
- Sag alone is not a sufficient research endpoint; the broader feature set is needed for future switching-vs-fault generalization studies.
- Current and sequence-like features are the right direction for harder datasets even if the present ablation tie hides their value.
- WMU placement for waveform analytics should maximize discriminative information, not only detection coverage.

## 12. Next research steps

- Add LoadSwitch levels at 5%, 15%, and 25%.
- Vary fault resistance, clearing time, and inception angle.
- Expand the dataset so train/test separation is meaningful.
- Extend the study toward weak-grid or DER-integrated operating conditions.
