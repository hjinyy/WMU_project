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

## 14. Feature diagnostics extension

The current diagnostics pass adds a **feature-interpretation layer** on top of the baseline classification/localization pipeline.

Important framing:
- the current results are **not** presented as a final classification-performance claim;
- the emphasis is on **feature separability**, **Normal / LoadSwitch / Fault overlap structure**, and **dataset limitations** in the 83-case run.

### Added diagnostics figures and the questions they answer

- `figures/feature_boxplot_core_features.png`  
  Which core case-level features show visibly different class-wise distributions?
- `figures/feature_violin_core_features.png`  
  How wide or narrow is each class distribution, especially with the tiny Normal set?
- `figures/scatter_dv_di.png`  
  Do voltage-change energy and current-change energy separate switching from faults in 2D?
- `figures/scatter_sag_di.png`  
  Is sag severity alone enough, or does it need current disturbance strength together?
- `figures/scatter_i0_v0.png`  
  Do sequence-like zero-sequence responses separate unbalanced vs. balanced events?
- `figures/scatter_res_hf.png`  
  Do residual-frequency and high-frequency disturbance ratios provide extra morphology clues?
- `figures/scatter_zdrop_di.png`  
  Does apparent-impedance collapse help distinguish switching from faults?
- `figures/scatter_unbalance.png`  
  Do voltage/current unbalance features form event-type clusters?
- `figures/normal_misclassification_core_features.png`  
  Where do the three misclassified Normal cases sit relative to Normal / LoadSwitch / SLG / ThreePhase distributions?
- `figures/feature_correlation_heatmap.png`  
  Which core features are redundant or strongly coupled?
- `figures/randomforest_top30_feature_importance.png`  
  Which wide-table features dominate the fitted RandomForest decision surface?
- `figures/important_wmu_bus_count.png`  
  Which observed buses appear most often among the top-ranked RandomForest features?
- `figures/heatmap_slg_dv_energy.png`, `figures/heatmap_threephase_dv_energy.png`, `figures/heatmap_loadswitch_dv_energy.png`  
  Where does voltage disturbance energy become strongest across TargetBus × ObservedBus?
- `figures/heatmap_slg_i0_ratio.png`  
  Where is the zero-sequence-like current response strongest for SLG events?
- `figures/heatmap_threephase_di_energy.png`, `figures/heatmap_loadswitch_di_energy.png`  
  How does current disturbance energy propagate spatially for three-phase faults and load switching?
- `figures/waveform_representative_normal.png`, `figures/waveform_representative_loadswitch.png`, `figures/waveform_representative_slg.png`, `figures/waveform_representative_threephase.png`  
  What do representative raw Va/Vb/Vc and Ia/Ib/Ic traces look like around the event window?

### Added diagnostics reports

- `reports/feature_distribution_summary.csv`
- `reports/normal_misclassification_feature_values.csv`
- `reports/feature_correlation_matrix.csv`
- `reports/randomforest_feature_importance.csv`
- `reports/important_wmu_bus_count.csv`
- `reports/feature_diagnostics_summary.md`

These artifacts should be interpreted as a structured diagnostic package for understanding **why** features separate classes (or fail to do so), not only **how well** a classifier scores on the current small dataset.

## 15. Normal/SLG debugging update

A dedicated debugging pass was added because the earlier flat full-WMU confusion matrix predicted all three `Normal` cases as `SLG_Fault`.

Current interpretation:
- this is **not** treated as a result that can simply be ignored;
- the pipeline now explicitly audits whether the issue comes from feature extraction, label parsing, event-window handling, or raw-data integrity;
- the current tracked results emphasize **feature separability and dataset-integrity diagnosis**, not a final classification claim.

### New debugging artifacts and the questions they answer

- `reports/normal_raw_sanity_check.csv`  
  Are the three Normal cases steady-state, low-trigger, and free of obvious event-like changes?
- `figures/normal_waveform_sanity_check.png`  
  Do `Va/Vb/Vc/Ia/Ib/Ic` remain normal around 0.45-0.65 s for the Normal cases?
- `reports/label_parsing_check.csv`  
  Are `Normal`, `LoadSwitch`, `SLG_Fault`, and `ThreePhase_Fault` parsed into the intended labels and target buses?
- `reports/normal_vs_event_feature_values.csv`  
  How do Normal case-level feature values compare numerically against event-labeled cases?
- `figures/normal_vs_slg_core_feature_boxplot.png`  
  Are the Normal and SLG case-level summaries genuinely separable, or are they numerically overlapped?
- `reports/classification_flat_vs_hierarchical_metrics.csv`  
  How does the original flat classifier compare to the new hierarchical `Normal-vs-Event -> event-type` structure?
- `figures/confusion_matrix_hierarchical_full_wmu.png`  
  Does the hierarchical trigger improve Normal recall without hiding other failure modes?
- `figures/feature_ablation_normal_recall.png`  
  Which feature groups help the hierarchical pipeline preserve Normal recall?
- `figures/sensor_count_normal_recall.png`  
  Under limited WMU counts, how stable is Normal recall after the trigger-first change?

### Current conclusion from the debugging pass

- the Normal raw waveforms remain low-trigger / near-steady-state;
- the current issue is driven less by Normal raw behavior itself and more by the fact that many `SLG_Fault` cases appear indistinguishable from `Normal` under the present raw dataset / feature tables;
- therefore the current results must be read as a **pipeline + dataset integrity diagnostic**, not as a final event-classification conclusion.

## 16. Direct SLG regeneration update (R2025b)

The previous `SLG_Fault_Bus*.xlsx` dataset was archived after diagnostics showed that `Normal` and `SLG_Fault` shared the same feature signature. A new direct-parameter MATLAB R2025b automation path was then used to regenerate `SLG_Fault_Bus01.csv` ~ `SLG_Fault_Bus30.csv`.

Key outcome:
- the old `Normal == SLG` corruption problem is resolved;
- the direct smoke tests show clear AG-fault waveforms and strong event-trigger features;
- but the current classifier now swaps `SLG_Fault` and `ThreePhase_Fault`, so the research bottleneck has moved from **data corruption** to **fault subtype separability**.

Important framing:
- these regenerated results should still be treated as **diagnostic / iterative research outputs**, not a final classification claim.
