# Waveform Event Analysis Results

This folder contains the compact, git-trackable outputs from the corrected 83-file 3-phase waveform event analysis pipeline.

## Dataset basis for the current tracked results

- Total raw event files: 83
- Normal: 3
- LoadSwitch: 20
- SLG_Fault: 30
- ThreePhase_Fault: 30

The LoadSwitch cases correspond to the regenerated **15% abrupt load increase** dataset at load buses only.

## Reports

Stored in `reports/`:
- `data_quality_summary.md` - dataset quality snapshot
- `classification_full_wmu_metrics.csv` - full-WMU LOO classification metrics
- `classification_full_wmu_report.txt` - narrative classification report
- `confusion_matrix_full_wmu.csv` - confusion-matrix counts for the best full-WMU model
- `misclassified_cases_full_wmu.csv` - case-level misclassifications for the best full-WMU model
- `feature_ablation_metrics.csv` - feature-group comparison table
- `feature_ablation_used_columns.txt` - actual feature columns/counts per ablation group
- `sensor_count_curve.csv` - limited-WMU performance curves
- `selected_wmu_by_k.csv` - greedy WMU order by sensor count
- `fault_localization_preliminary.csv` - self-matching and strict-LOO localization summary
- `fault_localization_debug.csv` - per-case localization debug table
- `fault_localization_notes.txt` - localization caveats
- `final_analysis_summary.md` - one-page summary of the current run

## Figures

Stored in `figures/`:
- `confusion_matrix_full_wmu.png`
- `feature_ablation_macro_f1.png`
- `feature_ablation_loadswitch_recall.png`
- `sensor_count_macro_f1.png`
- `sensor_count_balanced_accuracy.png`
- `sensor_count_loadswitch_recall.png`
- `sensor_count_fault_precision.png`
- `fault_localization_confusion_matrix.png`

## Not tracked here

The following remain outside git because of size:
- raw xlsx/csv waveform files
- the full external output directory
- large intermediate tables such as full feature tables
- `sensor_selection_debug.csv` from the external output directory

Regenerate the external outputs with:

```bash
python scripts/run_waveform_event_analysis.py \
  --input-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_raw" \
  --output-dir "C:\Users\user\Documents\MATLAB\WMU_test\WMU_batch_data" \
  --f0 50
```

## Feature diagnostics extension

The tracked results now also include a **feature diagnostics package** whose main goal is to explain how selected waveform features separate:
- Normal
- LoadSwitch
- SLG_Fault
- ThreePhase_Fault

Important framing:
- the emphasis is **not** a final claim about classification performance;
- the emphasis **is** feature separability, Normal false-alarm interpretation, feature redundancy, spatial response structure, and dataset limitations.

### Added reports

- `reports/feature_distribution_summary.csv` - EventType-wise mean/std/median/q25/q75 for core features
- `reports/normal_misclassification_feature_values.csv` - feature values and class-relative positions for the three Normal→SLG false alarms
- `reports/feature_correlation_matrix.csv` - case-level core feature correlation matrix
- `reports/randomforest_feature_importance.csv` - fitted RandomForest feature importance over the wide feature table
- `reports/important_wmu_bus_count.csv` - observed-bus frequency among top-30 important features
- `reports/feature_diagnostics_summary.md` - narrative diagnostics summary

### Added figures and the questions they answer

- `figures/feature_boxplot_core_features.png` / `figures/feature_violin_core_features.png`  
  Which core features differ by class, and how stable are those distributions?
- `figures/scatter_dv_di.png`, `figures/scatter_sag_di.png`, `figures/scatter_i0_v0.png`, `figures/scatter_res_hf.png`, `figures/scatter_zdrop_di.png`, `figures/scatter_unbalance.png`  
  Which 2D feature combinations visually separate switching from faults or balanced from unbalanced events?
- `figures/normal_misclassification_core_features.png`  
  Why were all three Normal cases predicted as `SLG_Fault`?
- `figures/feature_correlation_heatmap.png`  
  Which core features are strongly redundant?
- `figures/randomforest_top30_feature_importance.png` / `figures/important_wmu_bus_count.png`  
  Which wide-table features and buses dominate the fitted RandomForest model?
- `figures/heatmap_*.png`  
  How do disturbance signatures spread across `TargetBus × ObservedBus`?
- `figures/waveform_representative_*.png`  
  What do representative raw waveforms look like around the event window?

## Normal/SLG debugging update

A dedicated debugging pass was added because the earlier flat classifier predicted all three `Normal` cases as `SLG_Fault`.

Important framing:
- the current tracked results are **not** a final claim that the event taxonomy is already solved;
- they are a **diagnostic package** for checking trigger behavior, class overlap, and possible raw-data / label-integrity issues.

### Additional tracked reports

- `reports/normal_raw_sanity_check.csv`
- `reports/label_parsing_check.csv`
- `reports/normal_vs_event_feature_values.csv`
- `reports/classification_flat_vs_hierarchical_metrics.csv`

### Additional tracked figures

- `figures/normal_waveform_sanity_check.png`
- `figures/normal_vs_slg_core_feature_boxplot.png`
- `figures/confusion_matrix_hierarchical_full_wmu.png`
- `figures/feature_ablation_normal_recall.png`
- `figures/sensor_count_normal_recall.png`

### What these new artifacts are for

- verify that Normal raw waveforms are genuinely low-trigger and steady-state;
- check whether `SLG_Fault` cases are numerically separable from `Normal` in the current dataset;
- compare the original flat classifier with a new hierarchical trigger-first classifier;
- keep the emphasis on **feature separability and dataset limitations**, not on a final performance claim.

## Direct SLG regeneration update

The tracked results now reflect a regenerated direct-automation SLG dataset.

What changed:
- the previous bad SLG raw files were archived out of the active raw dataset;
- a new direct exact-parameter MATLAB R2025b path regenerated `SLG_Fault_Bus01.csv` ~ `SLG_Fault_Bus30.csv`;
- the raw dataset is back to `83` event files.

What improved:
- `Normal` and `SLG_Fault` no longer share the same feature signature.
- direct smoke tests show clearly fault-like AG behavior on representative buses.

What remains unsolved:
- the current classifier still swaps `SLG_Fault` and `ThreePhase_Fault` almost perfectly.
- so the present outputs should still be interpreted as a **feature/dataset diagnostic package**, not as a final event-classification benchmark.

## Event-classification vs localization split

A later subtype-debug pass concluded that the original wide case representation can over-emphasize target-bus fingerprint.

What that means:
- a feature table that is good for localization may be overly location-dominant for subtype classification;
- subtype classification should rely more on phase asymmetry / sequence / event-summary descriptors;
- localization can keep the explicit bus-wide fingerprint.

Added outputs:
- `reports/fault_pair_distance_debug.csv`
- `reports/classification_grouped_by_targetbus_metrics.csv`
- `reports/classification_feature_representation_comparison.csv`
- `reports/fault_subtype_binary_metrics.csv`
- `reports/event_type_feature_importance.csv`
- `reports/slg_threephase_feature_summary.csv`
- `figures/fault_pair_distance_boxplot.png`
- `figures/confusion_matrix_grouped_by_targetbus.png`
- `figures/confusion_matrix_event_type_summary.png`
- `figures/confusion_matrix_fault_subtype_binary.png`
- `figures/event_type_top30_feature_importance.png`
- `figures/slg_threephase_subtype_feature_boxplot.png`
- `figures/sensor_count_event_classification_macro_f1.png`
- `figures/sensor_count_localization_accuracy.png`
