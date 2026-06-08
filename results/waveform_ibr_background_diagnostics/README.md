# IBR-background waveform diagnostics

## 1. Purpose

The prior 84-case IBR-like SSO background study obtained macro-F1 = 1.0 for event classification, but that headline result did not explain feature-level separability, the immediate k=1 sensor-count saturation, or the weak strict-LOO exact-bus localization. This additive diagnostics pass addresses those gaps without rerunning Simulink or modifying raw waveforms.

## 2. Dataset and reused inputs

- Raw dataset (read-only reference): `/mnt/c/Users/user/Documents/MATLAB/WMU_final/WMU_batch_raw_ibr_background`
- Reused feature tables: `/mnt/c/Users/user/Documents/MATLAB/WMU_final/WMU_batch_data_ibr_background`
- Cases: SSO_Normal 3, SSO_LoadSwitch 21, SSO_SLG_Fault 30, SSO_ThreePhase_Fault 30 (84 total)
- Deterministic conditions: one SSO setting, one 15% load-switch setting, fixed fault parameters, no injected measurement noise.

## 3. Feature distribution and separability

The plots in `figures/feature_distribution_*.png` show individual samples over boxplots, so the three Normal cases remain visible. Symlog is used only when a feature spans more than four orders of magnitude. The strongest univariate separators are:

| Feature | MaxAbsCliffsDelta | TopSeparatingClassPair |
| --- | --- | --- |
| Delta_I_unbalance | 1 | SSO_SLG_Fault vs SSO_ThreePhase_Fault |
| mean_sag | 1 | SSO_SLG_Fault vs SSO_ThreePhase_Fault |
| I0_ratio | 1 | SSO_SLG_Fault vs SSO_ThreePhase_Fault |
| dV_Res_ratio_3ph_max | 1 | SSO_SLG_Fault vs SSO_ThreePhase_Fault |
| max_sag | 1 | SSO_Normal vs SSO_ThreePhase_Fault |
| I2_ratio | 1 | SSO_Normal vs SSO_LoadSwitch |
| dI_energy_3ph_mean | 1 | SSO_Normal vs SSO_ThreePhase_Fault |
| Delta_V_unbalance | 1 | SSO_Normal vs SSO_LoadSwitch |
| V2_ratio | 1 | SSO_Normal vs SSO_ThreePhase_Fault |
| dV_energy_3ph_mean | 1 | SSO_Normal vs SSO_ThreePhase_Fault |

`reports/single_feature_dv_energy_performance.csv` directly tests whether one dV-energy feature alone reproduces the four-class LOO result. The ranking should not be read as causal proof: many bus/phase features are correlated because all cases were generated under a narrow deterministic design.

## 4. Feature importance

RandomForest impurity importance was computed on the full wide WMU table. Cross-validated permutation importance was then applied to the top 60 impurity candidates to avoid thousands of low-information permutations on only 84 cases.

| Feature | Importance |
| --- | --- |
| Bus27__I_rms_pre_C | 0.0180464 |
| Bus11__I_rms_pre_B | 0.0173839 |
| Bus30__I2_ratio | 0.0122471 |
| Bus24__dV_HF_ratio_A | 0.0120743 |
| Bus17__dI_SSO20_30_ratio_B | 0.00991212 |
| Bus16__dI_SSC5_55_ratio_B | 0.0090906 |
| Bus16__dV_HF_ratio_A | 0.00818777 |
| Bus17__dI_HF_ratio_A | 0.0081297 |
| Bus17__dI_Res_ratio_B | 0.00780907 |
| Bus13__I_rms_jump_B | 0.00773321 |
| Bus27__sag_C | 0.00727114 |
| Bus29__dI_SSO20_30_ratio_B | 0.00715572 |
| Bus14__dV_E28_ratio_A | 0.00666469 |
| Bus30__dV_SSC5_55_ratio_B | 0.00650234 |
| Bus05__Delta_V_unbalance | 0.00643116 |
| Bus28__dI_HF_ratio_A | 0.00640424 |
| Bus02__I_rms_post_A | 0.00638787 |
| Bus08__Delta_Z_app | 0.00637389 |
| Bus16__I_rms_pre_A | 0.0061957 |
| Bus21__dI_SSO20_30_ratio_B | 0.00611213 |

Class-wise one-vs-rest importances are in `classwise_feature_importance.csv`. Voltage disturbance energy mainly exposes event severity; SSO-band ratios help distinguish the persistent background signature; current magnitude/jump features help separate switching from faults; sequence/unbalance features are especially relevant to SLG versus balanced three-phase faults; impedance changes contribute to fault versus switching discrimination.

Several top full-WMU impurity features are pre-event current magnitudes. That is
a warning that operating-point/location fingerprints and correlated predictors
also help the classifier; it strengthens the case for grouped validation and
prevents interpreting the importance ranking as purely event-physics evidence.

## 5. Sensor-count diagnostics

- Best single WMU: bus 1 (macro-F1 1.0000)
- Worst single WMU: bus 18 (macro-F1 0.9798)
- Bus 1: macro-F1 1.0000, balanced accuracy 1.0000
- Leave-target-location-out at k=1: macro-F1 1.0000, balanced accuracy 1.0000

The k=1 plateau is therefore a property of this deterministic, high-severity, full-network-observation feature design rather than evidence that one WMU is generally sufficient. The grouped split removes every event at the held target location and tests each Normal case once; with only three Normal records, Normal estimates remain high variance. The selected-bus ranking is diagnostic, not a universal placement optimum.

## 6. Fault-localization diagnostics

- Exact: 0.0000
- Top-3: 0.0000
- 1-hop: 0.4667
- 2-hop: 0.8000
- Mean/median graph-distance error: 1.833 / 2.000 hops
- Zone accuracy: 0.7333; zone macro-F1: 0.7273

Exact-bus localization remains preliminary because each bus has only one case per fault type and strict LOO becomes nearest-neighbor fingerprint transfer across different event instances. Neighbor/zone metrics are more realistic for this dataset, but they do not replace validation across fault resistance, inception angle, SSO amplitude/frequency, operating point, and noise.

## 7. Research interpretation

Event classification succeeds under the current controlled conditions. The feature distributions and importance tables explain why: broad severity, sequence, current, and impedance signatures create strong and partly redundant class separation. Placement is correspondingly easy and saturates early. Localization should be framed as neighborhood/zone screening rather than exact-bus identification until richer repeated conditions are available.

## 8. Limitations and next experiments

- Only 3 Normal cases.
- One SSO condition and one 15% load-switch magnitude.
- Fixed fault resistance/parameters and deterministic waveforms.
- No measurement noise, missing channels, timing jitter, or topology uncertainty.
- Strong feature correlation and many features relative to 84 cases.

Next experiments should vary SSO magnitude/frequency/damping, fault resistance and inception angle, load-switch size, operating point, and measurement noise. Repeated cases per bus are required for a defensible exact-bus localization benchmark.

## 9. Figure organization

The original generated files remain unchanged under `figures/`. Curated copies
are organized into:

- `figures_keep_main/`: figures recommended for the main text.
- `figures_appendix/`: supporting figures recommended for an appendix or supplement.
- `figures_deprecated/`: redundant or lower-priority figures retained for traceability but not recommended for publication.

Every copy operation, source/destination path, byte count, and SHA-256 checksum
is recorded in `reports/figure_organization_log.csv`.

### Main-text figures

| Figure | Recommended use |
| --- | --- |
| `feature_distribution_dv_energy.png` | Explain voltage-disturbance energy separation across classes. |
| `feature_distribution_di_energy.png` | Explain current-disturbance separation, especially switching versus faults. |
| `feature_distribution_sso_band_energy.png` | Show the role and overlap of SSO/spectral features. |
| `feature_distribution_sequence_unbalance.png` | Show the strongest physical separation between SLG and three-phase faults. |
| `sensor_count_macro_f1_extended.png` | Present the k=1 saturation result with alternative sensor-count baselines. |
| `localization_onehop_twohop_summary.png` | Summarize exact, neighbor, two-hop, and zone localization performance. |
| `localization_distance_cdf.png` | Show the cumulative graph-distance localization error. |
| `localization_zone_confusion_matrix.png` | Present the more defensible zone-level localization result. |

### Appendix figures

| Figure | Supporting role |
| --- | --- |
| `single_wmu_macro_f1_by_bus.png` | Detailed bus-by-bus single-WMU performance. |
| `single_wmu_class_recall_heatmap.png` | Bus-specific class-recall detail. |
| `localization_distance_by_fault_type.png` | SLG versus three-phase localization comparison. |
| `localization_distance_histogram.png` | Alternative view of graph-distance errors. |
| `localization_normalized_confusion_matrix.png` | Full exact-bus confusion structure. |
| `localization_predicted_bus_frequency.png` | Prediction concentration and bus-frequency bias. |
| `feature_importance_randomforest_top20.png` | Full-WMU impurity importance detail. |
| `feature_distribution_sag.png` | Supporting voltage-sag distribution evidence. |
| `feature_distribution_impedance.png` | Supporting impedance-feature distributions. |

### Deprecated figures retained for traceability

| Figure | Reason not recommended for main presentation |
| --- | --- |
| `single_wmu_balanced_accuracy_by_bus.png` | Largely duplicates the single-WMU macro-F1 ranking. |
| `sensor_count_balanced_accuracy_extended.png` | Duplicates the principal macro-F1 sensor-count conclusion. |
| `sensor_count_class_recall_extended.png` | Adds limited information under near-perfect class recall. |
| `grouped_cv_macro_f1_by_sensor_count.png` | Flat 1.0 curve is less informative than the extended comparison. |
| `classwise_feature_importance_heatmap.png` | Dense and difficult to interpret in the main narrative. |
| `permutation_importance_top20.png` | Near-zero values reflect feature redundancy and provide little visual ranking. |
| `bus1_vs_other_bus_feature_comparison.png` | The normalized comparison is less direct than the single-WMU result tables. |
| `localization_graph_distance_heatmap.png` | Visually dense and less interpretable than the CDF and zone matrix. |
