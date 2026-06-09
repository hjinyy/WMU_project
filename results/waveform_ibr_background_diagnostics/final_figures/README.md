# Final figures: classification-oriented minimum WMU placement

## Research objective

This study is not a minimum-sensor fault detector, and exact fault-bus
localization is not its primary objective. The objective is:

> In the modified IEEE 30-bus system with an IBR-like SSO background, identify
> the smallest WMU bus set that maintains four-class event-classification
> performance.

Formally, minimize `|S|` subject to maintained event-classification performance,
where `S` is the selected WMU bus set. The classes are `SSO_Normal`,
`SSO_LoadSwitch`, `SSO_SLG_Fault`, and `SSO_ThreePhase_Fault`.

## Dataset

- 84 deterministic cases
- IBR-like 25 Hz SSO background in every class
- 3 Normal, 21 LoadSwitch, 30 SLG-fault, and 30 three-phase-fault cases
- One 15% LoadSwitch condition
- Fixed fault parameters
- No measurement noise

## Main figures

### Fig01_feature_separation_panel.png

Representative Bus 1 waveform features explain all four classes:

- **LoadSwitch:** the phase-A 67–77 Hz voltage cycle-difference ratio
  (`dV_E72_ratio_A`) is high for LoadSwitch and achieved 1.0 three-fold
  one-feature balanced accuracy in this deterministic dataset.
- **Fault severity:** voltage disturbance energy is much larger for faults.
- **SSO-band response:** the 20–30 Hz ratio shows how event-induced waveform
  changes occupy the shared 25 Hz SSO-background band. It is not an SSO-presence
  detector because SSO is present in every class.
- **Fault subtype:** zero-sequence current separates SLG from balanced
  three-phase faults.

The 60–80 Hz hypothesis was supported at Bus 1: phase-A 60–80 Hz, 67–77 Hz, and
70–80 Hz voltage ratios each achieved 1.0 one-feature balanced accuracy.
However, many bands also separate perfectly because the dataset is narrow and
deterministic. The 67–77 Hz feature was retained because it already exists in
the feature table and provides a direct, reproducible LoadSwitch panel.

### Fig02_sensor_count_saturation.png

This is the principal result. It shows four-class macro-F1 versus selected WMU
count for feature-aware greedy placement, leave-target-location-out evaluation,
random subsets, and the worst single-WMU baseline. Under the current dataset,
classification performance is maintained from `k=1`; the feature-aware greedy
selection uses Bus 1.

This is a preliminary dataset-specific minimum, not evidence that one WMU is
generally sufficient. The current data use one SSO condition, one 15%
LoadSwitch condition, fixed fault parameters, no noise, and limited operating
condition diversity.

## Supplementary localization diagnostics

`Fig03`, `Fig04`, and `Fig05` are retained only as supplementary diagnostics.
Exact-bus localization is weak with the present feature set, while 1-hop,
2-hop, and topology-community results retain some location information. Exact
localization is not claimed as the research contribution.

- `Fig03_localization_tolerance_summary.png`: exact, neighborhood, and community tolerance.
- `Fig04_localization_zone_confusion.png`: confusion across topology communities.
- `Fig05_localization_distance_cdf.png`: graph-distance distribution of localization errors.

The four zones are unweighted-topology communities from NetworkX greedy
modularity detection, not manually selected bus-number groups and not
electrical-distance clusters. See `../reports/zone_definition_notes.md`.

## Limitations

- One SSO magnitude/frequency condition
- One 15% LoadSwitch condition
- Fixed fault parameters
- No measurement noise
- Limited operating-point diversity
- Only three Normal cases

## Next experiments

Vary LoadSwitch magnitude, SSO magnitude/frequency, fault resistance, fault
inception angle, measurement noise, missing channels, and operating point.
Then repeat the placement optimization to test whether the
classification-oriented minimum remains `k=1`.
