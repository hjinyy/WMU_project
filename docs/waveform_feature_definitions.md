# Waveform Feature Definitions

## Event timing assumptions

- `SLG_Fault`, `ThreePhase_Fault`: `t_event = 0.5 s`
- `LoadSwitch`: `t_event = 0.2 s`
- `Normal`: stored as `EventType=Normal`, with `t_event = 0.5 s` used only as a reference split
- Base frequency: `f0 = 50 Hz`
- One-cycle delay for cycle-difference features: `1/f0 = 0.02 s`

## 1. Voltage time-domain features

### `sag_A`, `sag_B`, `sag_C`
Per-phase voltage sag:
- compute one-cycle RMS,
- compare pre-event RMS to minimum post-event RMS,
- `sag = max(0, 1 - Vpost_min / Vpre)`.

Physical meaning:
- captures phase-specific post-event voltage depression.

### `max_sag`, `mean_sag`
Aggregate sag severity across phases.

### `dV_energy_A/B/C`
RMS energy of the one-cycle delayed voltage difference waveform.

Physical meaning:
- measures abrupt waveform change after removing the nominal periodic component one cycle earlier.

### `dV_energy_3ph_mean`, `dV_energy_3ph_max`
Three-phase aggregate of the per-phase cycle-difference voltage energy.

## 2. Current time-domain features

### `dI_energy_A/B/C`
Current counterpart of `dV_energy_*`.

### `dI_energy_3ph_mean`, `dI_energy_3ph_max`
Three-phase aggregate current disturbance strength.

### `I_rms_pre_*`, `I_rms_post_*`
Per-phase one-cycle RMS current before and after the event.

### `I_rms_jump_*`, `I_rms_jump_3ph_max`
Post-minus-pre RMS current change.

Physical meaning:
- highlights load pickup and fault-current injection.

## 3. Frequency-domain features

All frequency features are computed on the cycle-difference waveform in the event window.

### `dV_HF_ratio_*`
Energy ratio in the `200-2000 Hz` band.

Physical meaning:
- captures high-frequency disturbance content from sharp switching/fault transitions.

### `dV_E28_ratio_*`
Energy ratio in the `23-33 Hz` band.

### `dV_E72_ratio_*`
Energy ratio in the `67-77 Hz` band.

### `dV_Res_ratio_*`
`E28 + E72` relative energy.

Physical meaning:
- tracks sideband/residual content around the 50 Hz fundamental.

### `dI_HF_ratio_*`, `dI_Res_ratio_*`
Current-domain versions of the same idea.

## 4. Unbalance and sequence-like features

### `V_unbalance_pre/post`, `I_unbalance_pre/post`
Coefficient-of-variation style unbalance:
- standard deviation of phase RMS values divided by their mean.

### `Delta_V_unbalance`, `Delta_I_unbalance`
Post-event minus pre-event unbalance.

### `V0_ratio`, `I0_ratio`
Zero-sequence RMS divided by the mean per-phase RMS.

Physical meaning:
- useful for distinguishing balanced vs. unbalanced disturbances.

### `V2_ratio`, `I2_ratio`
Approximate negative-sequence to positive-sequence ratio from a fundamental phasor estimate over the post-event window.

## 5. Apparent impedance features

### `Z_app_pre`, `Z_app_post`
Mean phase RMS voltage divided by mean phase RMS current before and after the event.

### `Delta_Z_app`
`Z_app_post - Z_app_pre`

### `Z_drop_ratio`
`(Z_app_pre - Z_app_post) / Z_app_pre`

Physical meaning:
- faults tend to collapse apparent impedance more strongly than switching events.

## 6. Lissajous features

### `liss_corr_abs_A/B/C`
Absolute voltage-current correlation over the event window.

### `liss_area_norm_A/B/C`
Shoelace-based loop area normalized by the voltage-current bounding box.

### `liss_corr_abs_min`, `liss_area_norm_min`
Minimum per-phase values.

Physical meaning:
- summarizes changes in V-I trajectory shape and hysteresis-like distortion.

## 7. Case-level summary features

### `star_bus_dV`, `star_bus_dI`
Observed bus with the largest three-phase voltage/current disturbance energy.

### `max_dV_energy`, `max_dI_energy`, `max_sag`
Case-level peak disturbance indicators.

### `max_HF_ratio`, `max_Res_ratio`
Largest observed spectral disturbance ratio across buses.

### `max_V0_ratio`, `max_I0_ratio`
Largest zero-sequence-style response across buses.

### `max_V_unbalance`, `max_I_unbalance`
Largest observed post-event unbalance.

### `max_Z_drop_ratio`, `min_Z_app_post`
Impedance-collapse summaries.

## 8. Wide case vector

`feature_table_by_case_wide.csv` flattens bus-level features into columns like:
- `Bus01__dV_energy_A`
- `Bus12__V0_ratio`
- `Bus30__Z_drop_ratio`

This table is the main input for classification, sensor-selection, and localization workflows.
