# Research Summary

## Problem setting

We evaluate a fixed set of 10 WMU candidate buses on IEEE 30-bus SLG fault cases. For each fault case, the WMUs observe:

- differential voltage signal `dV`
- differential current signal `dI`
- raw voltage signal `Vraw`
- raw current signal `Iraw`

The goal is to answer:

1. How well does the chosen WMU set detect faults?
2. Which WMUs are the most informative?
3. What is the minimum number of WMUs needed for a target coverage?
4. Which WMUs act as proximity proxies for different fault zones?

## Core metrics

### DV_energy

`DV_energy(i, j)` is the RMS energy of `dV_(bus)_A` for case `i` and WMU `j` over the saved post-fault window.

Interpretation:

- higher value means the WMU reacts more strongly to the fault
- useful for detection and for choosing `StarWMU`

### Sag

`Sag(i, j)` is derived from `Vraw_(bus)_A`:

1. estimate one-cycle RMS from raw A-phase voltage
2. compute pre-event RMS average
3. compute post-event minimum RMS
4. define sag as:

`Sag = max(0, 1 - Vpost_min / Vpre_mean)`

Interpretation:

- higher value means deeper voltage drop
- complements `DV_energy`

### Res_ratio

`Res_ratio(i, j)` is computed from the FFT of post-event `dV_(bus)_A`:

- numerator: energy in `23-33 Hz` and `67-77 Hz`
- denominator: total positive-frequency energy

Interpretation:

- helps separate cases with similar amplitude but different spectral shape

### StarWMU

For case `i`:

`StarWMU(i) = argmax_j DV_energy(i, j)`

Interpretation:

- proxy for the electrically closest or most dominant sensor for that fault

## Coverage definition

For threshold `thr_dv`:

`Detected(i) = max_j DV_energy(i, j) > thr_dv`

Coverage for a WMU subset `S`:

`Coverage(S) = mean(max_{j in S} DV_energy(i, j) > thr_dv)`

## Minimum-WMU logic

### A. Coverage-based minimum

Find the smallest WMU subset such that:

- `Coverage >= 95%`
- or `Coverage >= 99%`

### B. Greedy / set-cover style selection

Add WMUs one by one to maximize marginal gain in coverage.

Expected behavior:

- first sensors cover hub or central fault regions
- later sensors mostly recover edge or weak cases

### C. Robust one-drop minimum

Require that even after removing any one WMU from the chosen subset:

- coverage still stays above the target

This is stricter and usually increases the minimum count.

## Recommended figures

### Figure 1

Coverage vs number of WMUs.

### Figure 2

WMU ranking bars:

- mean `DV_energy`
- `StarWMU` count

### Figure 3

Case x WMU heatmaps:

- `DV_energy`
- `Sag`

### Figure 4

IEEE 30-bus network graph:

- candidate WMU buses marked
- node intensity by `StarWMU` count

### Figure 5

WMU DV distribution across cases.

### Figure 6

3D scatter of:

- `DV_energy_max`
- `Sag_max`
- `Res_ratio_max`

colored by `StarWMU`.

## Tuned threshold note

Using a very low threshold such as `0.02` can saturate coverage and hide the staircase behavior. A tuned threshold near `0.072` was found to produce a more informative progression:

- lower coverage for 1 sensor
- rapid rise for 2-4 sensors
- diminishing return after 5-6 sensors

This better supports minimum-WMU discussion.
