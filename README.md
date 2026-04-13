# WMU Project

Python tooling for WMU observability analysis on IEEE 30-bus SLG fault sweeps.

## Scope

This repository organizes the workflow discussed in the study:

- 10 candidate WMU buses are evaluated against 30 SLG fault cases
- each case stores differential A-phase signals and raw A-phase signals
- the main per-case, per-WMU metrics are:
  - `DV_energy`
  - `Sag`
  - `Res_ratio`
  - `StarWMU`
- coverage and minimum-WMU studies are then derived from those metrics

## Repository layout

- `docs/research_summary.md`
  Research notes, assumptions, metric definitions, and interpretation guide.
- `requirements.txt`
  Python dependencies.
- `data/ieee30_edges.csv`
  IEEE 30-bus graph edges used by the network plot.
- `src/wmu_project/analysis.py`
  Shared loading, metric extraction, coverage, and plotting helpers.
- `scripts/analyze_observability.py`
  End-to-end analysis that writes Excel summaries.
- `scripts/figure1_coverage_vs_num_wmu.py`
  Coverage vs number of WMUs.
- `scripts/figure2_wmu_ranking.py`
  WMU ranking bars for mean DV and Star count.
- `scripts/figure3_case_wmu_heatmap.py`
  Case-by-WMU heatmaps for DV and Sag.
- `scripts/figure4_network_graph.py`
  IEEE 30-bus network plot with WMU nodes and StarWMU counts.
- `scripts/figure5_wmu_boxplot.py`
  WMU DV distribution by case.
- `scripts/figure6_3d_scatter.py`
  3D scatter of `DV_energy_max`, `Sag_max`, `Res_ratio_max`.

## Expected input

Primary input is an Excel workbook like `WMU_fault_results_sag.xlsx` with sheets:

- `Summary`
- `Fault_1` ... `Fault_30`

Each fault sheet is expected to contain:

- `Time_s`
- `dV_<bus>_A`
- `dI_<bus>_A`
- `Vraw_<bus>_A`
- `Iraw_<bus>_A`

for the selected WMU buses.

## Recommended default settings

- WMU buses: `3, 5, 8, 10, 11, 12, 23, 26, 28, 29`
- fault type: `SLG`
- base frequency: `50 Hz`
- event time: `2.2 s`
- tuned detection threshold: `thr_dv = 0.072`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## End-to-end analysis

```bash
python scripts/analyze_observability.py \
  --input /path/to/WMU_fault_results_sag.xlsx \
  --output-dir outputs \
  --thr-dv 0.072
```

## Figure generation

```bash
python scripts/figure1_coverage_vs_num_wmu.py --input /path/to/WMU_fault_results_sag.xlsx --output outputs
python scripts/figure2_wmu_ranking.py --input /path/to/WMU_fault_results_sag.xlsx --output outputs
python scripts/figure3_case_wmu_heatmap.py --input /path/to/WMU_fault_results_sag.xlsx --output outputs
python scripts/figure4_network_graph.py --input /path/to/WMU_fault_results_sag.xlsx --output outputs
python scripts/figure5_wmu_boxplot.py --input /path/to/WMU_fault_results_sag.xlsx --output outputs
python scripts/figure6_3d_scatter.py --input /path/to/WMU_fault_results_sag.xlsx --output outputs
```

## Interpretation targets

The workflow is designed to support the following expected patterns:

- `Coverage vs #WMU` should look like a saturating staircase.
- `StarWMU` should be distributed across several buses rather than collapse to a single one.
- `Case x WMU` heatmaps should show block or zone structure.
- greedy or set-cover results should expose a core group of hub sensors and a smaller set of peripheral sensors.
- robust one-drop requirements should demand one or two more WMUs than plain coverage.
