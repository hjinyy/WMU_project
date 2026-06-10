from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "waveform_ibr_background_diagnostics"
REPORTS = RESULTS / "reports"
FINAL = RESULTS / "hard_constraint_final_figures"
DEFAULT_DATA = Path(
    "/mnt/c/Users/user/Documents/MATLAB/WMU_final/"
    "WMU_batch_data_ibr_background"
)
DEFAULT_DELIVERY = Path("/mnt/c/Users/user/Documents/MATLAB! WMU_final")
CLASS_ORDER = [
    "SSO_Normal",
    "SSO_LoadSwitch",
    "SSO_SLG_Fault",
    "SSO_ThreePhase_Fault",
]
CLASS_LABELS = ["Normal", "Load switch", "SLG fault", "3-phase fault"]
PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
FIGURES = [
    "Fig01_fault_nonfault_feature_separation.png",
    "Fig02_minimum_wmu_hard_constraint.png",
    "Fig03_selected_wmu_binary_confusion.png",
    "Fig04_per_fault_bus_coverage.png",
    "Fig05_margin_or_safety_map.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--delivery-dir", type=Path, default=DEFAULT_DELIVERY)
    return parser.parse_args()


def configure() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, filename: str, delivery: Path) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL / filename, facecolor="white")
    fig.savefig(delivery / filename, facecolor="white")
    plt.close(fig)


def class_boxplot(
    ax: plt.Axes,
    frame: pd.DataFrame,
    feature: str,
    title: str,
    ylabel: str,
    log: bool = False,
) -> None:
    sns.boxplot(
        data=frame,
        x="EventType",
        y=feature,
        order=CLASS_ORDER,
        hue="EventType",
        palette=PALETTE,
        legend=False,
        fliersize=0,
        width=0.58,
        ax=ax,
    )
    sns.stripplot(
        data=frame,
        x="EventType",
        y=feature,
        order=CLASS_ORDER,
        color="#202020",
        alpha=0.5,
        size=3,
        jitter=0.18,
        ax=ax,
    )
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(4), CLASS_LABELS, rotation=17, ha="right")
    ax.grid(axis="x", visible=False)


def figure01(data_dir: Path, delivery: Path) -> None:
    selected = pd.read_csv(REPORTS / "selected_minimum_wmu_set.csv").iloc[0]
    bus = int(str(selected["WMUSet"]).split("|")[0])
    by_bus = pd.read_csv(data_dir / "feature_table_by_bus.csv")
    frame = (
        by_bus.loc[by_bus["ObservedBus"] == bus]
        .drop_duplicates("CaseName")
        .sort_values("CaseName")
        .reset_index(drop=True)
    )
    frame["Balanced_E72_ratio"] = frame[
        ["dV_E72_ratio_A", "dV_E72_ratio_B", "dV_E72_ratio_C"]
    ].min(axis=1)
    predictions = pd.read_csv(
        REPORTS / "selected_minimum_wmu_case_predictions.csv"
    )
    score_plot = predictions.copy()
    score_plot["BinaryClass"] = score_plot["BinaryFaultLabel"].map(
        {0: "Non-fault", 1: "Fault"}
    )
    threshold = float(selected["BestThreshold"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    class_boxplot(
        axes[0, 0],
        frame,
        "Balanced_E72_ratio",
        "(a) LoadSwitch rejection: balanced 67–77 Hz dV ratio",
        "Minimum phase ratio",
    )
    class_boxplot(
        axes[0, 1],
        frame,
        "dV_energy_3ph_max",
        "(b) Fault severity: voltage disturbance energy",
        "dV energy (3-phase max)",
        log=True,
    )
    class_boxplot(
        axes[1, 0],
        frame,
        "I0_ratio",
        "(c) Fault sequence evidence: zero-sequence current",
        "I0 ratio",
    )
    sns.boxplot(
        data=score_plot,
        x="BinaryClass",
        y="FaultScore",
        order=["Non-fault", "Fault"],
        palette=["#4C78A8", "#E45756"],
        hue="BinaryClass",
        legend=False,
        fliersize=0,
        ax=axes[1, 1],
    )
    sns.stripplot(
        data=score_plot,
        x="BinaryClass",
        y="FaultScore",
        order=["Non-fault", "Fault"],
        color="#202020",
        alpha=0.55,
        jitter=0.18,
        size=3,
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(
        threshold,
        color="#111111",
        linestyle="--",
        label=f"Threshold = {threshold:.3f}",
    )
    axes[1, 1].set_title(
        "(d) Fixed physical FaultScore separation",
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("FaultScore")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(axis="x", visible=False)
    fig.suptitle(
        f"Fault/non-fault waveform-feature separation at selected WMU Bus {bus}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, FIGURES[0], delivery)


def figure02(delivery: Path) -> None:
    counts = pd.read_csv(REPORTS / "hard_constraint_feasible_counts_by_k.csv")
    selected = pd.read_csv(REPORTS / "selected_minimum_wmu_set.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    bars = ax.bar(
        counts["k"],
        counts["FeasibleSubsetCount"],
        color=["#E45756", "#72B7B2", "#4C78A8", "#B279A2"],
        width=0.65,
    )
    ax.bar_label(bars, padding=3)
    ax.set(
        title="Exhaustive hard-constraint search identifies the minimum WMU count",
        xlabel="Number of selected WMUs (k)",
        ylabel="Feasible subset count",
        xticks=counts["k"],
    )
    ax.annotate(
        f"First feasible level: k = {int(selected['k'])}\n"
        f"Selected set: Bus {selected['WMUSet']}\n"
        f"Margin: {selected['FaultScoreMargin']:.3f}",
        xy=(1, counts.loc[counts["k"] == 1, "FeasibleSubsetCount"].iloc[0]),
        xytext=(1.55, max(counts["FeasibleSubsetCount"]) * 0.78),
        arrowprops={"arrowstyle": "->", "color": "#222222"},
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, FIGURES[1], delivery)


def figure03(delivery: Path) -> None:
    predictions = pd.read_csv(
        REPORTS / "selected_minimum_wmu_case_predictions.csv"
    )
    matrix = pd.crosstab(
        predictions["BinaryFaultLabel"],
        predictions["PredictedBinaryFaultLabel"],
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        xticklabels=["Pred non-fault", "Pred fault"],
        yticklabels=["True non-fault", "True fault"],
        ax=ax,
    )
    ax.set(
        title=(
            "Selected minimum WMU set: binary hard-constraint result\n"
            "Normal + LoadSwitch false alarms = 0; missed faults = 0"
        ),
        xlabel="Prediction",
        ylabel="Truth",
    )
    fig.tight_layout()
    save(fig, FIGURES[2], delivery)


def figure04(delivery: Path) -> None:
    predictions = pd.read_csv(
        REPORTS / "selected_minimum_wmu_case_predictions.csv"
    )
    fault = predictions.loc[predictions["BinaryFaultLabel"] == 1].copy()
    fault["TargetBus"] = fault["TargetBus"].astype(int)
    coverage = fault.pivot(
        index="TargetBus",
        columns="EventType",
        values="PredictedBinaryFaultLabel",
    ).reindex(range(1, 31))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(1, 31)
    ax.scatter(
        x,
        coverage["SSO_SLG_Fault"],
        marker="o",
        s=52,
        color="#E45756",
        label="SLG fault",
    )
    ax.scatter(
        x,
        coverage["SSO_ThreePhase_Fault"] + 0.035,
        marker="s",
        s=45,
        color="#72B7B2",
        label="3-phase fault",
    )
    ax.set(
        title="Selected minimum WMU detects both fault types at every fault bus",
        xlabel="Fault target bus",
        ylabel="Detected",
        xticks=range(1, 31),
        yticks=[0, 1],
        yticklabels=["Missed", "Detected"],
        ylim=(-0.1, 1.15),
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, FIGURES[3], delivery)


def figure05(delivery: Path) -> None:
    margins = pd.read_csv(REPORTS / "fault_score_margin_by_single_wmu.csv")
    margins["Bus"] = margins["WMUSet"].astype(int)
    margins = margins.sort_values("FaultScoreMargin", ascending=False)
    colors = np.where(
        margins["Feasible"], "#4C78A8", "#BAB0AC"
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(
        margins["Bus"].astype(str),
        margins["FaultScoreMargin"],
        color=colors,
    )
    selected_index = margins.index[margins["Bus"] == 5][0]
    position = margins.index.get_loc(selected_index)
    bars[position].set_color("#E45756")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set(
        title="Single-WMU safety margin ranking for the fixed physical rule",
        xlabel="Observed WMU bus (sorted by margin)",
        ylabel="Separation margin (fault min − non-fault max)",
    )
    ax.tick_params(axis="x", rotation=90)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#E45756", label="Selected Bus 5"),
            plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="Other feasible"),
            plt.Rectangle((0, 0), 1, 1, color="#BAB0AC", label="Infeasible"),
        ],
        loc="lower left",
    )
    fig.tight_layout()
    save(fig, FIGURES[4], delivery)


def write_readme(delivery: Path) -> None:
    selected = pd.read_csv(REPORTS / "selected_minimum_wmu_set.csv").iloc[0]
    counts = pd.read_csv(REPORTS / "hard_constraint_feasible_counts_by_k.csv")
    count_text = ", ".join(
        f"k={int(row.k)}: {int(row.FeasibleSubsetCount)}"
        for row in counts.itertuples(index=False)
    )
    readme = f"""# Hard-constraint minimum WMU placement

## 1. Research objective

This analysis does not optimize simple event detection, a macro-F1 curve, or
exact fault-bus localization. It finds the smallest WMU set that detects every
fault while never classifying Normal or LoadSwitch as a fault.

## 2. Problem definition

`BinaryFaultLabel = 0` for `SSO_Normal` and `SSO_LoadSwitch`.
`BinaryFaultLabel = 1` for `SSO_SLG_Fault` and
`SSO_ThreePhase_Fault`.

A placement is feasible only when all four hard constraints hold:

1. Normal false positives = 0
2. LoadSwitch false positives = 0
3. SLG false negatives = 0
4. Three-phase-fault false negatives = 0

## 3. Dataset

- 84 cases: 3 Normal, 21 LoadSwitch, 30 SLG fault, 30 three-phase fault
- 30 observed WMU buses per case
- Existing feature tables only; Simulink and raw-data generation were not run

## 4. Feature strategy

The fixed physical score combines normalized fault evidence from voltage/current
disturbance energy, sag, RMS jump, sequence/unbalance, and apparent-impedance
features. LoadSwitch rejection uses the balanced 67–77 Hz voltage
cycle-difference ratio:

`FaultScore = max(fault evidence) - 0.5 * LoadSwitchEvidence`

Each feature is normalized independently per observed bus using label-independent
5th and 95th percentiles. `LoadSwitchEvidence` is the minimum of the three phase
67–77 Hz ratios, because the current deterministic LoadSwitch response is high
on all three phases. The coefficient 0.5 is fixed globally and is not optimized
per subset.

## 5. Exhaustive subset search

All combinations were evaluated through k=4:

- {count_text}

The first feasible level is k={int(selected['k'])}.
Feasible counts need not increase with k because the fixed set score takes the
maximum fault evidence and maximum LoadSwitch evidence across selected WMUs;
adding a WMU can increase the LoadSwitch penalty and reduce the separation
margin.

## 6. Final result

- Selected WMU set: Bus {selected['WMUSet']}
- Normal FP: {int(selected['NormalFP'])}
- LoadSwitch FP: {int(selected['LoadSwitchFP'])}
- SLG FN: {int(selected['SLGFN'])}
- Three-phase FN: {int(selected['ThreePhaseFN'])}
- Fault recall: {selected['FaultRecall']:.3f}
- Non-fault specificity: {selected['NonFaultSpecificity']:.3f}
- LoadSwitch specificity: {selected['LoadSwitchSpecificity']:.3f}
- Separation margin: {selected['FaultScoreMargin']:.6f}
- Threshold: {selected['BestThreshold']:.6f}

The threshold result is an in-dataset physical separability result. ML
leave-one-case-out, leave-target-location-out, and stratified three-fold results
are reported separately as supporting checks; they do not define the minimum.

## 7. Final figures

1. `Fig01_fault_nonfault_feature_separation.png`
2. `Fig02_minimum_wmu_hard_constraint.png`
3. `Fig03_selected_wmu_binary_confusion.png`
4. `Fig04_per_fault_bus_coverage.png`
5. `Fig05_margin_or_safety_map.png`

Previous localization figures are excluded from this main result. They remain
available only in the pre-existing diagnostics/archive directories.

## 8. Limitations

- One deterministic IBR-like SSO condition
- One 15% LoadSwitch condition
- Fixed fault resistance and inception conditions
- No measurement noise or missing channels
- Limited operating-point diversity
- Only three Normal cases
- The physical threshold and normalization were assessed on the current dataset;
  external-condition robustness is not yet established

## 9. Next steps

Repeat the complete hard-constraint subset search after varying LoadSwitch
magnitude, fault resistance, inception angle, SSO amplitude/frequency,
measurement noise, missing channels, and operating point. The key question is
whether Bus {selected['WMUSet']} and k={int(selected['k'])} remain feasible.
"""
    (FINAL / "README.md").write_text(readme)
    shutil.copy2(FINAL / "README.md", delivery / "README.md")


def main() -> None:
    args = parse_args()
    configure()
    FINAL.mkdir(parents=True, exist_ok=True)
    args.delivery_dir.mkdir(parents=True, exist_ok=True)
    for filename in FIGURES:
        (FINAL / filename).unlink(missing_ok=True)
        (args.delivery_dir / filename).unlink(missing_ok=True)
    figure01(args.data_dir, args.delivery_dir)
    figure02(args.delivery_dir)
    figure03(args.delivery_dir)
    figure04(args.delivery_dir)
    figure05(args.delivery_dir)
    write_readme(args.delivery_dir)


if __name__ == "__main__":
    main()
