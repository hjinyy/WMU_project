from __future__ import annotations

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
FINAL = RESULTS / "final_figures"
DATA = Path(
    "/mnt/c/Users/user/Documents/MATLAB/! WMU_final/"
    "WMU_batch_data_ibr_background"
)
DELIVERY = Path("/mnt/c/Users/user/Documents/MATLAB/! WMU_final")
CLASS_ORDER = [
    "SSO_Normal",
    "SSO_LoadSwitch",
    "SSO_SLG_Fault",
    "SSO_ThreePhase_Fault",
]
CLASS_LABELS = ["Normal", "Load switch", "SLG fault", "3-phase fault"]
PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
FINAL_FILES = [
    "Fig01_feature_separation_panel.png",
    "Fig02_sensor_count_saturation.png",
    "Fig03_localization_tolerance_summary.png",
    "Fig04_localization_zone_confusion.png",
    "Fig05_localization_distance_cdf.png",
]


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL / filename, facecolor="white")
    fig.savefig(DELIVERY / filename, facecolor="white")
    plt.close(fig)


def bus01_features() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "feature_table_by_bus.csv")
    return frame.loc[frame["ObservedBus"] == 1].drop_duplicates("CaseName")


def figure01() -> None:
    frame = bus01_features()
    panels = [
        (
            "dV_E72_ratio_A",
            "Load-switch signature: 67–77 Hz dV ratio",
            None,
        ),
        (
            "dV_energy_3ph_mean",
            "Fault severity: voltage disturbance energy",
            "log",
        ),
        (
            "dV_SSO20_30_ratio_A",
            "SSO-band response: 20–30 Hz dV ratio",
            None,
        ),
        (
            "I0_ratio",
            "Fault subtype: zero-sequence current ratio",
            None,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for index, (ax, (feature, title, scale)) in enumerate(
        zip(axes.flat, panels), start=1
    ):
        plot = frame[["EventType", feature]].dropna()
        sns.boxplot(
            data=plot,
            x="EventType",
            y=feature,
            order=CLASS_ORDER,
            hue="EventType",
            palette=PALETTE,
            legend=False,
            width=0.58,
            fliersize=0,
            linewidth=1,
            ax=ax,
        )
        sns.stripplot(
            data=plot,
            x="EventType",
            y=feature,
            order=CLASS_ORDER,
            color="#202020",
            alpha=0.52,
            size=3,
            jitter=0.18,
            ax=ax,
        )
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(f"({chr(96 + index)}) {title}", loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(feature.replace("_", " "))
        ax.set_xticks(range(4), CLASS_LABELS, rotation=17, ha="right")
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Representative waveform features explain event-class separation",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "All panels use existing features observed at Bus 1, the k=1 "
        "feature-aware greedy placement.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, FINAL_FILES[0])


def figure02() -> None:
    frame = pd.read_csv(REPORTS / "sensor_count_extended_summary.csv")
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    methods = [
        ("feature-aware greedy (existing LOO)", "Feature-aware greedy", "#4C78A8", "o"),
        ("grouped CV feature-aware rank", "Leave-target-location-out", "#E45756", "s"),
        ("random mean", "Random subsets (mean)", "#72B7B2", "^"),
        ("single worst", "Worst single WMU", "#9C755F", "D"),
    ]
    for method, label, color, marker in methods:
        part = frame.loc[frame["Method"] == method].sort_values("k")
        ax.plot(
            part["k"],
            part["macro_f1"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1)
    ax.annotate(
        "Minimum observed count: k = 1 (Bus 1)\n"
        "with classification performance maintained",
        xy=(1, 1.0),
        xytext=(3.1, 0.986),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=9.5,
    )
    ax.set(
        title=(
            "Event classification performance saturates with one WMU under the\n"
            "current deterministic IBR-SSO dataset"
        ),
        xlabel="Number of selected WMUs (k)",
        ylabel="Four-class macro-F1",
        xlim=(0.8, 10.2),
        ylim=(0.975, 1.0015),
        xticks=range(1, 11),
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    save(fig, FINAL_FILES[1])


def figure03() -> None:
    metrics = pd.read_csv(REPORTS / "localization_distance_metrics.csv").iloc[0]
    labels = ["Exact bus", "Top-3 bus", "≤1 hop", "≤2 hops", "Zone"]
    values = [
        metrics["exact_accuracy"],
        metrics["top3_accuracy"],
        metrics["one_hop_accuracy"],
        metrics["two_hop_accuracy"],
        metrics["zone_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(
        labels,
        values,
        color=["#BAB0AC", "#BAB0AC", "#4C78A8", "#72B7B2", "#F58518"],
        edgecolor="white",
    )
    ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3)
    ax.set(
        title="Supplementary localization diagnostic: tolerance summary",
        ylabel="Accuracy",
        ylim=(0, 0.9),
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, FINAL_FILES[2])


def figure04() -> None:
    matrix = pd.read_csv(
        REPORTS / "localization_zone_confusion_matrix.csv", index_col=0
    )
    metrics = pd.read_csv(REPORTS / "localization_zone_metrics.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Row-normalized fraction"},
        ax=ax,
    )
    ax.set(
        title=(
            "Supplementary localization diagnostic: zone confusion\n"
            f"accuracy = {metrics['zone_accuracy']:.1%}, "
            f"macro-F1 = {metrics['zone_macro_f1']:.1%}"
        ),
        xlabel="Predicted topology community",
        ylabel="True topology community",
    )
    fig.tight_layout()
    save(fig, FINAL_FILES[3])


def figure05() -> None:
    frame = pd.read_csv(REPORTS / "localization_per_case_distance.csv")
    distances = np.sort(frame["GraphDistance"].to_numpy())
    cdf = np.arange(1, len(distances) + 1) / len(distances)
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    ax.step(distances, cdf, where="post", color="#4C78A8", linewidth=2.4)
    ax.scatter(
        [1, 2],
        [(distances <= 1).mean(), (distances <= 2).mean()],
        color=["#4C78A8", "#72B7B2"],
        zorder=3,
    )
    ax.annotate(
        "46.7% within 1 hop",
        xy=(1, (distances <= 1).mean()),
        xytext=(1.35, 0.34),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        "80.0% within 2 hops",
        xy=(2, (distances <= 2).mean()),
        xytext=(2.35, 0.68),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set(
        title="Supplementary localization diagnostic: graph-distance CDF",
        xlabel="Graph-distance error (hops)",
        ylabel="Empirical cumulative probability",
        xlim=(0.8, max(5.2, distances.max() + 0.2)),
        ylim=(0, 1.03),
        xticks=range(1, int(distances.max()) + 1),
    )
    fig.tight_layout()
    save(fig, FINAL_FILES[4])


def write_zone_definition() -> None:
    source = pd.read_csv(REPORTS / "localization_zone_definition.csv")
    source = source.rename(columns={"Zone": "Community"})
    source["DefinitionBasis"] = (
        "Unweighted IEEE-30 topology; NetworkX greedy modularity communities"
    )
    source["ElectricalDistanceUsed"] = False
    source.to_csv(REPORTS / "zone_definition_final.csv", index=False)
    groups = source.groupby("Community")["Bus"].apply(list)
    lines = [
        "# Final zone/community definition",
        "",
        "The four localization zones were not defined by bus-number ranges and were "
        "not tuned to improve localization accuracy. They were generated from "
        "`data/ieee30_edges.csv` using NetworkX greedy modularity community "
        "detection on the unweighted IEEE 30-bus topology graph.",
        "",
        "## Membership",
        "",
    ]
    lines.extend(
        f"- **{community}:** {', '.join(map(str, buses))}"
        for community, buses in groups.items()
    )
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "This is a topology-community partition, not an electrical-distance "
            "clustering. Line impedance, operating point, voltage sensitivity, and "
            "IBR dynamics are not used. It is acceptable as a transparent "
            "supplementary diagnostic, but it should not be interpreted as an "
            "optimized protection-zone definition. An electrical-distance study "
            "would be more physically grounded, but must be fixed independently of "
            "localization labels and evaluated without tuning zones to improve the result.",
        ]
    )
    (REPORTS / "zone_definition_notes.md").write_text("\n".join(lines) + "\n")


def write_readme() -> None:
    readme = """# Final figures: classification-oriented minimum WMU placement

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
"""
    (FINAL / "README.md").write_text(readme)


def main() -> None:
    style()
    FINAL.mkdir(parents=True, exist_ok=True)
    for filename in FINAL_FILES:
        (FINAL / filename).unlink(missing_ok=True)
        (DELIVERY / filename).unlink(missing_ok=True)
    figure01()
    figure02()
    figure03()
    figure04()
    figure05()
    write_zone_definition()
    write_readme()
    shutil.copy2(FINAL / "README.md", DELIVERY / "README_final_figures.md")


if __name__ == "__main__":
    main()
