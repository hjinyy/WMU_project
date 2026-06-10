from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "waveform_ibr_background_diagnostics"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "bus5_explanation_figures"
DEFAULT_DATA = Path(
    "/mnt/c/Users/user/Documents/MATLAB/WMU_final/"
    "WMU_batch_data_ibr_background"
)
DEFAULT_DELIVERY = Path("/mnt/c/Users/user/Documents/MATLAB/WMU_final")
REQUIRED_REPORTS = [
    "single_wmu_hard_constraint_results.csv",
    "fault_score_margin_by_single_wmu.csv",
    "exhaustive_subset_hard_constraint_results.csv",
    "feasible_minimum_wmu_sets.csv",
    "selected_minimum_wmu_set.csv",
    "selected_minimum_wmu_case_predictions.csv",
    "hard_constraint_feature_mapping.csv",
    "hard_constraint_fault_scores_by_case_bus.csv",
]
FEATURES = {
    "Balanced 67-77 Hz dV ratio": [
        "dV_E72_ratio_A",
        "dV_E72_ratio_B",
        "dV_E72_ratio_C",
    ],
    "dV E72 ratio": ["dV_E72_ratio_A"],
    "60-80 Hz dV ratio proxy": ["dV_E72_ratio_A"],
    "70-80 Hz dV ratio proxy": ["dV_E72_ratio_A"],
    "I RMS jump": ["I_rms_jump_3ph_max"],
    "Switching-window dV energy": ["dV_energy_3ph_max"],
    "dV energy": ["dV_energy_3ph_max"],
    "dI energy": ["dI_energy_3ph_max"],
    "Maximum sag": ["max_sag"],
    "V RMS drop proxy": ["max_sag"],
    "Delta Z apparent": ["Delta_Z_app"],
    "I0 ratio": ["I0_ratio"],
    "I2 ratio": ["I2_ratio"],
    "V0 ratio": ["V0_ratio"],
    "V2 ratio": ["V2_ratio"],
    "Delta I unbalance": ["Delta_I_unbalance"],
    "Delta V unbalance": ["Delta_V_unbalance"],
}
CLASS_LABELS = {
    "SSO_Normal": "Normal",
    "SSO_LoadSwitch": "Load switch",
    "SSO_SLG_Fault": "SLG fault",
    "SSO_ThreePhase_Fault": "3-phase fault",
}


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
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def write_inventory(data_dir: Path) -> None:
    paths = [REPORTS / name for name in REQUIRED_REPORTS] + [
        data_dir / "feature_table_by_bus.csv",
        data_dir / "dataset_metadata.csv",
        ROOT / "data" / "ieee30_edges.csv",
    ]
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        rows.append(
            {
                "FileName": path.name,
                "Path": str(path),
                "Rows": len(frame),
                "Columns": len(frame.columns),
                "Bytes": path.stat().st_size,
                "Purpose": {
                    "single_wmu_hard_constraint_results.csv": "k=1 metrics and feasibility",
                    "fault_score_margin_by_single_wmu.csv": "single-WMU score boundaries",
                    "exhaustive_subset_hard_constraint_results.csv": "minimum-cardinality evidence",
                    "feasible_minimum_wmu_sets.csv": "six feasible k=1 candidates",
                    "selected_minimum_wmu_set.csv": "selected Bus 5 rule and margin",
                    "selected_minimum_wmu_case_predictions.csv": "Bus 5 case predictions",
                    "hard_constraint_feature_mapping.csv": "physical feature mapping",
                    "hard_constraint_fault_scores_by_case_bus.csv": "per-case/per-bus score components",
                    "feature_table_by_bus.csv": "waveform feature distributions",
                    "dataset_metadata.csv": "load-switch targets and load magnitude proxy",
                    "ieee30_edges.csv": "unweighted IEEE-30 topology",
                }.get(path.name, "input"),
            }
        )
    pd.DataFrame(rows).to_csv(
        REPORTS / "bus5_explanation_input_inventory.csv", index=False
    )


def enrich_single_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    single = pd.read_csv(REPORTS / "single_wmu_hard_constraint_results.csv")
    single["Bus"] = single["WMUSet"].astype(int)
    scores = pd.read_csv(REPORTS / "hard_constraint_fault_scores_by_case_bus.csv")
    score_summary = (
        scores.groupby(["ObservedBus", "EventType"])["FaultScore"]
        .agg(["min", "max", "median"])
        .reset_index()
    )
    pivot = score_summary.pivot(
        index="ObservedBus", columns="EventType", values=["min", "max", "median"]
    )
    pivot.columns = [f"{stat}_{event}" for stat, event in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"ObservedBus": "Bus"})
    single = single.merge(pivot, on="Bus", how="left")
    single["NormalMaxScore"] = single["max_SSO_Normal"]
    single["LoadSwitchMaxScore"] = single["max_SSO_LoadSwitch"]
    single["SLGMinScore"] = single["min_SSO_SLG_Fault"]
    single["ThreePhaseMinScore"] = single["min_SSO_ThreePhase_Fault"]
    single["Selected"] = single["Bus"] == 5
    feasible = single.loc[single["Feasible"]].copy().sort_values(
        "FaultScoreMargin", ascending=False
    )
    feasible[
        [
            "Bus",
            "NormalFP",
            "LoadSwitchFP",
            "SLGFN",
            "ThreePhaseFN",
            "FaultRecall",
            "NonFaultSpecificity",
            "LoadSwitchSpecificity",
            "FaultMinScore",
            "NonFaultMaxScore",
            "FaultScoreMargin",
            "BestThreshold",
            "LoadSwitchMaxScore",
            "NormalMaxScore",
            "SLGMinScore",
            "ThreePhaseMinScore",
            "Selected",
        ]
    ].to_csv(REPORTS / "feasible_single_wmu_comparison.csv", index=False)
    return single, scores


def failure_reason(row: pd.Series) -> str:
    failures = []
    if row["NormalFP"] > 0:
        failures.append("Normal false alarm")
    if row["LoadSwitchFP"] > 0:
        failures.append("LoadSwitch false alarm")
    if row["SLGFN"] > 0:
        failures.append("SLG missed fault")
    if row["ThreePhaseFN"] > 0:
        failures.append("ThreePhase missed fault")
    if not failures:
        return "Feasible"
    return failures[0] if len(failures) == 1 else "Combined failure"


def write_failure_analysis(single: pd.DataFrame) -> pd.DataFrame:
    out = single.copy()
    out["DominantFailureReason"] = out.apply(failure_reason, axis=1)
    out[
        [
            "Bus",
            "Feasible",
            "NormalFP",
            "LoadSwitchFP",
            "SLGFN",
            "ThreePhaseFN",
            "DominantFailureReason",
            "FaultScoreMargin",
            "NonFaultMaxScore",
            "FaultMinScore",
        ]
    ].to_csv(
        REPORTS / "single_wmu_failure_reason_analysis.csv", index=False
    )
    return out


def aggregate_feature(row: pd.Series, columns: list[str], name: str) -> float:
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(float)
    if name == "Balanced 67-77 Hz dV ratio":
        return float(np.nanmin(values))
    if name in {
        "I RMS jump",
        "Delta Z apparent",
        "Delta I unbalance",
        "Delta V unbalance",
    }:
        return float(np.nanmax(np.abs(values)))
    return float(np.nanmax(values))


def feature_level_comparison(
    by_bus: pd.DataFrame, single: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    feasible_buses = set(single.loc[single["Feasible"], "Bus"].astype(int))
    rows = []
    for record in by_bus.itertuples(index=False):
        series = pd.Series(record._asdict())
        bus = int(series["ObservedBus"])
        group = (
            "Selected Bus 5"
            if bus == 5
            else (
                "Other feasible single-WMU buses"
                if bus in feasible_buses
                else "Infeasible single-WMU buses"
            )
        )
        binary = (
            "Fault" if series["EventType"] in {
                "SSO_SLG_Fault",
                "SSO_ThreePhase_Fault",
            } else "Non-fault"
        )
        for name, columns in FEATURES.items():
            rows.append(
                {
                    "BusGroup": group,
                    "Bus": bus,
                    "CaseName": series["CaseName"],
                    "EventType": series["EventType"],
                    "BinaryClass": binary,
                    "Feature": name,
                    "Value": aggregate_feature(series, columns, name),
                    "MappedColumns": "|".join(columns),
                }
            )
    feature_rows = pd.DataFrame(rows)
    score_copy = scores.copy()
    score_copy["Bus"] = score_copy["ObservedBus"].astype(int)
    score_copy["BusGroup"] = score_copy["Bus"].map(
        lambda bus: (
            "Selected Bus 5"
            if bus == 5
            else (
                "Other feasible single-WMU buses"
                if bus in feasible_buses
                else "Infeasible single-WMU buses"
            )
        )
    )
    score_copy["BinaryClass"] = score_copy["BinaryFaultLabel"].map(
        {0: "Non-fault", 1: "Fault"}
    )
    score_rows = score_copy.rename(columns={"FaultScore": "Value"})[
        ["BusGroup", "Bus", "CaseName", "EventType", "BinaryClass", "Value"]
    ]
    score_rows["Feature"] = "FaultScore"
    score_rows["MappedColumns"] = "fixed hard-constraint score"
    combined = pd.concat([feature_rows, score_rows], ignore_index=True)
    summary = (
        combined.groupby(["BusGroup", "Feature", "EventType", "BinaryClass"])
        ["Value"]
        .agg(Mean="mean", Median="median", IQR=lambda x: x.quantile(.75)-x.quantile(.25),
             Minimum="min", Maximum="max", Count="size")
        .reset_index()
    )
    summary.to_csv(
        REPORTS / "bus5_feature_level_comparison.csv", index=False
    )
    return combined


def topology_metrics(
    single: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    edges = pd.read_csv(ROOT / "data" / "ieee30_edges.csv")
    graph = nx.from_pandas_edgelist(edges, "from_bus", "to_bus")
    degree = dict(graph.degree())
    degree_c = nx.degree_centrality(graph)
    between = nx.betweenness_centrality(graph)
    close = nx.closeness_centrality(graph)
    load_rows = metadata.loc[
        metadata["EventType"] == "SSO_LoadSwitch",
        ["TargetBus", "LoadAddActivePower", "LoadAddInductivePower"],
    ].dropna(subset=["TargetBus"])
    load_rows["TargetBus"] = load_rows["TargetBus"].astype(int)
    load_buses = set(load_rows["TargetBus"])
    threshold = load_rows["LoadAddActivePower"].quantile(.75)
    major_load_buses = set(
        load_rows.loc[
            load_rows["LoadAddActivePower"] >= threshold, "TargetBus"
        ]
    )
    load_map = load_rows.set_index("TargetBus")["LoadAddActivePower"].to_dict()
    rows = []
    for bus in sorted(graph):
        distances = nx.single_source_shortest_path_length(graph, bus)
        rows.append(
            {
                "Bus": bus,
                "Degree": degree[bus],
                "DegreeCentrality": degree_c[bus],
                "BetweennessCentrality": between[bus],
                "ClosenessCentrality": close[bus],
                "AverageShortestPathDistance": np.mean(
                    [distance for target, distance in distances.items() if target != bus]
                ),
                "MaximumShortestPathDistance": max(distances.values()),
                "MinimumDistanceToConfiguredLoadBus": min(
                    distances[target] for target in load_buses
                ),
                "AverageDistanceToConfiguredLoadBuses": np.mean(
                    [distances[target] for target in load_buses]
                ),
                "MinimumDistanceToMajorLoadBus": min(
                    distances[target] for target in major_load_buses
                ),
                "IsConfiguredLoadBus": bus in load_buses,
                "LoadAddActivePower": load_map.get(bus, np.nan),
                "GeneratorOrPVBus": "not available in repository data",
                "IBRCandidateBus": "not available in repository data",
            }
        )
    out = pd.DataFrame(rows).merge(
        single[["Bus", "Feasible", "FaultScoreMargin"]], on="Bus", how="left"
    )
    out["Selected"] = out["Bus"] == 5
    out.to_csv(REPORTS / "bus_topology_centrality_metrics.csv", index=False)
    correlations = {
        metric: float(out[metric].corr(out["FaultScoreMargin"], method="spearman"))
        for metric in [
            "DegreeCentrality",
            "BetweennessCentrality",
            "ClosenessCentrality",
            "AverageShortestPathDistance",
        ]
    }
    return out, correlations


def key_bus_comparison(
    single: pd.DataFrame,
    scores: pd.DataFrame,
    by_bus: pd.DataFrame,
    topology: pd.DataFrame,
) -> pd.DataFrame:
    representative_infeasible = 12
    buses = [5, 1, 6, 27, representative_infeasible]
    score_summary = (
        scores.groupby(["ObservedBus", "EventType"])
        .agg(
            FaultEvidenceMedian=("FaultEvidence", "median"),
            LoadSwitchEvidenceMedian=("LoadSwitchEvidence", "median"),
            FaultScoreMedian=("FaultScore", "median"),
        )
        .reset_index()
    )
    rows = []
    for bus in buses:
        metric = single.loc[single["Bus"] == bus].iloc[0]
        feature = by_bus.loc[by_bus["ObservedBus"] == bus]
        load = feature.loc[feature["EventType"] == "SSO_LoadSwitch"]
        fault = feature.loc[feature["EventType"].isin(
            ["SSO_SLG_Fault", "SSO_ThreePhase_Fault"]
        )]
        topo = topology.loc[topology["Bus"] == bus].iloc[0]
        rows.append(
            {
                "Bus": bus,
                "Role": {
                    5: "selected hard-constraint bus",
                    1: "previous four-class greedy bus",
                    6: "second margin feasible bus",
                    27: "third margin feasible bus",
                    representative_infeasible: "representative combined-failure bus",
                }[bus],
                "Feasible": metric["Feasible"],
                "Margin": metric["FaultScoreMargin"],
                "FaultMinScore": metric["FaultMinScore"],
                "NonFaultMaxScore": metric["NonFaultMaxScore"],
                "LoadSwitchFP": metric["LoadSwitchFP"],
                "MissedFaultCount": metric["SLGFN"] + metric["ThreePhaseFN"],
                "LoadSwitchBalancedE72Median": load[
                    ["dV_E72_ratio_A", "dV_E72_ratio_B", "dV_E72_ratio_C"]
                ].min(axis=1).median(),
                "FaultDVSeverityMedian": fault["dV_energy_3ph_max"].median(),
                "FaultI0RatioMedian": fault["I0_ratio"].median(),
                "Degree": topo["Degree"],
                "BetweennessCentrality": topo["BetweennessCentrality"],
                "ClosenessCentrality": topo["ClosenessCentrality"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(
        REPORTS / "key_bus_comparison_bus5_bus1_others.csv", index=False
    )
    return out


def save(fig: plt.Figure, filename: str, delivery: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, facecolor="white")
    fig.savefig(delivery / filename, facecolor="white")
    plt.close(fig)


def plot_margin(feasible: pd.DataFrame, delivery: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = ["#E45756" if bus == 5 else "#4C78A8" for bus in feasible["Bus"]]
    bars = ax.bar(feasible["Bus"].astype(str), feasible["FaultScoreMargin"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in feasible["FaultScoreMargin"]], padding=3)
    ax.set(
        title="Bus 5 provides the largest safety margin among feasible single-WMU placements",
        xlabel="Feasible single-WMU bus",
        ylabel="Separation margin",
        ylim=(0, feasible["FaultScoreMargin"].max() * 1.25),
    )
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "Fig06_feasible_single_wmu_margin_comparison.png", delivery)


def plot_failure_map(failure: pd.DataFrame, delivery: Path) -> None:
    order = [
        "Feasible",
        "Normal false alarm",
        "LoadSwitch false alarm",
        "SLG missed fault",
        "ThreePhase missed fault",
        "Combined failure",
    ]
    ymap = {name: idx for idx, name in enumerate(order)}
    colors = {
        "Feasible": "#4C78A8",
        "Normal false alarm": "#F58518",
        "LoadSwitch false alarm": "#ECA82C",
        "SLG missed fault": "#E45756",
        "ThreePhase missed fault": "#B279A2",
        "Combined failure": "#79706E",
    }
    fig, ax = plt.subplots(figsize=(12, 5.3))
    for reason, group in failure.groupby("DominantFailureReason"):
        ax.scatter(
            group["Bus"],
            [ymap[reason]] * len(group),
            s=75,
            color=colors[reason],
            label=reason,
            edgecolor="white",
        )
    ax.scatter([5], [ymap["Feasible"]], marker="*", s=280, color="#E45756",
               edgecolor="#222222", label="Selected Bus 5", zorder=5)
    ax.set(
        title="Why single-WMU candidates fail the hard constraints",
        xlabel="Observed WMU bus",
        ylabel="Outcome",
        xticks=range(1, 31),
        yticks=range(len(order)),
        yticklabels=order,
        ylim=(-0.5, len(order)-0.5),
    )
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "Fig07_single_wmu_failure_reason_map.png", delivery)


def plot_feature_explanation(
    by_bus: pd.DataFrame, scores: pd.DataFrame, selected: pd.Series, delivery: Path
) -> None:
    bus = by_bus.loc[by_bus["ObservedBus"] == 5].copy()
    bus["Class"] = bus["EventType"].map(CLASS_LABELS)
    bus["BalancedE72"] = bus[
        ["dV_E72_ratio_A", "dV_E72_ratio_B", "dV_E72_ratio_C"]
    ].min(axis=1)
    bus["AbsDeltaIUnbalance"] = bus["Delta_I_unbalance"].abs()
    score = scores.loc[scores["ObservedBus"] == 5].copy()
    score["BinaryClass"] = score["BinaryFaultLabel"].map({0: "Non-fault", 1: "Fault"})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    panels = [
        ("BalancedE72", "Balanced 67–77 Hz LoadSwitch evidence", False),
        ("dV_energy_3ph_max", "Fault severity: dV energy", True),
        ("I0_ratio", "Fault sequence evidence: I0 ratio", False),
    ]
    for ax, (feature, title, log) in zip(axes.flat[:3], panels):
        sns.boxplot(data=bus, x="Class", y=feature, order=list(CLASS_LABELS.values()),
                    hue="Class", palette=["#4C78A8","#F58518","#E45756","#72B7B2"],
                    legend=False, fliersize=0, ax=ax)
        sns.stripplot(data=bus, x="Class", y=feature, order=list(CLASS_LABELS.values()),
                      color="#222222", alpha=.5, jitter=.18, size=3, ax=ax)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=17)
        ax.grid(axis="x", visible=False)
    ax = axes[1, 1]
    sns.boxplot(data=score, x="BinaryClass", y="FaultScore",
                order=["Non-fault","Fault"], hue="BinaryClass",
                palette=["#4C78A8","#E45756"], legend=False, fliersize=0, ax=ax)
    sns.stripplot(data=score, x="BinaryClass", y="FaultScore",
                  order=["Non-fault","Fault"], color="#222222", alpha=.5,
                  jitter=.18, size=3, ax=ax)
    ax.axhline(selected["BestThreshold"], color="#111111", linestyle="--",
               label=f"Threshold {selected['BestThreshold']:.3f}")
    ax.axhspan(selected["NonFaultMaxScore"], selected["FaultMinScore"],
               color="#59A14F", alpha=.18, label=f"Margin {selected['FaultScoreMargin']:.3f}")
    ax.set_title("Final Bus 5 FaultScore and safety gap", loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.suptitle("How Bus 5 separates LoadSwitch/Normal from faults",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "Fig08_bus5_feature_level_explanation.png", delivery)


def plot_topology(topology: pd.DataFrame, correlations: dict[str, float], delivery: Path) -> None:
    metric = "ClosenessCentrality"
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    feasible = topology["Feasible"]
    ax.scatter(topology.loc[~feasible, metric], topology.loc[~feasible, "FaultScoreMargin"],
               color="#BAB0AC", s=50, label="Infeasible")
    ax.scatter(topology.loc[feasible, metric], topology.loc[feasible, "FaultScoreMargin"],
               color="#4C78A8", s=65, label="Feasible")
    bus5 = topology.loc[topology["Bus"] == 5].iloc[0]
    ax.scatter([bus5[metric]], [bus5["FaultScoreMargin"]], marker="*", s=260,
               color="#E45756", edgecolor="#222222", label="Bus 5", zorder=5)
    for row in topology.itertuples(index=False):
        if row.Bus in {1, 5, 6, 22, 24, 25, 27}:
            ax.annotate(str(row.Bus), (getattr(row, metric), row.FaultScoreMargin),
                        xytext=(4,4), textcoords="offset points", fontsize=8)
    ax.set(
        title="Relationship between topology centrality and hard-constraint margin",
        xlabel="Closeness centrality",
        ylabel="Separation margin",
    )
    ax.text(.02,.02, f"Spearman ρ = {correlations[metric]:.3f}",
            transform=ax.transAxes, bbox={"facecolor":"white","alpha":.85})
    ax.legend()
    fig.tight_layout()
    save(fig, "Fig09_margin_vs_topology_centrality.png", delivery)


def plot_key_buses(key: pd.DataFrame, delivery: Path) -> None:
    order = [5, 1, 6, 27, 12]
    data = key.set_index("Bus").loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    colors = ["#E45756" if bus == 5 else ("#4C78A8" if feasible else "#BAB0AC")
              for bus, feasible in zip(data["Bus"], data["Feasible"])]
    axes[0].bar(data["Bus"].astype(str), data["Margin"], color=colors)
    axes[0].axhline(0, color="#222222", linewidth=1)
    axes[0].set(title="Safety margin", xlabel="Bus", ylabel="Margin")
    x=np.arange(len(data)); width=.36
    axes[1].bar(x-width/2, data["NonFaultMaxScore"], width, label="Non-fault max", color="#4C78A8")
    axes[1].bar(x+width/2, data["FaultMinScore"], width, label="Fault min", color="#E45756")
    axes[1].set(title="Score boundaries", xlabel="Bus", ylabel="FaultScore",
                xticks=x, xticklabels=data["Bus"].astype(str))
    axes[1].legend()
    axes[2].bar(x-width/2, data["LoadSwitchBalancedE72Median"], width,
                label="LoadSwitch 67–77 Hz", color="#F58518")
    axes[2].bar(x+width/2, data["FaultI0RatioMedian"], width,
                label="Fault I0 ratio", color="#72B7B2")
    axes[2].set(title="Representative physical evidence", xlabel="Bus",
                ylabel="Median feature value", xticks=x,
                xticklabels=data["Bus"].astype(str))
    axes[2].legend()
    fig.suptitle("Why Bus 5 is preferred over Bus 1 and other candidates",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "Fig10_key_bus_comparison.png", delivery)


def write_topology_notes(topology: pd.DataFrame, correlations: dict[str, float]) -> None:
    bus5 = topology.loc[topology["Bus"] == 5].iloc[0]
    ranks = topology.rank(ascending=False, method="min")
    lines = [
        "# Bus 5 topology explanation",
        "",
        "Topology was computed only from `data/ieee30_edges.csv` as an unweighted graph.",
        "",
        "## Bus 5 metrics",
        "",
        f"- Degree: {int(bus5['Degree'])}",
        f"- Degree centrality: {bus5['DegreeCentrality']:.6f}",
        f"- Betweenness centrality: {bus5['BetweennessCentrality']:.6f}",
        f"- Closeness centrality: {bus5['ClosenessCentrality']:.6f}",
        f"- Average shortest-path distance: {bus5['AverageShortestPathDistance']:.3f}",
        f"- Maximum shortest-path distance: {int(bus5['MaximumShortestPathDistance'])}",
        f"- Configured LoadSwitch/load bus: {bool(bus5['IsConfiguredLoadBus'])}",
        f"- Configured added active power at Bus 5: {bus5['LoadAddActivePower']:.5f}",
        "",
        "Bus 5 is a degree-2 peripheral branch connected through Bus 2; it is not a "
        "topological hub. Bus 1 has identical degree, betweenness, closeness, and "
        "path-distance metrics in this unweighted topology, yet Bus 1 is infeasible "
        "while Bus 5 has the largest positive margin.",
        "Bus 5 is also the largest configured LoadSwitch target in the current "
        "metadata (added active-power setting 0.14130). This may help expose a "
        "stable local switching signature, but it is a dataset-specific operating "
        "condition rather than a general topology property.",
        "",
        "## Topology-margin relationship",
        "",
        *[
            f"- Spearman correlation for {metric}: {value:.3f}"
            for metric, value in correlations.items()
        ],
        "",
        "Therefore topology centrality alone does not explain the selection. The "
        "direct evidence is waveform-feature separation: Bus 5 suppresses all "
        "Normal/LoadSwitch scores while retaining a clear lower bound for every fault.",
        "",
        "Generator/PV and explicit IBR-candidate bus metadata were not available in "
        "the repository data used here, so no unsupported assignment was inferred.",
    ]
    (REPORTS / "bus5_topology_explanation.md").write_text("\n".join(lines) + "\n")


def write_readme(
    feasible: pd.DataFrame,
    failure: pd.DataFrame,
    topology: pd.DataFrame,
    correlations: dict[str, float],
    key: pd.DataFrame,
    delivery: Path,
) -> None:
    bus5 = feasible.loc[feasible["Bus"] == 5].iloc[0]
    bus1 = key.loc[key["Bus"] == 1].iloc[0]
    failure_counts = failure["DominantFailureReason"].value_counts()
    ranking = ", ".join(
        f"Bus {int(row.Bus)} ({row.FaultScoreMargin:.3f})"
        for row in feasible.itertuples(index=False)
    )
    readme = f"""# Why Bus 5 is the selected minimum WMU placement

## Purpose

This analysis explains why Bus 5 was selected in the fault/non-fault
hard-constraint minimum-WMU study. It does not change the objective to fault
localization.

## Existing result

- Minimum count: k=1
- Selected placement: Bus 5
- Margin: {bus5['FaultScoreMargin']:.6f}
- Normal FP / LoadSwitch FP / SLG FN / ThreePhase FN: 0 / 0 / 0 / 0

## Feasible single-WMU comparison

Six buses are feasible. Margin ranking: {ranking}. Bus 5 has the largest gap
because its non-fault maximum is only {bus5['NonFaultMaxScore']:.6f}, while its
fault minimum remains {bus5['FaultMinScore']:.6f}. Other feasible buses also
achieve zero errors, but their score boundaries are much closer.

## Why infeasible buses fail

- Normal false alarm only: {failure_counts.get('Normal false alarm', 0)} buses
- LoadSwitch false alarm only: {failure_counts.get('LoadSwitch false alarm', 0)} buses
- SLG missed fault only: {failure_counts.get('SLG missed fault', 0)} buses
- Three-phase missed fault only: {failure_counts.get('ThreePhase missed fault', 0)} buses
- Combined failure: {failure_counts.get('Combined failure', 0)} buses

The dominant weakness is therefore elevated Normal score rather than
LoadSwitch rejection. Bus 12 is the only single-WMU candidate with a
LoadSwitch false alarm under the selected rule.

## Feature-level explanation

At Bus 5, balanced 67–77 Hz voltage evidence is high for LoadSwitch and enters
the fixed score as a rejection penalty. Fault cases retain strong
voltage/current severity and sequence/unbalance evidence, particularly
zero-sequence current for SLG faults. The resulting non-fault maximum
({bus5['NonFaultMaxScore']:.6f}) and fault minimum
({bus5['FaultMinScore']:.6f}) leave a {bus5['FaultScoreMargin']:.6f} safety gap
around threshold {bus5['BestThreshold']:.6f}.

The processed table contains the direct 67–77 Hz (`E72`) band for every bus.
Requested 60–80 and 70–80 Hz terms are documented as nearest-band proxies;
raw waveforms were not recomputed in this task.

## Topology explanation

Bus 5 has degree 2, betweenness {topology.loc[topology.Bus==5,'BetweennessCentrality'].iloc[0]:.6f},
and closeness {topology.loc[topology.Bus==5,'ClosenessCentrality'].iloc[0]:.6f}.
It is not a topological hub. The Spearman correlation between closeness and
margin is {correlations['ClosenessCentrality']:.3f}. Centrality alone therefore
does not justify Bus 5; waveform separation is the direct selection basis.
Bus 5 is the largest configured LoadSwitch target in the current metadata
(added active-power setting 0.14130), which may strengthen its local switching
signature, but this remains a dataset-specific explanation.

## Bus 5 versus Bus 1

Bus 1 was sufficient for the earlier four-class macro-F1 greedy objective, but
that objective did not maximize the strict fault/non-fault safety margin.
Under the fixed hard-constraint score, Bus 1 has margin
{bus1['Margin']:.6f} and {int(bus1['LoadSwitchFP'])} LoadSwitch false alarms but
fails because Normal cases cross the fault decision region. Bus 5 and Bus 1
have identical unweighted topology centrality, confirming that their different
outcomes arise from waveform-feature behavior rather than graph position.

## Conclusion

Bus 5 is the most robust single-WMU placement in the current dataset because
it simultaneously provides strong LoadSwitch rejection and a retained lower
bound on fault evidence. This is a deterministic-dataset conclusion and must
be re-tested under varied LoadSwitch size, SSO conditions, fault parameters,
noise, missing channels, and operating points.
"""
    (FIGURES / "README.md").write_text(readme)
    shutil.copy2(
        FIGURES / "README.md",
        delivery / "README_bus5_explanation.md",
    )


def main() -> None:
    args = parse_args()
    configure()
    FIGURES.mkdir(parents=True, exist_ok=True)
    args.delivery_dir.mkdir(parents=True, exist_ok=True)
    write_inventory(args.data_dir)
    single, scores = enrich_single_metrics()
    failure = write_failure_analysis(single)
    by_bus = pd.read_csv(args.data_dir / "feature_table_by_bus.csv")
    feature_long = feature_level_comparison(by_bus, single, scores)
    metadata = pd.read_csv(args.data_dir / "dataset_metadata.csv")
    topology, correlations = topology_metrics(single, metadata)
    key = key_bus_comparison(single, scores, by_bus, topology)
    selected = pd.read_csv(REPORTS / "selected_minimum_wmu_set.csv").iloc[0]
    feasible = single.loc[single["Feasible"]].sort_values(
        "FaultScoreMargin", ascending=False
    )
    plot_margin(feasible, args.delivery_dir)
    plot_failure_map(failure, args.delivery_dir)
    plot_feature_explanation(by_bus, scores, selected, args.delivery_dir)
    plot_topology(topology, correlations, args.delivery_dir)
    plot_key_buses(key, args.delivery_dir)
    write_topology_notes(topology, correlations)
    write_readme(feasible, failure, topology, correlations, key, args.delivery_dir)


if __name__ == "__main__":
    main()
