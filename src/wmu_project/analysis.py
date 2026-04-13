from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_WMU_BUSES = [3, 5, 8, 10, 11, 12, 23, 26, 28, 29]
DEFAULT_WINDOWS_INPUT = Path(r"C:\Users\user\Documents\MATLAB\WMU_test\WMU_fault_results_sag.xlsx")


@dataclass
class AnalysisResult:
    case_summary: pd.DataFrame
    wmu_detail: pd.DataFrame
    dv_matrix: pd.DataFrame
    sag_matrix: pd.DataFrame
    res_matrix: pd.DataFrame
    wmu_risk: pd.DataFrame
    coverage_curve: pd.DataFrame
    minimal_k: pd.DataFrame
    threshold: float
    wmu_buses: list[int]


def load_fault_sheets(input_xlsx: str | Path) -> dict[int, pd.DataFrame]:
    xls = pd.ExcelFile(input_xlsx)
    sheets = {}
    for name in xls.sheet_names:
        if name.startswith("Fault_"):
            case_id = int(name.split("_")[1])
            sheets[case_id] = pd.read_excel(xls, sheet_name=name)
    return dict(sorted(sheets.items()))


def one_cycle_rms(x: np.ndarray, window: int) -> np.ndarray:
    power = x ** 2
    kernel = np.ones(window) / window
    rms2 = np.convolve(power, kernel, mode="full")[: len(x)]
    return np.sqrt(rms2)


def compute_res_ratio(signal: np.ndarray, fs: float) -> float:
    x = signal - np.nanmean(signal)
    if len(x) < 8:
        return 0.0
    X = np.fft.rfft(x)
    P = np.abs(X / len(x)) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    band = ((23 <= f) & (f <= 33)) | ((67 <= f) & (f <= 77))
    total = (f > 0) & (f <= fs / 2)
    den = np.nansum(P[total])
    return float(np.nansum(P[band]) / den) if den > 0 else 0.0


def analyze_fault_workbook(
    input_xlsx: str | Path,
    wmu_buses: Iterable[int] = DEFAULT_WMU_BUSES,
    t_event: float = 2.2,
    f0: float = 50.0,
    thr_dv: float = 0.072,
) -> AnalysisResult:
    wmu_buses = list(wmu_buses)
    sheets = load_fault_sheets(input_xlsx)
    rows_case = []
    rows_detail = []
    dv_rows = []
    sag_rows = []
    res_rows = []

    for case_id, df in sheets.items():
        t = df["Time_s"].to_numpy()
        fs = float(1.0 / np.median(np.diff(t)))
        n_cycle = max(1, round(fs / f0))
        pre = t < t_event
        post = t >= t_event

        dv_map = {}
        sag_map = {}
        res_map = {}

        for bus in wmu_buses:
            dv = df[f"dV_{bus}_A"].to_numpy(dtype=float)
            vraw = df[f"Vraw_{bus}_A"].to_numpy(dtype=float)

            dv_energy = float(np.sqrt(np.nanmean(dv ** 2)))
            vrms = one_cycle_rms(vraw, n_cycle)
            pre_v = float(np.nanmean(vrms[pre]))
            post_min = float(np.nanmin(vrms[post]))
            sag = max(0.0, 1.0 - post_min / pre_v) if pre_v > 0 else np.nan
            res_ratio = compute_res_ratio(dv[post], fs)

            dv_map[bus] = dv_energy
            sag_map[bus] = sag
            res_map[bus] = res_ratio
            rows_detail.append(
                {
                    "CaseID": case_id,
                    "WMU": bus,
                    "DV_Energy": dv_energy,
                    "Sag": sag,
                    "Res_ratio": res_ratio,
                }
            )

        star = max(dv_map, key=dv_map.get)
        max_dv = dv_map[star]
        max_sag = max(sag_map.values())
        max_res = max(res_map.values())

        rows_case.append(
            {
                "CaseID": case_id,
                "fs": fs,
                "Detected": max_dv > thr_dv,
                "StarWMU": star,
                "MaxDV": max_dv,
                "MaxSag": max_sag,
                "MaxRes": max_res,
            }
        )
        dv_rows.append({"CaseID": case_id, "StarWMU": star, **{f"WMU_{b}": dv_map[b] for b in wmu_buses}})
        sag_rows.append({"CaseID": case_id, "StarWMU": star, **{f"WMU_{b}": sag_map[b] for b in wmu_buses}})
        res_rows.append({"CaseID": case_id, "StarWMU": star, **{f"WMU_{b}": res_map[b] for b in wmu_buses}})

    case_summary = pd.DataFrame(rows_case).sort_values("CaseID").reset_index(drop=True)
    wmu_detail = pd.DataFrame(rows_detail).sort_values(["CaseID", "WMU"]).reset_index(drop=True)
    dv_matrix = pd.DataFrame(dv_rows).sort_values("CaseID").reset_index(drop=True)
    sag_matrix = pd.DataFrame(sag_rows).sort_values("CaseID").reset_index(drop=True)
    res_matrix = pd.DataFrame(res_rows).sort_values("CaseID").reset_index(drop=True)

    risk = (
        wmu_detail.groupby("WMU", as_index=False)
        .agg(
            MeanDV_Energy=("DV_Energy", "mean"),
            MedianDV_Energy=("DV_Energy", "median"),
            MaxDV_Energy=("DV_Energy", "max"),
            MeanSag=("Sag", "mean"),
            MedianSag=("Sag", "median"),
            MaxSag=("Sag", "max"),
            MeanRes_ratio=("Res_ratio", "mean"),
            MaxRes_ratio=("Res_ratio", "max"),
        )
    )
    star_count = case_summary["StarWMU"].value_counts().to_dict()
    solo_detect = {
        bus: int((dv_matrix[f"WMU_{bus}"] > thr_dv).sum()) for bus in wmu_buses
    }
    risk["StarCount"] = risk["WMU"].map(lambda x: star_count.get(x, 0))
    risk["DetectCountSolo"] = risk["WMU"].map(lambda x: solo_detect.get(x, 0))
    risk = risk.sort_values("MeanDV_Energy", ascending=False).reset_index(drop=True)

    coverage_curve = greedy_coverage(dv_matrix, wmu_buses, thr_dv)
    minimal_k = exhaustive_min_k(dv_matrix, wmu_buses, thr_dv, targets=(0.95, 0.99))

    return AnalysisResult(
        case_summary=case_summary,
        wmu_detail=wmu_detail,
        dv_matrix=dv_matrix,
        sag_matrix=sag_matrix,
        res_matrix=res_matrix,
        wmu_risk=risk,
        coverage_curve=coverage_curve,
        minimal_k=minimal_k,
        threshold=thr_dv,
        wmu_buses=wmu_buses,
    )


def greedy_coverage(dv_matrix: pd.DataFrame, wmu_buses: list[int], thr_dv: float) -> pd.DataFrame:
    remaining = list(wmu_buses)
    selected: list[int] = []
    rows = []
    for k in range(1, len(wmu_buses) + 1):
        best_bus = remaining[0]
        best_cov = -1.0
        for bus in remaining:
            cols = [f"WMU_{b}" for b in selected + [bus]]
            coverage = (dv_matrix[cols].max(axis=1) > thr_dv).mean()
            if coverage > best_cov:
                best_cov = coverage
                best_bus = bus
        selected.append(best_bus)
        remaining.remove(best_bus)
        rows.append(
            {
                "NumWMU": k,
                "Coverage": best_cov,
                "SelectedWMUs": str(selected),
            }
        )
    return pd.DataFrame(rows)


def exhaustive_min_k(
    dv_matrix: pd.DataFrame,
    wmu_buses: list[int],
    thr_dv: float,
    targets: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    rows = []
    for target in targets:
        found = False
        for k in range(1, len(wmu_buses) + 1):
            for subset in combinations(wmu_buses, k):
                cols = [f"WMU_{b}" for b in subset]
                coverage = (dv_matrix[cols].max(axis=1) > thr_dv).mean()
                if coverage >= target:
                    rows.append(
                        {"Method": "Coverage", "Target": target, "MinK": k, "Subset": str(list(subset)), "Coverage": coverage}
                    )
                    found = True
                    break
            if found:
                break
        if not found:
            rows.append({"Method": "Coverage", "Target": target, "MinK": np.nan, "Subset": "Not reached", "Coverage": np.nan})

        found = False
        for k in range(2, len(wmu_buses) + 1):
            for subset in combinations(wmu_buses, k):
                worst = 1.0
                for dropped in subset:
                    reduced = [b for b in subset if b != dropped]
                    cols = [f"WMU_{b}" for b in reduced]
                    worst = min(worst, (dv_matrix[cols].max(axis=1) > thr_dv).mean())
                if worst >= target:
                    rows.append(
                        {"Method": "Robust-1Drop", "Target": target, "MinK": k, "Subset": str(list(subset)), "Coverage": worst}
                    )
                    found = True
                    break
            if found:
                break
        if not found:
            rows.append({"Method": "Robust-1Drop", "Target": target, "MinK": np.nan, "Subset": "Not reached", "Coverage": np.nan})
    return pd.DataFrame(rows)


def save_excel(result: AnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.case_summary.to_excel(writer, sheet_name="CaseSummary", index=False)
        result.wmu_detail.to_excel(writer, sheet_name="WMUDetail", index=False)
        result.dv_matrix.to_excel(writer, sheet_name="DV_Matrix", index=False)
        result.sag_matrix.to_excel(writer, sheet_name="Sag_Matrix", index=False)
        result.res_matrix.to_excel(writer, sheet_name="Res_Matrix", index=False)
        result.wmu_risk.to_excel(writer, sheet_name="WMURisk", index=False)
        result.coverage_curve.to_excel(writer, sheet_name="CoverageCurve", index=False)
        result.minimal_k.to_excel(writer, sheet_name="MinimalK", index=False)


def finalize_figure(fig: plt.Figure, output_path: str | Path, show: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def save_figure1(result: AnalysisResult, output_dir: str | Path, show: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "figure1_coverage_vs_num_wmu.png"
    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(result.coverage_curve["NumWMU"], result.coverage_curve["Coverage"] * 100, marker="o", linewidth=2)
    plt.ylim(0, 105)
    plt.xlabel("# WMU")
    plt.ylabel("Coverage (%)")
    plt.title(f"Coverage vs #WMU (thr_dv={result.threshold:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return finalize_figure(fig, out, show=show)


def save_figure2(result: AnalysisResult, output_dir: str | Path, show: bool = True) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out1 = output_dir / "figure2_wmu_mean_dv_bar.png"
    out2 = output_dir / "figure2b_star_wmu_count.png"

    risk = result.wmu_risk.copy()
    fig1 = plt.figure(figsize=(8, 4.5))
    plt.bar(risk["WMU"].astype(str), risk["MeanDV_Energy"])
    plt.xlabel("WMU Bus")
    plt.ylabel("Mean DV Energy")
    plt.title("WMU Ranking by Mean DV Energy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig1, out1, show=show)

    star = risk.sort_values("StarCount", ascending=False)
    fig2 = plt.figure(figsize=(8, 4.5))
    plt.bar(star["WMU"].astype(str), star["StarCount"])
    plt.xlabel("WMU Bus")
    plt.ylabel("Star Count")
    plt.title("StarWMU Count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig2, out2, show=show)
    return out1, out2


def save_figure3(result: AnalysisResult, output_dir: str | Path, show: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "figure3_case_wmu_heatmap_sorted.png"
    dv = result.dv_matrix.copy().sort_values(["StarWMU", "CaseID"])
    arr = dv[[f"WMU_{b}" for b in result.wmu_buses]].to_numpy()
    fig = plt.figure(figsize=(8, 7))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="DV Energy")
    plt.xticks(range(len(result.wmu_buses)), [str(b) for b in result.wmu_buses])
    plt.yticks(range(len(dv)), dv["CaseID"].astype(str))
    plt.xlabel("WMU Bus")
    plt.ylabel("Fault Case (sorted by StarWMU)")
    plt.title("Case x WMU DV Energy Heatmap")
    plt.tight_layout()
    return finalize_figure(fig, out, show=show)


def save_figure4(result: AnalysisResult, output_dir: str | Path, edge_csv: str | Path, show: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "figure4_network_graph.png"
    edges = pd.read_csv(edge_csv)
    g = nx.Graph()
    g.add_edges_from(edges[["from_bus", "to_bus"]].itertuples(index=False, name=None))
    for b in range(1, 31):
        if b not in g:
            g.add_node(b)
    pos = nx.spring_layout(g, seed=7)
    star_count = result.case_summary["StarWMU"].value_counts().to_dict()
    sizes = [450 if n in result.wmu_buses else 180 for n in g.nodes()]
    colors = [star_count.get(n, 0) if n in result.wmu_buses else 0 for n in g.nodes()]
    fig = plt.figure(figsize=(10, 7))
    nx.draw_networkx_edges(g, pos, alpha=0.35, width=1.2)
    nodes = nx.draw_networkx_nodes(g, pos, node_size=sizes, node_color=colors, cmap="plasma")
    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.colorbar(nodes, label="StarWMU Count")
    plt.title("IEEE 30-Bus Graph with WMU Nodes")
    plt.axis("off")
    plt.tight_layout()
    return finalize_figure(fig, out, show=show)


def save_figure5(result: AnalysisResult, output_dir: str | Path, show: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "figure5_wmu_boxplot.png"
    data = [result.dv_matrix[f"WMU_{b}"].to_numpy() for b in result.wmu_buses]
    fig = plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, labels=[str(b) for b in result.wmu_buses])
    plt.xlabel("WMU Bus")
    plt.ylabel("DV Energy")
    plt.title("WMU DV Distribution by Case")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return finalize_figure(fig, out, show=show)


def save_figure6(result: AnalysisResult, output_dir: str | Path, show: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "figure6_3dscatter_dv_sag_resratio.png"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        result.case_summary["MaxDV"],
        result.case_summary["MaxSag"],
        result.case_summary["MaxRes"],
        c=result.case_summary["StarWMU"],
        s=70,
        cmap="tab10",
    )
    for _, row in result.case_summary.iterrows():
        ax.text(row["MaxDV"], row["MaxSag"], row["MaxRes"], str(int(row["CaseID"])), fontsize=7)
    ax.set_xlabel("DV_energy_max")
    ax.set_ylabel("Sag_max")
    ax.set_zlabel("Res_ratio_max")
    ax.set_title("3D Scatter of Fault Cases Colored by StarWMU")
    fig.colorbar(sc, ax=ax, label="StarWMU")
    plt.tight_layout()
    return finalize_figure(fig, out, show=show)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--thr-dv", type=float, default=0.072)
    parser.add_argument("--t-event", type=float, default=2.2)
    parser.add_argument("--f0", type=float, default=50.0)
    return parser


def resolve_default_paths(input_arg: str | None, output_arg: str | None, script_path: str | Path) -> tuple[Path, Path]:
    script_path = Path(script_path).resolve()
    repo_root = script_path.parents[1]

    if input_arg:
        input_path = Path(input_arg)
    else:
        candidates = [
            DEFAULT_WINDOWS_INPUT,
            repo_root / "WMU_fault_results_sag.xlsx",
            repo_root / "data" / "WMU_fault_results_sag.xlsx",
        ]
        input_path = None
        for candidate in candidates:
            if candidate.exists():
                input_path = candidate
                break
        if input_path is None:
            raise FileNotFoundError(
                "No input workbook was provided and no default workbook was found. "
                "Expected one of: "
                f"{DEFAULT_WINDOWS_INPUT}, "
                f"{repo_root / 'WMU_fault_results_sag.xlsx'}, "
                f"{repo_root / 'data' / 'WMU_fault_results_sag.xlsx'}"
            )

    if output_arg:
        output_dir = Path(output_arg)
    else:
        output_dir = repo_root / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_path, output_dir
