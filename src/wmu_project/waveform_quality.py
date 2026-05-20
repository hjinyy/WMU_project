from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd

from .waveform_io import load_waveform_case
from .waveform_utils import PHASES, SIGNAL_PREFIXES, CaseMetadata, detect_observed_buses, format_bus, signal_columns_for_bus


def inspect_case_quality(case: CaseMetadata, zero_tol: float = 1e-6) -> dict[str, object]:
    row: dict[str, object] = {
        "CaseName": case.case_name,
        "EventType": case.event_type,
        "TargetBus": case.target_bus,
        "InputFile": str(case.source_path),
    }
    warnings: list[str] = []
    try:
        df = load_waveform_case(case)
    except Exception as exc:
        row.update(
            {
                "SheetReadable": False,
                "TimeColumnExists": False,
                "TimeMonotonic": False,
                "SamplingIntervalMean": np.nan,
                "SamplingIntervalStd": np.nan,
                "SimulationDuration": np.nan,
                "ObservedBusCount": 0,
                "MissingSignalCount": np.nan,
                "NaNCount": np.nan,
                "InfCount": np.nan,
                "ZeroLikeBusCount": np.nan,
                "ZeroLikeBuses": "",
                "Status": "FAILED",
                "Notes": str(exc),
            }
        )
        return row

    t = df["Time"].to_numpy(dtype=float)
    dt = np.diff(t) if len(t) > 1 else np.array([], dtype=float)
    time_monotonic = bool(np.all(dt > 0)) if dt.size else False
    if not time_monotonic:
        warnings.append("time_not_monotonic")

    buses = detect_observed_buses(df.columns)
    missing_signal_count = 0
    zero_like_buses: list[int] = []
    for bus in buses:
        cols = signal_columns_for_bus(bus)
        missing = [name for name in cols.values() if name not in df.columns]
        missing_signal_count += len(missing)
        if missing:
            warnings.append(f"missing_bus_{bus:02d}_signals")
            continue
        vals = df[list(cols.values())].to_numpy(dtype=float)
        if np.nanmax(np.abs(vals)) <= zero_tol:
            zero_like_buses.append(bus)

    nan_count = int(df.isna().sum().sum())
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy(dtype=float)).sum())
    if nan_count > 0:
        warnings.append("contains_nan")
    if inf_count > 0:
        warnings.append("contains_inf")
    if zero_like_buses:
        warnings.append("zero_like_bus")
    if missing_signal_count > 0:
        warnings.append("missing_signals")

    status = "OK"
    if warnings:
        status = "WARNING"
    if not time_monotonic or not buses:
        status = "FAILED" if not buses else status

    row.update(
        {
            "SheetReadable": True,
            "TimeColumnExists": True,
            "TimeMonotonic": time_monotonic,
            "SamplingIntervalMean": float(np.mean(dt)) if dt.size else np.nan,
            "SamplingIntervalStd": float(np.std(dt)) if dt.size else np.nan,
            "SimulationDuration": float(t[-1] - t[0]) if len(t) else np.nan,
            "ObservedBusCount": len(buses),
            "ObservedBuses": ", ".join(format_bus(bus) for bus in buses),
            "MissingSignalCount": int(missing_signal_count),
            "NaNCount": nan_count,
            "InfCount": inf_count,
            "ZeroLikeBusCount": len(zero_like_buses),
            "ZeroLikeBuses": ", ".join(format_bus(bus) for bus in zero_like_buses),
            "Status": status,
            "Notes": "; ".join(dict.fromkeys(warnings)),
        }
    )
    return row


def _inspect_case_quality_task(case: CaseMetadata) -> dict[str, object]:
    return inspect_case_quality(case)


def build_quality_outputs(cases: list[CaseMetadata], max_workers: int | None = None) -> tuple[pd.DataFrame, str]:
    workers = 1 if len(cases) < 4 else (max_workers or min(4, os.cpu_count() or 1))
    if workers <= 1:
        rows = [inspect_case_quality(case) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_inspect_case_quality_task, cases))
    report = pd.DataFrame(rows)
    report = report.sort_values(["EventType", "TargetBus", "CaseName"]).reset_index(drop=True)
    status_counts = report["Status"].value_counts().to_dict()
    class_counts = report["EventType"].value_counts().to_dict()
    lines = [
        f"Total xlsx files: {len(report)}",
        "Class counts:",
    ]
    for event, count in class_counts.items():
        lines.append(f"  - {event}: {count}")
    lines.extend(
        [
            "Status counts:",
            f"  - OK: {status_counts.get('OK', 0)}",
            f"  - WARNING: {status_counts.get('WARNING', 0)}",
            f"  - FAILED: {status_counts.get('FAILED', 0)}",
            "Files with warnings/failures:",
        ]
    )
    flagged = report.loc[report["Status"] != "OK", ["CaseName", "Status", "Notes"]]
    if flagged.empty:
        lines.append("  - none")
    else:
        for _, row in flagged.iterrows():
            lines.append(f"  - {row['CaseName']}: {row['Status']} ({row['Notes']})")
    return report, "\n".join(lines) + "\n"


def write_quality_outputs(report: pd.DataFrame, summary_text: str, reports_dir: Path, repo_summary_path: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(reports_dir / "data_quality_report.csv", index=False)
    (reports_dir / "data_quality_summary.txt").write_text(summary_text, encoding="utf-8")
    repo_summary_path.parent.mkdir(parents=True, exist_ok=True)
    repo_summary_path.write_text("# Data Quality Summary\n\n```text\n" + summary_text + "```\n", encoding="utf-8")
