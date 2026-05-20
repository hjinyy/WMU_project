from __future__ import annotations

from pathlib import Path

import pandas as pd

from .waveform_utils import CaseMetadata, WaveformDataError, detect_time_column, list_waveform_case_files, to_local_path


def list_cases(input_dir: str | Path) -> list[CaseMetadata]:
    return list_waveform_case_files(input_dir)


def load_waveform_case(case: CaseMetadata) -> pd.DataFrame:
    try:
        if case.source_path.suffix.lower() == ".csv":
            df = pd.read_csv(case.source_path)
        else:
            df = pd.read_excel(case.source_path, sheet_name=0, engine="openpyxl")
    except Exception as exc:  # pragma: no cover - exercised by runtime data
        raise WaveformDataError(f"Failed to open {case.source_path.name}: {exc}") from exc
    if df.empty:
        raise WaveformDataError(f"Workbook {case.source_path.name} is empty")
    time_col = detect_time_column(df.columns)
    if time_col != "Time":
        df = df.rename(columns={time_col: "Time"})
    return df


def output_roots(output_dir: str | Path) -> tuple[Path, Path]:
    root = to_local_path(output_dir)
    reports = root / "reports"
    figures = root / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return reports, figures
