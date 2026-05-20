from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import math
import re
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

EVENT_CLASS_ORDER = ["Normal", "LoadSwitch", "SLG_Fault", "ThreePhase_Fault"]
FAULT_EVENTS = {"SLG_Fault", "ThreePhase_Fault"}
LOAD_SWITCH_ALIASES = {"LoadSwitch", "LoadSwitch15pct"}
FILENAME_RE = re.compile(
    r"^(?P<event>Normal|LoadSwitch(?:15pct)?|SLG_Fault|ThreePhase_Fault)"
    r"(?:_(?:Bus(?P<bus>\d+)|Case(?P<case>\d+)))?\.(?P<ext>xlsx|csv)$",
    re.IGNORECASE,
)
BUS_COLUMN_RE = re.compile(r"^(?P<signal>[VI][abc])_(?P<bus>\d+)$")
PHASES = ("A", "B", "C")
SIGNAL_PREFIXES = ("V", "I")


@dataclass(frozen=True)
class CaseMetadata:
    case_name: str
    event_type: str
    target_bus: float
    source_path: Path
    variant: str | None = None

    @property
    def target_bus_int(self) -> int | None:
        return None if math.isnan(self.target_bus) else int(self.target_bus)

    @property
    def event_time(self) -> float:
        if self.event_type == "LoadSwitch":
            return 0.2
        return 0.5


class WaveformDataError(RuntimeError):
    pass


NORMALIZED_SIGNAL_MAP = {
    "Va": "Va",
    "Vb": "Vb",
    "Vc": "Vc",
    "Ia": "Ia",
    "Ib": "Ib",
    "Ic": "Ic",
}


def to_local_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.exists():
        return path
    text = str(path_like)
    if re.match(r"^[A-Za-z]:\\", text):
        win = PureWindowsPath(text)
        drive = win.drive.rstrip(":").lower()
        translated = Path("/mnt") / drive / Path(*win.parts[1:])
        return translated
    return path


def parse_case_filename(path: str | Path) -> CaseMetadata:
    path = Path(path)
    match = FILENAME_RE.match(path.name)
    if not match:
        raise WaveformDataError(f"Unsupported waveform filename: {path.name}")
    event_raw = match.group("event")
    event_type = "LoadSwitch" if event_raw in LOAD_SWITCH_ALIASES else event_raw
    bus_text = match.group("bus")
    variant = event_raw if event_raw != event_type else None
    target_bus = float("nan") if bus_text is None else float(int(bus_text))
    return CaseMetadata(
        case_name=path.stem,
        event_type=event_type,
        target_bus=target_bus,
        source_path=path,
        variant=variant,
    )


def list_waveform_case_files(input_dir: str | Path) -> list[CaseMetadata]:
    root = to_local_path(input_dir)
    files = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".xlsx", ".csv"}
        ]
    )
    cases = []
    for path in files:
        try:
            cases.append(parse_case_filename(path))
        except WaveformDataError:
            continue
    return cases


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_time_column(columns: Iterable[str]) -> str:
    for candidate in ("Time", "Time_s", "time", "time_s"):
        if candidate in columns:
            return candidate
    raise WaveformDataError("Time column not found")


def detect_observed_buses(columns: Iterable[str]) -> list[int]:
    buses: set[int] = set()
    for column in columns:
        match = BUS_COLUMN_RE.match(str(column))
        if match:
            buses.add(int(match.group("bus")))
    return sorted(buses)


def signal_columns_for_bus(bus: int) -> dict[str, str]:
    columns = {}
    for prefix in SIGNAL_PREFIXES:
        for phase in PHASES:
            key = f"{prefix}{phase}"
            columns[key] = f"{prefix}{phase.lower()}_{bus}"
    return columns


def build_case_index_frame(cases: Iterable[CaseMetadata]) -> pd.DataFrame:
    rows = []
    for meta in cases:
        rows.append(
            {
                "CaseName": meta.case_name,
                "EventType": meta.event_type,
                "TargetBus": meta.target_bus,
                "InputFile": str(meta.source_path),
                "Variant": meta.variant or "",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["EventType"] = pd.Categorical(frame["EventType"], EVENT_CLASS_ORDER, ordered=True)
        frame = frame.sort_values(["EventType", "TargetBus", "CaseName"]).reset_index(drop=True)
    return frame


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    if den is None or not np.isfinite(den) or abs(den) < 1e-12:
        return default
    return float(num / den)


def nanmax(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return default
    return float(np.nanmax(arr))


def nanmean(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return default
    return float(np.nanmean(arr))


def format_bus(bus: int | float | None) -> str:
    if bus is None or (isinstance(bus, float) and math.isnan(bus)):
        return "NaN"
    return f"Bus{int(bus):02d}"


def load_ieee30_graph(edge_csv: str | Path) -> nx.Graph:
    edges = pd.read_csv(edge_csv)
    g = nx.Graph()
    for _, row in edges.iterrows():
        g.add_edge(int(row.iloc[0]), int(row.iloc[1]))
    return g


def one_hop_hit(graph: nx.Graph, true_bus: int, pred_bus: int) -> bool:
    if true_bus == pred_bus:
        return True
    if true_bus not in graph or pred_bus not in graph:
        return False
    return pred_bus in set(graph.neighbors(true_bus))
