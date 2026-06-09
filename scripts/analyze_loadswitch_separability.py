from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import kruskal
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(
    "/mnt/c/Users/user/Documents/MATLAB/! WMU_final/"
    "WMU_batch_data_ibr_background"
)
DEFAULT_RAW = Path(
    "/mnt/c/Users/user/Documents/MATLAB/! WMU_final/"
    "WMU_batch_raw_ibr_background"
)
DEFAULT_REPORTS = (
    ROOT / "results" / "waveform_ibr_background_diagnostics" / "reports"
)
CLASS_ORDER = [
    "SSO_Normal",
    "SSO_LoadSwitch",
    "SSO_SLG_Fault",
    "SSO_ThreePhase_Fault",
]
BANDS = [
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
    (50.0, 60.0),
    (60.0, 70.0),
    (70.0, 80.0),
    (60.0, 80.0),
    (67.0, 77.0),
    (80.0, 100.0),
    (100.0, 150.0),
]
META = {
    "CaseName",
    "EventType",
    "TargetBus",
    "ObservedBus",
    "EventTime",
    "SamplingRateHz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--observed-bus", type=int, default=1)
    return parser.parse_args()


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def one_feature_score(values: pd.Series, labels: pd.Series) -> tuple[float, float]:
    valid = values.notna() & np.isfinite(values)
    x = values.loc[valid].to_numpy().reshape(-1, 1)
    y = (labels.loc[valid] == "SSO_LoadSwitch").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)
    pred = cross_val_predict(model, x, y, cv=folds)
    return (
        float(balanced_accuracy_score(y, pred)),
        float(f1_score(y, pred, zero_division=0)),
    )


def event_masks(t: np.ndarray, event_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = (t >= max(t[0], event_time - 0.10)) & (t < event_time)
    post = (t >= event_time) & (t <= min(t[-1], event_time + 0.10))
    event = (t >= max(t[0], event_time - 0.04)) & (
        t <= min(t[-1], event_time + 0.12)
    )
    return pre, post, event


def rolling_rms(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    kernel = np.ones(samples_per_cycle) / samples_per_cycle
    return np.sqrt(np.maximum(np.convolve(signal**2, kernel, mode="same"), 0.0))


def cycle_difference(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    out = np.zeros_like(signal)
    out[samples_per_cycle:] = (
        signal[samples_per_cycle:] - signal[:-samples_per_cycle]
    )
    return out


def band_ratios(signal: np.ndarray, fs: float) -> dict[str, float]:
    x = detrend(np.asarray(signal, dtype=float), type="constant")
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total = power[(frequency > 0) & (frequency <= 2000)].sum()
    rows = {}
    for low, high in BANDS:
        label = f"{low:g}_{high:g}Hz"
        band = power[(frequency >= low) & (frequency < high)].sum()
        rows[label] = float(band / total) if total > 0 else 0.0
    return rows


def raw_case_features(
    csv_path: Path, event_time: float, observed_bus: int
) -> dict[str, float]:
    columns = ["Time"] + [
        f"{kind}{phase}_{observed_bus}"
        for kind in ("V", "I")
        for phase in ("a", "b", "c")
    ]
    frame = pd.read_csv(csv_path, usecols=columns)
    t = frame["Time"].to_numpy(float)
    fs = float(1.0 / np.median(np.diff(t)))
    pre, post, event = event_masks(t, event_time)
    # Match the existing feature-extraction pipeline (IEEE model nominal f0=50 Hz).
    samples_per_cycle = max(1, int(round(fs / 50.0)))
    output: dict[str, float] = {}
    pre_v, post_v, pre_i, post_i = [], [], [], []
    v_peak, i_peak, v_short_energy, i_short_energy = [], [], [], []
    spectra: dict[str, list[float]] = {}

    for phase in ("a", "b", "c"):
        v = frame[f"V{phase}_{observed_bus}"].to_numpy(float)
        i = frame[f"I{phase}_{observed_bus}"].to_numpy(float)
        vrms = rolling_rms(v, samples_per_cycle)
        irms = rolling_rms(i, samples_per_cycle)
        pre_v.append(float(np.mean(vrms[pre])))
        post_v.append(float(np.mean(vrms[post])))
        pre_i.append(float(np.mean(irms[pre])))
        post_i.append(float(np.mean(irms[post])))
        dv = cycle_difference(v, samples_per_cycle)[event]
        di = cycle_difference(i, samples_per_cycle)[event]
        v_peak.append(float(np.max(np.abs(dv))))
        i_peak.append(float(np.max(np.abs(di))))
        v_short_energy.append(float(np.sqrt(np.mean(dv**2))))
        i_short_energy.append(float(np.sqrt(np.mean(di**2))))
        for signal_name, signal in (("dV", dv), ("dI", di)):
            for band, ratio in band_ratios(signal, fs).items():
                output[f"raw_{signal_name}_band_{band}_ratio_{phase.upper()}"] = ratio
                spectra.setdefault(
                    f"raw_{signal_name}_band_{band}_ratio_3ph_max", []
                ).append(ratio)

    v_pre = float(np.mean(pre_v))
    v_post = float(np.mean(post_v))
    i_pre = float(np.mean(pre_i))
    i_post = float(np.mean(post_i))
    output.update(
        {
            "raw_delta_V_rms": v_post - v_pre,
            "raw_relative_delta_V_rms": (v_post - v_pre) / max(abs(v_pre), 1e-12),
            "raw_delta_I_rms": i_post - i_pre,
            "raw_relative_delta_I_rms": (i_post - i_pre) / max(abs(i_pre), 1e-12),
            "raw_apparent_power_proxy_pre": v_pre * i_pre,
            "raw_apparent_power_proxy_post": v_post * i_post,
            "raw_delta_apparent_power_proxy": v_post * i_post - v_pre * i_pre,
            "raw_dV_peak_deviation": max(v_peak),
            "raw_dI_peak_deviation": max(i_peak),
            "raw_dV_short_window_energy": max(v_short_energy),
            "raw_dI_short_window_energy": max(i_short_energy),
        }
    )
    output.update({name: max(values) for name, values in spectra.items()})
    return output


def build_case_matrix(
    data_dir: Path, raw_dir: Path, observed_bus: int
) -> pd.DataFrame:
    by_bus = pd.read_csv(data_dir / "feature_table_by_bus.csv")
    selected = by_bus.loc[by_bus["ObservedBus"] == observed_bus].copy()
    selected = selected.drop_duplicates("CaseName").reset_index(drop=True)
    metadata = pd.read_csv(raw_dir / "dataset_metadata.csv").set_index("CaseName")
    raw_rows = []
    for row in selected.itertuples(index=False):
        meta = metadata.loc[row.CaseName]
        path = raw_dir / Path(str(meta["OutputFile"]).replace("\\", "/")).name
        raw_rows.append(
            {
                "CaseName": row.CaseName,
                **raw_case_features(path, float(row.EventTime), observed_bus),
            }
        )
    return selected.merge(pd.DataFrame(raw_rows), on="CaseName", how="left")


def summarize_feature(
    frame: pd.DataFrame, feature: str, source: str
) -> dict[str, object]:
    values = pd.to_numeric(frame[feature], errors="coerce")
    load = values[frame["EventType"] == "SSO_LoadSwitch"].dropna().to_numpy()
    rest = values[frame["EventType"] != "SSO_LoadSwitch"].dropna().to_numpy()
    groups = [
        values[frame["EventType"] == event].dropna().to_numpy()
        for event in CLASS_ORDER
    ]
    pvalue = kruskal(*groups).pvalue if all(len(group) for group in groups) else np.nan
    balanced, f1 = one_feature_score(values, frame["EventType"])
    row: dict[str, object] = {
        "Feature": feature,
        "Source": source,
        "ObservedBus": int(frame["ObservedBus"].iloc[0]),
        "LoadSwitchVsRestCliffsDelta": cliffs_delta(load, rest),
        "LoadSwitchVsRestAbsCliffsDelta": abs(cliffs_delta(load, rest)),
        "LoadSwitchVsNormalCliffsDelta": cliffs_delta(load, groups[0]),
        "LoadSwitchVsSLGCliffsDelta": cliffs_delta(load, groups[2]),
        "LoadSwitchVsThreePhaseCliffsDelta": cliffs_delta(load, groups[3]),
        "KruskalWallisPValue": pvalue,
        "OneFeatureBalancedAccuracy": balanced,
        "OneFeatureF1": f1,
    }
    for event, group in zip(CLASS_ORDER, groups):
        prefix = event.removeprefix("SSO_")
        row[f"{prefix}_Mean"] = float(np.mean(group))
        row[f"{prefix}_Median"] = float(np.median(group))
        row[f"{prefix}_IQR"] = float(np.quantile(group, 0.75) - np.quantile(group, 0.25))
    return row


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    frame = build_case_matrix(args.data_dir, args.raw_dir, args.observed_bus)
    feature_columns = [
        column
        for column in frame.columns
        if column not in META and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = [
        summarize_feature(
            frame,
            feature,
            "raw-derived"
            if feature.startswith("raw_")
            else "existing feature_table_by_bus",
        )
        for feature in feature_columns
    ]
    ranking = pd.DataFrame(rows).sort_values(
        [
            "OneFeatureBalancedAccuracy",
            "LoadSwitchVsRestAbsCliffsDelta",
            "KruskalWallisPValue",
        ],
        ascending=[False, False, True],
    )
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(
        args.reports_dir / "loadswitch_feature_separability.csv", index=False
    )

    frequency = ranking[
        ranking["Feature"].str.contains(r"raw_d[VI]_band_", regex=True)
    ].copy()
    frequency["Signal"] = frequency["Feature"].str.extract(r"raw_(d[VI])_band_")
    frequency["BandHz"] = frequency["Feature"].str.extract(
        r"band_([0-9]+(?:_[0-9]+)?Hz)"
    )[0]
    frequency["Aggregation"] = frequency["Feature"].str.extract(
        r"_ratio_(A|B|C|3ph_max)$"
    )[0]
    frequency.to_csv(
        args.reports_dir / "frequency_band_sweep_loadswitch.csv", index=False
    )

    print(ranking.head(15).to_string(index=False))
    print("\nTop frequency bands:")
    print(frequency.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
