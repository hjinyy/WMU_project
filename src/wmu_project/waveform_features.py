from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
from scipy.signal import detrend

from .waveform_io import load_waveform_case
from .waveform_utils import (
    PHASES,
    CaseMetadata,
    detect_observed_buses,
    format_bus,
    nanmax,
    nanmean,
    safe_divide,
    signal_columns_for_bus,
)

EPS = 1e-12

BUS_FEATURE_GROUPS = {
    "DV_energy_only": ["dV_energy"],
    "Voltage_time_only": ["sag_", "max_sag", "mean_sag", "dV_energy", "V_unbalance", "Delta_V_unbalance"],
    "Voltage_time_freq": ["sag_", "max_sag", "mean_sag", "dV_energy", "V_unbalance", "Delta_V_unbalance", "HF_ratio_A", "HF_ratio_B", "HF_ratio_C", "HF_ratio_3ph_max", "E28_ratio", "E72_ratio", "Res_ratio"],
    "Voltage_current": ["sag_", "max_sag", "mean_sag", "dV_energy", "dI_energy", "I_rms_", "I_rms_jump", "V_unbalance", "Delta_V_unbalance"],
    "Voltage_current_unbalance_sequence": ["sag_", "max_sag", "mean_sag", "dV_energy", "dI_energy", "I_rms_", "I_rms_jump", "V_unbalance", "Delta_V_unbalance", "I_unbalance", "Delta_I_unbalance", "V0_ratio", "I0_ratio", "V2_ratio", "I2_ratio"],
    "Impedance_added": ["sag_", "max_sag", "mean_sag", "dV_energy", "dI_energy", "I_rms_", "I_rms_jump", "V_unbalance", "Delta_V_unbalance", "I_unbalance", "Delta_I_unbalance", "V0_ratio", "I0_ratio", "V2_ratio", "I2_ratio", "Z_app_", "Delta_Z_app", "Z_drop_ratio"],
    "All_features": [""],
}

CASE_SUMMARY_GROUPS = {
    "DV_energy_only": ["star_bus_dV", "max_dV_energy"],
    "Voltage_time_only": ["star_bus_dV", "max_dV_energy", "max_sag", "max_V_unbalance"],
    "Voltage_time_freq": ["star_bus_dV", "max_dV_energy", "max_sag", "max_HF_ratio", "max_Res_ratio", "max_V_unbalance"],
    "Voltage_current": ["star_bus_dV", "star_bus_dI", "max_dV_energy", "max_dI_energy", "max_sag", "max_HF_ratio", "max_Res_ratio", "max_V_unbalance", "max_I_unbalance"],
    "Voltage_current_unbalance_sequence": ["star_bus_dV", "star_bus_dI", "max_dV_energy", "max_dI_energy", "max_sag", "max_HF_ratio", "max_Res_ratio", "max_V0_ratio", "max_I0_ratio", "max_V_unbalance", "max_I_unbalance"],
    "Impedance_added": ["star_bus_dV", "star_bus_dI", "max_dV_energy", "max_dI_energy", "max_sag", "max_HF_ratio", "max_Res_ratio", "max_V0_ratio", "max_I0_ratio", "max_V_unbalance", "max_I_unbalance", "max_Z_drop_ratio", "min_Z_app_post"],
    "All_features": ["star_bus_dV", "star_bus_dI", "max_dV_energy", "max_dI_energy", "max_sag", "max_HF_ratio", "max_Res_ratio", "max_V0_ratio", "max_I0_ratio", "max_V_unbalance", "max_I_unbalance", "max_Z_drop_ratio", "min_Z_app_post"],
}


@dataclass
class FeatureTables:
    by_bus: pd.DataFrame
    by_case: pd.DataFrame
    by_case_wide: pd.DataFrame


@dataclass
class CaseFeatureBundle:
    metadata: CaseMetadata
    by_bus: pd.DataFrame
    by_case: pd.DataFrame
    by_case_wide: pd.DataFrame


class FeatureComputationError(RuntimeError):
    pass


def rolling_rms(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.abs(signal)
    kernel = np.ones(window) / window
    power = np.convolve(np.square(signal), kernel, mode="same")
    return np.sqrt(np.maximum(power, 0.0))


def cycle_difference(signal: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0 or delay >= len(signal):
        return np.zeros_like(signal)
    diff = np.zeros_like(signal)
    diff[delay:] = signal[delay:] - signal[:-delay]
    return diff


def spectral_ratios(signal: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return {"hf": 0.0, "e28": 0.0, "e72": 0.0, "res": 0.0}
    x = detrend(x, type="constant")
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total_mask = (freqs > 0) & (freqs <= min(fs / 2.0, 2000.0))
    total_power = np.sum(power[total_mask])
    if total_power <= 0:
        return {"hf": 0.0, "e28": 0.0, "e72": 0.0, "res": 0.0}
    hf = np.sum(power[(freqs >= 200.0) & (freqs <= min(2000.0, fs / 2.0))])
    e28 = np.sum(power[(freqs >= 23.0) & (freqs <= 33.0)])
    e72 = np.sum(power[(freqs >= 67.0) & (freqs <= 77.0)])
    return {
        "hf": float(hf / total_power),
        "e28": float(e28 / total_power),
        "e72": float(e72 / total_power),
        "res": float((e28 + e72) / total_power),
    }


def unbalance_ratio(rms_values: list[float]) -> float:
    mean_val = float(np.mean(rms_values)) if rms_values else 0.0
    if mean_val <= EPS:
        return 0.0
    return float(np.std(rms_values) / mean_val)


def zero_sequence_ratio(phases: list[np.ndarray]) -> float:
    phase_rms = [float(np.sqrt(np.nanmean(np.square(sig)))) for sig in phases]
    phase_mean = float(np.mean(phase_rms)) if phase_rms else 0.0
    v0 = np.mean(np.vstack(phases), axis=0)
    v0_rms = float(np.sqrt(np.nanmean(np.square(v0))))
    return safe_divide(v0_rms, phase_mean, 0.0)


def fundamental_phasor(signal: np.ndarray, fs: float, f0: float) -> complex:
    n = signal.size
    if n < 4:
        return 0.0j
    t = np.arange(n) / fs
    kernel = np.exp(-1j * 2.0 * np.pi * f0 * t)
    return complex(np.sum(signal * kernel) / n)


def negative_sequence_ratio(phases: list[np.ndarray], fs: float, f0: float) -> float:
    if not phases:
        return 0.0
    a = complex(np.exp(2j * np.pi / 3.0))
    va, vb, vc = [fundamental_phasor(sig, fs, f0) for sig in phases]
    v1 = (va + a * vb + a**2 * vc) / 3.0
    v2 = (va + a**2 * vb + a * vc) / 3.0
    return safe_divide(abs(v2), abs(v1), 0.0)


def lissajous_metrics(v: np.ndarray, i: np.ndarray) -> tuple[float, float]:
    if v.size < 4 or i.size < 4:
        return 0.0, 0.0
    corr = float(abs(np.corrcoef(v, i)[0, 1])) if np.nanstd(v) > 0 and np.nanstd(i) > 0 else 0.0
    x = np.asarray(v, dtype=float)
    y = np.asarray(i, dtype=float)
    area = 0.5 * abs(np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0])
    box = max((np.nanmax(x) - np.nanmin(x)) * (np.nanmax(y) - np.nanmin(y)), EPS)
    return corr, float(area / box)


def event_window_masks(t: np.ndarray, t_event: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = (t >= max(t_event - 0.10, t[0])) & (t < t_event)
    post = (t >= t_event) & (t <= min(t_event + 0.10, t[-1]))
    event = (t >= max(t_event - 0.04, t[0])) & (t <= min(t_event + 0.12, t[-1]))
    if pre.sum() < 10:
        pre = t < t_event
    if post.sum() < 10:
        post = t >= t_event
    if event.sum() < 10:
        event = np.ones_like(t, dtype=bool)
    return pre, post, event


def compute_bus_features(df: pd.DataFrame, bus: int, t_event: float, fs: float, f0: float) -> dict[str, float]:
    cols = signal_columns_for_bus(bus)
    missing = [column for column in cols.values() if column not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing columns for bus {bus}: {missing}")
    t = df["Time"].to_numpy(dtype=float)
    pre_mask, post_mask, event_mask = event_window_masks(t, t_event)
    n_cycle = max(1, int(round(fs / f0)))

    voltages = {phase: df[f"V{phase.lower()}_{bus}"].to_numpy(dtype=float) for phase in PHASES}
    currents = {phase: df[f"I{phase.lower()}_{bus}"].to_numpy(dtype=float) for phase in PHASES}

    features: dict[str, float] = {}
    sag_values = []
    dv_energies = []
    di_energies = []
    hf_ratios_v = []
    res_ratios_v = []
    hf_ratios_i = []
    res_ratios_i = []
    i_rms_jump = []
    liss_corrs = []
    liss_areas = []
    v_pre_rms_values = []
    v_post_rms_values = []
    i_pre_rms_values = []
    i_post_rms_values = []
    v_post_signals = []
    i_post_signals = []

    for phase in PHASES:
        v = voltages[phase]
        i = currents[phase]
        v_rms = rolling_rms(v, n_cycle)
        i_rms = rolling_rms(i, n_cycle)
        pre_v = float(np.nanmean(v_rms[pre_mask]))
        post_v = float(np.nanmean(v_rms[post_mask]))
        post_v_min = float(np.nanmin(v_rms[post_mask]))
        pre_i = float(np.nanmean(i_rms[pre_mask]))
        post_i = float(np.nanmean(i_rms[post_mask]))
        sag = max(0.0, 1.0 - safe_divide(post_v_min, pre_v, 1.0))
        dv = cycle_difference(v, n_cycle)
        di = cycle_difference(i, n_cycle)
        dv_event = dv[event_mask]
        di_event = di[event_mask]
        dv_energy = float(np.sqrt(np.nanmean(np.square(dv_event))))
        di_energy = float(np.sqrt(np.nanmean(np.square(di_event))))
        spec_v = spectral_ratios(dv_event, fs)
        spec_i = spectral_ratios(di_event, fs)
        corr, area = lissajous_metrics(v[event_mask], i[event_mask])

        features[f"sag_{phase}"] = sag
        features[f"dV_energy_{phase}"] = dv_energy
        features[f"dI_energy_{phase}"] = di_energy
        features[f"I_rms_pre_{phase}"] = pre_i
        features[f"I_rms_post_{phase}"] = post_i
        features[f"I_rms_jump_{phase}"] = post_i - pre_i
        features[f"dV_HF_ratio_{phase}"] = spec_v["hf"]
        features[f"dV_E28_ratio_{phase}"] = spec_v["e28"]
        features[f"dV_E72_ratio_{phase}"] = spec_v["e72"]
        features[f"dV_Res_ratio_{phase}"] = spec_v["res"]
        features[f"dI_HF_ratio_{phase}"] = spec_i["hf"]
        features[f"dI_Res_ratio_{phase}"] = spec_i["res"]
        features[f"liss_corr_abs_{phase}"] = corr
        features[f"liss_area_norm_{phase}"] = area

        sag_values.append(sag)
        dv_energies.append(dv_energy)
        di_energies.append(di_energy)
        hf_ratios_v.append(spec_v["hf"])
        res_ratios_v.append(spec_v["res"])
        hf_ratios_i.append(spec_i["hf"])
        res_ratios_i.append(spec_i["res"])
        i_rms_jump.append(post_i - pre_i)
        liss_corrs.append(corr)
        liss_areas.append(area)
        v_pre_rms_values.append(pre_v)
        v_post_rms_values.append(post_v)
        i_pre_rms_values.append(pre_i)
        i_post_rms_values.append(post_i)
        v_post_signals.append(v[post_mask])
        i_post_signals.append(i[post_mask])

    features["max_sag"] = nanmax(sag_values)
    features["mean_sag"] = nanmean(sag_values)
    features["dV_energy_3ph_mean"] = nanmean(dv_energies)
    features["dV_energy_3ph_max"] = nanmax(dv_energies)
    features["dI_energy_3ph_mean"] = nanmean(di_energies)
    features["dI_energy_3ph_max"] = nanmax(di_energies)
    features["I_rms_jump_3ph_max"] = nanmax(i_rms_jump)
    features["dV_HF_ratio_3ph_max"] = nanmax(hf_ratios_v)
    features["dV_Res_ratio_3ph_max"] = nanmax(res_ratios_v)
    features["dI_HF_ratio_3ph_max"] = nanmax(hf_ratios_i)
    features["dI_Res_ratio_3ph_max"] = nanmax(res_ratios_i)

    features["V_unbalance_pre"] = unbalance_ratio(v_pre_rms_values)
    features["V_unbalance_post"] = unbalance_ratio(v_post_rms_values)
    features["Delta_V_unbalance"] = features["V_unbalance_post"] - features["V_unbalance_pre"]
    features["I_unbalance_pre"] = unbalance_ratio(i_pre_rms_values)
    features["I_unbalance_post"] = unbalance_ratio(i_post_rms_values)
    features["Delta_I_unbalance"] = features["I_unbalance_post"] - features["I_unbalance_pre"]

    features["V0_ratio"] = zero_sequence_ratio(v_post_signals)
    features["I0_ratio"] = zero_sequence_ratio(i_post_signals)
    features["V2_ratio"] = negative_sequence_ratio(v_post_signals, fs, f0)
    features["I2_ratio"] = negative_sequence_ratio(i_post_signals, fs, f0)

    v_pre_mean = nanmean(v_pre_rms_values)
    v_post_mean = nanmean(v_post_rms_values)
    i_pre_mean = nanmean(i_pre_rms_values)
    i_post_mean = nanmean(i_post_rms_values)
    z_pre = safe_divide(v_pre_mean, i_pre_mean, 0.0)
    z_post = safe_divide(v_post_mean, i_post_mean, 0.0)
    features["Z_app_pre"] = z_pre
    features["Z_app_post"] = z_post
    features["Delta_Z_app"] = z_post - z_pre
    features["Z_drop_ratio"] = safe_divide(z_pre - z_post, z_pre, 0.0)

    features["liss_corr_abs_min"] = float(np.nanmin(liss_corrs)) if liss_corrs else 0.0
    features["liss_area_norm_min"] = float(np.nanmin(liss_areas)) if liss_areas else 0.0
    return features


def compute_case_features(case: CaseMetadata, f0: float = 50.0) -> CaseFeatureBundle:
    df = load_waveform_case(case)
    t = df["Time"].to_numpy(dtype=float)
    if t.size < 2:
        raise FeatureComputationError(f"Not enough samples in {case.case_name}")
    fs = float(1.0 / np.nanmedian(np.diff(t)))
    buses = detect_observed_buses(df.columns)
    bus_rows: list[dict[str, object]] = []
    all_feature_names: set[str] = set()
    for bus in buses:
        row = {
            "CaseName": case.case_name,
            "EventType": case.event_type,
            "TargetBus": case.target_bus,
            "ObservedBus": bus,
            "EventTime": case.event_time,
            "SamplingRateHz": fs,
        }
        row.update(compute_bus_features(df, bus, case.event_time, fs, f0))
        bus_rows.append(row)
        all_feature_names.update(set(row) - {"CaseName", "EventType", "TargetBus", "ObservedBus", "EventTime", "SamplingRateHz"})

    by_bus = pd.DataFrame(bus_rows).sort_values("ObservedBus").reset_index(drop=True)
    if by_bus.empty:
        raise FeatureComputationError(f"No observed buses detected for {case.case_name}")

    star_bus_dv_row = by_bus.loc[by_bus["dV_energy_3ph_max"].idxmax()]
    star_bus_di_row = by_bus.loc[by_bus["dI_energy_3ph_max"].idxmax()]
    by_case = pd.DataFrame(
        [
            {
                "CaseName": case.case_name,
                "EventType": case.event_type,
                "TargetBus": case.target_bus,
                "EventTime": case.event_time,
                "SamplingRateHz": fs,
                "ObservedBusCount": int(by_bus["ObservedBus"].nunique()),
                "star_bus_dV": int(star_bus_dv_row["ObservedBus"]),
                "star_bus_dI": int(star_bus_di_row["ObservedBus"]),
                "max_dV_energy": float(by_bus["dV_energy_3ph_max"].max()),
                "max_dI_energy": float(by_bus["dI_energy_3ph_max"].max()),
                "max_sag": float(by_bus["max_sag"].max()),
                "max_HF_ratio": float(by_bus["dV_HF_ratio_3ph_max"].max()),
                "max_Res_ratio": float(by_bus["dV_Res_ratio_3ph_max"].max()),
                "max_V0_ratio": float(by_bus["V0_ratio"].max()),
                "max_I0_ratio": float(by_bus["I0_ratio"].max()),
                "max_V_unbalance": float(by_bus["V_unbalance_post"].max()),
                "max_I_unbalance": float(by_bus["I_unbalance_post"].max()),
                "max_Z_drop_ratio": float(by_bus["Z_drop_ratio"].max()),
                "min_Z_app_post": float(by_bus["Z_app_post"].min()),
            }
        ]
    )

    wide_row = by_case.iloc[0].to_dict()
    feature_columns = sorted(all_feature_names)
    for _, row in by_bus.iterrows():
        bus = int(row["ObservedBus"])
        for feature_name in feature_columns:
            wide_row[f"Bus{bus:02d}__{feature_name}"] = row.get(feature_name, np.nan)
    by_case_wide = pd.DataFrame([wide_row])
    return CaseFeatureBundle(metadata=case, by_bus=by_bus, by_case=by_case, by_case_wide=by_case_wide)


def _compute_case_features_task(args: tuple[CaseMetadata, float]) -> CaseFeatureBundle:
    case, f0 = args
    return compute_case_features(case, f0=f0)


def compute_feature_tables(cases: list[CaseMetadata], f0: float = 50.0, max_workers: int | None = None) -> FeatureTables:
    workers = 1 if len(cases) < 4 else (max_workers or min(4, os.cpu_count() or 1))
    if workers <= 1:
        bundles = [compute_case_features(case, f0=f0) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            bundles = list(executor.map(_compute_case_features_task, [(case, f0) for case in cases]))
    by_bus = pd.concat([bundle.by_bus for bundle in bundles], ignore_index=True)
    by_case = pd.concat([bundle.by_case for bundle in bundles], ignore_index=True)
    by_case_wide = pd.concat([bundle.by_case_wide for bundle in bundles], ignore_index=True)
    by_bus = by_bus.sort_values(["EventType", "TargetBus", "CaseName", "ObservedBus"]).reset_index(drop=True)
    by_case = by_case.sort_values(["EventType", "TargetBus", "CaseName"]).reset_index(drop=True)
    by_case_wide = by_case_wide.sort_values(["EventType", "TargetBus", "CaseName"]).reset_index(drop=True)
    return FeatureTables(by_bus=by_bus, by_case=by_case, by_case_wide=by_case_wide)


def export_feature_tables(feature_tables: FeatureTables, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    feature_tables.by_bus.to_csv(root / "feature_table_by_bus.csv", index=False)
    feature_tables.by_case.to_csv(root / "feature_table_by_case.csv", index=False)
    feature_tables.by_case_wide.to_csv(root / "feature_table_by_case_wide.csv", index=False)


def select_feature_columns_by_group(by_case_wide: pd.DataFrame, group_name: str) -> list[str]:
    if group_name not in BUS_FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group: {group_name}")
    bus_patterns = BUS_FEATURE_GROUPS[group_name]
    case_columns = [col for col in CASE_SUMMARY_GROUPS[group_name] if col in by_case_wide.columns]
    bus_columns = []
    for column in by_case_wide.columns:
        if not column.startswith("Bus"):
            continue
        suffix = column.split("__", 1)[1] if "__" in column else column
        if group_name == "All_features":
            bus_columns.append(column)
        elif any(pattern in suffix for pattern in bus_patterns):
            bus_columns.append(column)
    feature_columns = case_columns + sorted(bus_columns)
    return feature_columns
