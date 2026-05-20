from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wmu_project.waveform_io import output_roots
from wmu_project.waveform_localization import evaluate_fault_localization
from wmu_project.waveform_utils import to_local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run waveform fault localization analysis")
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge-file", default="data/ieee30_edges.csv")
    args = parser.parse_args()

    by_case_wide = pd.read_csv(to_local_path(args.feature_table))
    reports_dir, figures_dir = output_roots(to_local_path(args.output_dir))
    evaluate_fault_localization(by_case_wide, Path(args.edge_file), reports_dir, figures_dir)
    print(reports_dir / "fault_localization_preliminary.csv")


if __name__ == "__main__":
    main()
