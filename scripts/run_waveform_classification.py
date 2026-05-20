from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wmu_project.waveform_classification import evaluate_feature_ablation, evaluate_full_classification
from wmu_project.waveform_io import output_roots
from wmu_project.waveform_utils import to_local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run waveform classification evaluations")
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    by_case_wide = pd.read_csv(to_local_path(args.feature_table))
    reports_dir, figures_dir = output_roots(to_local_path(args.output_dir))
    evaluate_full_classification(by_case_wide, reports_dir, figures_dir)
    evaluate_feature_ablation(by_case_wide, reports_dir, figures_dir)
    print(reports_dir)


if __name__ == "__main__":
    main()
