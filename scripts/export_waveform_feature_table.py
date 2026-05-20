from __future__ import annotations

import argparse

from wmu_project.waveform_features import compute_feature_tables, export_feature_tables
from wmu_project.waveform_io import list_cases
from wmu_project.waveform_utils import to_local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export waveform feature tables")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--f0", type=float, default=50.0)
    args = parser.parse_args()

    feature_tables = compute_feature_tables(list_cases(to_local_path(args.input_dir)), f0=args.f0)
    export_feature_tables(feature_tables, to_local_path(args.output_dir))
    print("feature_table_by_bus.csv")
    print("feature_table_by_case.csv")
    print("feature_table_by_case_wide.csv")


if __name__ == "__main__":
    main()
