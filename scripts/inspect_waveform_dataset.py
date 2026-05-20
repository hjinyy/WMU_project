from __future__ import annotations

import argparse

from wmu_project.waveform_io import list_cases, output_roots
from wmu_project.waveform_quality import build_quality_outputs, write_quality_outputs
from wmu_project.waveform_utils import build_case_index_frame, to_local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect waveform dataset quality")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = to_local_path(args.input_dir)
    output_dir = to_local_path(args.output_dir)
    reports_dir, _ = output_roots(output_dir)

    cases = list_cases(input_dir)
    build_case_index_frame(cases).to_csv(reports_dir / "dataset_case_index.csv", index=False)
    report, summary = build_quality_outputs(cases)
    write_quality_outputs(report, summary, reports_dir, reports_dir / "data_quality_summary.md")
    print(summary)


if __name__ == "__main__":
    main()
