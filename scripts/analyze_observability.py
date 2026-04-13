from pathlib import Path

from wmu_project.analysis import analyze_fault_workbook, build_argument_parser, save_excel


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    result = analyze_fault_workbook(args.input, thr_dv=args.thr_dv, t_event=args.t_event, f0=args.f0)
    save_excel(result, output_dir / "observability_sweep_summary.xlsx")


if __name__ == "__main__":
    main()
