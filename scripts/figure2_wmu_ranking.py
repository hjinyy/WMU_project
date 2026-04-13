from pathlib import Path

from wmu_project.analysis import analyze_fault_workbook, build_argument_parser, save_figure2


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    result = analyze_fault_workbook(args.input, thr_dv=args.thr_dv, t_event=args.t_event, f0=args.f0)
    save_figure2(result, Path(args.output_dir))


if __name__ == "__main__":
    main()
