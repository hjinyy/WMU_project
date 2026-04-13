from pathlib import Path

from wmu_project.analysis import analyze_fault_workbook, build_argument_parser, resolve_default_paths, save_figure5


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    # Default workbook/output path logic is shared across all figure scripts.
    input_path, output_dir = resolve_default_paths(args.input, args.output_dir, __file__)
    result = analyze_fault_workbook(input_path, thr_dv=args.thr_dv, t_event=args.t_event, f0=args.f0)
    # The distribution plot is saved and also displayed via plt.show().
    out = save_figure5(result, output_dir, show=True)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
