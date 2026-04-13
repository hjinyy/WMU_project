from pathlib import Path

from wmu_project.analysis import analyze_fault_workbook, build_argument_parser, resolve_default_paths, save_figure6


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    # If no arguments are given in PyCharm, the script uses the fixed default
    # workbook path and writes results into the repository outputs/ folder.
    input_path, output_dir = resolve_default_paths(args.input, args.output_dir, __file__)
    result = analyze_fault_workbook(input_path, thr_dv=args.thr_dv, t_event=args.t_event, f0=args.f0)
    # The 3D scatter is saved to disk and also shown as a popup/tool window.
    out = save_figure6(result, output_dir, show=True)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
