from pathlib import Path

from wmu_project.analysis import analyze_fault_workbook, build_argument_parser, resolve_default_paths, save_figure1


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    # If --input / --output-dir are not passed, this call automatically uses:
    # input  -> C:\Users\user\Documents\MATLAB\WMU_test\WMU_fault_results_sag.xlsx
    # output -> <repo>/outputs
    input_path, output_dir = resolve_default_paths(args.input, args.output_dir, __file__)
    result = analyze_fault_workbook(input_path, thr_dv=args.thr_dv, t_event=args.t_event, f0=args.f0)

    # show=True means the figure is both saved and displayed with plt.show().
    out = save_figure1(result, output_dir, show=True)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
