from pathlib import Path

from wmu_project.analysis import analyze_fault_workbook, build_argument_parser, resolve_default_paths, save_figure4


def main() -> None:
    parser = build_argument_parser()
    parser.add_argument("--edge-csv", default=str(Path(__file__).resolve().parents[1] / "data" / "ieee30_edges.csv"))
    args = parser.parse_args()
    input_path, output_dir = resolve_default_paths(args.input, args.output_dir, __file__)
    result = analyze_fault_workbook(input_path, thr_dv=args.thr_dv, t_event=args.t_event, f0=args.f0)
    out = save_figure4(result, output_dir, args.edge_csv, show=True)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
