import argparse

from fgpt import __author__, __license__, __version__


def _add_isolate_args(parser):
    parser.add_argument(
        "--rest_of_path",
        type=str,
        required=True,
        help="Relative path to the directory containing the target Fortran module.",
    )
    parser.add_argument(
        "--target_module",
        type=str,
        required=True,
        help="Name of the module to isolate (without .f90).",
    )
    parser.add_argument(
        "--work", type=str, required=True, help="Working directory root (e.g. $works)."
    )
    parser.add_argument(
        "--parent_subroutine",
        type=str,
        default="hydrol_main",
        help="Parent subroutine containing the targets.",
    )
    parser.add_argument(
        "--target_subroutines",
        type=str,
        nargs="+",
        required=True,
        help="List of subroutine names to isolate.",
    )
    parser.add_argument(
        "--f2py",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Also transpile to Python after isolation (True/False).",
    )
    parser.add_argument(
        "--openacc",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable OpenACC support (True/False).",
    )
    parser.add_argument(
        "--tapenade",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Prepare output for Tapenade auto-differentiation (True/False).",
    )


def _add_autodiff_args(parser):
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g. template.yaml).",
    )
    parser.add_argument(
        "--class_file",
        type=str,
        required=True,
        help="Path to the global module Python file.",
    )
    parser.add_argument(
        "--main_file",
        type=str,
        required=True,
        help="Path to the main driver Python file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["jax", "fwd", "bwd"],
        default="jax",
        help="Transformation mode (default: jax).",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Directory for benchmark outputs.",
    )


def _run_isolate(args):
    from fgpt.isolator import Isolator

    isolator = Isolator(
        rest_of_path=args.rest_of_path,
        target_module=args.target_module,
        work=args.work,
        openacc=args.openacc,
        tapenade=args.tapenade,
        f2py=args.f2py,
    )
    isolator.run(
        parent_subroutine=args.parent_subroutine,
        target_subroutines=args.target_subroutines,
    )


def _run_autodiff(args):
    from fgpt.autodiff import AutoDiff

    autodiff = AutoDiff(
        config_path=args.config_path,
        benchmark_dir=args.benchmark_dir,
        mode=args.mode,
    )
    autodiff.transform(
        class_file=args.class_file,
        main_file=args.main_file,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="fgpt",
        description=(
            "FGPT — Fortran-to-Python transpiler and JAX converter.\n\n"
            "Commands:\n"
            "  isolate   Extract and transpile a Fortran subroutine (Stages 1 & 2)\n"
            "  autodiff  Convert a NumPy Python module to JAX/Equinox (Stage 3)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"fgpt {__version__} — {__author__} — {__license__}",
    )

    # fgpt isolate
    isolate_parser = subparsers.add_parser(
        "isolate",
        help="Extract and transpile a Fortran subroutine (Stages 1 & 2).",
        description="Isolate a Fortran subroutine, validate it, and optionally transpile to Python.",
    )
    _add_isolate_args(isolate_parser)

    # fgpt autodiff
    autodiff_parser = subparsers.add_parser(
        "autodiff",
        help="Convert a NumPy Python module to JAX/Equinox (Stage 3).",
        description="Transform a NumPy-based Python class into a JAX/Equinox-compatible module.",
    )
    _add_autodiff_args(autodiff_parser)

    args = parser.parse_args()

    if args.command == "isolate":
        _run_isolate(args)
    elif args.command == "autodiff":
        _run_autodiff(args)


if __name__ == "__main__":
    main()
