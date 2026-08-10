# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

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

    parser.add_argument(
        "--py2jx",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "Enable Python to JAX conversion (True/False). "
            "Requires --f2py true, since JAX conversion operates on the "
            "transpiled Python output."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["jax", "fwd", "bwd"],
        default="jax",
        help="Transformation mode: 'jax' (default), 'fwd' (forward-mode AD), 'bwd' (reverse-mode AD).",
    )

    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Directory for benchmark outputs. Defaults to <cwd>/benchmark if not set.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the YAML config file containing code templates (e.g. template.yaml).",
    )

    parser.add_argument(
        "--vectorize",
        nargs="+",
        metavar="LOOP_BOUND",
        default=["kjpindex"],
        help=(
            "Loop upper-bound variables to vectorize. "
            "Example: --vectorize kjpindex nvm npts"
        ),
    )


def _add_autodiff_args(parser):
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
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

    parser.add_argument(
        "--vectorize",
        nargs="+",
        metavar="LOOP_BOUND",
        default=["kjpindex"],
        help=(
            "Loop upper-bound variables to vectorize. "
            "Example: --vectorize kjpindex nvm npts"
        ),
    )


def _run_isolate(args):
    from fgpt.isolator import Isolator

    # py2jx (Stage 3, Python -> JAX) operates on the transpiled Python
    # output produced by f2py (Stage 2), so f2py must be enabled whenever
    # py2jx is requested.
    if args.py2jx and not args.f2py:
        args.f2py = True
        print(
            "[fgpt] --py2jx true requires --f2py true; automatically enabling --f2py."
        )

    isolator = Isolator(
        rest_of_path=args.rest_of_path,
        target_module=args.target_module,
        work=args.work,
        openacc=args.openacc,
        tapenade=args.tapenade,
        f2py=args.f2py,
        py2jx=args.py2jx,
    )

    if args.tapenade:
        isolator.logger.warning(
            "Tapenade automatic differentiation is active. "
            "Please ensure the following requirements are met:\n"
            "  - Tapenade is installed (version 3.16+ recommended)\n"
            "  - TAPENADE_HOME environment variable is set\n"
            "  - Tapenade executable is available in PATH\n"
            "  - C compiler (mpicc) is available for building the runtime\n"
            "  - Tapenade runtime (adStack.c) can be compiled"
        )

    isolator.run(
        benchmark_dir=args.benchmark_dir,
        config_path=args.config_path,
        vectorize=args.vectorize,
        mode=args.mode,
        parent_subroutine=args.parent_subroutine,
        target_subroutines=args.target_subroutines,
    )


def _run_autodiff(args):
    from fgpt.autodiff import AutoDiff

    autodiff = AutoDiff(
        config_path=args.config_path,
        vectorize=args.vectorize,
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
            "  isolate   Extract and transpile a Fortran subroutine, "
            "optionally through Stages 1-3 (isolate, transpile, JAX conversion)\n"
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
        help="Extract, transpile, and (optionally) JAX-convert a Fortran subroutine (Stages 1-3).",
        description=(
            "Isolate a Fortran subroutine, validate it, and optionally run it "
            "through the full pipeline: transpile to Python (Stage 2, --f2py) "
            "and convert the result to JAX/Equinox (Stage 3, --py2jx)."
        ),
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

    if args.openacc:
        raise RuntimeError(
            "FGPT Fortran GPU porting via OpenACC is not available in the public version.\n"
            "Contact Kazem Ardaneh (kardaneh@ipsl.fr) for more information about "
            "the GPU-enabled version."
        )

    if args.command == "isolate":
        _run_isolate(args)
    elif args.command == "autodiff":
        _run_autodiff(args)


if __name__ == "__main__":
    main()
