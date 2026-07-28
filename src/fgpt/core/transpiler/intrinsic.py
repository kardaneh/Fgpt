# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Intrinsic function normalization and transformation layer.

This module defines a unified representation of Fortran intrinsic functions
and provides utilities to normalize their call signatures into a consistent
intermediate representation suitable for AST transformation and code lowering.

Each intrinsic is described using an IntrinsicSignature object that encodes:
- Canonical argument order
- Optional arguments
- Default values
- Variadic behavior (e.g. MIN/MAX)
- Fortran → Python argument mapping
- Optional transformation logic for semantic rewriting

This layer is used during transpilation to convert Fortran intrinsics into
equivalent Python / NumPy / AST constructs with consistent semantics.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class IntrinsicSignature:
    """
    Describes the signature and transformation rules of a Fortran intrinsic.

    This structure defines how a Fortran intrinsic function should be interpreted
    and normalized during AST transformation.
    """

    name: str
    args: list[str]  # canonical order
    optional: set[str]  # optional args
    defaults: dict[str, Any]  # default values if omitted
    varargs: bool = False  # e.g. MAX, MIN
    arg_map: dict[str, str] = None  # Fortran -> Python names
    transform: Callable | None = None


def normalize_intrinsic_call(
    signature: IntrinsicSignature, positional_args: list, keyword_args: dict
) -> dict:
    """
    Normalize a Fortran intrinsic call into a structured dictionary.

    This function aligns positional and keyword arguments with the canonical
    intrinsic signature, applies defaults, validates required arguments, and
    optionally applies semantic transformations.

    Parameters
    ----------
    signature : IntrinsicSignature
        Intrinsic definition used for normalization.
    positional_args : List
        Positional arguments from the parsed call.
    keyword_args : Dict
        Keyword arguments from the parsed call.

    Returns
    -------
    Dict
        Normalized argument dictionary ready for transformation.

    Raises
    ------
    ValueError
        If required arguments are missing.
    """
    # Step 1: initialize with defaults
    normalized = dict(signature.defaults)

    # Step 2: positional arguments
    if signature.varargs:
        normalized[signature.args[0]] = positional_args
    else:
        for arg_name, value in zip(signature.args, positional_args):
            normalized[arg_name] = value

    # Step 3: keyword arguments override
    for key, value in keyword_args.items():
        normalized[key.lower()] = value

    # Step 4: required args check
    for arg in signature.args:
        if arg not in signature.optional and normalized.get(arg) is None:
            raise ValueError(f"Missing required argument: {arg}")

    # Step 5: transform
    if signature.transform:
        normalized = signature.transform(normalized)

    return normalized


def dim_to_axis(dim_ast: ast.AST) -> ast.AST | None:
    """
    Convert Fortran-style DIM argument indexing to Python (0-based) axis.

    For integer constants, converts `dim` → `dim - 1`.
    For symbolic expressions, rewrites as a subtraction AST node.

    Parameters
    ----------
    dim_ast : ast.AST or None
        AST node representing the dimension argument.

    Returns
    -------
    ast.AST or None
        Converted AST expression suitable for Python indexing.
    """
    if dim_ast is None:
        return None

    if isinstance(dim_ast, ast.Constant) and isinstance(dim_ast.value, int):
        return ast.Constant(value=dim_ast.value - 1)

    # fallback: symbolic expression
    return ast.BinOp(left=dim_ast, op=ast.Sub(), right=ast.Constant(value=1))


SUM = IntrinsicSignature(
    name="SUM",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis", "mask": "where"},
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"]),
        "where": args["mask"],
    },
)

MAXLOC = IntrinsicSignature(
    name="MAXLOC",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {"array": args["array"], "axis": dim_to_axis(args["dim"])},
)

MINLOC = IntrinsicSignature(
    name="MINLOC",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {"array": args["array"], "axis": dim_to_axis(args["dim"])},
)

SQRT = IntrinsicSignature(
    name="SQRT", args=["array"], optional=set(), defaults={}, arg_map={"array": "a"}
)

MAXVAL = IntrinsicSignature(
    name="MAXVAL",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {"array": args["array"], "axis": dim_to_axis(args["dim"])},
)

MINVAL = IntrinsicSignature(
    name="MINVAL",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {"array": args["array"], "axis": dim_to_axis(args["dim"])},
)

RESHAPE = IntrinsicSignature(
    name="RESHAPE",
    args=["source", "shape", "pad", "order"],
    optional={"pad", "order"},
    defaults={"pad": None, "order": ast.Constant(value="F")},
    arg_map={"source": "a", "shape": "newshape", "order": "order"},
    transform=lambda args: {
        "source": args["source"],
        "newshape": args["shape"],
        "order": args["order"],
    },
)

PRODUCT = IntrinsicSignature(
    name="PRODUCT",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis", "mask": "where"},
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"]),
        "where": args["mask"],
    },
)

MIN = IntrinsicSignature(
    name="MIN",
    args=["values"],
    optional=set(),
    defaults={},
    varargs=True,
    arg_map={"values": "values"},
)

MAX = IntrinsicSignature(
    name="MAX",
    args=["values"],
    optional=set(),
    defaults={},
    varargs=True,
    arg_map={"values": "values"},
)

DOT_PRODUCT = IntrinsicSignature(
    name="DOT_PRODUCT",
    args=["a", "b"],
    optional=set(),
    defaults={},
    arg_map={"a": "a", "b": "b"},
)

MATMUL = IntrinsicSignature(
    name="MATMUL",
    args=["a", "b"],
    optional=set(),
    defaults={},
    arg_map={"a": "a", "b": "b"},
)

# Defines the ensemble of intrinsic signature which requires some modificaiton
# due to the fact that they don't represent mirror like transformation
intrinsic_signatures = {
    "SUM": SUM,
    "PRODUCT": PRODUCT,
    "MAXVAL": MAXVAL,
    "MINVAL": MINVAL,
    "MAXLOC": MAXLOC,
    "MINLOC": MINLOC,
    "MATMUL": MATMUL,
    "DOT_PRODUCT": DOT_PRODUCT,
    "RESHAPE": RESHAPE,
    "MIN": MIN,
    "MAX": MAX,
    "SQRT": SQRT,
}
