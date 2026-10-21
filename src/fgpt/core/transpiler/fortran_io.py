# Copyright 2026 CNRS
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Fortran I/O and derived-type runtime helpers.

This module provides Python-side runtime helpers used by the Fortran-to-Python
translation layer. It contains:

* runtime support for representing Fortran derived types as Python dataclasses;
* helpers for reading scalar derived types from unformatted Fortran files;
* helpers for reading arrays of derived types stored in Fortran order;
* metadata classes used to track the state of Fortran I/O units during
  translation, including file status, form, access mode, and file position.

The AST-builder functions in this module return Python ``ast`` nodes or lists
of nodes that can be inserted into generated Python modules. The generated
runtime helpers rely on ``numpy`` and ``dataclasses`` being available in the
execution environment.

The I/O metadata classes are used by the translation layer to distinguish
formatted and unformatted Fortran I/O and to select the corresponding Python
runtime representation.
"""

import ast
from dataclasses import dataclass

FORTRAN_DERIVED_TYPE = """
class FortranDerivedType:

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "FortranDerivedType":
        kwargs = {}
        for f in fields(cls):
            default = (
                f.default_factory() if f.default_factory is not MISSING
                else f.default
            )
            if isinstance(default, np.ndarray):
                full_shape, dtype = shape + default.shape, default.dtype
            else:
                full_shape, dtype = shape, np.asarray(default).dtype
            kwargs[f.name] = np.zeros(full_shape, dtype=dtype)

        instance = cls(**kwargs)
        instance._shape = tuple(shape)
        return instance

    def __getitem__(self, idx):
        return type(self)(**{
            f.name: getattr(self, f.name)[idx] for f in fields(self)
        })

    def __setitem__(self, idx, value):
        for f in fields(self):
            getattr(self, f.name)[idx] = getattr(value, f.name)
"""


def build_fortran_derived_type() -> ast.ClassDef:
    """
    Build the AST for the ``FortranDerivedType`` runtime helper.

    The helper is defined from :data:`FORTRAN_DERIVED_TYPE` and returned as
    a class definition suitable for insertion into a generated Python module.

    Returns
    -------
    ast.ClassDef
        AST representation of the ``FortranDerivedType`` class.
    """
    return ast.parse(FORTRAN_DERIVED_TYPE).body[0]


# Contains as well the reading functions necessary for the derived type reading for Scalar derived types
# and Struct of Arrays with mixed components

DERIVED_TYPE_SCALAR = """
def _struct_dtype_scalar(instance):
    spec = []
    for f in fields(instance):
        val = getattr(instance, f.name)
        if isinstance(val, np.ndarray):
            spec.append((f.name, val.dtype, val.size))
        else:
            spec.append((f.name, np.asarray(val).dtype))
    return np.dtype(spec)

def read_fortran_struct_scalar(ffile, instance):
    dtype = _struct_dtype_scalar(instance)
    rec = ffile.read_record(dtype)[0]
    for f in fields(instance):
        val = getattr(instance, f.name)
        if isinstance(val, np.ndarray):
            raw = rec[f.name]
            val[...] = raw.reshape(val.shape, order="F")
        else:
            setattr(instance, f.name, type(val)(rec[f.name]))
"""


def build_fortan_derived_type_scalar_read() -> list:
    return ast.parse(DERIVED_TYPE_SCALAR).body


DERIVED_TYPE_SOA = """
def _struct_dtype(instance):
    outer_ndim = len(instance._shape)
    spec = []
    for f in fields(instance):
        arr = getattr(instance, f.name)
        inner_shape = arr.shape[outer_ndim:]
        spec.append((f.name, arr.dtype, inner_shape) if inner_shape else (f.name, arr.dtype))
    return np.dtype(spec)

def _unpack_fortran_order(flat, shape):
    inner_shape = flat.shape[1:]
    reshaped = flat.reshape(inner_shape + tuple(shape), order="F")
    inner_ndim = len(inner_shape)
    axes = (tuple(range(inner_ndim, inner_ndim + len(shape))) + tuple(range(inner_ndim)))
    return reshaped.transpose(axes)

def read_fortran_struct_array(ffile, instance):
    dtype = _struct_dtype(instance)
    raw = ffile.read_record(dtype)
    shape = instance._shape
    for f in fields(instance):
        getattr(instance, f.name)[...] = _unpack_fortran_order(raw[f.name], shape)
"""


def build_fortan_derived_type_array_read() -> list:
    return ast.parse(DERIVED_TYPE_SOA).body


class IOStatus:
    REPLACE = "replace"
    OLD = "old"
    NEW = "new"
    UNKNOWN = "unknown"
    SCRATCH = "scratch"


class IOForm:
    FORMATTED = "formatted"
    UNFORMATTED = "unformatted"


class IOAccess:
    SEQUENTIAL = "sequential"
    DIRECT = "direct"
    STREAM = "stream"


class IOPosition:
    ASIS = "asis"
    REWIND = "rewind"
    APPEND = "append"


@dataclass
class IOUnit:
    """
    Tracks the OPEN-time state of one Fortran I/O unit for the duration of
    the enclosing subroutine (unit variables are subroutine-scoped, so
    this is reset per subroutine in :meth:`F2NP.handle_subroutine_stmt`),
    so later WRITE/READ/CLOSE statements referencing the same unit
    variable know whether it's formatted or unformatted, and whether it's
    a real file at all (as opposed to stdout, per
    :meth:`F2NP._is_supported_print_target`).
    """

    unit_name: str
    file_expr: ast.expr
    status: str = IOStatus.UNKNOWN
    form: str = IOForm.FORMATTED
    access: str = IOAccess.SEQUENTIAL
    position: str = IOPosition.ASIS
    recl_expr: ast.expr | None = None
    iostat_expr: ast.expr | None = None
