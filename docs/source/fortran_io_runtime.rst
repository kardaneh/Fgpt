Fortran I/O Runtime Support
===========================

This page documents the Python runtime support used by the Fortran-to-Python
translation layer for Fortran derived types, unformatted file I/O, I/O unit
metadata, and NetCDF operations.

Contents
--------

.. contents::
   :local:
   :depth: 2

Overview
--------

The runtime support is organized into four main areas:

* **Fortran derived-type representation:** Helpers for representing Fortran
  derived types as Python dataclasses and for creating scalar or
  array-shaped instances.
* **Derived-type binary I/O:** Helpers for reading derived-type data from
  unformatted Fortran files, including nested structures and structures
  containing mixed components.
* **Fortran I/O unit metadata:** Classes for tracking the state of Fortran
  I/O units during translation, including file status, form, access mode,
  and position.
* **NetCDF support:** Metadata definitions, a Python runtime
  (``fortran_netcdf.py``) and translator wiring for the Fortran netCDF-2
  (``nc*``), netCDF-3 (``nf_*``) and ``nf__*`` interfaces.

Fortran Derived-Type Representation
-----------------------------------

``FortranDerivedType`` is the base class of every generated derived-type
dataclass, whether the type is used as a single instance, as an array, or
nested inside another type. It provides:

* ``zeros(shape)`` creates an instance in *structure-of-arrays* form:
  ``shape`` becomes a prefix of the dimensions of every leaf field, at any
  nesting depth. It calls ``_expand`` on a default instance.
* ``_expand(default, shape)`` is a static helper that rebuilds a default
  instance recursively. Nested dataclass fields are expanded field by field,
  and each leaf is replaced by ``np.zeros(shape + leaf.shape)`` with the
  leaf's dtype. Without it, a nested dataclass default would become an
  ``object`` array.
* ``_shape`` is a read-only **property**, not a stored attribute. It is
  derived from the leaves (leaf shape minus the field's own declared
  dimensions), and is ``()`` for a scalar struct. Because it is not a
  dataclass field, it stays correct after ``__getitem__`` and survives
  ``jax.tree_util.tree_map``.
* ``__getitem__(idx)`` indexes every leaf and returns a new instance. It
  recurses through nested dataclasses. Basic indices give views and fancy
  indices give copies.
* ``__setitem__(idx, value)`` assigns the fields of another instance at
  ``idx``. It is NumPy-only, because JAX arrays are immutable.

Generated code never writes through chained access such as
``x[i].items.a[k] = v``. Every write is emitted as one flat subscript on the
leaf (``x.items.a[i, k] = v``), which always updates the original array.

The implementation uses Python dataclasses to inspect the fields and NumPy
to build the arrays. The class is stored as source text in
``FORTRAN_DERIVED_TYPE`` and converted to an ``ast.ClassDef`` by
``build_fortran_derived_type()``. The generated module must import ``np``,
``fields``, ``is_dataclass`` and ``MISSING``.

For the generated dataclasses and their translation rules, see
:doc:`derived_types`.

Derived-Type Binary I/O
-----------------------

The module provides a unified set of runtime helpers for reading Fortran
derived-type data from unformatted files.

The implementation is defined in ``DERIVED_TYPE_READ`` and includes the
following functions:

* ``_struct_dtype(inst, depth)`` constructs a NumPy structured dtype from
  the fields of a derived-type instance. It handles nested dataclass fields
  and array-valued components.
* ``_fortran_axes(groups)`` determines the axis ordering required to
  account for differences between Fortran and NumPy array storage order.
* ``_assign_struct(inst, raw, groups)`` assigns the values read from the
  file to the corresponding fields of the derived-type instance. It handles
  nested structures, array-valued fields, and scalar fields.
* ``read_fortran_struct(ffile, inst)`` reads a record from an unformatted
  Fortran file and populates the supplied derived-type instance.

The unified ``read_fortran_struct()`` implementation handles both scalar
and array-shaped derived-type instances. It also supports nested derived
types and structures with mixed components, rather than relying on separate
read implementations for scalar and array cases.

The function uses the instance's shape and field information to construct
the appropriate structured dtype, read the record, and assign the resulting
data to the instance.

AST Generation
--------------

The module provides two AST-builder functions for generating the derived-type
runtime helpers:

* ``build_fortran_derived_type()``
    Builds and returns an ``ast.ClassDef`` node representing the
    ``FortranDerivedType`` class.

* ``build_fortran_struct_read()``
    Builds and returns a list of ``ast.FunctionDef`` nodes representing the
    derived-type reading helpers defined in ``DERIVED_TYPE_READ``.

These AST nodes can be inserted into generated Python modules by the
translation layer.

Fortran I/O Unit Metadata
-------------------------

The module defines metadata classes used to track the state of Fortran I/O
units during translation.

IOStatus
~~~~~~~~

``IOStatus`` defines the supported file status values:

* ``REPLACE``
* ``OLD``
* ``NEW``
* ``UNKNOWN``
* ``SCRATCH``

IOForm
~~~~~~

``IOForm`` describes the format of an I/O operation:

* ``FORMATTED``
* ``UNFORMATTED``

IOAccess
~~~~~~~~

``IOAccess`` defines the supported access modes:

* ``SEQUENTIAL``
* ``STREAM``

IOPosition
~~~~~~~~~~

``IOPosition`` defines the file positioning options:

* ``ASIS``
* ``REWIND``
* ``APPEND``

IOUnit
~~~~~~

``IOUnit`` is a dataclass that tracks the OPEN-time state of a Fortran I/O
unit during the translation of a subroutine.

It stores the following information:

* ``unit_name``: Name of the Fortran unit variable.
* ``file_expr``: AST expression representing the associated file.
* ``status``: File status, using ``IOStatus`` values.
* ``form``: File format, using ``IOForm`` values.
* ``access``: Access mode, using ``IOAccess`` values.
* ``position``: Initial file position, using ``IOPosition`` values.
* ``recl_expr``: AST expression representing the record length, when
  specified.
* ``iostat_expr``: AST expression representing the IOSTAT expression,
  when specified.

The metadata is maintained for the duration of the enclosing subroutine.
This allows subsequent READ, WRITE, and CLOSE statements referring to the
same unit variable to use the information established by the OPEN statement.

The translation layer uses this information to distinguish formatted from
unformatted I/O and to determine whether a unit represents a real file or a
supported output target such as standard output.

NetCDF Runtime Support
----------------------

The NetCDF support has three parts:

* ``netcdf_map.py``: metadata used by the translator (names, constants,
  argument roles).
* ``fortran_netcdf.py``: the Python runtime that the generated code calls.
  It is a thin layer over ``netCDF4`` and NumPy.
* ``F2NP``: rewrites Fortran calls into calls to that runtime.

Supported interfaces
~~~~~~~~~~~~~~~~~~~~

Three Fortran interfaces are supported.

* **netCDF-3 (``nf_*``)**: file, dimension, variable, attribute and data
  routines, including the typed variants (``_text``, ``_int1``, ``_int2``,
  ``_int``, ``_real``, ``_double``) of the ``var``, ``var1``, ``vara``,
  ``vars`` and ``varm`` access routines.
* **``nf__*`` variants**: ``nf__create``, ``nf__create_mp``, ``nf__open``,
  ``nf__open_mp``, ``nf__enddef``, ``nf_inq_base_pe`` and
  ``nf_set_base_pe``. Their performance hints (initial size, buffer or
  chunk size, alignment, base PE) have no equivalent in ``netCDF4`` and are
  accepted but ignored.
* **netCDF-2 (``nc*``)**: ``nccre``, ``ncopn``, ``ncvdef``, ``ncvpt``,
  ``ncvgt``, the generalized ``ncvptg`` and ``ncvgtg``, the character
  routines (``ncvptc``, ``ncvgtc``, ...), the attribute routines and the
  option routines ``ncpopt`` and ``ncgopt``. Each one wraps the matching
  ``nf_*`` function.

Supported file formats are ``NETCDF3_CLASSIC``, ``NETCDF3_64BIT_OFFSET``,
``NETCDF4`` and ``NETCDF4_CLASSIC``. The two netCDF-4 formats require a
netCDF-C library built with HDF5. The format is chosen from the ``cmode``
flags (``NF_64BIT_OFFSET``, ``NF_NETCDF4``, ``NF_CLASSIC_MODEL``) or from
the default set by ``nf_set_default_format``.

Metadata (``netcdf_map.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``ALLOWED_NETCDF_NAMES``: every NetCDF name the translator recognizes.
  It includes the netCDF-3 and netCDF-2 constants and all function names.
* ``IMPLEMENTED_NETCDF_FUNCTIONS``: functions that have a runtime
  implementation. A test checks that every allowed function name that is
  not a constant is implemented.
* ``NF3_CONSTANTS`` and ``NF2_CONSTANTS``: constant names. ``is_constant``
  accepts both sets. ``NF3_FUNCTIONS`` is the set of allowed ``nf_*`` names
  that are not constants.
* ``NF_SCALAR_OUTS``: number of trailing scalar outputs for ``nf_*``
  functions.
* ``NF_ARG_ROLES``: explicit roles for functions that do not follow the
  trailing-output rule (``nf_inq_var`` and the ``nf__*`` functions).
* ``NC2_FUNCTION_ROLES``: roles for every netCDF-2 function.
* ``NF_VALUE_FUNCTIONS``: functions that return a value instead of a
  status (``nf_strerror``, ``nf_inq_libvers``, ``nf_issyserr``).
* ``NC2_VALUE_FUNCTIONS``: netCDF-2 functions whose result is the function
  value (``nccre``, ``ncopn``, ``ncddef``, ``ncvdef``, ...).
* ``arg_roles(name, n_args)``: returns the role of each argument and
  raises ``NotImplementedError`` if the argument count is wrong.

Argument roles
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Role
     - Meaning in the generated Python
   * - ``in``
     - Passed as an argument.
   * - ``out``
     - Returned by the function and assigned to the Fortran variable.
   * - ``inout_array``
     - Passed as an argument and filled in place (for example the
       ``dimids`` array of ``nf_inq_var``).
   * - ``inout_scalar``
     - Passed as an argument and also returned, then assigned back to the
       same variable (the buffer or chunk size of ``nf__create`` and
       ``nf__open``).

Constants and aliases
~~~~~~~~~~~~~~~~~~~~~

The netCDF-3 constants (``NF_*``) are the single source of truth. The
netCDF-2 names (``NCNOERR``, ``NCCLOB``, ``NCLONG``, ...) are aliases of
them, because the two interfaces share the same numeric values. For
example ``NCNOCLOB == NF_NOCLOBBER == 4`` and ``NCEEXIST == NF_EEXIST ==
-35``. Names that only exist in netCDF-2 (``NCSYSERR``, ``NCFOOBAR``) are
defined separately. Constants used in Fortran expressions are emitted in
upper case and recorded for the runtime copy.

Return conventions
~~~~~~~~~~~~~~~~~~

Fortran passes outputs through arguments, while Python returns them. The
translator rewrites each call accordingly.

.. list-table::
   :header-rows: 1

   * - Fortran
     - Generated Python
   * - ``status = nf_create(f, m, ncid)``
     - ``(status, ncid) = nf_create(f, m)``
   * - ``status = nf_get_vara_real(ncid, v, s, c, a)``
     - ``status = nf_get_vara_real(ncid, v, s, c, a)`` (``a`` filled in
       place)
   * - ``status = nf__create(f, m, 0, bufrsize, ncid)``
     - ``(status, bufrsize, ncid) = nf__create(f, m, 0, bufrsize)``
   * - ``ncid = ncopn(f, m, rcode)``
     - ``(ncid, rcode) = ncopn(f, m)``
   * - ``call ncclos(ncid, rcode)``
     - ``rcode = ncclos(ncid)``
   * - ``call ncpopt(0)``
     - ``ncpopt(0)``

The rules are:

* ``nf_*`` functions return ``(status, outputs...)``, or just the status
  when there are no outputs.
* netCDF-2 functions return ``(function_value, outputs..., rcode)``, with
  ``rcode`` always last. Routines without a function value return
  ``(outputs..., rcode)``, or just ``rcode``.
* Assignment targets are listed in argument order.
* A failed ``ncopn`` or ``nccre`` returns ``None`` as the dataset, and
  netCDF-2 functions that return an id return ``-1`` on failure.

Error handling
~~~~~~~~~~~~~~

``nf_*`` functions never raise for NetCDF errors; they return the status
code. ``nf_strerror`` returns the message for any status. Positive values
are operating-system errors, as in the C library, and ``nf_issyserr``
detects them.

The netCDF-2 interface follows ``NCOPTS``. The default is
``NCFATAL | NCVERBOS``: a failing call logs a message through the
``logging`` module and raises ``RuntimeError``. Call ``ncpopt(0)`` first to
get the status in ``rcode`` instead. ``ncgopt()`` returns the current
value. Translated programs that check ``rcode`` should therefore call
``ncpopt(0)`` before the first NetCDF call, as real netCDF-2 programs do.

Behaviour that differs from ``netCDF4``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runtime reproduces the Fortran library, not the Python one.

* **Ids are 1-based**, for dimensions, variables and attributes. They stay
  stable when a dimension or variable is renamed.
* **Dimension order is reversed.** Fortran lists the fastest-varying
  dimension first, and NetCDF files store the slowest first. A Fortran
  array ``a(nx, ny)`` is stored as ``a(y, x)`` in the file, which is what
  ``ncdump -h`` shows. Data are transposed on write and read, so values
  round-trip unchanged. Start, count, stride and index arguments are 1-based
  and in Fortran order.
* **Define and data mode** are tracked by the runtime. For the netCDF-3
  formats and ``NETCDF4_CLASSIC`` the rules are strict: ``nf_enddef`` and
  ``nf_redef`` are required, and a wrong call returns
  ``NF_ENOTINDEFINE`` or ``NF_EINDEFINE``. In ``NETCDF4`` files the switch
  is implicit, as in netCDF-C.
* **Variables are created lazily.** ``netCDF4`` only accepts ``_FillValue``
  when a variable is created, but Fortran sets it afterwards with
  ``nf_put_att``. ``nf_def_var`` therefore records the definition, and the
  variable is created at ``nf_enddef`` or on the first call that needs it.
* **Format rules** are enforced: one unlimited dimension and it must be the
  first one for the netCDF-3 formats, and the netCDF-4 types
  (``NF_INT64``, ``NF_STRING``, ...) are rejected with ``NF_EBADTYPE`` in
  classic files.
* **Type conversion** follows netCDF: out-of-range values return
  ``NF_ERANGE``, and mixing text and numbers returns ``NF_ECHAR``.
* **Hyperslabs** (``vara``, ``vars``, ``varm``) check start, count and
  stride in netCDF-C order and return ``NF_EINVALCOORDS``, ``NF_EEDGE`` or
  ``NF_ESTRIDE``. Writing past the end of the unlimited dimension extends
  it; reading past the end is an error.

Mapped access (``imap``)
^^^^^^^^^^^^^^^^^^^^^^^^

``nf_put_varm_*`` and ``nf_get_varm_*`` take ``imap`` **in elements**.
The netCDF-2 routines ``ncvptg``, ``ncvgtg``, ``ncvpgc`` and ``ncvggc``
take it **in bytes**, as their documentation specifies, and the runtime
divides by the item size of the *memory buffer*, not of the file variable.
Byte offsets that are not a multiple of the item size return
``NF_EINVAL``. In the netCDF-2 routines an all-zero ``stride`` or
``imap`` selects the default layout. Negative ``imap`` values are
rejected with ``NF_EINVAL``.

Accepted but ignored
^^^^^^^^^^^^^^^^^^^^

* ``NF_SHARE``, ``NF_LOCK`` and ``NF_NOFILL`` mode flags.
* The size, alignment and base-PE hints of the ``nf__*`` functions.
  ``nf_inq_base_pe`` always reports 0.

Translator wiring (``F2NP``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``handle_netcdf_call`` rewrites ``x = nf_xxx(...)`` and
  ``x = ncxxx(...)`` assignments.
* ``handle_netcdf_call_stmt`` rewrites ``CALL ncxxx(...)`` statements.
* Both use ``arg_roles`` and the helper that splits arguments by role.
  ``nf_get_att_*`` is special-cased because its last argument is an array
  or a scalar depending on the variable.
* Every function and constant that is used is recorded in
  ``netcdf_names_used``.
* A ``nf_*`` function called inside an expression, not as
  ``status = nf_xxx(...)``, raises ``NotImplementedError``. The functions
  in ``NF_VALUE_FUNCTIONS`` are the exception.
* A recognized ``nf_*`` name without a runtime implementation raises
  ``NotImplementedError``.

Runtime inclusion in generated code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``emit_netcdf_runtime(out_dir, names_used)`` copies ``fortran_netcdf.py``
next to the generated code when ``names_used`` is not empty. The generated
module imports the names it uses from that file. Every emitted name must
exist in the runtime, and a test checks this.

Limitations
~~~~~~~~~~~

* netCDF-4 groups, user-defined types and the parallel and compression
  routines are not in the allow-list.
* netCDF-4 formats need HDF5. A library built without it is not reported as
  ``NF_ENOTBUILT``; a failed create returns an operating-system error.
* Output arrays of the netCDF-2 attribute routines are returned, not filled
  in place: ``ncagt`` returns the values, so a dummy array argument is
  rebound rather than updated.
* Character outputs (``ncvgtc``, ``ncagtc``) are returned as strings,
  padded or truncated to the requested length.


See Also
--------

* :doc:`derived_types`
* :doc:`transformation`
* :doc:`jax_conversion`
* :doc:`architecture`
