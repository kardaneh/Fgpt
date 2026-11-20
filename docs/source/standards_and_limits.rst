Standards and Limits
====================

This page describes the coding conventions that Fortran sources must follow
to be transformed reliably by FGPT, and the constructs that are currently
unsupported or only partially supported.

.. contents:: On this page
   :local:
   :depth: 2

Coding Standards
----------------

FGPT relies on predictable, explicit source code. Sources that follow the
standards below translate reliably; deviations may be mistranslated or rejected.

The list of standards is expected to grow as the framework evolves.

Fortran I/O Standard
~~~~~~~~~~~~~~~~~~~~

All I/O in code to be transformed should follow a generic template
(see ``io_generic.f90``). Every ``OPEN`` statement uses **only the specifiers
whose defaults differ from the desired behaviour**. Everything else is left
at the compiler default.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Specifier
     - Default
     - When to specify
   * - ``newunit=``
     - none
     - Always
   * - ``file=``
     - ``fort.n``
     - Always
   * - ``status=``
     - ``'unknown'``
     - Whenever the intent is ``replace``, ``old`` or ``new``
   * - ``form=``
     - ``'formatted'``
     - Only for ``unformatted``
   * - ``access=``
     - ``'sequential'``
     - Only for ``direct`` or ``stream``
   * - ``position=``
     - ``'asis'``
     - Only for ``append``
   * - ``recl=``
     - none
     - Only for direct access
   * - ``iostat=``
     - none
     - Only when errors must be handled

Example:

.. code-block:: fortran

   ! Formatted sequential file, replaced if it exists
   open(newunit=u, file='out.dat', status='replace')

   ! Unformatted direct-access file
   open(newunit=u, file='data.bin', form='unformatted', &
        access='direct', recl=1024)

Why this matters
^^^^^^^^^^^^^^^^

Because only non-default specifiers appear, the transformer can infer the
intended I/O mode directly from the presence or absence of each specifier,
without having to evaluate defaults. This keeps the generated Python I/O
calls (see :doc:`fortran_io_runtime`) simple and predictable.

Fortran Language Standard
~~~~~~~~~~~~~~~~~~~~~~~~~

Input sources are parsed through ``fparser`` and should conform to
Fortran 2003/2008.

Current Limits
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Feature
     - Status
     - Notes
   * - Pointers, classes, blocks
     - Not supported
     - The AST handles them, but the transformation pipeline does not yet.
   * - Derived types
     - Supported, with limits
     - Simple, array and nested derived types work (see :doc:`derived_types`).
       Not supported: deferred-shape or ``ALLOCATABLE`` components, pointer
       components, polymorphic types (``CLASS``) and initializers on arrays
       of derived types. Non-default lower bounds on components are recorded
       in the metadata, but index adjustment for them is still being
       completed.
   * - Interface blocks
     - Partial
     - Explicit interfaces are read; generic resolution is not.
   * - NetCDF I/O
     - Supported, with limits
     - The netCDF-2 (``nc*``), netCDF-3 (``nf_*``) and ``nf__*`` interfaces
       are supported (see :doc:`fortran_io_runtime`). netCDF-4 groups,
       user-defined types and the parallel and compression routines are not
       supported.
   * - HDF5 I/O
     - Planned
     - Not supported today, apart from the HDF5 library that NetCDF-4 files
       need underneath.

Derived-Type Dump Standard
~~~~~~~~~~~~~~~~~~~~~~~~~~

Input data for derived-type variables is read from unformatted dumps. Each
derived-type variable must be written as **one unformatted record**: a
scalar struct is one record, and an array of structs is one record holding
all elements, written with a single ``write(unit)`` of the whole variable,
without repacking or reshaping on the Fortran side. ``LOGICAL`` components
are stored as 4-byte integers, and the layout follows the compiler's natural
alignment. A wrong layout raises a record-size error when the dump is read
(see :doc:`fortran_io_runtime`).

Component shapes must be known when the type is defined. Deferred-shape or
``ALLOCATABLE`` components are rejected.

.. note::

   FGPT is a proof of concept under active development. This table reflects
   the current state and will change as features are added.

Reporting Unsupported Constructs
--------------------------------

If a valid Fortran construct fails to translate, please open an issue on the
`GitHub repository <https://github.com/kardaneh/FGPT.git>`_ with a minimal
reproducing example.
