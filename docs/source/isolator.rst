Isolator
========

The :class:`~fgpt.isolator.Isolator` is the entry point of the FGPT
transpilation pipeline. Its role is to extract a specific Fortran subroutine
or function from a large scientific codebase and produce a self-contained,
compilable and executable unit that can be transpiled to Python independently
of the rest of the original code.

It can also be run as a standalone command-line tool, making it usable
without writing any Python driver code.

.. contents:: Contents
   :local:
   :depth: 2

Purpose
-------

Production scientific Fortran codebases — such as land-surface models — are
typically structured as large modules where subroutines share global state
through ``USE`` associations and module-level variables. Transpiling or
testing a single routine in isolation requires:

- Locating the target subroutine within its module.
- Resolving all variables it references, whether declared locally, passed as
  dummy arguments, or imported from other modules via ``USE``.
- Recursively resolving the same dependencies for any subroutines it calls.
- Generating a standalone Fortran program that declares all required
  variables, calls the target routine, and writes its outputs.
- Compiling and running that program to validate the isolated routine before
  any transpilation occurs.

The Isolator automates this entire workflow. Optionally, if a
:class:`~fgpt.core.transpiler.transformer.Transformer` instance is supplied,
it also triggers the Fortran-to-Python transpilation of each isolated
routine immediately after the Fortran isolation step completes. If the
``py2jx`` flag is additionally enabled, the resulting Python output is then
passed to the JAX/Equinox conversion stage (Stage 3), producing a
JAX-traceable Equinox module directly from the isolated Fortran routine
without requiring a separate manual invocation.

Isolation Workflow
------------------

The full isolation process for a single target subroutine proceeds as
follows:

.. code-block:: text

        Target module (.f90)
                │
                ▼
        ┌────────────────────────────┐
        │ core.frontend.processor    │  parse module into fparser AST
        │   Processor                │
        └────────┬───────────────────┘
                 │
                 ▼
        ┌────────────────────────────┐
        │ core.frontend.extractor    │  find all subroutines; build call graph
        │   Extractor                │
        └────────┬───────────────────┘
                 │
                 ▼
        ┌───────────────────────────────────────────┐
        │   isolate_procedure  (recursive)          │
        │                                           │
        │  1. find_variables — classify dummy,      │
        │     global, and local variables           │
        │  2. find_global_variables — resolve USE   │
        │     imports via core.frontend.navigator   │
        │     .Navigator                            │
        │  3. process_declaration_variables —       │
        │     extract type/shape declarations       │
        │  4. extract_all_array_info — collect      │
        │     array metadata (core.analysis.shaper) │
        │  5. recurse into nested subroutine calls  │
        │  6. extract_intent / clean_subroutine /   │
        │     extract_modified_variables            │
        │  7. update_global_module — write          │
        │     standalone global Fortran file        │
        │  8. update_main_program — write           │
        │     standalone driver program             │
        │  9. compile_and_run — validate output     │
        │ 10. (optional) core.transpiler.transformer│
        │     .Transformer — transpile to Python    │
        │     immediately                           │
        │ 11. (optional) core.autodiff.AutoDiff     │
        │     convert Stage 2's Python output to    │
        │     JAX/Equinox (Stage 3, py2jx=True;     │
        │     requires f2py=True)                   │
        └────────┬──────────────────────────────────┘
                 │
                 ▼
        Isolated subroutine directory
        ├── global_module_<routine>.f90   (and optionally .py)
        └── main_<routine>.f90           (and optionally .py)

Recursive Isolation
~~~~~~~~~~~~~~~~~~~

A key aspect of the Isolator is that ``isolate_procedure`` is recursive.
When the target subroutine contains ``CALL`` statements to other subroutines
within the same module, each callee is isolated first (depth-first) before
the parent isolation is completed. Global variable declarations discovered in
nested routines are propagated upward via ``collect_global_vars_decl``, so
the parent's standalone module file includes everything it and its callees
need. Already-isolated subroutines are skipped to avoid redundant work.

The complete set of subroutines reachable from a target — the transitive
closure of its call graph — is computed by ``collect_all_subroutines`` using
a breadth-first traversal before the recursive isolation begins.

Output Structure
~~~~~~~~~~~~~~~~

For each isolated subroutine, the Isolator creates a directory under
``<target_module>/``:

.. code-block:: text

    <target_module>/
    └── <subroutine_name>/
        ├── global_module_<subroutine_name>.f90         standalone module with all dependencies
        ├── main_<subroutine_name>.f90                  driver program: declares vars, calls routine, writes outputs
        ├── global_module_<subroutine_name>.py          (if f2py=True) transpiled Python class
        ├── main_<subroutine_name>.py                   (if f2py=True) transpiled Python driver
        ├── global_module_<subroutine_name>_<mode>.py   (if py2jx=True) transformed NumPy code to JAX based on the mode
        └── main_<subroutine_name>_<mode>.py            (if py2jx=True) transformed NumPy code to JAX based on the mode

The Fortran driver is compiled and executed automatically as a validation
step. If compilation or execution fails, an assertion error is raised and
the pipeline stops before any Python output is written.

Usage
-----

Python API
~~~~~~~~~~

The typical entry point is :meth:`~fgpt.isolator.Isolator.run`, which calls
:meth:`~fgpt.isolator.Isolator.create_target_directory` and then
:meth:`~fgpt.isolator.Isolator.process_subroutines` internally:

.. code-block:: python

   from fgpt.isolator import Isolator

   isolator = Isolator(
       rest_of_path="modipsl/modeles/ORCHIDEE/src_sechiba/",
       target_module="hydrol",
       work="/scratch/user/runs",
   )

   isolator.run(
       parent_subroutine="hydrol_main",
       target_subroutines=["hydrol_soil", "hydrol_alma"],
   )

To also transpile each isolated routine to Python immediately after
isolation, enable the ``f2py`` flag:

.. code-block:: python

   isolator = Isolator(
       rest_of_path="modipsl/modeles/ORCHIDEE/src_sechiba/",
       target_module="hydrol",
       work="/scratch/user/runs",
       f2py=True,
   )

   isolator.run(
       parent_subroutine="hydrol_main",
       target_subroutines=["hydrol_soil"],
   )

With ``f2py=True``, a :class:`~fgpt.core.transpiler.transformer.Transformer`
is instantiated internally and called after each Fortran isolation step,
writing ``global_module_<routine>.py`` and ``main_<routine>.py`` alongside
the Fortran output files.

To run the complete pipeline through JAX/Equinox conversion (Stage 3),
enable ``py2jx`` in addition to ``f2py``:

.. code-block:: python

   isolator = Isolator(
       rest_of_path="modipsl/modeles/ORCHIDEE/src_sechiba/",
       target_module="hydrol",
       work="/scratch/user/runs",
       f2py=True,
       py2jx=True,
   )

   isolator.run(
       parent_subroutine="hydrol_main",
       target_subroutines=["hydrol_soil"],
       vectorize=["kjpindex"],
       config_path="template.yaml",     # yours or the packaged default template
       mode="jax",                      # "jax" | "fwd" | "bwd"
       benchmark_dir="benchmark_dir"
   )

With ``py2jx=True``, the JAX conversion stage operates directly on the
Python output produced by the preceding ``f2py`` step (``global_module_
<routine>.py`` and ``main_<routine>.py``), producing corresponding
JAX/Equinox output files alongside them. If ``f2py`` is not explicitly set
to ``True`` when ``py2jx=True``, it is enabled automatically, consistent
with the CLI's behavior in ``fgpt.cli``.

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The Isolator can also be invoked directly from the command line without
writing any Python code:

.. code-block:: bash

   python -m fgpt.cli isolate \
       --rest_of_path modipsl/modeles/ORCHIDEE/src_sechiba/ \
       --target_module hydrol \
       --work /scratch/user/runs \
       --parent_subroutine hydrol_main \
       --target_subroutines hydrol_soil hydrol_alma \
       --f2py True \
       --openacc False \
       --tapenade False \
       --py2jx False \
       --mode jax \
       --config_path template.yaml \
       --vectorize kjpindex \
       --benchmark_dir benchmark/ \

All arguments mirror the Python API parameters. ``--target_subroutines``
accepts a space-separated list of routine names.

Parameters and Flags
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``rest_of_path``
     - *(required)*
     - Relative path from ``work`` to the directory containing the target
       ``.f90`` module file.
   * - ``target_module``
     - *(required)*
     - Name of the Fortran module to operate on, without the ``.f90``
       extension.
   * - ``work``
     - ``$works``
     - Root working directory. Combined with ``rest_of_path`` to locate
       the source file.
   * - ``f2py``
     - ``False``
     - If ``True``, a :class:`~fgpt.core.transpiler.transformer.Transformer`
       is instantiated and each isolated subroutine is immediately
       transpiled to Python.
   * - ``openacc``
     - ``False``
     - If ``True``, OpenACC directives are preserved and included in the
       isolated output.
   * - ``tapenade``
     - ``False``
     - If ``True``, the isolated Fortran output is prepared for Tapenade
       automatic differentiation rather than direct execution. This is
       presumably related to the top-level :mod:`~fgpt.autodiff` module
       (described elsewhere as the "JAX/Tapenade conversion pipeline"),
       though the exact hand-off point between ``Isolator`` and
       ``autodiff`` is not documented here and should be confirmed.
   * - ``py2jx``
     - ``False``
     - If ``True``, a JAX/Equinox conversion (Stage 3) is triggered
       immediately after the Fortran-to-Python transpilation step,
       operating on the Python output produced by ``f2py``. Since Stage 3
       consumes Stage 2's output, ``py2jx=True`` requires ``f2py=True``;
       if ``f2py`` is not explicitly enabled, it is automatically set to
       ``True`` with a warning.


Source Preservation
-------------------

On first run, the Isolator copies the original ``.f90`` file to a
``_org.fgpt`` backup before any modifications are made. On subsequent runs,
the backup is used as the parse source rather than the (potentially already
modified) ``.f90`` file, ensuring that re-runs always start from the
original unmodified Fortran source.

Relationship to Other Components
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Role in isolation
   * - :class:`~fgpt.core.frontend.processor.Processor`
     - Parses the ``.f90`` file into an ``fparser`` AST; compiles and runs
       the isolated Fortran driver; writes modified module trees back to
       disk.
   * - :class:`~fgpt.core.frontend.extractor.Extractor`
     - Performs all static analysis: finds subroutines, builds the call
       graph, classifies variables (dummy / global / local / modified),
       extracts array metadata (via :mod:`~fgpt.core.analysis.shaper`),
       and resolves intents.
   * - :class:`~fgpt.core.frontend.navigator.Navigator`
     - Called by :class:`~fgpt.core.frontend.extractor.Extractor` to
       resolve variable declarations and subroutine definitions that are
       imported from other modules via ``USE`` associations.
   * - :class:`~fgpt.core.transpiler.transformer.Transformer`
     - Optionally invoked (when ``f2py=True``) after each subroutine is
       isolated to produce the corresponding Python class and driver
       files.
   * - :class:`~fgpt.autodiff.AutoDiff`
     - Optionally invoked (when ``py2jx=True``) after the Fortran-to-Python
       transpilation step to convert the transpiled Python output into a
       JAX/Equinox-compatible module (Stage 3). Requires ``f2py=True``,
       since it consumes the Python files produced by
       :class:`~fgpt.core.transpiler.transformer.Transformer`.

See Also
--------

* :doc:`transformation` — The Fortran-to-Python transpilation stage that
  the Isolator feeds into when ``f2py=True``.
* :doc:`jax_conversion` — The Python-to-JAX transformation stage to a
  jax compatible ``py2jx=True``.
* :doc:`architecture` — How the Isolator fits into the overall FGPT
  pipeline.
* :doc:`extractor` — Process of extraction.
* :doc:`api/fgpt` — Full API reference for
  :class:`~fgpt.isolator.Isolator` and its methods.
