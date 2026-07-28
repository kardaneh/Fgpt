
Quick Start
===========

This guide demonstrates the typical FGPT workflow: isolating a Fortran
subroutine, extracting its dependencies, and generating executable
Python code, with an optional JAX conversion step.

.. note::

   **Maturity levels of the pipeline stages:**

   - **Stages 1–2** (Isolation → Transpilation → NumPy Python) are
     production-ready and have been validated on real ORCHIDEE land-surface
     model subroutines. The generated NumPy-based Python code is numerically
     equivalent to the original Fortran.

   - **Stage 3** (JAX conversion) is a **work in progress** though it has
     already been validated on real ORCHIDEE land-surface model subroutines.
     It supports a substantial subset of generated NumPy patterns; however,
     complex control flow, certain loop constructs, and some edge cases in
     the generated code may not convert correctly, so it should be used with caution.

     Validation is performed by comparing outputs against a reference Fortran
     implementation. Specifically, NumPy-based test suites—originally designed to
     validate the Fortran model—are also translated into JAX-compatible form.
     This allows results from the JAX-translated code to be directly compared against
     the Fortran outputs, ensuring consistency across implementations.


Prerequisites
-------------

Install FGPT and its dependencies:

.. code-block:: bash

    git clone https://github.com/kardaneh/IPSL-Fgpt.git
    cd fgpt
    uv venv .venv --python 3.10
    source .venv/bin/activate
    uv pip install -e ".[dev]"

or with pip:

.. code-block:: bash

    pip install -e ".[dev]"


Step 1: Configure the Isolation Target
---------------------------------------

Create a configuration script describing the Fortran module and
subroutines you wish to isolate.

.. code-block:: bash

    TARGET_MODULE="hydrol"
    PARENT_SUBROUTINE="hydrol_main"
    TARGET_SUBROUTINES="
        hydrol_alma
        hydrol_vegupd
        hydrol_canop
        hydrol_flood
        hydrol_hydraulic_arch_tuzet_calc
        hydrol_soil
        explicitsnow_main
    "
    OPENACC="False"
    F2PY="True"
    TAPENADE="False"

The configuration identifies:

- The source Fortran module.
- The parent driver subroutine.
- The set of dependent routines to isolate.
- Optional flags: ``F2PY`` triggers Fortran-to-Python transpilation
  immediately after isolation; ``OPENACC`` and ``TAPENADE`` enable
  GPU and automatic differentiation paths respectively.


Step 2: Run the Isolator
-------------------------

Generate an isolated, standalone version of the target routines:

.. code-block:: bash

    fgpt isolate \
        --work "$WORK_DIR" \
        --rest_of_path "$REST_OF_PATH" \
        --target_module "$TARGET_MODULE" \
        --parent_subroutine "$PARENT_SUBROUTINE" \
        --target_subroutines $TARGET_SUBROUTINES \
        --openacc False \
        --f2py True \
        --tapenade False \
        --py2jx False \
        --mode jax \
        --config_path template.yaml \
        --vectorize kjpindex \
        --benchmark_dir benchmark/ \

The isolator:

- Extracts the requested routines and resolves all cross-module
  dependencies.
- Generates standalone compilable Fortran files
  (``global_module_<name>.f90``, ``main_<name>.f90``).
- Compiles and runs the standalone Fortran to produce binary reference
  input/output files.
- If ``--f2py True``: immediately transpiles each isolated routine to
  Python (see Step 3).
- If ``--py2jx True``: Uses the transpiled code to transform it into a
  JAX-compatible version.

.. tip::

   Instead of invoking ``fgpt isolate`` directly with all its flags, you
   can use the provided ``setup`` script to generate a reusable driver
   script from a single configuration block. See
   `Automating CLI Invocations with the setup Script`_ near the end of
   this guide.


Step 3: Fortran-to-Python Transpilation
-----------------------------------------

.. note::

   This step is **production-ready**. The generated NumPy-based Python
   code has been validated against the original Fortran binary outputs
   on real ORCHIDEE subroutines. If ``--f2py True`` was passed to the
   isolator, this step runs automatically and can be skipped here.

   The generated Python code is designed not only to faithfully reproduce
   the original Fortran semantics, but also to maximize compatibility with
   the subsequent JAX conversion stage. Where appropriate, the transpiler
   emits NumPy constructs and coding patterns that can be translated more
   reliably into JAX-compatible code.

If running standalone, the transpilation can also be triggered via the
Python API. However, it still requires the use of the Isolator and
Extractor classes, as the latter are required by the Transformer class
to ensure proper transpilation.

.. note::

   This standalone Python API workflow bypasses the ``fgpt`` CLI
   entirely and is therefore **not** covered by the ``setup`` script
   described later in this guide, which only automates CLI invocations.

.. code-block:: python

    from fgpt.isolator import Isolator
    from fgpt.extractor import Extractor
    from fgpt.transformer import Transformer

    # These instances are normally created by the Isolator
    isolator  = Isolator(...)
    extractor = Extractor(...)

    transformer = Transformer(
        benchmark_dir="benchmark",
        isolator=isolator,
        extractor=extractor,
        ignore_case=None,
        config_path=None,    # uses bundled default template or yours
    )

    out_module = transformer.update_global_python(subroutine_key="hydrol_soil")
    transformer.transfer_to_pyfile(out_module, "hydrol_soil", folder_name="hydrol")

    main_tree = transformer.update_main_python(out_module=out_module, subroutine_key="hydrol_soil")
    transformer.transfer_to_pyfile(main_tree, "hydrol_soil", folder_name="hydrol", python_file_type="main")

During transpilation, FGPT:

- Parses Fortran using ``fparser`` and builds a Python :mod:`ast` tree.
- Converts declarations, loops, conditionals, and expressions
  statement-by-statement.
- Maps Fortran intrinsic functions to their NumPy equivalents.
- Adjusts 1-based Fortran array indexing to 0-based Python indexing.
- Rewrites unqualified global references to ``self.attr`` accesses.
- Produces a NumPy-based Python class and a standalone driver script.

The output files are placed under ``<target_module>/<subroutine_name>/``:

.. code-block:: text

    hydrol/
    └── hydrol_soil/
        ├── global_module_hydrol_soil.f90   standalone Fortran module
        ├── main_hydrol_soil.f90            Fortran driver program
        ├── global_module_hydrol_soil.py    ← transpiled Python class
        └── main_hydrol_soil.py             ← transpiled Python driver

Compatibility of the generated Python files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generated Python class is designed to be a faithful, directly
executable translation of the original Fortran. In practice:

- **Array operations** are translated to NumPy and are fully compatible
  with standard Python tooling.
- **Control flow** (``DO`` loops, ``IF``/``ELSE``, ``WHERE``) is
  translated to standard Python ``for``/``if`` constructs and
  ``np.where``.
- **The generated files are valid, importable Python modules** and can
  be used directly in any NumPy-based workflow without modification.
- **JAX compatibility.** The transpiler emits NumPy code using
  JAX-friendly patterns whenever possible, making the generated Python
  well suited for automatic conversion. As a result, a substantial
  fraction of generated modules can be converted directly to JAX.
  However, some constructs—particularly those involving some complex
  control flow, implicit shape assumptions, or unsupported Python
  features, may still require manual adjustments before successful
  conversion. See Step 4.

Step 4: JAX Conversion
------------------------

.. warning::

   **This step is a work in progress.** While the JAX conversion layer
   handles a significant subset of patterns generated by the transpiler
   and has been validated on real ORCHIDEE subroutines used to verify
   the Python transpilation, it is not yet robust enough for unsupervised
   production use. Specifically:

   - Broadcasting issues and attribute mismatches
   - Certain NumPy or Python intrinsics are not yet implemented
     (e.g., ``dataclass``, ``AugAssign``, etc.)
   - Mixed scalar/array conditional branches may generate incorrect
     ``lax.cond`` signatures
   - Subroutines with large numbers of class attributes may exceed
     Equinox pytree limitations
   - The ``fwd`` and ``bwd`` modes produce structurally valid
     ``eqx.Module`` subclasses but do not yet generate corresponding
     ``jax.jvp`` or ``jax.grad`` call sites

   **Always validate JAX outputs numerically against the Fortran
   reference implementations** before using them in any downstream computation.

JAX conversion can be applied independently to any compatible NumPy-based Python
class. It does not require the Fortran transpilation steps to be run first,
although it is strongly recommended to use the previously generated NumPy code.

.. note::

   It is recommended to have at least 6 GiB of memory, as this is sufficient
   to test the most demanding routines, including ``hydrol_soil``,
   ``explicitsnow_main``, and ``hydrol_hydraulic_arch_tuzet_calc``.
   These high-level routines have multiple child callees and extensive use
   of multidimensional arrays.

The Python API below bypasses the ``fgpt`` CLI entirely, in the same way
as Step 3's standalone transpilation workflow, and is likewise not
covered by the ``setup`` script.

.. code-block:: python

    from fgpt.autodiff import AutoDiff

    autodiff = AutoDiff(
        config_path=None,       # uses bundled default template
        mode="jax",             # "jax" | "fwd" | "bwd"
        vectorize=["kjpindex"]  # default, ["kjpindex"]
    )

    autodiff.transform(
        class_file="hydrol/hydrol_soil/global_module_hydrol_soil.py",
        main_file="hydrol/hydrol_soil/main_hydrol_soil.py",
    )

Or via the CLI:

.. code-block:: bash

    fgpt autodiff \
        --class_file hydrol/hydrol_soil/global_module_hydrol_soil.py \
        --main_file  hydrol/hydrol_soil/main_hydrol_soil.py \
        --mode jax

.. tip::

   This CLI form can also be generated automatically using the
   ``setup`` script (with ``COMMAND="autodiff"``) — see
   `Automating CLI Invocations with the setup Script`_ below.

Another option is to provide the class and main modules directly.
In this case, you must also pass `routine_dir`, which specifies the
directory where the generated files should be saved.
This directory can be the same as the one containing the original
NumPy-based modules.

.. code-block:: python

    from fgpt.autodiff import AutoDiff

    autodiff = AutoDiff(
        config_path=None,       # uses bundled default template
        mode="jax",             # "jax" | "fwd" | "bwd"
        vectorize=["kjpindex"]  # default, ["kjpindex"]
    )

    autodiff.transform(
        class_file=class_module,          # AST module
        main_file=main_module,            # AST module
        routine_dir="hydrol/hydrol_soil/" # Path inside which we need to save
    )

Supported modes:

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Mode
     - Output suffix
     - Status
   * - ``jax``
     - ``_jax.py``
     - Functional for most transpiled subroutines. Produces an
       XLA-compilable ``eqx.Module``. **Validate outputs before use.**
   * - ``fwd``
     - ``_d.py``
     - Structural scaffolding only. ``jax.jvp`` wrappers not yet
       emitted. **Experimental.**
   * - ``bwd``
     - ``_b.py``
     - Structural scaffolding with checkpointed while loops.
       ``jax.grad`` wrappers not yet emitted. **Experimental.**


Automating CLI Invocations with the ``setup`` Script
------------------------------------------------------

The ``fgpt isolate`` command shown in Step 2, and the ``fgpt autodiff``
CLI invocation shown in Step 4, can both be generated from a single
configuration script, ``setup``, provided in the repository, rather
than typing out each command by hand.

.. important::

   This script only automates the **CLI paths** (Step 2's ``fgpt
   isolate``, and Step 4's ``fgpt autodiff`` CLI form). It does **not**
   apply to the standalone Python API workflows described in Step 3, or
   to the Python-API portion of Step 4 — those bypass the CLI entirely
   and must be invoked directly as shown above.

The script is dual-purpose: setting ``COMMAND="isolator"`` generates a
driver script equivalent to Step 2's ``fgpt isolate`` invocation, while
``COMMAND="autodiff"`` generates a driver equivalent to Step 4's ``fgpt
autodiff`` invocation.

.. code-block:: bash

    #!/usr/bin/env bash

    # Choose which fgpt subcommand this setup should drive.
    #   "isolator" -> fgpt isolate  ... (Stages 1 - 3)
    #   "autodiff" -> fgpt autodiff ... (Stage 3)
    COMMAND="isolator"

    # How to invoke the CLI. If fgpt is installed (pip install -e .) and
    # exposes a console entry point, "fgpt" works directly. Otherwise use:
    #   FGPT_CMD="python -m fgpt.cli"
    FGPT_CMD="fgpt"

    # Path configuration
    WORK_DIR="$work"
    REST_OF_PATH="modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    TARGET_MODULE="hydrol"

    # Parent and target subroutines
    PARENT_SUBROUTINE="hydrol_main"
    TARGET_SUBROUTINES="hydrol_alma hydrol_vegupd hydrol_canop hydrol_flood \
        hydrol_hydraulic_arch_tuzet_calc hydrol_soil explicitsnow_main"

    # Isolation options
    OPENACC="False"
    F2PY="True"
    TAPENADE="False"
    PY2JX="True"

    # Autodiff configuration (used only when COMMAND="autodiff")
    CONFIG_PATH="template.yaml"
    CLASS_FILE="hydrol/hydrol_vegupd/global_module_hydrol_vegupd.py"
    MAIN_FILE="hydrol/hydrol_vegupd/main_hydrol_vegupd.py"
    MODE="jax"
    BENCHMARK_DIR="benchmark"
    VECTORIZE="kjpindex"

Running the script generates a corresponding driver script, e.g.:

.. code-block:: bash

    ./setup
    ./run_isolator.sh

The generated ``run_isolator.sh`` (or ``run_autodiff.sh``) is a
self-contained, timestamped script that echoes the active configuration
before invoking the underlying ``fgpt`` command, and reports completion
time once finished — useful for keeping a record of exactly which
parameters were used for a given run.

.. note::

   ``PY2JX="True"`` requires ``F2PY="True"``, since Stage 3 (JAX
   conversion) operates on the Python output produced by the Stage 2
   transpilation. If ``F2PY`` is left ``False`` while ``PY2JX`` is
   ``True``, the CLI enables ``--f2py`` automatically and emits a
   warning.

To generate an autodiff driver instead, set ``COMMAND="autodiff"`` and
adjust ``CLASS_FILE``/ ``MAIN_FILE`` to point to the already-transpiled
NumPy modules you want to convert (see Step 4 above, and
:doc:`jax_conversion` for details on Stage 3 behavior and limitations).


Using the Example Notebooks
---------------------------

The repository includes several example notebooks, such as
``Test_F2NP.ipynb`` and ``Test_JAX_Converter.ipynb``, which demonstrate
different features and workflows.

Before running the notebooks, ensure you have completed the installation
steps above. Then register the project's virtual environment as a Jupyter
kernel:

.. code-block:: bash

   source .venv/bin/activate

   uv run ipython kernel install --user \
       --env VIRTUAL_ENV "$(pwd)/.venv" \
       --name=project

You can then launch JupyterLab:

.. code-block:: bash

   uv run --with jupyter jupyter lab

Alternatively, you can open the notebooks directly in Visual Studio Code.
VS Code will automatically detect the project's ``.venv``. Simply select
the ``project`` kernel (or the corresponding virtual environment) when
prompted.

Running Tests
-------------

The isolator generates binary input and output files that serve as
reference data for validating the translated NumPy-based Python code.

The generated Python implementation can be tested by executing the
corresponding driver script:

.. code-block:: bash

   python hydrol/isolated_procedure/main_*.py

The output produced by the translated NumPy code is compared against the
reference binary data generated from the original Fortran
implementation, ensuring that the transformation preserves the
numerical behaviour of the source routine.

The same validation drivers are also translated into JAX-compatible
form. This allows the JAX-translated implementation to be executed with
the same reference inputs and its outputs compared directly against the
Fortran reference data (or equivalently the validated NumPy outputs),
providing a consistent validation workflow across the Fortran, NumPy,
and JAX implementations.

.. code-block:: bash

   python hydrol/isolated_procedure/main_*_jax.py

To run the full test suite:

.. code-block:: bash

    pytest

This runs the project's development test suite. It includes tests for the
parser, AST transformations, code generation, JAX conversion, and other
core components used to build the transpilation pipeline.

Typical Workflow
----------------

A complete workflow typically follows:

.. code-block:: text

    Fortran source (.f90)
           │
           ▼
       Isolator
           │
           ▼
    Static analysis
    (Navigator + Extractor)
           │
           ▼
    F2NP + Transformer
           │
           ▼
    NumPy Python class (.py)
           │
           ▼  ⚠ work in progress
    JAX conversion                  ← validate outputs carefully
    (AutoDiff + JaxConverter)
           │
           ▼
    eqx.Module (_jax.py)


Next Steps
----------
- See :doc:`transformation` for a detailed description of the
  Fortran-to-Python transpilation pipeline (Stages 1–3).
- See :doc:`jax_conversion` for the JAX conversion layer internals and
  known limitations (Stage 4).
- See :doc:`architecture` for how all components fit together.
- See :doc:`testing` for development and testing guidelines.
