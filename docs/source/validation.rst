Validation Against Fortran
==========================

The Python and JAX outputs are validated numerically against the original
Fortran. This requires reference data (inputs and outputs) produced by
running the isolated Fortran procedures. This page describes that workflow
and demonstrates it on ``examples/example_02.py``.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

Isolation produces standalone Fortran sources, but **not** the binary data
needed to run them. The reference data is generated in three steps, after
which the Python and JAX numerical comparison can run.

.. code-block:: text

    Isolation
        │   (isolated Fortran procedures, no .bin files yet)
        ▼
    make  →  ./out
        │   (generates the global and dummy input .bin files)
        ▼
    python executive.py --target_module <module>
        │   (runs the isolated Fortran with these inputs
        │    and generates output.bin)
        ▼
    Python / JAX numerical comparison against output.bin

Step 1: Isolate the Procedures
------------------------------

Run the isolation as described in :doc:`quickstart` Step 1 and 2. The isolated folder
(for example ``hydrol/``) contains the isolated Fortran procedures, but
**no input data** yet by setting all flags to False.

Step 2: Generate the Inputs
---------------------------

Use the provided ``Makefile`` to build and run the isolated Fortran:

.. code-block:: bash

    make
    ./out

Running ``./out`` writes the **global** and **dummy** input files
(``<name>.bin``). These are the inputs shared by the Fortran, Python and
JAX runs.

Step 3: Generate the Reference Outputs
--------------------------------------

Run the executive script on the module containing the isolated procedures:

.. code-block:: bash

    python src/fgpt/core/common/executive.py --target_module <module_name>

This runs the isolated Fortran procedures with the inputs from Step 2 and
produces ``output.bin``, the reference output.

Step 4: Compare Python and JAX
------------------------------

With the inputs and reference outputs in place, run the numerical
compatibility tests for the generated code (see :doc:`quickstart`,
"Running Tests") or simply re-run the full process with now the isolation
of Fortran, the transpilation to Python and the transformation to JAX,
which now runs the tests for the numerical comparing for each procedure
at each process using the ``output.bin``.

Worked Example: ``example_02.py``
---------------------------------

The ``examples/`` folder contains a complete case. ``example_02.py`` drives
the isolator from Python (instead of the ``fgpt isolate`` CLI) ``example_01``
module. The workflow has three phases: isolate the Fortran, generate the
reference data, then re-run the isolation with the Python and JAX
transformations enabled.

Phase 1: Isolate the Fortran only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``example_02.py``, the transformation options are all disabled, so
the isolator only generates the standalone Fortran sources:

.. code-block:: python

   sys.argv = [
       "my_program",
       "--rest_of_path", "Fgpt/examples",
       "--target_module", "example_01",
       "--work", "/home/kardaneh",
       "--target_model", "general",
       "--parent_subroutine", "test_procedures",
       "--target_subroutines", "test_procedures",
       "--openacc", "False",
       "--f2py", "False",      # no Python transpilation yet
       "--py2jx", "False",     # no JAX conversion yet
       "--tapenade", "False",
       "--mode", "jax",
       "--benchmark_dir", "/home/kardaneh/benchmark",
       "--config_path", "/home/kardaneh/Fgpt/template.yaml",
       "--vectorize", "kjpindex", "nvm", "npts",
   ]

Run it from the repository root:

.. code-block:: bash

   python examples/example_02.py

.. note::

   Adapt ``--work``, ``--rest_of_path`` and ``--config_path`` to your own
   machine.

Phase 2: Generate the reference data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The isolation does not produce any ``.bin`` files. Build and run the
isolated Fortran to generate the inputs (global and dummy), then run the
executive script to generate the reference outputs:

.. code-block:: bash

   cd examples/
   make example_02
   ./example_02
   cd ..
   python src/fgpt/core/common/executive.py --target_module <module_name>

``./example_02`` writes the input ``.bin`` files, and ``executive.py`` runs
the isolated Fortran with those inputs to produce ``output.bin``.

Phase 3: Enable the Python and JAX transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return to ``example_02.py`` and set ``--f2py`` and/or ``--py2jx`` to
``"True"``:

.. code-block:: python

       "--f2py", "True",       # Fortran -> NumPy
       "--py2jx", "True",      # NumPy -> JAX (requires f2py)

then run the script again:

.. code-block:: bash

   python examples/example_02.py

This repeats the isolation and additionally performs the Python
transpilation and/or the JAX conversion. Each time, the generated code is
run against the reference data from Phase 2 and the numerical
compatibility with the Fortran outputs is verified.

.. note::

   ``--py2jx True`` requires ``--f2py True``, since the JAX conversion
   operates on the transpiled Python (see :doc:`quickstart`).

Summary of Generated Files
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - File
     - Produced by
     - Purpose
   * - Global / dummy ``.bin`` files
     - Running the isolated Fortran driver
     - Input data shared by the Fortran, Python, and JAX implementations.
   * - ``output.bin``
     - ``executive.py``
     - Reference output generated by the isolated Fortran procedures.
   * - Python / JAX results
     - Generated-code test drivers
     - Results compared numerically against the Fortran reference output.

Troubleshooting
---------------

* **Missing ``.bin`` files:** ``make`` and ``./out`` were not run.
* **``executive.py`` cannot find the module:** check that ``--target_module``
  matches the isolated folder name.
