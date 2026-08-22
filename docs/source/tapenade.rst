Tapenade Automatic Differentiation Integration
==============================================

Tapenade is an Automatic Differentiation (AD) tool that transforms Fortran
code to compute derivatives efficiently. It generates tangent-linear (forward
mode) and adjoint (reverse mode) code from a given Fortran subroutine or
function.

For more information, see the `Tapenade documentation <https://inria.hal.science/hal-00913983/document/>`_.

Requirements
------------

Before using FGPT with Tapenade automatic differentiation, ensure:

1. **Tapenade is installed** on your system
2. **Tapenade executable** is available in your PATH
3. **C compiler** is configured for building the Tapenade runtime

Required Paths and Compilers
----------------------------

Based on typical configurations, the following environment variables should be set:

.. code-block:: bash

   # Set Tapenade installation path
   export TAPENADE_HOME=/path/to/tapenade

   # Ensure Tapenade executable is in PATH
   export PATH=$TAPENADE_HOME/bin:$PATH

   # C compiler for Tapenade runtime
   export CC=mpicc

Tapenade Runtime
----------------

Tapenade requires a runtime library (``adStack.c``) that must be compiled
and linked with the differentiated code:

.. code-block:: bash

   # Tapenade installation location
   TAPENADE_HOME=/path/to/tapenade
   TAPENADE_LIB=$(TAPENADE_HOME)/ADFirstAidKit
   TAPENADE_SRC=$(TAPENADE_LIB)/adStack.c

   # Compile the runtime
   $(CC) -fopenmp -c $(TAPENADE_SRC) -o adStack.o

FGPT Integration
----------------

FGPT isolates the target subroutine together with all its cross-dependencies,
producing a clean, self-contained Fortran file that Tapenade can process
without external library dependencies. Tapenade emits two differentiated versions of the isolated subroutine:

* ``{module_name}_tgt.f90`` - Tangent-linear (forward mode) derivative code
* ``{module_name}_b.f90`` - Adjoint (reverse mode) derivative code

Both files are generated in the same directory as the isolated procedure,
ready for compilation and linking with the Tapenade runtime (``adStack.o``).

TapenadePass Class
------------------

The ``TapenadePass`` class is responsible for post-processing Tapenade-generated
code. It cleans and prepares the differentiated Fortran code by removing
unnecessary statements, resolving array shapes, and replacing Tapenade-specific
placeholders with actual values.

**Purpose:**

- Remove unnecessary USE and EXTERNAL statements introduced by Tapenade
- Identify and replace ``ISIZE`` dimensions with actual array bounds
- Infer array shapes from assignments and reduction operations
- Handle nested reductions (``SUM``, ``MINLOC``, ``MAXLOC``, etc.)
- Process CALL statements with ``ISIZE`` arguments
- Map declarations to correct shapes or scalars

**Key Design:**

- Processes each subroutine individually to avoid variable name conflicts
- Maintains three levels of array information: original, module-level, and subroutine-level
- Tracks current subroutine context during processing
- Handles ``ALLOCATABLE`` arrays by inheriting shapes from base arrays

**Main Methods:**

- ``generate_adjoint_and_tangent()``: Generates both tangent and adjoint code
- ``clean_tapenade_statements()``: Main entry point for cleaning
- ``get_array_info()``: Searches array information across all levels
- ``set_array_info()``: Stores array information at the appropriate level

Usage Example
-------------

.. code-block:: python

   from fgpt.core.common.logger import Logger
   from fgpt.core.passes.tapenade import TapenadePass

   # Initialize logger
   logger = Logger()

   # Define array information
   all_array_info = {
       "my_array": [
           {"dim_str": "1", "dim_end": "10"},
           {"dim_str": "1", "dim_end": "20"},
       ],
   }

   # Create Tapenade pass
   tapenade_pass = TapenadePass(
       logger=logger,
       allowed_external_subroutines=[],
       all_array_info=all_array_info,
   )

   # Generate tangent and adjoint code
   tapenade_pass.generate_adjoint_and_tangent(
       file_path="my_module.f90",
       module_name="my_module",
       subroutine_name="my_subroutine"
   )

Workflow
--------

The Tapenade integration workflow consists of the following steps:

1. **Isolation**: FGPT isolates the target subroutine with all dependencies
2. **Differentiation**: Tapenade generates tangent and adjoint code
3. **Post-processing**: The ``TapenadePass`` class cleans and fixes the generated code

Compilation
-----------

Compilation of the generated code with the Tapenade runtime are integerated into the ``Makefile``.


Limitations
-----------

- ALLOCATABLE arrays must be declared at module level (not inside subroutines)
- Only Fortran 2003 is fully supported
- Tapenade must be installed and available in PATH
- The runtime library (adStack.c) must be compiled separately


References
----------

- Tapenade Documentation: https://inria.hal.science/hal-00913983/document/
- Tapenade Website: https://www.tapenade.inria.fr/
- FGPT Documentation: https://fgpt.readthedocs.io/
