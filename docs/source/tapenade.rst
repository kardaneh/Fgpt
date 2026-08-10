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
* ``{module_name}_adj.f90`` - Adjoint (reverse mode) derivative code

Both files are generated in the same directory as the isolated procedure,
ready for compilation and linking with the Tapenade runtime (``adStack.o``).
