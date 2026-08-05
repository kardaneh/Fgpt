Configuration
=============

FGPT's code generators (both the Python/NumPy middle-end and the JAX
backend) do not hard-code the Python source they emit. Instead, they read
it from a YAML file of code templates, conventionally named
``template.yaml``. This page documents its structure and every built-in
template.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

``template.yaml`` is passed as the ``config_path`` argument wherever a
generator is constructed, e.g.:

.. code-block:: python

   from fgpt.autodiff import AutoDiff

   autodiff = AutoDiff(config_path="template.yaml", mode="jax")

There are two ways to supply it:

- **User-supplied file.** Point ``config_path`` at your own YAML file.
  This file can override individual built-in templates and/or add new
  ones; templates you don't override fall back to the package defaults.
- **Package default.** If you omit ``config_path`` (or point to a file
  that doesn't redefine a given template), FGPT falls back to the default
  templates bundled inside the package at build time.

This means the *safe* way to customize generation is to copy only the
templates you need to change into your own ``template.yaml`` — everything
else continues to use the tested package defaults.

Top-Level Structure
--------------------

The file has two top-level sections, one per backend:

.. code-block:: yaml

   Python_templates:
       <template_key>:
           name: "<template_key>"
           template: |
               <raw Python source, as a YAML block scalar>

   JAX_templates:
       <template_key>:
           name: "<template_key>"
           template: |
               <raw Python source, as a YAML block scalar>

Each entry under ``Python_templates`` or ``JAX_templates`` is a named
template with two fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``name``
     - The template's key, repeated as a string. Must match the
       dictionary key it's nested under.
   * - ``template``
     - A YAML block scalar (``|``) containing the raw Python source to
       emit. May contain placeholders — see
       :ref:`template-placeholders` below — that are substituted by the
       generator before the code is written out.

.. _template-placeholders:

Placeholders
-------------

Templates are not static text: they contain placeholders that the
generator fills in at code-generation time. Two placeholder styles
appear in the built-in templates:

- **f-string style** (``{variable_name}``) — resolved using Python's own
  f-string / ``.format()`` semantics against values known at generation
  time, e.g. ``{benchmark_dir}``, ``{subroutine_name}``, ``{variable}``.
- **``$``-prefixed style** (``$path``) — substituted directly by the
  generator (for example, the output path for a timing log), independent
  of Python's string-formatting machinery.

Some templates also contain an empty list literal, e.g. ``for attr_name
in []:``. This is intentional: the generator populates this list
programmatically (e.g. with the attribute names discovered during shape
inference) before the template is emitted, so the ``[]`` in the source
template is a placeholder to be replaced with the real list, not a bug.

``Python_templates``
---------------------

Templates used by the Fortran-to-NumPy middle-end when assembling the
NumPy-based ``.py`` output.

``Python_global_normal_template``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module-level (non-class) scaffold for the global-variables module: plain
module-level constants (``nice``, ``kjpindex``, etc.) and a
``declaration_initialization()`` function stub. Used when the target
routine is generated in flat/procedural style rather than as a class.

``Python_global_class_template``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Class-based equivalent of the above: the same set of global variables are
instead assigned as ``self.*`` attributes inside ``__init__``, with a
``declaration_initialization(self)`` method stub. This is the template
used for the class-based output that :doc:`jax_conversion` subsequently
consumes (since the JAX backend operates on a NumPy *class*, not a flat
module).

``Python_main_template``
~~~~~~~~~~~~~~~~~~~~~~~~

Minimal driver-script scaffold: a ``main()`` function and the standard
``if __name__ == "__main__":`` entry point. This is the base that the
generator's ``main_<routine>.py`` output is built from.

``Python_read_for_loop_class_template``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads reference input data for a class instance from a Fortran binary
file (``global.bin``) using ``scipy.io.FortranFile``, matching each
attribute's dtype (``float64`` → reals, ``int32``/``bool`` → ints) and
reshaping arrays in Fortran (column-major, ``order='F'``) order. The
``for attr_name in []:`` list is populated by the generator with the
actual attribute names to restore.

``Python_read_dummy_template``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the above, but reads *dummy argument* values (from
``dummy.bin``) rather than class attributes — i.e. it populates the
input arguments passed into the isolated subroutine under test, again
matched by dtype and reshaped in Fortran order.

``Python_timer_template``
~~~~~~~~~~~~~~~~~~~~~~~~~

A ``functools.wraps``-based decorator that times a wrapped function with
``time.perf_counter()``, prints the duration, and appends a
``[TIMER] ...`` line to a log file at ``$path``. This is the timing
mechanism behind the Fortran-vs-Python figures in :doc:`benchmark`.

``Python_test_output_template``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compares a Python-side value against the corresponding Fortran reference
value read from the same binary stream, using ``numpy.allclose`` for
arrays and ``numpy.isclose`` for scalars (see :doc:`benchmark` for the
tolerances used). On mismatch, prints the maximum absolute error and the
min/max of both the Python and Fortran values to aid debugging.

``JAX_templates``
-----------------

Templates used by the JAX backend (:doc:`jax_conversion`) when
instrumenting generated JAX code.

``JAX_timer_template``
~~~~~~~~~~~~~~~~~~~~~~

The JAX equivalent of ``Python_timer_template``. Differs in two
JAX-specific ways:

- It calls ``jax.tree_util.tree_map`` over the function's output,
  invoking ``.block_until_ready()`` on every leaf that supports it. This
  is required because JAX dispatches asynchronously — without blocking,
  a naive ``perf_counter()`` measurement would time only dispatch
  overhead, not actual execution, understating the true runtime.
- It resolves the function's display name via ``getattr(func,
  "__name__", func.func.__name__)`` to handle both plain functions and
  ``functools.partial``-wrapped ones (as commonly produced by
  ``eqx.filter_jit``-decorated methods).

Like its Python counterpart, it writes a ``[TIMER JAX] ...`` line to
``$path``. This is the timing mechanism behind the JAX-vs-Fortran figures
in :doc:`benchmark`, and is what makes the post-warm-up (steady-state)
timing methodology described there possible: since blocking on
``block_until_ready()`` forces completion of the traced computation, it
correctly separates JIT-compilation time from execution time as long as
the timed call is not the first (compiling) call.

See Also
--------

* :doc:`jax_conversion` — How ``config_path`` is consumed by
  :class:`~fgpt.autodiff.AutoDiff`, and how the ``vectorize`` parameter
  interacts with the JAX templates above.
* :doc:`transformation` — The Fortran-to-NumPy middle-end that consumes
  the ``Python_templates`` section.
* :doc:`benchmark` — How the timer and test-output templates above are
  used to produce the runtime and correctness figures.
