JAX Conversion
==============

The JAX conversion layer — the JAX backend in FGPT's compiler-style
pipeline (see :doc:`architecture`) — transforms a NumPy-based Python class
produced by the FGPT transpiler into a JAX/Equinox-compatible module ready
for XLA-compiled execution on CPU, GPU, or TPU. It can also be applied
standalone to any compatible hand-written NumPy class, independently of the
Fortran transpilation pipeline.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The conversion is orchestrated by :class:`~fgpt.autodiff.AutoDiff` and
driven method-by-method by
:class:`~fgpt.core.backends.jax_converter.converter.JaxConverter`. The two
classes have clearly separated responsibilities:

:class:`~fgpt.autodiff.AutoDiff` handles **class-level restructuring** —
the same role :class:`~fgpt.core.transpiler.transformer.Transformer` plays in
the Fortran-to-NumPy middle-end. It rewrites the class declaration to
inherit from ``eqx.Module``, converts ``np`` calls and type annotations to
their ``jnp`` equivalents, classifies attributes as Equinox static or
dynamic fields, and emits the final output file.

:class:`~fgpt.core.backends.jax_converter.converter.JaxConverter` handles
**all control-flow and expression rewriting** within each method. It is an
``ast.NodeTransformer`` composed of focused mixin classes covering array
updates, conditionals, loop transpiler, call rewriting, and scope management.
Unlike the middle-end's largely 1:1 statement translation, this rewriting is
analysis-driven and optimizing in nature — loops and conditionals are
classified before a rewrite strategy is chosen. See :doc:`architecture` for
the full breakdown of its internals and the translation-vs-optimization
distinction between the two stages.

The user can specify which loop upper-bound variables (e.g. ``kjpindex``,
``nvm``, ``npts``) should trigger vectorization via the ``vectorize``
parameter of :class:`~fgpt.autodiff.AutoDiff` — see
:ref:`the-vectorize-parameter` below for details.

The output is a sibling file suffixed according to the chosen ``mode``:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Mode
     - Output suffix
     - Description
   * - ``jax``
     - ``_jax.py``
     - XLA-compiled JAX module. Loops become ``lax.scan`` or ``vmap``;
       conditionals become ``lax.cond`` or ``jnp.where``; in-place array
       updates become ``.at[].set()``.
   * - ``fwd``
     - ``_d.py``
     - Same structural transformation as ``jax``, scaffolded for
       forward-mode (tangent-linear) differentiation. ``jax.jvp`` call
       sites are not yet emitted.
   * - ``bwd``
     - ``_b.py``
     - Same structural transformation as ``jax``, using checkpointed
       ``while`` loops to reduce memory during reverse-mode AD.
       ``jax.grad`` / ``jax.vjp`` call sites are not yet emitted.

.. note::

   **Current differentiation status.** All three modes produce valid,
   XLA-compilable ``eqx.Module`` subclasses. However, FGPT does not yet
   emit ``jax.grad``, ``jax.jvp``, or ``jax.vjp`` call sites, because the
   variables with respect to which gradients should be taken are not yet
   specified in the pipeline configuration. Explicit differentiation wrappers
   are planned for a future release once the input specification interface is
   defined.

What the Conversion Produces
----------------------------

Given a NumPy class such as:

.. code-block:: python

   import numpy as np

   class HydrolAlma:
       def __init__(self):
           self.kjpindex = 100
           self.soil_temp = np.zeros(100)

       def compute(self):
           for ji in range(0, self.kjpindex):
               self.soil_temp[ji] = self.soil_temp[ji] + 1.0

The conversion produces an Equinox module where:

- The class inherits from ``eqx.Module``.
- ``np`` operations become ``jnp`` operations.
- Attributes are annotated as static or dynamic Equinox fields.
- The ``for`` loop is lowered to ``lax.scan``.
- The in-place update ``soil_temp[ji] = ...`` becomes
  ``soil_temp = soil_temp.at[ji].set(...)``.
- The outermost public method is decorated with ``eqx.filter_jit``.

.. code-block:: python

   import jax.numpy as jnp
   import equinox as eqx
   from jax import lax

   class HydrolAlma(eqx.Module):
       kjpindex: int = eqx.field(static=True)
       soil_temp: jnp.ndarray

       @eqx.filter_jit
       def compute(self):
           def _scan_body_0(carry, ji):
               soil_temp = carry
               soil_temp = soil_temp.at[ji].set(soil_temp[ji] + 1.0)
               return soil_temp, None

           self.soil_temp, _ = lax.scan(
               _scan_body_0, self.soil_temp, jnp.arange(0, self.kjpindex)
           )

Usage
-----

Basic Usage
~~~~~~~~~~~

Pass the paths of the class file and main driver file produced by the FGPT
transpiler (or any compatible NumPy class files) to
:meth:`~fgpt.autodiff.AutoDiff.transform`:

.. code-block:: python

   from fgpt.autodiff import AutoDiff

   autodiff = AutoDiff(
       config_path="template.yaml",
       mode="jax",
   )

   routine = "hydrol_alma"
   autodiff.transform(
       class_file=f"hydrol/{routine}/global_module_{routine}.py",
       main_file=f"hydrol/{routine}/main_{routine}.py",
   )

This writes ``global_module_hydrol_alma_jax.py`` and
``main_hydrol_alma_jax.py`` alongside the originals.

.. note::

   ``config_path`` points to a YAML file describing the code templates
   used during generation (global module, main driver, I/O boilerplate,
   timers, and so on). It can either be a file supplied by the user
   (combining the package's default templates with user-added ones) or
   omitted to fall back on the default templates bundled with the FGPT
   package at build time. See :doc:`configuration` for the full
   ``template.yaml`` schema and a description of every built-in template.

Choosing a Mode
~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard JAX execution (default)
   autodiff = AutoDiff(config_path="template.yaml", mode="jax")

   # Scaffolded for forward-mode AD (fwd) — jax.jvp wrappers not yet emitted
   autodiff = AutoDiff(config_path="template.yaml", mode="fwd")

   # Scaffolded for reverse-mode AD (bwd) — jax.grad wrappers not yet emitted
   autodiff = AutoDiff(config_path="template.yaml", mode="bwd")

.. _the-vectorize-parameter:

The ``vectorize`` Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vectorize`` controls how a given loop's index variable is treated during
lowering:

- If a loop's lower-bound variable (e.g. ``kjpindex``) is listed in
  ``vectorize``, its loop index is rewritten to a full-slice (``:``) over
  that axis, and the loop body is lowered to a **vectorised body** —
  the loop itself disappears and the body operates directly over the
  batch axis (see the "Vectorised body (loop removed)" row in
  :ref:`transformation-details` below).
- If the loop's lower-bound variable is **not** listed in ``vectorize``,
  the loop is instead lowered directly to ``lax.scan``, preserving its
  sequential, state-carrying semantics.

.. code-block:: python

   # kjpindex is vectorized: loop index becomes ":", loop disappears
   autodiff = AutoDiff(config_path="template.yaml", vectorize=["kjpindex"], mode="jax")

   # kjpindex is not vectorized: loop lowers to lax.scan instead
   autodiff = AutoDiff(config_path="template.yaml", mode="jax")

.. note::

   Vectorization via ``vectorize`` currently produces a full-slice
   (``:``) rewrite rather than an explicit ``jax.vmap`` call.

Working with AST Objects
~~~~~~~~~~~~~~~~~~~~~~~~

Advanced users integrating FGPT into custom code-generation pipelines can
apply the transformation directly to Python :mod:`ast` objects rather than
source files. This avoids a round-trip through the file system and allows
the converted AST to be composed with other programmatic transformations
before being unparsed:

.. code-block:: python

   import ast
   from fgpt.autodiff import AutoDiff
   from fgpt.core.backends.jax_converter.converter import JaxConverter

   autodiff = AutoDiff(config_path="template.yaml", vectorize=["kjpindex"], mode="jax")

   routine = "hydrol_alma"
   # Parse source to AST
   with open(f"hydrol/{routine}/global_module_{routine}.py") as f:
       class_tree = ast.parse(f.read())

   with open(f"hydrol/{routine}/main_{routine}.py") as f:
       main_tree = ast.parse(f.read())

   # Apply restructuring directly to the file through the
   # ast.Module
   autodiff.transform(
        class_file = class_tree,
        main_file = main_tree,
    )
   # Unparse to source global
   class_source = ast.unparse(ast.fix_missing_locations(class_tree))
   main_source = ast.unparse(ast.fix_missing_locations(main_tree))

.. _transformation-details:

Transformation Details
----------------------

The following table summarises the key rewriting rules applied by
:class:`~fgpt.core.backends.jax_converter.converter.JaxConverter`:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Fortran / NumPy construct
     - JAX equivalent
     - Strategy
   * - ``a[i] = v``
     - ``a = a.at[i].set(v)``
     - Functional update; also handles ``.add``, ``.multiply``, etc.
   * - ``if cond: x = a else: x = b``
     - ``x = jnp.where(cond, a, b)``
     - Pure value-select; no side effects.
   * - ``if cond: <stateful block>``
     - ``lax.cond(cond, _if_true_N, _if_false_N, ...)``
     - Index-dependent or stateful branch; synthetic helpers generated.
   * - ``if cond: a[mask] = v``
     - ``a = jnp.where(mask, v, a)``
     - Element-wise masked assignment.
   * - ``for i in range(...): <sequential>``
     - ``lax.scan(_scan_body_N, carry, indices)``
     - State-carrying sequential loop; the default when the loop's
       upper-bound variable is not listed in ``vectorize``.
   * - ``for i in range(...): <independent>``
     - Vectorised body (loop removed); index rewritten to ``:``
     - Applies when the loop's upper-bound variable is listed in
       ``vectorize`` (see :ref:`the-vectorize-parameter`). A ``vmap``-based
       rewrite is planned for a future release.
   * - ``for i in ...: while <cond(i)>:``
     - ``jax.vmap(eqx.internal.while_loop(...))``
     - Vectorised axis inside a dynamic ``while``; supported for
       forward- and reverse-mode (``fwd`` / ``bwd``) differentiation.
   * - ``np.zeros / np.ones / ...``
     - ``jnp.zeros / jnp.ones / ...``
     - Library alias replacement by :class:`~fgpt.autodiff.AutoDiff`.
   * - ``print(...) / logging.info(...)``
     - *(removed)*
     - Stripped by ``RemoveLogging`` before tracing begins.

.. _known-limitations:

Known Limitations
-----------------

The JAX conversion layer is under active development. The following
constructs are not yet supported, or are only partially supported:

- **Class definitions (``ClassDef``) inside a converted method.** Nested
  or locally-defined classes are not rewritten and will not lower
  correctly.
- **Augmented assignment (``AugAssign``).** Only the explicit form
  ``a = a + 1`` is currently rewritten correctly. The shorthand form
  ``a += 1`` is not yet handled and should be avoided (or rewritten to
  the explicit form) in code destined for JAX conversion.
- **Some ``if`` constructs.** A subset of conditional patterns beyond
  those listed in :ref:`transformation-details` above are not yet
  implemented.
- **Dynamic loops** (i.e. loops whose trip count is not known until
  runtime) are supported.
- **``while`` loops** are supported specifically via
  ``eqx.internal.while_loop``, which allows both forward-mode and
  reverse-mode differentiation through the loop.
- **Differentiation call sites.** As noted above, ``jax.grad``,
  ``jax.jvp``, and ``jax.vjp`` are not yet emitted automatically in any
  mode.

If you encounter a construct that silently produces incorrect output
rather than a clear error, please report it — the priority for upcoming
releases is converting silent failures into explicit, actionable errors
at conversion time.

See Also
--------

* :doc:`architecture` — How the JAX backend fits into the overall FGPT
  pipeline, and the translation-vs-optimization distinction between the
  middle-end and this stage.
* :doc:`transformation` — The preceding Fortran-to-NumPy transpilation stage
  whose output this layer consumes.
* :doc:`configuration` — Full schema of ``template.yaml``, including the
  built-in Python and JAX code templates referenced by ``config_path``.
* :doc:`benchmark` — Runtime and correctness comparisons between the
  Fortran, Python, and JAX outputs of this pipeline, including how the
  ``vectorize`` parameter affects measured speed-ups.
* :doc:`project_structure` — Full package layout, including the
  ``core/backends/jax_converter/`` module breakdown.
* :doc:`api/fgpt` — Full API reference for :class:`~fgpt.autodiff.AutoDiff`
  and :class:`~fgpt.core.backends.jax_converter.converter.JaxConverter`.
