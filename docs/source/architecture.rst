Architecture
============

FGPT is a source-to-source **transpiler** that translates production
Fortran scientific code into executable NumPy-based Python, and then
optionally **lowers** that intermediate representation into
JAX/Equinox-compatible modules suitable for GPU acceleration and automatic
differentiation. Internally, it is organised like a conventional compiler:
a pipeline of independently testable phases — frontend, middle-end, and
backend — each communicating through well-defined intermediate
representations (IRs) rather than shared mutable state. The Fortran → NumPy
phase is largely a direct syntax-level translation, while the NumPy → JAX
phase goes further, applying genuine optimizing-backend transformations
(vectorisation, loop/conditional rewriting) rather than a 1:1 mapping —
see the callout in the *JAX Backend* section below.

.. contents:: Contents
   :local:
   :depth: 2

High-Level Pipeline
-------------------

.. code-block:: text

    Fortran Source (.f90)
            │
            ▼
    ┌───────────────────┐
    │     Frontend      │  Processor · Isolator · Navigator · Extractor
    │  (parse + IR gen) │
    └────────┬──────────┘
             │  standalone Fortran AST (IR) + metadata
             ▼
    ┌───────────────────┐
    │    Middle-End     │  F2NP · Transformer
    │  (IR → NumPy IR)  │  → NumPy-based Python class
    └────────┬──────────┘
             │  .py source file (target-neutral IR)
             ▼
    ┌───────────────────┐
    │   JAX Backend     │  AutoDiff · JaxConverter
    │  (target lowering)│  → lax.scan · lax.cond · vmap · .at[].set()
    └────────┬──────────┘
             │  _jax / _d / _b .py file
             ▼
    Executable JAX/Equinox Module


Frontend — Parsing and IR Construction
--------------------------------------

The frontend locates a target Fortran subroutine within a large codebase,
resolves all of its cross-module dependencies, and produces a self-contained
AST — FGPT's intermediate representation — ready for the middle-end. It is
composed of five components that run in sequence, mirroring a classical
compiler frontend's lexing/parsing, symbol resolution, and semantic analysis
stages.

:class:`~fgpt.core.frontend.processor.Processor` is the **parser**: the
shared primitive used throughout the frontend. It wraps ``fparser`` with a
Fortran 2008 grammar and exposes utilities for building ASTs from files,
strings, or individual statements. Every other frontend component depends
on it.

:class:`~fgpt.isolator.Isolator` is the frontend's **driver / entry point**.
It identifies the target module, invokes
:class:`~fgpt.core.frontend.processor.Processor` to parse it, and produces
both the original and a re-parsed copy of the AST for structural
transformation. It also tracks internal call relationships and compilation
error flags used by downstream passes.

:class:`~fgpt.core.frontend.navigator.Navigator` performs **symbol
resolution** for names not defined in the immediately parsed module. It runs
a breadth-first search over the module hierarchy, following ``USE`` chains,
to locate variable declarations and subroutine definitions across file
boundaries. It handles interface blocks and avoids redundant traversal via a
visited-modules set.

:class:`~fgpt.core.frontend.extractor.Extractor` performs **semantic
analysis**: static analysis over the parsed module that builds every
metadata structure consumed downstream — dummy argument lists, call graphs,
array shapes, loop patterns, scope classifications (global / local /
modified), and implicit shape information. It is stateful and should be
re-instantiated per independent analysis session. Array-shape and dimension
inference specifically is delegated to
:class:`~fgpt.core.analysis.shaper.Shaper`.

**Key IR artefacts produced:**

- Standalone Fortran AST (``fparser`` representation) for the target subroutine.
- ``cls_info`` — symbol table mapping variable names to types, dimensions,
  dtypes, and scope.
- Loop, array, and dependency metadata consumed by the middle-end transpiler
  pass, :class:`~fgpt.core.transpiler.f2np.F2NP`.

Middle-End — Lowering Fortran IR to NumPy
-----------------------------------------

The middle-end is the core of FGPT's code generation: it lowers the
Fortran AST into structurally equivalent NumPy-based Python — a
target-neutral IR that both the reference NumPy execution path and the JAX
backend can consume. The output is a standalone Python class whose methods
directly correspond to the original Fortran subroutines. This phase is split
between a statement-level code generator and a module-level orchestrator.

:class:`~fgpt.core.transpiler.f2np.F2NP` is the **statement/expression-level
code generator**. It walks the Fortran AST subroutine by subroutine and
incrementally builds a raw Python :mod:`ast` tree. Control-flow constructs
(``DO``, ``IF``, ``SELECT CASE``) are tracked via an explicit stack rather
than Python's call stack, since Fortran's block-closing statements must be
matched against nested and chained constructs. Expression-level translation is
centralised in ``handle_expr``, which dispatches literals, binary/unary
operations, array part references, and intrinsic function calls — mapped to
NumPy equivalents via :mod:`~fgpt.core.transpiler.intrinsic` — to dedicated
handlers. Fortran ``WHERE`` constructs are lowered to
``if mask.any(): ...`` blocks with boolean-mask subscripting.

:class:`~fgpt.core.transpiler.transformer.Transformer` is the middle-end's
**module-level orchestrator**, operating above the statement level. It takes
the raw Python AST from :class:`~fgpt.core.transpiler.f2np.F2NP` and wraps it
in a proper class structure: converting Fortran ``SPECIFICATION PART``
declarations into class attributes, resolving cross-subroutine dependencies,
rewriting ``CALL`` statements as method invocations on the correct instance,
generating binary I/O boilerplate, and emitting the final ``.py`` source
file. It applies two IR-to-IR post-processing passes before code emission:

- :class:`~fgpt.core.support.utils.ReplaceGlobals` rewrites unqualified
  variable and method references into ``self.attr`` or ``instance.attr``
  accesses.
- :class:`~fgpt.core.support.utils.AdjustIndices` compensates for Fortran's
  1-based array indexing by inserting ``-1`` offsets at every array
  subscript and adjusting loop bounds correspondingly.

**Key IR artefacts produced:**

- A ``.py`` source file containing a Python class whose methods correspond
  to the translated Fortran subroutines, using NumPy for array operations —
  the target-neutral IR consumed by the JAX backend.

JAX Backend — Target Lowering and Code Generation
-------------------------------------------------

The JAX backend lowers the target-neutral NumPy IR produced by the
middle-end into a JAX/Equinox-compatible module — analogous to a compiler
backend translating a generic IR into target-specific machine code. The
output is a sibling file suffixed with ``_jax``, ``_d`` for forward-mode, or
``_b`` for reverse-mode, whose class inherits from ``eqx.Module`` and
whose outermost method is decorated with ``eqx.filter_jit``.

.. note::

   **Translation vs. optimization.** The Fortran → NumPy middle-end
   (Stage 2) is close to a direct, structural translation: each Fortran
   statement maps to a corresponding Python/NumPy statement, and
   :class:`~fgpt.core.transpiler.transformer.Transformer`'s post-processing
   passes (``ReplaceGlobals``, ``AdjustIndices``) are correctness fixups
   rather than optimizations. The JAX backend (Stage 3) does more than
   translate: :class:`~fgpt.core.backends.jax_converter.converter.JaxConverter`
   performs analysis-driven **optimizing transformations** — classifying
   loops and conditionals before rewriting them (``VectorizationAnalyzer``),
   choosing between ``jnp.where``, ``lax.cond``, and masked vectorised
   assignment based on that classification, and inferring reduction axes
   and broadcast shapes rather than emitting a fixed mapping. This is why
   the JAX phase is described as a backend performing target-specific code
   generation and optimization, whereas the NumPy phase is closer to a
   conventional transpiler pass.

This backend has two components with clearly separated responsibilities:
:class:`~fgpt.autodiff.AutoDiff` handles class-level restructuring (target
ABI setup), and :class:`~fgpt.core.backends.jax_converter.converter.JaxConverter`
handles all control-flow and expression code generation.

AutoDiff — Backend Driver
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fgpt.autodiff.AutoDiff` plays a role analogous to
:class:`~fgpt.core.transpiler.transformer.Transformer` in the middle-end: it
is the backend's high-level driver, restructuring the module as a whole
rather than translating individual statements. Its responsibilities are:

- Rewriting the class declaration to inherit from ``eqx.Module`` instead
  of a plain Python class.
- Converting ``np`` array operations and type annotations to their ``jnp``
  equivalents (e.g. ``np.zeros`` → ``jnp.zeros``).
- Classifying and annotating class attributes as Equinox static or dynamic
  fields.
- Delegating all control-flow code generation to
  :class:`~fgpt.core.backends.jax_converter.converter.JaxConverter` method
  by method.
- Emitting the final ``_jax.py`` / ``_d.py`` / ``_b.py`` target file.

The ``mode`` parameter (``'jax'``, ``'fwd'``, ``'bwd'``) controls the output
suffix and selects between standard ``lax`` while loops and checkpointed
variants intended for reverse-mode AD. Before any code generation begins,
``RemoveLogging`` (in
:mod:`~fgpt.core.backends.jax_converter.converter`) strips ``print`` and
``logging.*`` calls from function bodies, since these are incompatible with
JAX tracing.

.. note::

   **Current status of differentiation modes.** FGPT can currently produce
   ``_jax``, ``_d``, and ``_b`` output files with the appropriate
   structural scaffolding, but does not yet emit ``jax.grad``,
   ``jax.jacfwd``, or ``jax.jacrev`` call sites. This is because the
   differentiation inputs — the variables with respect to which gradients
   are to be taken — are not yet specified as part of the pipeline
   configuration. Forward-mode (``jax.jvp``) and backward-mode
   (``jax.vjp`` / ``jax.grad``) differentiation wrappers are planned for a
   future release once the input specification interface is defined.

JaxConverter — Backend Code Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fgpt.core.backends.jax_converter.converter.JaxConverter` is an
``ast.NodeTransformer`` composed of several mixin classes — one per target
lowering concern, each living in its own module under
``core/backends/jax_converter/``. Together they perform all the statement-
and expression-level code generation that makes the emitted code traceable
by XLA:

- **Array updates** (:mod:`~fgpt.core.backends.jax_converter.array_updates`)
  — in-place subscript assignments (``a[i] = v``) are rewritten to
  ``a = a.at[i].set(v)`` (or ``.add``, ``.multiply``, etc.) to satisfy
  JAX's functional immutability requirement.
- **Conditionals** (:mod:`~fgpt.core.backends.jax_converter.conditionals`,
  :mod:`~fgpt.core.backends.jax_converter.masking`) — ``if`` / ``elif`` /
  ``else`` blocks are classified and lowered into ``jnp.where`` for pure
  value-select patterns, ``lax.cond`` with synthetic ``_if_true_N`` /
  ``_if_false_N`` helper functions for stateful branches, or vectorised
  mask assignments for element-wise branches.
- **Loops** (:mod:`~fgpt.core.backends.jax_converter.loops`,
  :mod:`~fgpt.core.backends.jax_converter.dynamic_loops`) — ``for`` loops
  are classified and lowered into ``lax.scan`` for sequential index loops,
  vectorised body expansion for batch-axis loops, or ``jax.vmap`` wrapping
  an ``eqx.internal.while_loop`` for loops whose ``while`` condition
  depends on the vectorised axis.
- **Call rewriting** (:mod:`~fgpt.core.backends.jax_converter.call_rewriting`)
  — call sites are updated to thread mutated attributes and arguments
  through helper functions correctly.
- **Scope management** (:mod:`~fgpt.core.backends.jax_converter.scope_utils`)
  — argument-name stacks and context stacks track lexical scope to decide
  whether a name should be passed by value or read from ``self``.
- **Vectorization** (:mod:`~fgpt.core.backends.jax_converter.vectorization`)
  — converts scalar functions into vectorized equivalents where profitable.

Synthetic helper functions (``_scan_body_N``, ``_if_true_N``, etc.)
generated during the visit pass are collected in ``_pending_helpers`` and
finalised by ``process_helpers`` after the main pass completes.

Supporting **analysis utilities**, in
:mod:`~fgpt.core.backends.jax_converter.analysis` and
:mod:`~fgpt.core.backends.utils`, assist the code generator — the backend's
equivalent of a compiler's target-specific analysis passes:

- ``VectorizationAnalyzer`` classifies each ``for`` loop and ``if`` block as
  ``index_loop``, ``vector``, ``masked``, ``lax.cond``-candidate, etc.,
  before any transformation occurs. It performs structural inspection only and
  does not modify the IR.
- ``ReductionHandler`` infers the correct ``axis=`` argument for reductions
  (``sum``, ``mean``, ``max``, ``min``) given array shapes, vectorisation
  context, and broadcasting semantics.
- ``MaybeAddIndexTransformer`` inserts broadcast and indexing operations
  (``:``, ``None``) to reconcile mismatched tensor ranks introduced by
  scalar lifting and batch-dimension promotion.
- ``WhileVectorToScalar`` rewrites vectorised ``x = x.at[idx].set(expr)``
  updates inside while loops into scalar accumulator assignments, and
  records which variables must be threaded through loop state.
- ``Control`` is a lightweight metadata container pushed onto
  ``_control_stack`` for each active loop or if-block scope, carrying the
  construct kind, loop bounds, transform type, vectorisation axis, and any
  additional analysis results needed by downstream passes.

**Key IR/target artefacts produced:**

- A ``_jax.py`` / ``_d.py`` / ``_b.py`` file containing an
  ``eqx.Module`` subclass with ``jnp``-based array operations, ``lax``-based
  control flow, functional array updates, and Equinox static/dynamic field
  annotations.

Package Structure
-----------------

The package layout mirrors the compiler pipeline directly: ``frontend/``
holds parsing and symbol resolution, ``analysis/`` holds shape/semantic
analysis shared across phases, ``passes/`` holds IR-to-IR transformation
passes, ``transpiler/`` holds the Fortran → NumPy IR transpiler (middle-end),
``backends/`` holds target-specific code generators (currently JAX), and
``support/`` holds cross-cutting infrastructure used by every phase.

.. code-block:: text

    fgpt/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py                     # Command-line interface (driver)
    ├── version.py                 # Package version
    ├── isolator.py                # Frontend driver: Fortran isolation pipeline
    ├── autodiff.py                # Backend driver: JAX pipeline
    │
    ├── core/
    │   ├── frontend/                    # Parsing + symbol resolution
    |   |   ├── __init__.py
    │   │   ├── processor.py             # Fortran parser (fparser wrapper)
    │   │   ├── extractor.py             # Semantic analysis / metadata extraction
    │   │   └── navigator.py             # Cross-module symbol resolution
    │   │
    │   ├── analysis/                    # Shared static-analysis utilities
    |   |   ├── __init__.py
    │   │   └── shaper.py                # Array shape/dimension analysis
    │   │
    │   ├── passes/                      # IR-to-IR transformation passes
    |   |   ├── __init__.py
    │   │   └── modifier.py              # Fortran AST transformation passes
    │   │
    │   ├── transpiler/                    # Middle-end: Fortran IR → NumPy IR
    |   |   ├── __init__.py
    │   │   ├── transformer.py           # Module-level transpiler orchestrator
    │   │   ├── f2np.py                  # Statement/expression-level codegen
    │   │   └── intrinsic.py             # Fortran intrinsic → NumPy mapping
    │   │
    │   ├── backends/                    # Target-specific code generators
    |   |   ├── __init__.py
    |   |   ├── utils.py                 # Shared helper functions across backends
    |   |   └── jax_converter/           # JAX target backend
    |   |       ├── converter.py         # Backend entry point: orchestrates JAX codegen
    |   |       ├── analysis.py          # Target-specific analysis (shape inference, deps)
    |   |       ├── array_updates.py     # Array mutation → JAX-safe functional update codegen
    |   |       ├── call_rewriting.py    # Call-site codegen for JAX-compatible primitives
    |   |       ├── conditionals.py      # if/else → lax.cond / jnp.where codegen
    |   |       ├── dynamic_loops.py     # Runtime-dependent loop-bound codegen
    |   |       ├── loops.py             # Static/structured loop → lax.scan codegen
    |   |       ├── masking.py           # Branchless masking codegen
    |   |       ├── scope_utils.py       # Variable-scope tracking during codegen
    |   |       └── vectorization.py     # Scalar → vectorized function codegen
    │   │
    │   └── common/                     # Cross-cutting infrastructure
    |       ├── __init__.py
    │       ├── executive.py             # Pipeline/workflow orchestration
    │       ├── logger.py                # Logging infrastructure
    │       ├── line_length.py           # Fortran line-length utilities
    │       └── utils.py                 # Shared helper utilities (ReplaceGlobals, AdjustIndices)
    │
    └── templates/
        └── default.yaml

Design Principles
-----------------

FGPT follows a strict staged compiler architecture where each phase
communicates with the next through well-defined IR artefacts (ASTs, symbol
tables/metadata dicts, source files) rather than shared mutable state.

**Separation of concerns.** Parsing, semantic analysis, Fortran-to-NumPy
transpilation, and JAX target transformation are implemented in distinct
frontend/middle-end/backend components with no circular dependencies. This
means each phase can be developed, tested, and executed independently, and
new backends could in principle be added alongside the JAX one without
touching the frontend or middle-end.

**IR-centric transformations.** All rewriting — whether at the NumPy-IR level
(``ReplaceGlobals``, ``AdjustIndices``), or the JAX target level
(``JaxConverter``) — operates on AST representations rather than source text,
ensuring structural correctness and enabling precise node-level introspection.

**Progressive metadata enrichment.** The frontend's ``Extractor`` builds the
symbol table and metadata structures (array shapes, scope maps, loop
patterns) once, and every downstream phase consumes them read-only. The JAX
backend extends this with its own ``cls_info``-derived analysis
(vectorisation axes, rank inference, scalar lifting) without revisiting the
Fortran source.

**Differentiation-mode portability.** The ``mode`` parameter (``'jax'``,
``'fwd'``, ``'bwd'``) threads through ``AutoDiff`` and ``JaxConverter`` to
select between standard ``lax`` while loops and checkpointed variants. The
structural scaffolding for all three modes is already generated; explicit
``jax.grad`` / ``jax.jvp`` / ``jax.vjp`` call sites are planned for a
future release once the differentiation input specification interface is
defined.

See Also
--------

* :doc:`transformation` — Detailed documentation of the Fortran-to-NumPy
  transpilation pipeline (frontend and middle-end).
* :doc:`jax_conversion` — Detailed documentation of the JAX target backend.
