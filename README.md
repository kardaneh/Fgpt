# Fgpt


[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-green.svg)](./docs)

**FGPT** is a source-to-source transpiler that converts production scientific
Fortran code into executable NumPy-based Python, and optionally into
JAX/Equinox-compatible modules for GPU-accelerated and differentiable
computation.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Pipeline Overview](#pipeline-overview)
4. [Core Components](#core-components)
   - [Processor](#processor)
   - [Isolator](#isolator)
   - [Navigator](#navigator)
   - [Extractor](#extractor)
   - [Modifier](#modifier)
   - [F2NP](#f2np)
   - [Transformer](#transformer)
   - [AutoDiff & JAX Conversion](#autodiff--jax-conversion)
   - [Executive](#executive)
   - [Shaper](#shaper)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Testing](#testing)
8. [Development](#development)
9. [License](#license)
10. [Authors](#authors)
---

## Introduction

FGPT was built to modernise large scientific Fortran codebases — such as
land-surface models — without requiring manual rewriting. It operates in
three stages:

1. **Isolation** : A target Fortran subroutine is extracted from its
   module, its cross-module dependencies are resolved, and a standalone
   compilable unit is produced and validated.
2. **Transpilation** : The isolated Fortran AST is translated
   statement-by-statement and expression-by-expression into a structurally
   equivalent NumPy-based Python class, preserving the original numerical
   semantics.
3. **JAX conversion** : The generated Python class is rewritten into a
   JAX/Equinox module: loops become `lax.scan` or `vmap`, conditionals
   become `lax.cond` or `jnp.where`, and in-place array updates become
   `.at[].set()`, enabling XLA compilation and automatic differentiation.

The translation is AST-based throughout. Fortran source is parsed into an
`fparser` AST; Python output is assembled as a `ast.Module` and unparsed
to source — never via string manipulation — ensuring syntactic correctness
and enabling precise, auditable transformations at every stage.

---


## Project Structure
```
   fgpt/
   ├── src/fgpt/                 # Main package
   │   ├── __init__.py
   |   ├── processor.py          # Fortran parser (fparser wrapper)
   |   ├── isolator.py           # Subroutine extraction
   |   ├── navigator.py          # Cross-module symbol resolution
   |   ├── extractor.py          # Static analysis and metadata extraction
   |   ├── modifier.py           # Optional Fortran AST rewriting
   |   ├── f2np.py               # Statement/expression-level translator
   |   ├── transformer.py        # Pipeline orchestrator; emits .py files
   |   ├── autodiff.py           # JAX conversion driver
   |   ├── jax_utils.py          # VectorizationAnalyzer, ReductionHandler, …
   │   ├── templates/
   │   │   └── default.yaml      # bundled default, never edited by users
   |   ├── jax_converter/
   |   │   ├── converter.py      # JaxConverter (ast.NodeTransformer)
   |   │   ├── analysis.py
   |   │   ├── array_updates.py
   |   │   ├── call_rewriting.py
   |   │   ├── conditionals.py
   |   │   ├── dynamic_loops.py
   |   │   ├── loops.py
   |   │   ├── masking.py
   |   │   ├── scope_utils.py
   |   │   └── vectorization.py
   |   ├── utils.py              # ReplaceGlobals, AdjustIndices, helpers
   |   ├── executive.py          # Workflow orchestration
   |   ├── logger.py             # Logging infrastructure
   |   ├── line_length.py        # Fortran line-length utilities
   |   ├── intrinsic.py          # Fortran intrinsic → NumPy mapping
   |   ├── shaper.py             # Array shape/dimension handling
   |   └── version.py
   ├── tests/                    # Unit tests
   │   ├── conftest.py
   │   ├── test_autodiff.py
   │   ├── test_extractor.py
   │   ├── test_f2np.py
   │   ├── test_intrinsic.py
   │   ├── test_jaxconverter.py
   │   ├── test_jax_utils.py
   │   ├── test_navigator.py
   │   ├── test_processor.py
   │   ├── test_shaper.py
   │   ├── test_utils.py
   │   └── test_transformer.py
   ├── docs/                     # Documentation
   │   ├── source/
   │   └── build/
   ├── .github/workflows/        # CI/CD pipelines
   │   ├── ci.yaml
   │   └── docs.yml
   ├── setup                     # Setup file for transformation
   ├── arch-nvhpc_HAL.env
   ├── arch-nvhpc_LEONARDO.env
   ├── arch-nvhpc_spirit.env
   ├── Makefile                  # Run isolated procedures
   ├── template.yaml             # Code generation templates(user-facing, can be customised freely)
   ├── pyproject.toml            # Package configuration
   ├── README.md                 # Project README
   └── LICENSE                   # CC BY-NC-SA 4.0
```

---

## Pipeline Overview

```
             Fortran Source (.f90)
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Isolation & Analysis         │
│  Processor → Isolator                   │
│       ├── Navigator  ─┐                 │
│       └── Extractor ◄─┘  → Modifier     │
└────────────────────┬────────────────────┘
                     │ corrected Fortran AST
                     ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Transpilation                │
│  F2NP → Transformer                     │
│       ├── ReplaceGlobals                │
│       └── AdjustIndices                 │
└────────────────────┬────────────────────┘
                     │ .py source file
                     ▼
┌─────────────────────────────────────────┐
│  Stage 3 — JAX Conversion (optional)    │
│  AutoDiff → JaxConverter                │
│       ├── lax.scan / vmap               │
│       ├── lax.cond / jnp.where          │
│       └── .at[].set()                   │
└────────────────────┬────────────────────┘
                     │
                     ▼
        JAX/Equinox Module (_jax.py)
```

---

## Core Components

### Processor

`processor.py` is the shared parsing primitive used throughout all three
pipeline stages. It wraps `fparser` with a Fortran 2008 grammar and provides
a unified interface for building ASTs from files, strings, or individual
statements.

**Key responsibilities:**

- **AST construction** — parses Fortran source from files, raw strings, or
  individual statements into `fparser` AST objects.
- **Declaration handling** — splits, duplicates, and recombines entity
  declarations and allocations (`separate_entity_declaration`,
  `combine_allocate_declaration`, `map_declaration`) to support both
  CPU and GPU code paths.
- **Output generation** — writes the standalone `global_module.f90` and
  `main.f90` files that are compiled and run to validate each isolated
  subroutine (`update_global_module`, `update_main_program`).
- **Compilation & execution** — compiles and runs the generated Fortran
  programs via `compile_and_run`, asserting correctness before any Python
  output is produced.
- **Utility methods** — generates `CALL` statements (`create_call_stmt`),
  handles variable initialisation and I/O instrumentation
  (`initialization_statement`), and queues declarations in a consistent
  order (scalars first, then arrays, then parameters) via `process_queue`.
---

### Isolator

`isolator.py` is the entry point of the pipeline. Given a target module and
a list of subroutine names, it extracts each routine into a self-contained,
independently compilable unit.

**Key responsibilities:**

- Locates the target module and parses it via `Processor`. A
  `_org.fgpt` backup of the original source is created on first run so
  that re-runs always start from unmodified Fortran.
- Recursively isolates nested subroutine calls depth-first, propagating
  global variable declarations upward via `collect_global_vars_decl`,
  so every parent's standalone module includes everything it and its
  callees need.
- Calls `Extractor` for static analysis, `Navigator` for cross-module
  symbol resolution, and optionally `Modifier` for Fortran-level
  rewrites before generating output.
- Produces a `global_module_<name>.f90` (shared declarations) and
  `main_<name>.f90` (driver program) for each routine, compiles and
  runs them to validate output, and optionally triggers `Transformer`
  to emit Python immediately when `f2py=True`.
- Can be run as a command-line tool — see [Usage](#usage).
---

### Navigator

`navigator.py` resolves variable declarations and subroutine definitions
that are not present in the immediately parsed module. It is called
exclusively by `Extractor` whenever a symbol cannot be resolved in the
current scope.

**Key responsibilities:**

- Performs a breadth-first search over the Fortran module hierarchy,
  following `USE` chains across file boundaries.
- Handles external subroutine interface blocks and avoids redundant
  traversal via a visited-modules set.
- Exposes `find_variable_in_module`, `variable_finder`, and
  `find_var_in_child_modules` for variable resolution, and
  `find_external_subroutines_in_module` / `external_subroutine_finder`
  for subroutine discovery.
---

### Extractor

`extractor.py` performs static and structural analysis of the parsed Fortran
module and builds all metadata structures consumed by the transpilation
stage.

**Key responsibilities:**

- **Subroutine discovery** — `find_subroutines` identifies all subroutines
  and functions, maps dummy argument lists, and distinguishes internal from
  external routines.
- **Variable classification** — `find_variables` classifies every variable
  in a subroutine as dummy argument (with intent IN / OUT / INOUT), global
  (imported via `USE`), or local. `find_global_variables` delegates
  cross-module resolution to `Navigator`.
- **Array metadata** — `extract_all_array_info` collects dimensional and shape
  information for all arrays, determines allocation requirements, and
  unifies allocatable declarations via `Processor.combine_allocate_declaration`.
- **Loop analysis** — `extract_loop_vect` retrieves variables usable as
  global loop variables; `extract_intent` extracts DO loop index names
  and their bounds.
- **AST normalisation** — `clean_subroutine` removes redundant or
  inconsistent declarations and verifies structural integrity before
  transpilation.
The Extractor is stateful; re-instantiate it for each independent
transpilation session.

---

### Modifier

`modifier.py` is an optional pass that rewrites the Fortran AST *before*
transpilation. It operates entirely within the Fortran representation and
produces no Python output — its purpose is to normalise the source so that
`F2NP` encounters only constructs it can translate directly.

**Key responsibilities:**

- Replaces unsupported intrinsic functions (`MAXLOC`, `MINLOC`, etc.) with
  equivalent manual loops compatible with the target environment.
- Converts array colon-slicing into indexed loops
  (`replace_vec_colon_with_index`) and merges vector loop bodies
  (`merge_vector_loop`).
- Adds `DO` loops for implicit array assignments (`add_dos`) and adjusts
  array bounds and subscript lists for GPU memory layouts
  (`modify_colon_array`, `modify_colon_array_vec`).
- Strips or replaces GPU-incompatible I/O operations (`WRITE`, `OPEN`,
  `CLOSE`) with flag-based alternatives (`replace_gpu_unsupported`).
- Modifies `SPECIFICATION PART` declarations to add assumed-shape specs
  and OpenACC/GPU attributes (`modify_specification_part`).
---

### F2NP

`f2np.py` is the core of the transpiler. It performs the actual
source-to-source translation, walking the Fortran AST for a single
subroutine and incrementally building the equivalent Python `ast` tree,
statement by statement and expression by expression.

**Fortran → Python/NumPy mappings:**

| Fortran construct | Python / NumPy equivalent |
|---|---|
| `DO i = a, b` | `for i in range(a, b)` |
| `IF / ELSE IF / ELSE` | `if / elif / else` |
| `WHERE (mask)` | `if mask.any(): ...` with boolean-mask subscripting |
| `SELECT CASE` | `if / elif` chain |
| `CALL sub(args)` | `sub(args)` |
| `REAL, DIMENSION(n) :: A` | `A = np.zeros(n, dtype=np.float64)` |
| `ABS`, `SQRT`, `MAXVAL`, … | `np.abs`, `np.sqrt`, `np.max`, … |
| `arr(i)` (1-based) | `arr[i]` (corrected by `AdjustIndices`) |

Control-flow constructs are tracked via an explicit stack and per-construct
counters rather than Python's call stack, since Fortran's block-closing
statements (`END IF`, `END DO`, `END SELECT`) must be matched against
possibly nested and chained (`ELSE IF`) constructs.

The main entry point is `recursive_ast`, which dispatches each Fortran
statement type to a dedicated `handle_*` method. Expression-level
translation is centralised in `handle_expr`.

---

### Transformer

`transformer.py` is the pipeline orchestrator. It consumes the raw Python
AST emitted by `F2NP` and assembles it into a coherent, importable Python
module — playing the same role as a compiler back-end.

**Key responsibilities:**

- **Class scaffolding** — converts Fortran `SPECIFICATION PART` declarations
  into Python class attributes and `__init__` assignments; pre-initialises
  dependent variables so the generated class is self-consistent.
- **Dependency resolution** — builds `cls_info` (variable → owning class
  mapping) from Fortran declarations and `USE` imports; consumed by
  `ReplaceGlobals` and call-site rewriting.
- **Call-site rewriting** — rewrites `CALL` statements as method invocations
  on the correct Python class instance, injecting `self` or instance
  arguments and correcting argument order.
- **Binary I/O generation** — synthesises NumPy binary-read boilerplate for
  subroutines that consume Fortran binary data files.
- **Post-processing** — applies `ReplaceGlobals` (unqualified names →
  `self.attr`) and `AdjustIndices` (1-based → 0-based subscripts) before
  emitting the final `.py` file.
---

### AutoDiff & JAX Conversion

`autodiff.py` and `jax_converter/` transform the NumPy-based Python class
produced by Stage 2 into a JAX/Equinox module.

**`AutoDiff`** handles class-level restructuring (analogous to `Transformer`
in Stage 2):

- Rewrites the class declaration to inherit from `eqx.Module`.
- Converts `np` operations and type annotations to `jnp` equivalents.
- Classifies attributes as Equinox static or dynamic fields.
- Strips `print` / `logging.*` calls (incompatible with JAX tracing) before
  conversion begins.
- Emits the output file suffixed with `_jax`, `_fwd`, or `_bwd` according
  to the `mode` parameter.
**`JaxConverter`** handles all control-flow and expression rewriting:

| NumPy construct | JAX equivalent | Strategy |
|---|---|---|
| `a[i] = v` | `a = a.at[i].set(v)` | Functional update |
| `if cond: x = a else: x = b` | `x = jnp.where(cond, a, b)` | Value select |
| `if cond: <stateful>` | `lax.cond(cond, _if_true_N, _if_false_N, ...)` | Synthetic helpers |
| `for i in range(...):` (sequential) | `lax.scan(_scan_body_N, carry, indices)` | State-carrying loop |
| `for i in range(...):` (independent) | vectorised body (loop removed) | Batch axis |
| `np.zeros / np.ones` | `jnp.zeros / jnp.ones` | Library alias |

> **Note on differentiation modes.** All three modes (`jax`, `fwd`, `bwd`)
> produce valid, XLA-compilable `eqx.Module` subclasses. Explicit
> `jax.grad` / `jax.jvp` / `jax.vjp` call sites are planned for a future
> release once the differentiation input specification interface is defined.

---

### Executive

`executive.py` validates isolated subroutines after they have been compiled
and produced binary output files. It iterates over the subroutines in the
target folder and runs them in either CPU or GPU mode to verify correctness
and performance.

---

### Shaper

`shaper.py` reconstructs accurate argument declarations for subroutines and
functions that are called from outside their defining module. It uses
`Navigator` to locate the call site, analyses each argument's shape and
usage context, and generates the corresponding Fortran declaration
statements. This is particularly useful for cross-module routines whose
dummy argument dimensions cannot be inferred from the definition alone.

---

## Installation

### Requirements

- Python 3.10+
- A Fortran compiler (e.g. `gfortran`, `nvhpc`) accessible on `PATH`
- `fparser2` for Fortran AST construction

### Using pip

```bash
git clone https://github.com/kardaneh/IPSL-Fgpt.git
cd fgpt
pip install -e .
```

### Using uv (recommended)

```bash
# Install uv (via pip or curl)
pip install uv
# or (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Clone the repository
git clone https://github.com/kardaneh/IPSL-Fgpt.git
cd fgpt

# Create virtual environment and activate
uv venv --python 3.10
source .venv/bin/activate

# Install FGPT in editable mode
uv pip install -e

# Optional: install extra dependencies
uv pip install -e ".[dev]" # development
uv pip install -e ".[notebooks]" # notebooks
uv pip install -e ".[doc]" # documentation(Sphinx)
```

---

## Usage

### Isolating and transpiling a subroutine

```python
from fgpt.isolator import Isolator

isolator = Isolator(
    rest_of_path="modipsl/modeles/ORCHIDEE/src_sechiba/",
    target_module="hydrol",
    work="/scratch/user/runs",
    f2py=True,          # also produce Python output
)

isolator.run(
    parent_subroutine="hydrol_main",
    target_subroutines=["hydrol_soil", "hydrol_alma"],
)
```

### Command-line interface

The CLI exposes two subcommands corresponding to the two stages of the pipeline.

**Stage 1 & 2 — Isolation and transpilation:**

```bash
fgpt isolate \
    --rest_of_path modipsl/modeles/ORCHIDEE/src_sechiba/ \
    --target_module hydrol \
    --work /scratch/user/runs \
    --parent_subroutine hydrol_main \
    --target_subroutines hydrol_soil hydrol_alma \
    --f2py True \
    --openacc False \
    --tapenade False
```

The key flags control which transformation path is taken:

| Flag | Default | Description |
|---|---|---|
| `--f2py` | `False` | Also transpile the isolated Fortran to NumPy Python |
| `--openacc` | `False` | Preserve OpenACC directives for GPU Fortran output |
| `--tapenade` | `False` | Prepare output for Tapenade automatic differentiation |

These three flags are mutually independent — for example, `--f2py True --openacc True` produces both a Python translation and an OpenACC-annotated Fortran output.

**Stage 3 — JAX conversion:**

```bash
fgpt autodiff \
    --config_path template.yaml \
    --class_file hydrol/hydrol_soil/global_module_hydrol_soil.py \
    --main_file hydrol/hydrol_soil/main_hydrol_soil.py \
    --mode jax
```

The `--mode` flag selects the transformation target:

| Mode | Output file suffix | Description |
|---|---|---|
| `jax` | `_jax.py` | XLA-compiled JAX module (default) |
| `fwd` | `_fwd.py` | Scaffolded for forward-mode differentiation |
| `bwd` | `_bwd.py` | Scaffolded for reverse-mode differentiation with checkpointing |

**Version and help:**

```bash
fgpt --version      # show version information
fgpt --help         # show available commands
fgpt isolate --help # show all isolate flags
fgpt autodiff --help # show all autodiff flags
```

### JAX conversion

```python
from fgpt.autodiff import AutoDiff

autodiff = AutoDiff(config_path="template.yaml", mode="jax")

autodiff.transform(
    class_file="hydrol/hydrol_soil/global_module_hydrol_soil.py",
    main_file="hydrol/hydrol_soil/main_hydrol_soil.py",
)
# produces global_module_hydrol_soil_jax.py and main_hydrol_soil_jax.py
```

---

## Testing

FGPT uses `pytest` for comprehensive testing of the transpilation pipeline,
metadata extraction, JAX conversion, and automatic differentiation workflows.

```bash
# Full test suite
pytest tests/ -v

# Specific module
pytest tests/test_f2np.py -v
pytest tests/test_transformer.py -v
pytest tests/test_autodiff.py -v

# Specific class or test
pytest tests/test_autodiff.py::TestAutoDiff -v
pytest tests/test_autodiff.py::TestAutoDiff::test_add_jax_imports -v

# Coverage report
pytest --cov=fgpt --cov-report=term-missing
```

---

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to **share** and **adapt** the material under the following
terms: **Attribution**, **NonCommercial**, and **ShareAlike**. See the
`LICENSE` file for full details.

---

## Authors

**Kazem Ardaneh**
CNRS / IPSL / Sorbonne University
kardaneh@ipsl.fr

**Shivamshan Sivanesan**
CNRS / IPSL
ssivanesan@ipsl.fr
