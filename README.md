# FGPT


[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-green.svg)](./docs)
[![CI](https://github.com/kardaneh/FGPT/actions/workflows/ci.yaml/badge.svg)](https://github.com/kardaneh/FGPT/actions/workflows/ci.yaml)

**FGPT** is a source-to-source transpiler that converts production scientific
Fortran code into executable NumPy-based Python, and optionally into
JAX/Equinox-compatible modules for GPU-accelerated and differentiable
computation. The current state of the project is a proof of concept, which has been tested on large modules of the IPSL land surface model.
The project has received support from the AI4PEX project.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Pipeline Overview](#pipeline-overview)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Development](#development)
8. [License](#license)
9. [Authors](#authors)
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
   ├── src/
   │   └── fgpt/
   │       ├── __init__.py
   │       ├── __main__.py
   │       ├── cli.py                     # Command-line interface
   │       ├── version.py                 # Package version
   │       ├── isolator.py                # Fortran isolation pipeline
   │       ├── autodiff.py                # JAX/Tapenade conversion pipeline
   │       │
   │       ├── core/
   │       │   ├── frontend/
   |       |   |   ├── __init__.py
   │       │   │   ├── processor.py       # Fortran parser (fparser wrapper)
   │       │   │   ├── extractor.py       # Static analysis and metadata extraction
   │       │   │   └── navigator.py       # Cross-module symbol resolution
   │       │   │
   │       │   ├── analysis/
   |       |   |   ├── __init__.py
   │       │   │   └── shaper.py          # Array shape/dimension analysis
   │       │   │
   │       │   ├── passes/
   |       |   |   ├── __init__.py
   │       │   │   └── modifier.py        # Fortran AST transformation passes
   │       │   │
   │       │   ├── lowering/
   |       |   |   ├── __init__.py
   │       │   │   ├── transformer.py     # Fortran → Python pipeline
   │       │   │   ├── f2np.py            # Statement/expression-level translation
   │       │   │   └── intrinsic.py       # Fortran intrinsic → NumPy mapping
   │       │   │
   │       │   ├── backends/
   |       |   |   ├── __init__.py
   |       |   |   ├── utils.py                  # Shared helper functions used across backend modules
   |       |   |   └── jax_converter/
   |       |   |       ├── converter.py          # Main entry point: orchestrates conversion of code into JAX representations
   |       |   |       ├── analysis.py           # Static/dynamic analysis utilities (shape inference, dependency tracking, etc.)
   |       |   |       ├── array_updates.py      # Handles array mutation patterns and converts them to JAX-safe updates
   |       |   |       ├── call_rewriting.py     # Rewrites function calls into JAX-compatible primitives or transformations
   |       |   |       ├── conditionals.py       # Transforms if/else logic into JAX control-flow primitives (e.g., lax.cond)
   |       |   |       ├── dynamic_loops.py      # Deals with loops whose bounds depend on runtime values (dynamic control flow)
   |       |   |       ├── loops.py              # Handles static/structured loop transformations
   |       |   |       ├── masking.py            # Implements masking strategies for conditional execution without branching
   |       |   |       ├── scope_utils.py        # Utilities for managing variable scope during transformation/rewrite passes
   |       |   |       └── vectorization.py      # Converts scalar functions into vectorized versions
   │       │   └── common/
   |       |       ├── __init__.py
   │       │       ├── executive.py       # Workflow orchestration
   │       │       ├── logger.py          # Logging infrastructure
   │       │       ├── line_length.py     # Fortran line-length utilities
   │       │       └── utils.py           # Shared helper utilities
   │       │
   │       └── templates/
   │           └── default.yaml
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
   |
   ├── notebooks/                         # Example notebooks, tutorials, and development prototypes
   │   ├── autodiff_principles.ipynb      # Introduction to JVP and VJP concepts
   │   ├── prototype.ipynb                # Experimental notebook with autodifferenciation
   │   ├── fortran_to_numpy.ipynb         # F2NP translation examples
   │   ├── jax_converter.ipynb            # JAX conversion pipeline examples
   │   └── jax_examples.ipynb             # JAX experiments and demonstrations
   |
   ├── docs/                     # Documentation
   │   ├── source/
   │   └── build/
   ├── .github/workflows/        # CI/CD pipelines
   │   └── ci.yaml
   |
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
│       └── Extractor ◄─┘                 │
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

## Installation

### Requirements

- Python 3.10+
- A Fortran compiler (e.g. `gfortran`, `nvhpc`) accessible on `PATH`
- `fparser2` for Fortran AST construction

### Using pip

```bash
git clone https://github.com/kardaneh/IPSL-FGPT.git
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
git clone https://github.com/kardaneh/IPSL-FGPT.git
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

**Stage 1 - 3 — Isolation, transpilation and JAX conversion**

```bash
fgpt isolate \
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
```

The key flags control which transformation path is taken:

| Flag | Default | Description |
|---|---|---|
| `--f2py` | `False` | Also transpile the isolated Fortran to NumPy Python |
| `--openacc` | `False` | Preserve OpenACC directives for GPU Fortran output |
| `--tapenade` | `False` | Prepare output for Tapenade automatic differentiation |
| `--py2jx` | `False` | Prepare output for JAX transformation and optimization |

These three flags are mutually independent, except that `py2jx` requires `f2py` to be enabled. For example, `--f2py True --openacc True` produces both a Python translation and an OpenACC-annotated Fortran output.

**Stage 3 — JAX conversion:**

```bash
fgpt autodiff \
    --config_path template.yaml \
    --class_file hydrol/hydrol_soil/global_module_hydrol_soil.py \
    --main_file hydrol/hydrol_soil/main_hydrol_soil.py \
    --vectorize kjpindex
    --mode jax
```

The `--mode` flag selects the transformation target:

| Mode | Output file suffix | Description |
|---|---|---|
| `jax` | `_jax.py` | XLA-compiled JAX module (default) |
| `fwd` | `_d.py` | Scaffolded for forward-mode differentiation |
| `bwd` | `_d.py` | Scaffolded for reverse-mode differentiation with checkpointing |

The `--vectorize` option specifies the lower-bound loops that the user wants to vectorize.
By default it's set to `["kjpindex"]`

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

The `isolate` command can perform the complete pipeline, including the JAX conversion. Alternatively, it can be used to execute only stages 1 and 2, with the `autodiff` command handling the final stage.

## Notebooks

The repository includes several example notebooks, such as `Test_F2NP.ipynb` and `Test_JAX_Converter.ipynb`, which demonstrate different features and workflows.

Before using the notebooks, complete the steps described in the [Installation](#Installation) section. Then activate the virtual environment and register it as a Jupyter kernel:

```bash
source .venv/bin/activate
uv run ipython kernel install --user \
    --env VIRTUAL_ENV "$(pwd)/.venv" \
    --name=project
```

Once the kernel has been installed, you can launch JupyterLab with:

```bash
uv run --with jupyter jupyter lab
```

Alternatively, you can open the notebooks directly in Visual Studio Code. VS Code will automatically detect the project's `.venv`. Simply select the `project` kernel (or the corresponding virtual environment) when prompted.

---

## Testing

FGPT uses `pytest` for comprehensive testing of the transpilation pipeline,
metadata extraction, JAX conversion, and the futur automatic differentiation workflows.

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

## Citation

If you use FGPT in your research, please cite the software.

### BibTeX

```bibtex
@software{ardaneh_fgpt_2026,
  author = {Kazem Ardaneh and Shivamshan Sivanesan},
  title = {FGPT: A Fortran-to-Python and JAX Transpiler for Scientific Codes},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/kardaneh/IPSL-FGPT}
}
```
