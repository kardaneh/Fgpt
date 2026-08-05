Installation
============

Requirements
------------

- Python 3.8 or higher
- fparser (Fortran AST parsing backend)
- NumPy (for intermediate transformations)
- Optional: CUDA-capable GPU (for GPU/JAX conversion pipelines)
- Optional: JAX (for GPU-accelerated transformations)

Installation from source (recommended)
---------------------------------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/kardaneh/IPSL-Fgpt.git
      cd fgpt

2. **Create a virtual environment**:

   Using `uv` (recommended):

   .. code-block:: bash

      uv venv --python 3.8
      source .venv/bin/activate

   Or using `venv`:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate

3. **Install core dependencies**:

   .. code-block:: bash

      uv pip install -e .

   This installs the FGPT core framework including parsing, extraction,
   transformation, and utility modules.

4. **Install optional development dependencies**:

   .. code-block:: bash

      uv pip install -e ".[dev]"

   Enables testing, linting, and development tools.

5. **Install documentation dependencies (optional)**:

   .. code-block:: bash

      uv pip install -e ".[docs]"

   Required for building the Sphinx documentation.

6. **Verify installation**:

   .. code-block:: bash

      python -c "import fgpt; print(fgpt.__version__)"

   Or check CLI (if enabled):

   .. code-block:: bash

      fgpt --help

Installation with GPU / JAX support (optional)
----------------------------------------------

If you plan to use GPU acceleration or JAX-based conversion pipelines, see the official JAX installation guide:
https://docs.jax.dev/en/latest/installation.html

1. Install JAX GPU (NVIDIA, CUDA 13):

    * **GPU (NVIDIA, CUDA 13)**

    .. code-block:: bash

        pip install -U "jax[cuda13]"

    * **GPU (AMD, ROCm)**

    .. code-block:: bash

        pip install -U "jax[rocm7-local]"

    * **TPU (Google Cloud TPU VM)**

    .. code-block:: bash

        pip install -U "jax[tpu]"


2. Ensure CUDA is available:

   .. code-block:: bash

      nvidia-smi

3. Run a small test conversion pipeline to verify GPU readiness.


Building Documentation
----------------------

To build the HTML documentation locally:

1. **Install documentation dependencies**:

   .. code-block:: bash

      uv pip install -e ".[docs]"

2. **Navigate to the docs directory**:

   .. code-block:: bash

      cd docs

3. **Build the HTML documentation**:

   .. code-block:: bash

      make clean
      make html

   The output will be generated in:

   .. code-block:: text

      docs/build/html/

4. **View the documentation locally**:

   Option 1 — open directly:

   .. code-block:: bash

      firefox build/html/index.html

   Option 2 — serve via HTTP server:

   .. code-block:: bash

      python -m http.server --directory build/html 8000

   Option 3 — VS Code Preview (recommended for developers)

   If you use Visual Studio Code, you can preview the documentation directly inside the editor.

    Install: Live Preview (Microsoft extension)

    .. code-block:: bash

        https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server

    Then:

    1. Build docs:

    .. code-block:: bash

        make html

    2. Open VS Code:

    .. code-block:: bash

        code .

    3. Open:

        **docs/build/html/index.html**

    .. code-block:: text

        Right-click → “Open with Show Preview”

    This provides an in-editor rendering of the documentation with automatic refresh on rebuild.
