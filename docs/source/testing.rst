Testing
=======

FGPT uses the `pytest` framework for comprehensive testing of the
Fortran-to-Python translation pipeline, metadata extraction utilities,
JAX conversion passes, and automatic differentiation workflow.

Running Tests
-------------

Run the full test suite:

.. code-block:: bash

    pytest tests -v

Run a specific test module:

.. code-block:: bash

    pytest tests/test_f2np.py -v
    pytest tests/test_transformer.py -v
    pytest tests/test_extractor.py -v
    pytest tests/test_autodiff.py -v

Run a specific test class:

.. code-block:: bash

    pytest tests/test_autodiff.py::TestAutoDiff -v

Run a single test:

.. code-block:: bash

    pytest tests/test_autodiff.py::TestAutoDiff::test_add_jax_imports -v

Generate a coverage report:

.. code-block:: bash

    pytest --cov=fgpt --cov-report=term-missing

Test Coverage
-------------

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Test Module
      - Coverage
    * - `test_f2np.py`
      - Translation of Fortran statements, expressions, loops, conditionals, and intrinsic functions
    * - `test_transformer.py`
      - End-to-end pipeline orchestration, class generation, dependency resolution, and AST insertion
    * - `test_extractor.py`
      - Extraction of declarations, array metadata, dummy arguments, and call dependencies
    * - `test_autodiff.py`
      - JAX/Equinox conversion, differentiation modes, and generated AST transformations
    * - `test_intrinsic.py`
      - Mapping and validation of Fortran intrinsic functions
    * - `test_jaxconverter.py`
      - JAX converter transformation passes and generated code correctness
    * - `test_jax_utils.py`
      - JAX utility functions and helper transformations
    * - `test_navigator.py`
      - AST traversal and navigation utilities
    * - `test_processor.py`
      - Source preprocessing and parsing functionality
    * - `test_shaper.py`
      - Array shape analysis and dimension inference
    * - `test_utils.py`
      - Utility functions and AST post-processing transformers

Testing Strategy
----------------

Tests are organized using pytest fixtures and class-based test suites.
Shared resources are initialized through fixtures and injected into test
classes using `@pytest.mark.usefixtures`.

Example test structure
----------------------

.. code-block:: python

    @pytest.fixture(scope="class")
    def test_env(request):
    autodiff = AutoDiff(
    config_path="template.yaml",
    benchmark_dir="./examples",
    logger=Logger(),
    mode="fwd"
    )
    request.cls.autodiff = autodiff
    yield

    @pytest.mark.usefixtures("test_env")
    class TestAutoDiff:

        def test_add_jax_imports(self):
            module = ast.parse("import logging\nx = 1")

            self.autodiff._add_jax_imports(module)

            import_names = [
                alias.name
                for node in module.body
                if isinstance(node, ast.Import)
                for alias in node.names
            ]

            assert "equinox" in import_names
            assert "jax.numpy" in import_names
            assert "jax" in import_names

Continuous Integration
----------------------

GitHub Actions automatically executes the test suite for:

* Pushes to the main branch.
* Pull requests.
* Manual workflow dispatches.

The continuous integration workflow ensures that all modules remain
compatible and that generated Python code continues to match the
expected Fortran semantics.
