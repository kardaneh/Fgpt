Contributing
============

Contributions to FGPT are welcome. Whether you are improving the
Fortran-to-Python translation pipeline, extending JAX support, fixing
bugs, improving documentation, or adding tests, please follow the
guidelines below before submitting changes.

Development Workflow
--------------------

1. Create a feature branch from the latest main branch.

.. code-block:: bash

    git checkout main
    git pull origin main
    git checkout -b feature/my-feature

2. Install development dependencies.

.. code-block:: bash

    uv sync --extra dev

3. Make your changes and add or update tests as needed.

Code Quality Checks
-------------------

Before committing, run the project's pre-commit hooks:

.. code-block:: bash

    pre-commit run --all-files

These checks enforce formatting, linting, import ordering, and other
repository standards.

Running Tests
-------------

Run the complete test suite:

.. code-block:: bash

    pytest tests -v

Generate a coverage report:

.. code-block:: bash

    pytest --cov=fgpt --cov-report=term-missing

All tests must pass before a contribution is submitted.

Documentation
-------------

If your changes affect user-facing functionality, please update the
relevant documentation in `docs/source`.

In particular, update:

* API documentation for new public classes or methods.
* Usage examples for new features.
* Developer documentation when modifying internal workflows.

Commit Messages
---------------

FGPT follows the Conventional Commits style:

.. code-block:: text

    feat: add support for WHERE statement translation
    fix: correct array index adjustment in loops
    docs: update transformer pipeline documentation
    test: add coverage for intrinsic function handling
    refactor: simplify dependency resolution logic

Submitting Changes
------------------

Before opening a pull request, ensure that:

* All tests pass.
* Pre-commit checks succeed.
* Documentation has been updated when necessary.
* New functionality includes appropriate test coverage.

Then push your branch and open a pull request against the main branch.

.. code-block:: bash

    git push origin feature/my-feature

The pull request description should clearly explain:

* The motivation for the change.
* The implementation approach.
* Any impacts on existing functionality.
* Relevant tests or benchmarks.

Reporting Issues
----------------

Bug reports, feature requests, and documentation improvements are all
welcome. When reporting an issue, please include:

* A minimal reproducible example.
* The FGPT version.
* Python version and operating system.
* Relevant error messages or stack traces.
