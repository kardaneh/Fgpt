import contextlib
import io
import pytest
import os, sys

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield
