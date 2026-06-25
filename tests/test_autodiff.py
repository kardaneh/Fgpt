import ast
import os
import tempfile

import pytest

from fgpt.autodiff import AutoDiff
from fgpt.jax_utils import contains_name, get_name
from fgpt.logger import Logger


@pytest.fixture(scope="class")
def test_env(request):
    test_dir = tempfile.mkdtemp()
    benchmark_dir = os.path.join(test_dir, "benchmark")
    autodiff = AutoDiff(
        config_path="template.yaml",
        benchmark_dir=benchmark_dir,
        logger=Logger(),
        mode="fwd",  # <- has three modes, jax, fwd, bwd
    )
    request.cls.autodiff = autodiff
    yield


def _parse(code: str) -> ast.Module:
    """Parse a code string into an ``ast.Module``."""
    return ast.parse(code)


def _func(code: str) -> ast.FunctionDef:
    """Parse a code string and return its first ``FunctionDef``."""
    return _parse(code).body[0]


@pytest.mark.usefixtures("test_env")
class TestAutoDiff:
    def test_add_jax_imports(self):
        # equinox, jax.numpy and jax imports are prepended
        module = _parse("import logging\nx = 1")
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

        # Named imports are stripped from the module body
        module = _parse("import logging\nimport time\nx = 1")
        self.autodiff._add_jax_imports(module, import_remove=["logging", "time"])
        remaining = [
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert "logging" not in remaining
        assert "time" not in remaining

        # Imports not in *import_remove* are kept
        module = _parse("import os\nimport logging")
        self.autodiff._add_jax_imports(module, import_remove=["logging"])
        remaining = [
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert "os" in remaining

    def test_get_args_shape(self):
        # Tuple shape from ``np.zeros`` is resolved correctly
        code = "x = np.zeros((kjpindex, nnobio))"
        tree = _func(f"def main():\n    {code}")
        result = self.autodiff._get_args_shape(tree, ["x"])
        assert result == {"x": ["kjpindex", "nnobio"]}

        # Plain scalar assignments produce no shape entry
        code = "x = 42"
        tree = _func(f"def main():\n    {code}")
        result = self.autodiff._get_args_shape(tree, ["x"])
        assert result == {}

        # Args not present in the tree return an empty dict
        code = "y = np.zeros((10, 20))"
        tree = _func(f"def main():\n    {code}")
        result = self.autodiff._get_args_shape(tree, ["x"])
        assert result == {}

        # ``jnp.zeros`` is also recognised as an array constructor
        code = "x = jnp.zeros((a, b))"
        tree = _func(f"def main():\n    {code}")
        result = self.autodiff._get_args_shape(tree, ["x"])
        assert result == {"x": ["a", "b"]}

    def test_extract_args(self):
        # Plain ``ast.Name`` arguments are extracted by id
        call = ast.parse("f(x, y)").body[0].value
        result = self.autodiff._extract_args(call)
        assert result == ["x", "y"]

        # ``ast.Attribute`` arguments are extracted by attr name
        call = ast.parse("f(self.x)").body[0].value
        result = self.autodiff._extract_args(call)
        assert result == ["x"]

        # Complex expressions fall back to ``ast.dump``
        call = ast.parse("f(a + b)").body[0].value
        result = self.autodiff._extract_args(call)
        assert len(result) == 1
        assert "BinOp" in result[0]

        # A call with no arguments returns an empty list
        call = ast.parse("f()").body[0].value
        result = self.autodiff._extract_args(call)
        assert result == []

    def test_index_functions(self):
        # All functions are indexed by name
        code = "def foo(): pass\ndef bar(): pass"
        functions = [
            n for n in ast.walk(_parse(code)) if isinstance(n, ast.FunctionDef)
        ]
        result = self.autodiff._index_functions(functions)
        assert set(result.keys()) == {"foo", "bar"}
        assert isinstance(result["foo"], ast.FunctionDef)

        # An empty list returns an empty dict
        result = self.autodiff._index_functions([])
        assert result == {}

    def test_get_name(self):
        node = ast.parse("self.x").body[0].value
        assert get_name(node) == "x"

        node = ast.parse("x").body[0].value
        assert get_name(node) == "x"

        node = ast.parse("x[0]").body[0].value
        assert get_name(node) == "x"

        # Unknown returns None
        node = ast.parse("1 + 2").body[0].value
        assert get_name(node) is None

    def test_extract_single_dim(self):
        node = ast.parse("self.kjpindex").body[0].value
        assert self.autodiff._extract_single_dim(node) == "kjpindex"

        node = ast.parse("nnobio").body[0].value
        assert self.autodiff._extract_single_dim(node) == "nnobio"

        node = ast.parse("3").body[0].value
        assert self.autodiff._extract_single_dim(node) == 3

        # Unknown returns None
        node = ast.parse("a + b").body[0].value
        assert self.autodiff._extract_single_dim(node) is None

    def test_extract_shape_arg(self):
        # Tuple
        node = ast.parse("(a, b, 3)").body[0].value
        result = self.autodiff._extract_shape_arg(node)
        assert result == ["a", "b", 3]

        # Single
        node = ast.parse("128").body[0].value
        result = self.autodiff._extract_shape_arg(node)
        assert result == [128]

        # List
        node = ast.parse("[a, b]").body[0].value
        result = self.autodiff._extract_shape_arg(node)
        assert result == ["a", "b"]

    def test_resolve_inline_shape(self):
        # np.zeros
        node = ast.parse("np.zeros((a, b))").body[0].value
        result = self.autodiff._resolve_inline_shape(node)
        assert result == ["a", "b"]

        node = ast.parse("x").body[0].value
        assert self.autodiff._resolve_inline_shape(node) is None

        node = ast.parse("np.sum(x)").body[0].value
        assert self.autodiff._resolve_inline_shape(node) is None

        node = ast.parse("np.zeros()").body[0].value
        assert self.autodiff._resolve_inline_shape(node) is None

    def test_infer_subscript_shape(self):
        parent = ["a", "b", "c"]
        slice_node = ast.Constant(value=0)
        result = self.autodiff._infer_subscript_shape(parent, slice_node)
        assert result == ["b", "c"]

        parent = ["a", "b"]
        slice_node = ast.Slice()
        result = self.autodiff._infer_subscript_shape(parent, slice_node)
        assert result == ["a", "b"]

        result = self.autodiff._infer_subscript_shape([], ast.Constant(value=0))
        assert result is None

        result = self.autodiff._infer_subscript_shape(
            "not_a_list", ast.Constant(value=0)
        )
        assert result is None

    def test_check_dimension_static(self):
        """Missing attribute defaults to static."""
        result = self.autodiff._check_dimension_static({}, "missing")
        assert result is True

        attrs = {"x": {"type": "int"}}
        assert self.autodiff._check_dimension_static(attrs, "x") is True

        attrs = {"x": {"dep_value": "y"}, "y": {"type": "int"}}
        assert self.autodiff._check_dimension_static(attrs, "x") is False

        """dep_value referencing an unknown attribute is still static."""
        attrs = {"x": {"dep_value": "unknown"}}
        assert self.autodiff._check_dimension_static(attrs, "x") is True

    def test_get_read_call(self):
        node = ast.parse("f.read_ints(n)").body[0].value
        call, dtype = self.autodiff._get_read_call(node)
        assert call is node
        assert dtype == "int32"

        node = ast.parse("f.read_reals(n)").body[0].value
        call, dtype = self.autodiff._get_read_call(node)
        assert call is node
        assert dtype == "float64"

        node = ast.parse("f.read_ints(n)[0]").body[0].value
        call, dtype = self.autodiff._get_read_call(node)
        assert call is node
        assert dtype == "int32"

        node = ast.parse("f.compute(n)").body[0].value
        call, dtype = self.autodiff._get_read_call(node)
        assert call is None
        assert dtype is None

    def test_check_read_rhs_assign_value(self):
        # Direct read
        node = ast.parse("f.read_reals(n)").body[0].value
        assert self.autodiff._check_read_rhs_assign_value(node) is True

        # Nested read
        node = ast.parse("jnp.float64(f.read_reals(n))").body[0].value
        assert self.autodiff._check_read_rhs_assign_value(node) is True

        # Plain call
        node = ast.parse("np.zeros((10,))").body[0].value
        assert self.autodiff._check_read_rhs_assign_value(node) is False

    def test_correct_name(self):
        node = ast.parse("self.kjpindex").body[0].value
        result = self.autodiff._correct_name(node, ["kjpindex"])
        assert isinstance(result, ast.Name)
        assert result.id == "kjpindex"

        node = ast.parse("self.other").body[0].value
        result = self.autodiff._correct_name(node, ["kjpindex"])
        assert isinstance(result, ast.Attribute)

        node = ast.parse("self.a + self.b").body[0].value
        result = self.autodiff._correct_name(node, ["a", "b"])
        assert isinstance(result.left, ast.Name)
        assert isinstance(result.right, ast.Name)

    def test_correct_tuple_shape(self):
        node = ast.parse("(self.a, self.b)").body[0].value
        result = self.autodiff._correct_tuple_shape(node, ["a", "b"])
        assert all(isinstance(e, ast.Name) for e in result.elts)
        assert [e.id for e in result.elts] == ["a", "b"]

        node = ast.parse("(self.a, self.c)").body[0].value
        result = self.autodiff._correct_tuple_shape(node, ["a"])
        assert isinstance(result.elts[0], ast.Name)
        assert isinstance(result.elts[1], ast.Attribute)

    def test_extract_main_context(self):
        """Class instantiation is captured in ``instances``."""
        code = "def main():\n    model = MyModel()\n    model.run(x)"
        func = _func(code)
        result = self.autodiff.extract_main_context(func)
        assert "model" in result["instances"]
        assert result["instances"]["model"] == "MyModel"

        # Method calls on tracked instances appear in ``method_calls``
        code = "def main():\n    model = MyModel()\n    model.run(x)"
        func = _func(code)
        result = self.autodiff.extract_main_context(func)
        methods = [mc["method"] for mc in result["method_calls"]]
        assert "run" in methods

        # ``test_*`` functions are captured in ``test_calls``
        code = "def main():\n    model = MyModel()\n    test_output(model, x)"
        func = _func(code)
        result = self.autodiff.extract_main_context(func)
        tests = [tc["test"] for tc in result["test_calls"]]
        assert "test_output" in tests

        # Attribute names used as np.zeros dimensions are collected
        code = (
            "def main():\n"
            "    model = MyModel()\n"
            "    x = np.zeros((model.kjpindex, model.nnobio))\n"
        )
        func = _func(code)
        result = self.autodiff.extract_main_context(func)
        assert "kjpindex" in result["attributes_used"]
        assert "nnobio" in result["attributes_used"]

        # The same method call is not recorded twice
        code = (
            "def main():\n"
            "    model = MyModel()\n"
            "    result = model.run(x)\n"
            "    model.run(x)\n"
        )
        func = _func(code)
        result = self.autodiff.extract_main_context(func)
        run_calls = [mc for mc in result["method_calls"] if mc["method"] == "run"]
        assert len(run_calls) == 1

    def test_contains_name(self):
        # Name found
        node = ast.parse("x = y + z").body[0]
        assert contains_name(node, "y") is True

        # Name not found
        node = ast.parse("x = 1").body[0]
        assert contains_name(node, "missing") is False

    def test_create_call_statement(self):
        # Tuple return produces an unpacking assignment
        code = "def run(self, a, b):\n    return (a, b, self)\n"
        func = _func(code)
        result = self.autodiff._create_call_statement(func, "model_eqx")
        assert isinstance(result, ast.Assign)
        assert isinstance(result.targets[0], ast.Tuple)

        # Name return produces a single ``_d``-suffixed assignment
        code = "def run(self, a):\n    return out\n"
        func = _func(code)
        result = self.autodiff._create_call_statement(func, "model_eqx")
        assert isinstance(result, ast.Assign)
        assert result.targets[0].id == "out_d"

        # Call return assigns the result to the instance name
        code = "def run(self, a):\n    return other(a)\n"
        func = _func(code)
        result = self.autodiff._create_call_statement(func, "model_eqx")
        assert isinstance(result, ast.Assign)
        assert result.targets[0].id == "model_eqx"

        # An unrecognised return type raises ``NotImplementedError``
        code = "def run(self):\n    return 42\n"
        func = _func(code)
        with pytest.raises(NotImplementedError):
            self.autodiff._create_call_statement(func, "model_eqx")

    def test_write_to_file(self, tmp_path):
        # Output file is created and contains the shebang line
        out = tmp_path / "out.py"
        tree = ast.parse("x = 1")
        self.autodiff.write_to_file(str(out), tree)
        assert out.exists()
        content = out.read_text()
        assert content.startswith("#!/usr/bin/env python3")

        # Output file has owner execute permission set
        import stat

        out = tmp_path / "out.py"
        tree = ast.parse("x = 1")
        self.autodiff.write_to_file(str(out), tree)
        mode = os.stat(out).st_mode
        assert mode & stat.S_IXUSR

        # Unparsed AST content is present in the written file
        out = tmp_path / "out.py"
        tree = ast.parse("my_variable = 99")
        self.autodiff.write_to_file(str(out), tree)
        assert "my_variable = 99" in out.read_text()
