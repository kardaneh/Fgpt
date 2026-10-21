# Copyright 2026 CNRS
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
import ast
import os
import tempfile

import jax.numpy as jnp
import pytest

from fgpt.autodiff import AutoDiff
from fgpt.core.backends.utils import CallEdge, contains_name, convert_np_jnp, get_name
from fgpt.core.common.logger import Logger


@pytest.fixture(scope="class")
def test_env(request):
    test_dir = tempfile.mkdtemp()
    benchmark_dir = os.path.join(test_dir, "benchmark")
    autodiff = AutoDiff(
        config_path="template.yaml",
        vectorize=[],
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


CLASS_SRC = """
class Mod:
    def __init__(self):
        self.which_rt = '                    '
        self.other = 3

    def run(self, x):
        if self.which_rt == 'Matrixial':
            x = x + 1
        elif 'Iterative' == self.which_rt:
            x = x + 2
        flag = self.which_rt != 'Matrixial'
        self.which_rt = 'Matrixial'
        return x
"""


@pytest.mark.usefixtures("test_env")
class TestAutoDiff:
    def test_correct_main(self):
        code = """
def main():
    model = MyModel()
    x = np.zeros((10,))
    out = model.run(x)
    test_output(model, x)
        """

        func = _func(code)

        details = {
            "instances": {"model": "MyModel"},
            "method_calls": [
                {
                    "instance": "model",
                    "method": "run",
                    "args": ["x"],
                }
            ],
            "test_calls": [
                {
                    "test": "test_output",
                    "instance": "model",
                    "args": ["model", "x"],
                }
            ],
        }

        self.autodiff.correct_main(func, details)
        ast.fix_missing_locations(func)
        source = ast.unparse(func)

        assert "jax.config.update" in source
        assert "x_d = jnp.asarray(x)" in source
        assert "model.run(x_d)" in source
        assert "test_output(model, x_d)" in source

    def test_consumer_before_assignment(self):
        code = """
def main():
    model = MyModel()
    x = np.zeros((10,))
    out = model.run(x)
    test_output(model, x)
        """

        func = _func(code)

        details = {
            "instances": {"model": "MyModel"},
            "method_calls": [{"instance": "model", "method": "run", "args": ["x"]}],
            "test_calls": [
                {"test": "test_output", "instance": "model", "args": ["model", "x"]}
            ],
        }

        self.autodiff.correct_main(func, details)
        ast.fix_missing_locations(func)
        source = ast.unparse(func)
        assert "x_d = jnp.asarray(x)" in source
        assert source.index("x_d") > source.index("np.zeros")

    def test_multiple_calls_first_consumer_only(self):
        code = """
def main():
    model = MyModel()
    x = np.zeros((10,))
    a = model.other()
    b = model.run(x)
    c = model.run(x)
    test_output(model, x)
        """

        func = _func(code)

        details = {
            "instances": {"model": "MyModel"},
            "method_calls": [{"instance": "model", "method": "run", "args": ["x"]}],
            "test_calls": [],
        }

        self.autodiff.correct_main(func, details)
        ast.fix_missing_locations(func)
        source = ast.unparse(func)
        # This x_d needs to be present before the b
        assert source.index("x_d") > source.index("b")

    def test_multiple_input_args(self):
        code = """
def main():
    model = MyModel()
    x = np.zeros((10,))
    y = np.ones((10,))
    out = model.run(x, y)
    test_output(model, x, y)
        """

        func = _func(code)

        details = {
            "instances": {"model": "MyModel"},
            "method_calls": [
                {"instance": "model", "method": "run", "args": ["x", "y"]}
            ],
            "test_calls": [],
        }

        self.autodiff.correct_main(func, details)
        ast.fix_missing_locations(func)
        source = ast.unparse(func)

        assert "x_d" in source
        assert "y_d" in source

    def test_call_inside_assign(self):
        code = """
def main():
    model = MyModel()
    x = np.zeros((10,))
    out = model.run(x)
    wrapper = model.run(x)
    test_output(model, x)
        """

        func = _func(code)

        details = {
            "instances": {"model": "MyModel"},
            "method_calls": [{"instance": "model", "method": "run", "args": ["x"]}],
            "test_calls": [],
        }

        self.autodiff.correct_main(func, details)
        ast.fix_missing_locations(func)
        source = ast.unparse(func)

        assert "x_d = jnp.asarray(x)" in source

    def test_correct_test_func(self):
        code = """
def test_output():
    x = reader.read_reals(10)
    y = reader.read_reals(20).reshape((4,5))
        """

        func = _func(code)

        self.autodiff.correct_test_func(func)

        source = ast.unparse(func)
        assert "jnp.asarray" in source

    def test_modify_ast(self):
        code = """
def main():
    model = MyModel()
    model.run(x)
        """

        func = _func(code)

        class_modif = {
            "model": [
                "model_eqx",
                ast.ClassDef(
                    name="MyModel_eqx",
                    bases=[],
                    keywords=[],
                    body=[],
                    decorator_list=[],
                ),
            ]
        }

        self.autodiff.modify_ast(func, class_modif)

        source = ast.unparse(func)

        assert "model_eqx" in source

    def test_get_class_dep(self):
        code = """
def a():
    b()
    c()
def b():
    pass
def c():
    pass
        """

        tree = _parse(code)

        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        graph = self.autodiff.get_class_dep(funcs)

        callees = {e.callee for e in graph["a"]}

        assert callees == {"b", "c"}

    def test_propagate_shapes(self):
        code = """
def a(x):
    b(x)

def b(y):
    return y
        """

        tree = _parse(code)

        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        graph = self.autodiff.get_class_dep(funcs)
        fn_index = self.autodiff._index_functions(funcs)

        method_calls = [{"method": "a", "args_shape": {"x": ["n"]}}]
        self.autodiff.cls_info = {}
        shapes = self.autodiff._propagate_shapes(graph, fn_index, method_calls)

        assert shapes["a"] == {"x": ["n"]}
        assert shapes["b"] == {"y": ["n"]}

    def test_prepare_class(self, monkeypatch):
        code = """
class Model:

    def __init__(self):
        pass

    def run(self):
        return ns
        """

        module = _parse(code)

        monkeypatch.setattr(
            self.autodiff,
            "transform_procedure",
            lambda *args, **kwargs: None,
        )

        details = {
            "method_calls": [
                {
                    "instance": "m",
                    "method": "run",
                    "args": [],
                    "args_shape": {},
                }
            ],
            "attributes_used": set(),
        }
        self.autodiff.struct_classes = {}
        self.autodiff._prepare_class(module, details)

        source = ast.unparse(module)

        assert "eqx.Module" in source
        assert "class Model_eqx" in source

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

        ad = self.autodiff
        ad.vectorize = ["kjpindex"]
        p = ["kjpindex", "nvm", "nlevels_tot"]

        def sl(s):
            return ast.parse(s).body[0].value.slice

        assert ad._infer_subscript_shape(p, sl("x[:, ivm, 0]")) == ["kjpindex"]
        assert ad._infer_subscript_shape(p, sl("x[ji, :, :]")) == ["nvm", "nlevels_tot"]
        assert (
            ad._infer_subscript_shape(p, sl("x[ji, :, :]"), keep_loop_index=True) == p
        )
        assert ad._infer_subscript_shape(p, sl("x[0]")) == ["nvm", "nlevels_tot"]
        assert ad._infer_subscript_shape(p, sl("x[1, 2, 3, 4]")) is None  # too many
        assert ad._infer_subscript_shape([], sl("x[0]")) is None

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

    def test_fortran_reshape_to_jax(self):
        code = """
import numpy as np

def fortran_reshape(source, shape, pad=None, order=None):
    source = np.asarray(source)
    shape = tuple(shape)
    total = 1
    for s in shape:
        total *= s
    flat = source.reshape(-1, order="F")
    if flat.size < total:
        if pad is None:
            raise ValueError("SOURCE too small and no PAD given")
        pad_flat = np.asarray(pad).reshape(-1, order="F")
        needed = total - flat.size
        reps = -(-needed // pad_flat.size)
        flat = np.concatenate([flat, np.tile(pad_flat, reps)[:needed]])
    else:
        flat = flat[:total]
    if order is None:
        return flat.reshape(shape, order="F")
    order0 = [o - 1 for o in order]
    tmp_shape = tuple(shape[i] for i in order0)
    tmp = flat.reshape(tmp_shape, order="F")
    perm = sorted(range(len(order0)), key=lambda i: order0[i])
    return tmp.transpose(perm)

class Global_module_test:
    def __init__(self):
        self.a = np.zeros((6,), dtype=np.float64)
"""
        class_module = ast.parse(code)
        module_helpers = self.autodiff._split_module_level_functions(class_module)

        for i, stmt in enumerate(class_module.body):
            if isinstance(stmt, ast.FunctionDef) and stmt in module_helpers:
                if stmt.name == "fortran_reshape":
                    class_module.body[i] = convert_np_jnp(stmt)

        reshape_fn = next(
            n
            for n in class_module.body
            if isinstance(n, ast.FunctionDef) and n.name == "fortran_reshape"
        )
        src = ast.unparse(reshape_fn)

        # Must still use jnp for actual array-data operations
        assert "jnp.asarray" in src
        assert "jnp.concatenate" in src
        assert "jnp.tile" in src

        # Executable correctness check: shape/order bookkeeping stays concrete,
        # actual computation still matches the NumPy reference for the same
        # inputs used in our earlier worked example.
        namespace = {"jnp": jnp}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[reshape_fn], type_ignores=[])
                ),
                "<test>",
                "exec",
            ),
            namespace,
        )
        result = namespace["fortran_reshape"]([1, 2, 3, 4, 5, 6], (2, 3), order=[2, 1])
        assert result.tolist() == [[1, 2, 3], [4, 5, 6]]

        # Executes fine even under jit, confirming no tracer-leakage from the
        # bookkeeping arithmetic
        import jax

        jitted = jax.jit(namespace["fortran_reshape"], static_argnums=(1, 3))
        result_jit = jitted(jnp.array([1, 2, 3, 4, 5, 6]), (2, 3), None, (2, 1))
        assert result_jit.tolist() == [[1, 2, 3], [4, 5, 6]]

    def test_get_args_shape_struct_zeros_constant_and_single(self):
        self.autodiff.struct_classes = {"laieff_type"}
        tree = ast.parse(
            "a = laieff_type.zeros((kjpindex, 3))\n"
            "b = laieff_type.zeros(kjpindex)\n"
            "c = laieff_type.zeros((gm.kjpindex, nvm))\n"
        )
        shapes = self.autodiff._get_args_shape(tree, ["a", "b", "c"])
        assert shapes["a"] == ["kjpindex", 3]
        assert shapes["b"] == ["kjpindex"]
        assert shapes["c"] == ["kjpindex", "nvm"]

    def test_struct_attribute_gets_annassign(self):
        self.autodiff.struct_classes = {"laieff_type"}
        self.autodiff.str_codes = {}
        self.autodiff.cls_info = {
            "Model": {
                "attributes": {
                    "kjpindex": {"type": "int"},
                    "laieff_fit": {
                        "type": "struct",
                        "struct_type": "laieff_type",
                        "dimensions": ["kjpindex", "nvm"],
                    },
                    "soil": {"type": "jnp.ndarray", "dimensions": ["kjpindex"]},
                },
                "methods": {},
            }
        }

        [fields] = self.autodiff._define_dynamic_and_field_data([], set())
        src = [ast.unparse(f) for f in fields]

        # struct field is emitted exactly once, annotated with its type
        assert src.count("laieff_fit: laieff_type") == 1
        assert "soil: jnp.ndarray" in src
        assert any(s.startswith("kjpindex: int") for s in src)

        # the struct is not mistaken for a static eqx.field
        assert not any("laieff_fit" in s and "eqx.field" in s for s in src)

        # unknown struct type is ignored
        self.autodiff.struct_classes = set()
        [fields] = self.autodiff._define_dynamic_and_field_data([], set())
        assert not any("laieff_fit" in ast.unparse(f) for f in fields)

    def test_propagate_struct_args_element_and_field(self):
        self.autodiff.vectorize = ["kjpindex"]
        self.autodiff.struct_classes = {"laieff_type"}
        self.autodiff.cls_info = {
            "M": {
                "attributes": {},
                "methods": {
                    "outer": {
                        "args": ["laieff_fit"],
                        "local_arr": {},
                        "struct_args": {
                            "laieff_fit": {
                                "type": "laieff_type",
                                "dimensions": ["kjpindex", "nvm", "nlevels_tot"],
                            }
                        },
                    },
                    "inner": {"args": ["angle", "elem"], "local_arr": {}},
                },
            }
        }
        call = ast.parse("self.inner(a, laieff_fit[ji, ivm, ilevel])").body[0].value
        edge = CallEdge(
            caller="outer",
            callee="inner",
            call_node=call,
            arg_shapes={},
            func_args=["angle", "elem"],
        )
        self.autodiff._propagate_struct_args({"outer": [edge], "inner": []}, {})
        assert self.autodiff.cls_info["M"]["methods"]["inner"]["struct_args"][
            "elem"
        ] == {"type": "laieff_type", "dimensions": ["kjpindex"]}

    def _edge(self, caller, callee, call_src, func_args):
        call = ast.parse(call_src).body[0].value
        return CallEdge(
            caller=caller,
            callee=callee,
            call_node=call,
            arg_shapes={},
            func_args=func_args,
        )

    def _setup_struct_cls(self, vectorize=("kjpindex",)):
        ad = self.autodiff
        ad.vectorize = list(vectorize)
        ad.struct_classes = {"laieff_type"}
        ad.cls_info = {
            "M": {
                "attributes": {},
                "methods": {
                    "outer": {
                        "args": ["laieff_fit"],
                        "local_arr": {},
                        "struct_args": {
                            "laieff_fit": {
                                "type": "laieff_type",
                                "dimensions": ["kjpindex", "nvm", "nlevels_tot"],
                            }
                        },
                    },
                    "mid": {"args": ["angle", "elem"], "local_arr": {}},
                    "inner": {"args": ["angle", "elem"], "local_arr": {}},
                },
            }
        }
        return ad

    def test_propagate_without_vectorization_gives_scalar_element(self):
        ad = self._setup_struct_cls(vectorize=())
        e = self._edge(
            "outer",
            "inner",
            "self.inner(a, laieff_fit[ji, ivm, ilevel])",
            ["angle", "elem"],
        )
        ad._propagate_struct_args({"outer": [e], "inner": []}, {})
        assert (
            ad.cls_info["M"]["methods"]["inner"]["struct_args"]["elem"]["dimensions"]
            == []
        )

    def test_propagate_slice_keeps_axis_in_both_modes(self):
        for vec in [("kjpindex",), ()]:
            ad = self._setup_struct_cls(vectorize=vec)
            e = self._edge(
                "outer",
                "inner",
                "self.inner(a, laieff_fit[:, ivm, ilevel])",
                ["angle", "elem"],
            )
            ad._propagate_struct_args({"outer": [e], "inner": []}, {})
            assert ad.cls_info["M"]["methods"]["inner"]["struct_args"]["elem"][
                "dimensions"
            ] == ["kjpindex"]

    def test_propagate_whole_struct_unchanged(self):
        ad = self._setup_struct_cls()
        e = self._edge("outer", "inner", "self.inner(a, laieff_fit)", ["angle", "elem"])
        ad._propagate_struct_args({"outer": [e], "inner": []}, {})
        assert ad.cls_info["M"]["methods"]["inner"]["struct_args"]["elem"][
            "dimensions"
        ] == ["kjpindex", "nvm", "nlevels_tot"]

    def test_propagate_partial_index_keeps_trailing_dims(self):
        ad = self._setup_struct_cls()
        e = self._edge(
            "outer", "inner", "self.inner(a, laieff_fit[ji, ivm])", ["angle", "elem"]
        )
        ad._propagate_struct_args({"outer": [e], "inner": []}, {})
        assert ad.cls_info["M"]["methods"]["inner"]["struct_args"]["elem"][
            "dimensions"
        ] == ["kjpindex", "nlevels_tot"]

    def test_propagate_conflict_raises(self):
        ad = self._setup_struct_cls()
        e1 = self._edge(
            "outer",
            "inner",
            "self.inner(a, laieff_fit[ji, ivm, ilevel])",
            ["angle", "elem"],
        )
        e2 = self._edge(
            "outer", "inner", "self.inner(a, laieff_fit)", ["angle", "elem"]
        )
        with pytest.raises(ValueError, match="Conflicting struct_args"):
            ad._propagate_struct_args({"outer": [e1, e2], "inner": []}, {})

    def test_propagate_non_struct_arg_ignored(self):
        ad = self._setup_struct_cls()
        e = self._edge("outer", "inner", "self.inner(a, other[ji])", ["angle", "elem"])
        ad._propagate_struct_args({"outer": [e], "inner": []}, {})
        assert "struct_args" not in ad.cls_info["M"]["methods"]["inner"]

    def _child(self, call_src, arg_shapes):
        ad = self.autodiff
        caller = ast.parse(
            "def f(self, a, laieff_fit, snowrho, rhosnew):\n    pass"
        ).body[0]
        callee = ast.parse("def g(self, p, q):\n    pass").body[0]
        call = ast.parse(call_src).body[0].value
        return ad._infer_child_args(call, arg_shapes, caller, callee)

    def test_child_args_element_gets_no_shape(self):
        self._setup_struct_cls()
        s = {"a": ["kjpindex"], "laieff_fit": ["kjpindex", "nvm", "nlevels_tot"]}
        arg_shapes = self._child("self.g(a, laieff_fit[ji, ivm, ilevel])", s)
        assert "q" in arg_shapes
        assert arg_shapes.get("q") == ["kjpindex", "nvm", "nlevels_tot"]

    def test_child_args_slice_subscript_keeps_passthrough(self):
        self._setup_struct_cls()
        s = {"snowrho": ["kjpindex", "nsnow"]}
        assert self._child("self.g(a, snowrho[ji, :])", s)["q"] == ["kjpindex", "nsnow"]

    def test_child_args_1d_subscript_keeps_passthrough(self):
        self._setup_struct_cls()
        s = {"rhosnew": ["kjpindex"]}
        assert self._child("self.g(a, rhosnew[ji])", s)["q"] == ["kjpindex"]

    def test_child_args_field_access(self):
        ad = self._setup_struct_cls()
        ad.struct_vars = {
            "laieff_fit": {
                "type": "laieff_type",
                "dimensions": ["kjpindex", "nvm", "nlevels_tot"],
            }
        }
        ad.struct_info = {
            "laieff_type": {
                "a": {"inner_dims": [], "dtype": "float64"},
                "b": {"inner_dims": [10], "dtype": "float64"},
            }
        }
        s = {}
        assert self._child("self.g(a, laieff_fit.a[:, ivm, ilevel])", s)["q"] == [
            "kjpindex"
        ]
        assert self._child("self.g(a, laieff_fit.b[:, ivm, ilevel])", s)["q"] == [
            "kjpindex",
            10,
        ]
        assert self._child("self.g(a, laieff_fit.a[ji, ivm, ilevel])", s).get("q") == []

    def _encode(self, src=CLASS_SRC):
        self.autodiff.struct_classes = set()
        mod = _parse(src)
        codes = self.autodiff._encode_string_attrs(mod)
        return codes, ast.unparse(mod)

    def test_encode_assign_and_init(self):
        codes, out = self._encode()
        c = codes["which_rt"]
        assert c[0] == ""  # blank-padded init maps to code 0
        assert "self.which_rt = jnp.int32(0)" in out
        assert f"self.which_rt = jnp.int32({c.index('Matrixial')})" in out

    def test_encode_if_test_compare(self):
        codes, out = self._encode()
        c = codes["which_rt"]
        assert f"self.which_rt == {c.index('Matrixial')}" in out
        assert "'Matrixial'" not in out.replace("STR_CODES", "")

    def test_encode_reversed_operand_and_elif(self):
        codes, out = self._encode()
        c = codes["which_rt"]
        assert f"{c.index('Iterative')} == self.which_rt" in out

    def test_encode_compare_inside_assign(self):
        codes, out = self._encode()
        c = codes["which_rt"]
        assert f"flag = self.which_rt != {c.index('Matrixial')}" in out

    def test_compare_only_strings_are_registered(self):
        codes, _ = self._encode()
        assert "Iterative" in codes["which_rt"]  # never assigned, only compared

    def test_codebook_unique_and_blank_first(self):
        codes, _ = self._encode()
        c = codes["which_rt"]
        assert c[0] == "" and len(c) == len(set(c))

    def test_non_string_attrs_untouched(self):
        codes, out = self._encode()
        assert "other" not in codes
        assert "self.other = 3" in out

    def test_no_string_attrs_returns_empty(self):
        codes, out = self._encode(
            "class A:\n    def __init__(self):\n        self.n = 1\n"
        )
        assert codes == {}
        assert "self.n = 1" in out

    def test_declaration_init_uses_encode_str(self):
        fn = _func(
            "def declaration_initialization(self):\n"
            "    ffile = FortranFile(path, 'r')\n"
            "    self.which_rt = ffile.read_record('S80')[0].decode('ascii').rstrip()\n"
        )
        self.autodiff._add_jax_for_file_reading(fn)
        out = ast.unparse(ast.fix_missing_locations(fn))
        assert "_encode_str('which_rt'" in out
        assert "'which_rt': which_rt" in out

    def test_test_func_decodes_codes(self):
        self.autodiff.str_codes = {"which_rt": ["", "Matrixial"]}
        fn = _func("def test_x(gmasme, a):\n    vals = [a, gmasme.which_rt]\n")
        self.autodiff.correct_test_func(fn)
        out = ast.unparse(ast.fix_missing_locations(fn))
        assert "STR_CODES['which_rt'][int(gmasme.which_rt)]" in out
        self.autodiff.str_codes = {}
