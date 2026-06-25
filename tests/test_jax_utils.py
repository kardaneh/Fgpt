import ast

import pytest

from fgpt.jax_utils import (
    MaybeAddIndexTransformer,
    ReductionHandler,
    VectorizationAnalyzer,
    WhileVectorToScalar,
)


def _make_cls_info():
    """Minimal but representative cls_info for a single class `Model`."""
    return {
        "Model": {
            "attributes": {
                "kjpindex": {"type": "int"},
                "nnobio": {"type": "int"},
                "soil_temp": {
                    "type": "jnp.ndarray",
                    "dimensions": ["kjpindex"],
                    "dtype": "float64",
                },
                "soil_moist": {
                    "type": "jnp.ndarray",
                    "dimensions": ["kjpindex", "nnobio"],
                    "dtype": "float64",
                },
                "mask": {
                    "type": "jnp.ndarray",
                    "dimensions": ["kjpindex"],
                    "dtype": "bool",
                },
            },
            "methods": {
                "compute": {
                    "args": ["self", "x"],
                    "local_arr": {
                        "tmp": {
                            "dimensions": ["kjpindex"],
                            "dtype": "float64",
                            "type": "jnp.ndarray",
                        },
                    },
                },
                "helper": {
                    "args": ["self", "y"],
                    "local_arr": {},
                },
            },
        }
    }


@pytest.fixture(scope="class")
def test_env(request):
    analyzer = VectorizationAnalyzer()
    request.cls.analyzer = analyzer

    indextransformer = MaybeAddIndexTransformer(
        cls_info=_make_cls_info(),
        cls_name="Model",
        func_name="compute",
        ranks={},
        target_rank=2,
        vect_context={
            "loop_info": {"kjpindex": "i"},
            "vectorization_axis": {"i": [0]},
        },
        local_defined_variables={},
        func_input_dim={},
        dynamic_variable_lift={},
        inferred_ranks={},
    )
    request.cls.indexer = indextransformer

    whiletransformer = WhileVectorToScalar()
    request.cls.whiletransformer = whiletransformer

    reduction_handler = ReductionHandler(
        cls_info=_make_cls_info(),
        cls_name="Model",
        func_name="compute",
        vectorization_axis={"i": {0}},
    )

    reduction_handler.func_input_dim = {}
    reduction_handler.dynamic_variable_lift = {}

    request.cls.reducer = reduction_handler

    yield


def _parse(code: str) -> ast.Module:
    """Parse a code string into an ``ast.Module``."""
    return ast.parse(code)


def _func(code: str) -> ast.FunctionDef:
    """Parse a code string and return its first ``FunctionDef``."""
    return _parse(code).body[0]


def _expr(code: str) -> ast.AST:
    """Parse a code string and return its first expression value."""
    return _parse(code).body[0].value


def _stmt(code: str) -> ast.stmt:
    """Parse a code string and return its first statement."""
    return _parse(code).body[0]


@pytest.mark.usefixtures("test_env")
class TestVectorizationAnalyzer:
    def test_collect_loop_vars_from_for(self):
        node = _stmt("for i in range(0, self.kjpindex):\n    pass")
        assert self.analyzer._collect_loop_vars_from_for(node) == {"i"}

        node = _stmt("for i in range(0, kjpindex):\n    pass")
        assert self.analyzer._collect_loop_vars_from_for(node) == {"i"}

        # Non vector loop
        node = _stmt("for i in range(0, nslm):\n    pass")
        assert self.analyzer._collect_loop_vars_from_for(node) == set()

        # Tuple targets
        node = _stmt("for i, j in range(0, self.kjpindex):\n    pass")
        assert self.analyzer._collect_loop_vars_from_for(node) == {"i", "j"}

    def test_name_used(self):
        expr = _expr("a + i")
        assert self.analyzer._name_used(expr, "i")

        expr = _expr("a + b")
        assert not self.analyzer._name_used(expr, "i")

    def test_index_uses_loop_vars(self):
        assert self.analyzer._index_uses_loop_vars(
            ast.Name(id="i"),
            {"i"},
        )

        expr = _expr("i + 1")
        assert self.analyzer._index_uses_loop_vars(
            expr,
            {"i"},
        )

        expr = _expr("f(i, j)")
        assert self.analyzer._index_uses_loop_vars(
            expr,
            {"i"},
        )

        expr = _expr("j + 1")
        assert not self.analyzer._index_uses_loop_vars(
            expr,
            {"i"},
        )

    def test_subscript_uses_loop_vars(self):
        expr = _expr("a[i]")
        assert self.analyzer._subscript_uses_loop_vars(
            expr,
            {"i"},
        )

        expr = _expr("a[j]")
        assert not self.analyzer._subscript_uses_loop_vars(
            expr,
            {"i"},
        )

    def test_static_expr(self):
        assert self.analyzer._is_static_expr(ast.Constant(1))
        assert self.analyzer._is_static_expr(ast.Name(id="x"))

        expr = _expr("x + y")
        assert self.analyzer._is_static_expr(expr)

    def test_check_for_while(self):
        node = _stmt("for i in range(10):\n    while x:\n        pass")
        assert self.analyzer.check_for_while(node)

        node = _stmt("for i in range(10):\n    pass")
        assert not self.analyzer.check_for_while(node)

    def test_classify_for(self):
        # TODO: IN the case we don't send only have
        # Vector
        node = _stmt("for i in range(0, self.kjpindex):\n    a[i] = 1")

        assert self.analyzer.classify_for(node) == "vector"
        # Non vector loop
        node = _stmt("for i in range(0, nslm):\n    a[i] = 1")
        assert self.analyzer.classify_for(node) == "index_loop"

        # Vector while
        node = _stmt("for i in range(0, self.kjpindex):\n    while x:\n        pass")
        assert self.analyzer.classify_for(node) == "vector_while"

    def test_is_index_selection_if(self):
        node = _stmt("if cond:\n    x = 1\nelse:\n    x = 2")
        assert self.analyzer._is_index_selection_if(
            node,
            {"i"},
        )

        node = _stmt("if cond:\n    a[i] = 1\nelse:\n    a[i] = 2")
        assert self.analyzer._is_index_selection_if(
            node,
            {"i"},
        )

        node = _stmt("if cond:\n    x = 1\nelse:\n    y = 2")
        assert not self.analyzer._is_index_selection_if(
            node,
            {"i"},
        )

    def test_is_masked_where_if(self):
        # Masked inline
        node = _stmt("if (b > c).any():\n    a[b > c] = 1")
        assert self.analyzer._is_masked_where_if(node)

        # Mask variable
        node = _stmt("if (b > c).any():\n    mask = b > c\n    a[mask] = 1")
        assert self.analyzer._is_masked_where_if(node)

        node = _stmt("if cond:\n    a[i] = 1")
        assert not self.analyzer._is_masked_where_if(node)

    def test_classify_if(self):
        # Masked
        node = _stmt("if cond:\n    x = 1\nelse:\n    x = 2")
        assert self.analyzer.classify_if(node) == "masked"

        self.analyzer.loop_stack = [{"i"}]
        node = _stmt("if cond:\n    a[i] = 1\nelse:\n    a[i] = 2")
        assert self.analyzer.classify_if(node) == "masked"
        # Masked where
        node = _stmt("if (b > c).any():\n    a[b > c] = 1")
        assert self.analyzer.classify_if(node) == "masked_where"
        # Scalar
        node = _stmt("if flag:\n    x = 1\n    y = 2")
        assert self.analyzer.classify_if(node) == "scalar"
        # Index loop
        self.analyzer.loop_stack = [{"i"}]
        node = _stmt("if a[i] > 0:\n    x = 1")
        assert self.analyzer.classify_if(node) == "index_loop"
        # Vector
        self.analyzer.loop_stack = [{"i"}]
        node = _stmt("if cond:\n   x = a[j] + b\n")
        assert self.analyzer.classify_if(node) == "vector"

    def test_structurally_equal(self):
        a = _expr("x + y")
        b = _expr("x + y")
        assert self.analyzer._structurally_equal(a, b)

        a = _expr("x + y")
        b = _expr("x - y")
        assert not self.analyzer._structurally_equal(a, b)


@pytest.mark.usefixtures("test_env")
class TestMaybeAddIndexTransformer:
    def test_is_full_slice(self):
        assert self.indexer.is_full_slice(ast.Slice())
        assert not self.indexer.is_full_slice(ast.Slice(lower=ast.Constant(0)))

    def test_is_lifted(self):
        self.indexer.dynamic_variable_lift = {"tmp": {"batched_axis": {0}}}
        node = ast.Name(id="tmp", ctx=ast.Load())
        assert self.indexer._is_lifted(node)

        self.indexer.dynamic_variable_lift = {"soil_temp": {}}
        node = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="soil_temp",
            ctx=ast.Load(),
        )
        assert self.indexer._is_lifted(node)

        self.indexer.dynamic_variable_lift = {"soil_temp": {}}
        node = ast.Subscript(
            value=ast.Name(id="soil_temp", ctx=ast.Load()),
            slice=ast.Name(id="i", ctx=ast.Load()),
            ctx=ast.Load(),
        )
        assert self.indexer._is_lifted(node)

    def test_extract_names(self):
        names = self.indexer.extract_names("kjpindex + nnobio")
        assert names == {"kjpindex", "nnobio"}
        # Invalid names
        assert self.indexer.extract_names("%%%") == set()

    def test_get_declared_dims(self):
        node = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="soil_temp",
            ctx=ast.Load(),
        )
        assert self.indexer.get_declared_dims(node) == ["kjpindex"]
        # Local array
        node = ast.Name(id="tmp", ctx=ast.Load())
        assert self.indexer.get_declared_dims(node) == ["kjpindex"]

        # Subscripts
        node = ast.parse("soil_moist[i]").body[0].value
        assert self.indexer.get_declared_dims(node) == [
            "kjpindex",
            "nnobio",
        ]

    def test_is_arange_over_dim(self):
        node = ast.parse("jnp.arange(self.kjpindex)").body[0].value
        assert self.indexer._is_arange_over_dim(node)

        node = ast.parse("jnp.arange(kjpindex)").body[0].value
        assert self.indexer._is_arange_over_dim(node)

        node = ast.parse("jnp.arange(n)").body[0].value
        assert not self.indexer._is_arange_over_dim(node)

    def test_get_active_dims(self):
        node = ast.Name(id="tmp", ctx=ast.Load())
        assert self.indexer.get_active_dims(node) == ["kjpindex"]

        node = ast.parse("soil_moist[i, j]").body[0].value
        assert self.indexer.get_active_dims(node) == []

        node = ast.parse("soil_moist[:, j]").body[0].value

        assert self.indexer.get_active_dims(node) == ["kjpindex"]

        node = ast.parse("soil_moist[:, :]").body[0].value

        assert self.indexer.get_active_dims(node) == [
            "kjpindex",
            "nnobio",
        ]

    def test_visit(self):
        node = ast.Name(id="tmp", ctx=ast.Load())
        result = self.indexer.visit(node)
        assert result is node

        node = ast.Name(id="tmp", ctx=ast.Load())
        self.indexer.ranks[node] = 2
        result = self.indexer.visit(node)
        assert result is node

        # Rank promotion
        node = ast.Name(id="tmp", ctx=ast.Load())
        self.indexer.ranks[node] = 1
        self.indexer.target_rank = 2
        result = self.indexer.visit(node)
        assert isinstance(result, ast.Subscript)

    def test_visit_binop(self):
        expr = ast.parse("soil_temp + tmp", mode="eval").body
        self.indexer.ranks[expr.left] = 1
        self.indexer.ranks[expr.right] = 1
        result = self.indexer.visit_BinOp(expr)
        assert isinstance(result, ast.BinOp)

        expr = ast.parse("a + b", mode="eval").body
        self.indexer.ranks[expr.left] = 1
        self.indexer.ranks[expr.right] = 1
        self.indexer.local_defined_var = {
            "a": ["kjpindex"],
            "b": ["nnobio"],
        }
        self.indexer.vect_context = {
            "loop_info": {
                "kjpindex": "i",
                "nnobio": "j",
            },
            "vectorization_axis": {
                "i": [0],
                "j": [0],
            },
        }
        result = self.indexer.visit_BinOp(expr)
        assert isinstance(result, ast.BinOp)

        # Rank promotion on the left
        expr = ast.parse("soil_temp + soil_moist", mode="eval").body
        self.indexer.ranks[expr.left] = 1
        self.indexer.ranks[expr.right] = 2
        result = self.indexer.visit_BinOp(expr)
        assert isinstance(result.left, ast.Subscript)


@pytest.mark.usefixtures("test_env")
class TestReductionHandler:
    def test_get_array_info(self):
        # Local array
        dims = self.reducer._get_array_info("tmp")
        assert dims == ["kjpindex"]
        # Attribute
        dims = self.reducer._get_array_info("soil_moist")
        assert dims == ["kjpindex", "nnobio"]
        # Unknown array
        assert self.reducer._get_array_info("does_not_exist") is None

    def test_extract_axes(self):
        node = ast.Name(id="soil_moist", ctx=ast.Load())
        axes = self.reducer._extract_axes(node)
        assert axes == {0, 1}

        node = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="soil_temp",
            ctx=ast.Load(),
        )
        axes = self.reducer._extract_axes(node)
        assert axes == {0}

        # Slice
        node = _expr("arr[:, 1]")
        axes = self.reducer._extract_axes(node)
        assert axes == {0}

        # 2D Slice
        node = _expr("arr[:, :]")
        axes = self.reducer._extract_axes(node)
        assert axes == {0, 1}

    def test_axis_matches(self):
        assert self.reducer._axis_matches(
            ast.Constant(0),
            {0},
        )

        # Tuple
        assert self.reducer._axis_matches(
            ast.Tuple(
                elts=[ast.Constant(0), ast.Constant(1)],
                ctx=ast.Load(),
            ),
            {0, 1},
        )

        assert not self.reducer._axis_matches(
            ast.Constant(1),
            {0},
        )

    def test_broadcastable(self):
        # Same shape
        assert self.reducer._broadcastable(
            [10, 20],
            [10, 20],
        )
        # Keepdims
        assert self.reducer._broadcastable(
            [10, 1],
            [10, 20],
        )

        assert not self.reducer._broadcastable(
            [10, 3],
            [10, 20],
        )

    def test_compute_reduction_shapes(self):
        no_keep, keep = self.reducer._compute_reduction_shapes(
            [100, 20],
            {1},
        )
        assert no_keep == [100]
        assert keep == [100, 1]

    def test_infer_axes(self):
        node = _expr("jnp.sum(self.soil_moist)")
        axes = self.reducer._infer_axes(node)
        # soil_moist -> {0,1}
        # vectorized axis -> {0}
        assert axes == {1}

    def test_is_reduction_call(self):
        node = _expr("jnp.sum(x)")
        is_red, is_special = self.reducer._is_reduction_call(node)
        assert is_red
        assert not is_special

        # Not reduction call
        node = _expr("jnp.exp(x)")
        is_red, is_special = self.reducer._is_reduction_call(node)
        assert not is_red

    def test_process_call(self):
        node = _expr("jnp.sum(self.soil_moist)")
        result = self.reducer.process_call(node)
        axis_kw = next(kw for kw in result.keywords if kw.arg == "axis")
        assert isinstance(axis_kw.value, ast.Constant)
        assert axis_kw.value.value == 1


@pytest.mark.usefixtures("test_env")
class TestWhileVectorToScalar:
    def test_get_name(self):
        node = ast.Name(id="x", ctx=ast.Load())
        assert self.whiletransformer._get_name(node) == "x"
        # Attribute
        node = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="soil_temp",
            ctx=ast.Load(),
        )
        assert self.whiletransformer._get_name(node) == "soil_temp"
        # Invalid
        node = ast.Constant(value=1)
        assert self.whiletransformer._get_name(node) is None

    def test_visit_name(self):
        # Vector array
        self.whiletransformer.vector_arrays = {"x"}
        node = ast.Name(id="x", ctx=ast.Load())
        result = self.whiletransformer.visit_Name(node)
        assert isinstance(result, ast.Name)
        assert result.id == "x"

        # Store context
        self.whiletransformer.vector_arrays = {"x"}
        node = ast.Name(id="x", ctx=ast.Store())
        result = self.whiletransformer.visit_Name(node)
        assert result.id == "x"
        assert isinstance(result.ctx, ast.Store)

    def test_visit_subscript(self):
        # remove vectorized axis
        self.whiletransformer.vectorization_axis = {"i": [0]}
        node = _expr("arr[i, j]")
        result = self.whiletransformer.visit(node)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Name)
        assert result.slice.id == "j"

        # Remove all axes
        self.whiletransformer.vectorization_axis = {"i": [0]}
        node = _expr("arr[i]")
        result = self.whiletransformer.visit(node)
        assert isinstance(result, ast.Name)
        assert result.id == "arr"

    def test_visit_subscript_tracks_loop_dependency(self):
        self.whiletransformer.loop_index = "i"
        node = _expr("arr[i]")
        self.whiletransformer.visit(node)
        assert "arr" in self.whiletransformer.ji_dependent_vars

    def test_visit_while(self):
        self.whiletransformer._in_while = False
        node = _stmt(
            """
while cond:
    x = 1
            """
        )

        self.whiletransformer.visit(node)
        assert self.whiletransformer._in_while is False

    def test_visit_assign(self):
        node = _stmt("x = y")
        result = self.whiletransformer.visit(node)
        assert ast.unparse(result) == "x = y"

        self.whiletransformer.var_to_replace = {"x": "x_scalar"}
        node = _stmt("x = x.at[i].set(y)")
        result = self.whiletransformer.visit(node)
        assert ast.unparse(ast.fix_missing_locations(result)) == "x_scalar = y"

        # Add
        self.whiletransformer.var_to_replace = {"x": "x_scalar"}
        node = _stmt("x = x.at[i].add(y)")
        result = self.whiletransformer.visit(node)
        assert (
            ast.unparse(ast.fix_missing_locations(result)) == "x_scalar = x_scalar + y"
        )

        # Self reference
        self.whiletransformer.var_to_replace = {"x": "x_scalar"}
        node = _stmt("x = x.at[i].add(x[i])")
        result = self.whiletransformer.visit(node)
        assert ast.unparse(ast.fix_missing_locations(result)) == (
            "x_scalar = x_scalar + x_scalar"
        )

    def test_visit_assign_tracks_while_variables(self):
        tree = _stmt(
            """
while cond:
    x = y
            """
        )
        self.whiletransformer.visit(tree)
        assert "x" in self.whiletransformer.while_used_vars
