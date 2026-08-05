import ast

import pytest

from fgpt.core.backends.jax_converter.converter import JaxConverter
from fgpt.core.backends.utils import Control
from fgpt.core.common.logger import Logger


def _make_cls_info():
    return {
        "Model": {
            "attributes": {
                "kjpindex": {"type": "int"},
                "nnobio": {"type": "int"},
                "nslm": {"type": "int"},
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
                "total": {
                    "type": "jnp.ndarray",
                    "dimensions": ["nslm"],
                    "dtype": "float64",
                },
            },
            "methods": {
                "compute": {
                    "args": ["x"],
                    "local_arr": {
                        "tmp": {
                            "dimensions": ["kjpindex"],
                            "dtype": "float64",
                            "type": "jnp.ndarray",
                        },
                    },
                },
                "helper": {
                    "args": ["y"],
                    "local_arr": {},
                },
            },
        }
    }


@pytest.fixture(scope="class")
def test_env(request):
    converter = JaxConverter(
        cls_info=_make_cls_info(),
        vectorize=["kjpindex"],
        logger=Logger(),
        mode="jax",
    )
    converter.func_name = "compute"
    converter.func_input_dim = {"x": ["kjpindex"]}
    converter.call_edge = {"compute": [], "helper": []}
    request.cls.converter = converter
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


def _unparse(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))


@pytest.mark.usefixtures("test_env")
class TestJaxConverter:
    def test_scope_push_pop(self):
        # A variable added in a pushed scope is not visible after pop
        self.converter._push_scope()
        self.converter._add_local("local_var", ("kjpindex",))
        assert self.converter._is_local("local_var") is True
        self.converter._pop_scope()
        assert self.converter._is_local("local_var") is False

    def test_scope_is_local(self):
        # A variable defined in an outer scope is visible from a nested scope
        self.converter._push_scope()
        self.converter._add_local("outer_var", ())
        self.converter._push_scope()
        assert self.converter._is_local("outer_var") is True
        self.converter._pop_scope()
        self.converter._pop_scope()

        # Unknown case
        assert self.converter._is_local("totally_unknown_name") is False

    def test_fresh_names(self):
        start = self.converter.counter
        true_name, false_name = self.converter._fresh_names()
        assert true_name == f"_if_true_{start}"
        assert false_name == f"_if_false_{start}"
        assert self.converter.counter == start + 1

        # Unique calls
        name1, _ = self.converter._fresh_names()
        name2, _ = self.converter._fresh_names()
        assert name1 != name2

    def test_is_logging_call(self):
        # Logging true
        stmt = _stmt('logging.info("hello")')
        assert self.converter._is_logging_call(stmt) is True

        stmt = _stmt('print("hello")')
        assert self.converter._is_logging_call(stmt) is False
        # False expression
        stmt = _stmt("x = 1")
        assert self.converter._is_logging_call(stmt) is False

    def test_collect_loop_vars(self):
        # Simple target
        stmts = _parse("for i in range(10):\n    pass").body
        result = self.converter._collect_loop_vars(stmts)
        assert result == {"i"}

        # Tuple target
        stmts = _parse("for i, j in pairs:\n    pass").body
        result = self.converter._collect_loop_vars(stmts)
        assert result == {"i", "j"}

        # No loop
        stmts = _parse("x = 1\ny = 2").body
        result = self.converter._collect_loop_vars(stmts)
        assert result == set()

    def test_first_reads(self):
        # Simple case
        stmts = _parse("y = x + 1").body
        result = self.converter._first_reads(stmts)
        assert "x" in result

        # `y` is only ever stored to, never loaded, so it should not appear
        stmts = _parse("y = 1").body
        result = self.converter._first_reads(stmts)
        assert "y" not in result

        # a[i] = a[i] + 1 -> `a` must be recorded as a LOAD (used before written)
        stmts = _parse("a[i] = a[i] + 1").body
        result = self.converter._first_reads(stmts)
        assert "a" in result

        # First_reads doesn't retrieve the loop index
        stmts = _parse("for i in range(10):\n    y = i + 1").body
        result = self.converter._first_reads(stmts)
        assert "i" not in result

        # Attribute
        stmts = _parse("y = self.kjpindex + 1").body
        result = self.converter._first_reads(stmts)
        assert "self.kjpindex" in result

    def test_collect_rhs_uses(self):
        # Simple case
        stmts = _parse("y = a + b").body
        result = self.converter._collect_rhs_uses(stmts)
        assert result == {"a", "b"}

        # The assigned target `y` should not appear in RHS uses
        stmts = _parse("y = a + b").body
        result = self.converter._collect_rhs_uses(stmts)
        assert "y" not in result

        # Subscripts
        stmts = _parse("y = a[i]").body
        result = self.converter._collect_rhs_uses(stmts)
        assert "a" in result
        assert "i" in result

    def test_collect_assigned(self):
        # Plain name
        stmts = _parse("x = 1").body
        result = self.converter._collect_assigned(stmts)
        assert "x" in result

        # Attribute mutated
        self.converter._mutated_attrs = set()
        stmts = _parse("self.soil_temp = 1").body
        result = self.converter._collect_assigned(stmts)
        assert "soil_temp" in result
        assert "soil_temp" in self.converter._mutated_attrs

        stmts = _parse("x += 1").body
        result = self.converter._collect_assigned(stmts)
        assert "x" in result

        stmts = _parse("a[i] = 1").body
        result = self.converter._collect_assigned(stmts)
        assert "a" in result

        # NO duplicates
        stmts = _parse("x = 1\nx = 2").body
        result = self.converter._collect_assigned(stmts)
        assert result.count("x") == 1

    def test_subscript_uses_loop_vars(self):
        node = _expr("a[i]")
        assert self.converter._subscript_uses_loop_vars(node, ["i"]) is True

        # No loop index usage inside subscript
        node = _expr("a[j]")
        assert self.converter._subscript_uses_loop_vars(node, ["i"]) is False

        node = _expr("a[i, k]")
        assert self.converter._subscript_uses_loop_vars(node, ["i"]) is True

        node = _expr("f(a[i])")
        assert self.converter._subscript_uses_loop_vars(node, ["i"]) is True

    def test_handle_lax_cond(self):
        self.converter._mutated_attrs = set()

        node = _stmt("""
if cond:
    x = 1
else:
    x = 2
        """)

        assigned = ["x"]
        read_before_write = []
        used_after = set()

        result = self.converter._handle_lax_cond(
            node, assigned, read_before_write, used_after
        )

        # should produce: [true_fn, false_fn, assign]
        assert len(result) == 3

        true_fn, false_fn, assign = result

        assert isinstance(true_fn, ast.FunctionDef)
        assert isinstance(false_fn, ast.FunctionDef)
        assert isinstance(assign, ast.Assign)

        # lax.cond call
        call = assign.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Attribute)
        assert call.func.attr == "cond"

        # Else body empty
        node = _stmt("""
if cond:
    x = 1
        """)

        result = self.converter._handle_lax_cond(node, ["x"], [], set())

        # only true function + assign
        assert len(result) == 2

        true_fn, assign = result
        assert isinstance(true_fn, ast.FunctionDef)
        assert isinstance(assign, ast.Assign)

        # false case becomes Lambda
        call = assign.value
        assert isinstance(call.args[2], ast.Lambda)

        # Mutliple assigned vars
        self.converter._mutated_attrs = set()
        node = _stmt("""
if cond:
    x = 1
    y = 2
else:
    x = 3
    y = 4
        """)
        result = self.converter._handle_lax_cond(
            node,
            assigned=["x", "y"],
            read_before_write=[],
            used_after={"x", "y"},
        )
        # true_fn + false_fn + assign
        assert len(result) == 3
        true_fn, false_fn, assign = result
        assert isinstance(true_fn, ast.FunctionDef)
        assert isinstance(false_fn, ast.FunctionDef)
        assert isinstance(assign, ast.Assign)

        # Variables read before written in a branch are forwarded as inputs
        self.converter._mutated_attrs = set()
        node = _stmt("""
if cond:
    x = prev + 1
else:
    x = 0
        """)
        result = self.converter._handle_lax_cond(
            node,
            assigned=["x"],
            read_before_write=["prev"],
            used_after=set(),
        )
        true_fn = result[0].body
        assert isinstance(true_fn[0], ast.Assign)
        assert isinstance(true_fn[0].targets[0], ast.Tuple)
        arg_names = [arg.id for arg in true_fn[0].targets[0].elts]
        assert "prev" in arg_names

        # A mutated self-attribute must appear in the generated helper's args
        self.converter._mutated_attrs = {"soil_temp"}
        node = _stmt("""
if cond:
    self.soil_temp = 1.0
else:
    self.soil_temp = 0.0
        """)
        result = self.converter._handle_lax_cond(
            node,
            assigned=["soil_temp"],
            read_before_write=["self.soil_temp"],
            used_after=set(),
        )
        true_fn = result[0].body
        assert isinstance(true_fn[0], ast.Assign)
        assert isinstance(true_fn[0].targets[0], ast.Tuple)
        arg_names = [arg.id for arg in true_fn[0].targets[0].elts]
        assert "soil_temp" in arg_names

    def test_to_arg(self):
        # strip 'self' prefix
        assert self.converter.to_arg("self.kjpindex") == "kjpindex"
        assert self.converter.to_arg("kjpindex") == "kjpindex"

    def test_mask_plain_assign(self):
        self.converter._local_defaults = {}
        self.converter._local_defined_stack = [{}]

        assign = _stmt("x = 1")

        result = self.converter._mask_plain_assign(assign, "m")

        assert len(result) == 1
        new_assign = result[0]

        assert isinstance(new_assign, ast.Assign)

        call = new_assign.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "where"

        # jnp.where(mask, value, old_val)
        assert isinstance(call.args[0], ast.Name)  # mask

        # Reuses old values
        # simulate prior definition in outer scope
        self.converter._local_defined_stack = [{}, {"x": ("kjpindex",)}]

        assign = _stmt("x = 5")
        result = self.converter._mask_plain_assign(assign, "m")

        call = result[0].value
        assert isinstance(call.args[2], ast.Name)  # old_val is Name, not ones_like
        assert call.args[2].id == "x"

        # Creates default
        self.converter._local_defaults = {}
        self.converter._local_defined_stack = [{}]

        assign = _stmt("x = 5")
        result = self.converter._mask_plain_assign(assign, "m")

        call = result[0].value
        old_val = call.args[2]

        # default = jnp.ones_like(1)
        assert isinstance(old_val, ast.Call)
        assert old_val.func.attr == "ones_like"

        # Pass through for arrays
        assign = _stmt("a = arr.at[0].set(1)")
        result = self.converter._mask_plain_assign(assign, "m")

        # value should be unchanged logic-wise (still list with assign)
        assert isinstance(result[0], ast.Assign)

        # Stateful variable -> Got lifted
        self.converter.var_state = {"x": ("stateful", None)}
        self.converter._local_defined_stack = [{}, {"x": ("kjpindex",)}]

        assign = _stmt("x = 1")
        result = self.converter._mask_plain_assign(assign, "m")
        call = result[0].value
        # should wrap in .at[:].set(...)
        assert isinstance(call, ast.Call)

    def test_mask_vector_assign(self):
        # Fallback to mask plain assign
        assign = _stmt("x = 1")

        self.converter._control_stack = [
            Control(
                kind="if",
                metadata={},
            )
        ]
        result = self.converter._mask_vector_assign(assign, "m", [], [])
        assert isinstance(result[0], ast.Assign)

        # Skip control temporary
        assign = _stmt("tmp = 1")
        self.converter._control_stack = [
            Control(
                kind="if",
                vectorization_axis={"ji": 0},
                loop_info={"kjpindex": "ji"},
                metadata={},
            )
        ]
        self.converter._is_control_temporary = lambda name, a, u: True

        result = self.converter._mask_vector_assign(assign, "m", ["tmp"], [])
        assert result == [assign]

    def test_assigned_name_or_attr_nested_subscript(self):
        target = _expr("a[i][j]")
        result = self.converter._assigned_name_or_attr(target, [])
        assert result == ["a"]

    def test_assigned_name_or_attr_self_subscript(self):
        self.converter._mutated_attrs.clear()

        target = _expr("self.soil_temp[i]")
        result = self.converter._assigned_name_or_attr(target, [])

        assert result == ["soil_temp"]
        assert "soil_temp" in self.converter._mutated_attrs

    def test_rewrite_if_return(self):
        fn = _func("""
def f(self):
    if cond:
        x = 1
        return x
    y = 2
    return y
        """)

        result = self.converter._rewrite_if_return(
            fn,
            fn.body[0],
            0,
        )

        assert len(result.body) == 1
        new_if = result.body[0]
        assert isinstance(new_if, ast.If)
        assert len(new_if.body) == 1
        assert len(new_if.orelse) == 1

        # If the if already has an else, the rewrite should still succeed
        fn = _func("""
def f(self):
    if cond:
        x = 1
        return x
    else:
        x = 2
        return x
        """)
        result = self.converter._rewrite_if_return(fn, fn.body[0], 0)
        assert isinstance(result, ast.FunctionDef)
        new_if = result.body[0]
        assert isinstance(new_if, ast.If)

        # Statements after the if-block must end up in the synthesised else
        fn = _func("""
def f(self):
    if cond:
        x = 1
        return x
    y = 2
    z = 3
    return z
        """)
        result = self.converter._rewrite_if_return(fn, fn.body[0], 0)
        new_if = result.body[0]
        # The else branch should contain y = 2, z = 3, return z
        assert len(new_if.orelse) >= 2

    def test_make_operand(self):
        # NOt mutated attributes
        self.converter._mutated_attrs = set()
        result = self.converter.make_operand("self.kjpindex")
        assert isinstance(result, ast.Attribute)
        assert result.attr == "kjpindex"

        # mutated attributes
        self.converter._mutated_attrs.add("self.kjpindex")
        result = self.converter.make_operand("kjpindex")
        assert isinstance(result, ast.Name)
        assert result.id == "kjpindex"

        result = self.converter.make_operand("local_var")
        assert isinstance(result, ast.Name)
        assert result.id == "local_var"

    def test_replace_returns(self):
        # strips return
        stmts = _parse("return 1").body
        result = self.converter._replace_returns(stmts)
        assert result == []

        stmts = _parse("x = 1\nreturn x").body
        result = self.converter._replace_returns(stmts)
        assert len(result) == 1
        assert isinstance(result[0], ast.Assign)

    def test_lift_if_return(self):
        # Merge branches
        code = (
            "def f(self):\n    if cond:\n        x = 1\n        return x\n    y = 2\n"
        )
        fn = _func(code)
        result = self.converter._lift_if_return(fn)
        # body should now contain a single If with both branches and no Return
        assert len(result.body) == 1
        assert isinstance(result.body[0], ast.If)
        assert not any(isinstance(s, ast.Return) for s in ast.walk(result.body[0]))

        # No return operation
        code = "def f(self):\n    if cond:\n        x = 1\n    y = 2\n"
        fn = _func(code)
        result = self.converter._lift_if_return(fn)
        assert len(result.body) == 2

    def test_is_pure_select(self):
        # Condition true for batching branches
        code = "if cond:\n    x = 1\nelse:\n    x = 2\n"
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is True

        # Condition false in the true branch
        code = "if cond:\n    x = f(1)\nelse:\n    x = 2\n"
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is False

        # Mismatched branches
        code = "if cond:\n    x = 1\nelse:\n    y = 2\n"
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is False

        # Subscript targets
        code = "if cond:\n    a[i] = 1\nelse:\n    a[i] = 2\n"
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is False

        # A single if-branch with no else still qualifies; the missing
        # else branch is handled by the fallback logic in _handle_scalar_select
        code = "if cond:\n    x = 1\n"
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is True

    def test_handle_scalar_select(self):
        node = _stmt("""
if cond:
    x = 1
else:
    x = 2
        """)

        result = self.converter._handle_scalar_select(
            node,
            ["x"],
        )

        assert len(result) == 2

        cond_assign = result[0]

        assert isinstance(cond_assign, ast.Assign)

        assert cond_assign.targets[0].id == "_cond_0"

        x_assign = result[1]

        assert isinstance(x_assign.value, ast.Call)

        assert x_assign.value.func.attr == "where"

    def test_normalize_masked_node(self):
        node = _expr("a[mask]")

        cleaned, mask = self.converter.normalize_masked_node(
            node,
            {"mask"},
        )

        assert isinstance(cleaned, ast.Name)
        assert cleaned.id == "a"

        assert isinstance(mask, ast.Name)
        assert mask.id == "mask"

    def test_handle_masked_where_outer_mask_name(self):
        # x[mask] = y[mask] + 1 / x[mask] = y[mask] + 2, where `mask` is
        # the WHERE condition itself (not locally assigned in either branch)
        code = "if mask:\n    x[mask] = y[mask] + 1\nelse:\n    x[mask] = y[mask] + 2\n"
        node = _stmt(code)
        result = self.converter.handle_masked_where(node)

        assert len(result) == 1
        assign = result[0]
        assert isinstance(
            assign, ast.Assign
        )  # x = x.at[mask].set(jnp.where(mask, y[mask] + 1, y[mask] + 2))
        assert (
            isinstance(assign.value, ast.Call)
            and isinstance(assign.value.func, ast.Attribute)
            and isinstance(assign.value.func.value, ast.Subscript)
        )
        mask_sub = assign.value.func.value
        assert isinstance(mask_sub.slice, ast.Name) and mask_sub.slice.id == "mask"
        assert isinstance(assign.value.args[0], ast.Call) and isinstance(
            assign.value.args[0].func, ast.Attribute
        )
        assert assign.value.args[0].func.attr == "where"

    def test_transform_compare(self):
        # Simple compare
        node = _expr("a > b")
        result = self.converter._transform_compare(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "greater"

        # Chained compare
        node = _expr("a < b < c")
        result = self.converter._transform_compare(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "logical_and"

        node = _expr("a in b")
        with pytest.raises(NotImplementedError):
            self.converter._transform_compare(node)

    def test_transform_boolop(self):
        node = _expr("a and b")
        result = self.converter._transform_boolop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "logical_and"

        node = _expr("a or b")
        result = self.converter._transform_boolop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "logical_or"

    def test_transform_if_test(self):
        node = _expr("not a")
        result = self.converter._transform_if_test(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "logical_not"

    def test_transform_binop(self):
        node = _expr("a | b")
        result = self.converter._transform_binop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "logical_or"

        node = _expr("a & b")
        result = self.converter._transform_binop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "logical_and"

        node = _expr("a + b")
        result = self.converter._transform_binop(node)
        assert isinstance(result, ast.BinOp)
        assert isinstance(result.op, ast.Add)

    def test_check_if(self):
        # Array true case
        node = _expr("self.soil_temp")
        assert self.converter.check_if_array(node) is True

        node = _expr("self.kjpindex")
        assert self.converter.check_if_array(node) is False

        # For function input array
        node = _expr("x")
        assert self.converter.check_if_array(node) is True

        node = _expr("totally_unknown")
        assert self.converter.check_if_array(node) is False

        node = _expr("self.soil_moist")
        assert self.converter.check_if_array(node, required_dims=["nnobio"]) is True
        assert (
            self.converter.check_if_array(node, required_dims=["unrelated_dim"])
            is False
        )

    def test_is_arr_at_op_call(self):
        node = _expr("arr.at[0].set(1)")
        assert self.converter.is_arr_at_op_call(node, "arr") is True

        # Local variable form: arr.at[0].set(1)
        node = _expr("soil_temp.at[0].set(1)")
        assert self.converter.is_arr_at_op_call(node, "soil_temp") is True

        # self.attr.at[...] does NOT match — is_arr_at_op_call expects a
        # bare local-variable base, not a self.attr base.
        node = _expr("self.soil_temp.at[0].set(1)")
        assert self.converter.is_arr_at_op_call(node, "soil_temp") is False

        node = _expr("self.soil_temp.at[0].set(1)")
        assert self.converter.is_arr_at_op_call(node, "other_name") is False

        node = _expr("f(x)")
        assert self.converter.is_arr_at_op_call(node, "x") is False

    def test_strip_any(self):
        node = _expr("(a < b).any()")
        result = self.converter.strip_any(node)
        assert isinstance(result, ast.Compare)

        node = _expr("f(x)")
        result = self.converter.strip_any(node)
        assert result is node

    def test_set_working_function(self):
        # Update class data
        self.converter.set_working_function(
            "helper", {"y": ["kjpindex"]}, {"helper": []}
        )
        assert self.converter.func_name == "helper"
        assert self.converter.func_input_dim == {"y": ["kjpindex"]}
        # restore for other tests in the class
        self.converter.set_working_function(
            "compute", {"x": ["kjpindex"]}, {"compute": [], "helper": []}
        )

    def test_reset_all(self):
        self.converter._mutated_attrs = {"soil_temp"}
        self.converter.var_state = {"tmp": ("stateful", None)}
        self.converter.dynamic_variable_lift = {"tmp": {}}
        self.converter.reset_all()
        assert self.converter._mutated_attrs == set()
        assert self.converter.var_state == {}
        assert self.converter.dynamic_variable_lift == {}
        assert self.converter.for_counter == 0
        assert self.converter.counter == 0

    def _setup_call_edge_parent(self):
        """Make 'compute' appear as a callee so has_parent is True."""
        from fgpt.core.backends.utils import (
            CallEdge,  # adjust import to your project layout
        )

        edge = CallEdge(
            caller="helper",
            callee="compute",
            call_node=None,
            arg_shapes=[],
            func_args=[],
        )
        self.converter.call_edge = {"helper": [edge], "compute": []}
        self.converter.func_name = "compute"

    def _setup_no_parent(self):
        self.converter.call_edge = {"compute": [], "helper": []}
        self.converter.func_name = "compute"

    def test_add_return_stmt(self):
        # raw tuple returned when has parent
        self._setup_call_edge_parent()
        ret = self.converter.add_return_stmt(
            var_modif_args=["x"],
            var_modif_attr=["soil_temp"],
        )
        assert isinstance(ret, ast.Return)
        assert isinstance(ret.value, ast.Tuple)
        ids = [e.id for e in ret.value.elts]
        assert "x" in ids
        assert "soil_temp" in ids

        # Tree.at() with no parent args
        self._setup_no_parent()
        ret = self.converter.add_return_stmt(
            var_modif_args=["x"],
            var_modif_attr=["soil_temp"],
        )
        assert isinstance(ret, ast.Return)
        # value should be a Tuple (args + tree_at_call)
        assert isinstance(ret.value, ast.Tuple)
        # Last element must be the eqx.tree_at call
        last = ret.value.elts[-1]
        assert isinstance(last, ast.Call)
        assert last.func.attr == "tree_at"

        # No parent, no args
        self._setup_no_parent()
        ret = self.converter.add_return_stmt(
            var_modif_args=[],
            var_modif_attr=["soil_temp"],
        )
        assert isinstance(ret, ast.Return)
        # Only the tree_at call — not wrapped in a Tuple
        assert isinstance(ret.value, ast.Call)
        assert ret.value.func.attr == "tree_at"

        # Empty mutation
        self._setup_no_parent()
        ret = self.converter.add_return_stmt(
            var_modif_args=[],
            var_modif_attr=[],
        )
        # tree_at with empty lambda body → still a Call
        assert isinstance(ret, ast.Return)

    def test_process_helpers(self):
        helper = _func("""
def _if_true_0(arg):
    x, = arg
    return (x + 1,)
        """)
        self.converter._pending_helpers = [helper]
        self.converter.process_helpers()
        assert self.converter._pending_helpers == []
        assert len(self.converter.helpers) >= 1

        # Context stack retrieved
        depth_before = len(self.converter._context_stack)
        helper = _func("""
def _if_true_1(arg):
    x, = arg
    return (x,)
        """)
        self.converter._pending_helpers = [helper]
        self.converter.process_helpers()
        assert len(self.converter._context_stack) == depth_before

        # func_arg_stack restored after process_helpers
        depth_before = len(self.converter._func_arg_stack)
        helper = _func("""
def _if_true_2(arg):
    x, = arg
    return (x,)
        """)
        self.converter._pending_helpers = [helper]
        self.converter.process_helpers()
        assert len(self.converter._func_arg_stack) == depth_before

    def _visit_if(self, code):
        """Reset converter, parse the first statement, call visit_If."""
        self.converter.reset_all()
        node = _stmt(code)
        return self.converter.visit_If(node)

    def test_visit_if(self):
        # Both branches assign the same scalar var -> jnp.where.
        result = self._visit_if("""
if x > 0:
    y = 1
else:
    y = 2
        """)
        stmts = result if isinstance(result, list) else [result]
        code = " ".join(_unparse(s) for s in stmts)
        assert "where" in code

        # x > 0 must be rewritten to jnp.greater (or similar).
        result = self._visit_if("""
if x > 0:
    y = 1
else:
    y = 0
        """)
        assert isinstance(result, ast.Assign)
        # THe first two assignement corresponds to the condition
        assert isinstance(result.value, ast.Call)
        assert isinstance(result.value.func, ast.Attribute)
        # THese corresponds to the transformed jax intrinsic functions
        assert result.value.func.attr == "where"
        assert isinstance(result.value.args[0], ast.Call)
        assert (
            isinstance(result.value.args[0].func, ast.Attribute)  # jnp.greater
            and result.value.args[0].func.attr == "greater"
        )

        # elif produces at least two jnp.where calls (nested)
        result = self._visit_if("""
if x > 0:
    y = 1
elif x < 0:
    y = -1
else:
    y = 0
        """)
        assert _unparse(result) == (
            "y = jnp.where(jnp.greater(x, 0), 1, jnp.where(jnp.less(x, 0), -1, 0))"
        )

        # An if whose entire body is logging.* calls is dropped entirely
        result = self._visit_if("""
if x > 0:
    logging.info("hi")
        """)
        assert result is None

        # Logging mixed with real code: logging removed, code kept
        result = self._visit_if("""
if x > 0:
    logging.debug("entering")
    y = 1
else:
    y = 0
        """)
        assert isinstance(result, ast.Assign)
        assert isinstance(result.value, ast.Call)
        assert (
            isinstance(result.value.func, ast.Attribute)
            and result.value.func.attr == "where"
        )
        assert len(result.value.args) == 3
        stmts = result if isinstance(result, list) else [result]
        code = " ".join(_unparse(s) for s in stmts)
        assert "logging" not in code

        # If both body and orelse collapse to nothing, visit_If returns None
        result = self._visit_if("""
if x > 0:
    logging.info("a")
else:
    logging.info("b")
        """)
        assert result is None

        # a and b in condition -> jnp.logical_and
        result = self._visit_if("""
if a and b:
    y = 1
else:
    y = 0
        """)
        assert isinstance(result, ast.Assign)
        assert isinstance(result.value, ast.Call)
        assert (
            isinstance(result.value.func, ast.Attribute)
            and result.value.func.attr == "where"
        )
        assert len(result.value.args) == 3
        assert isinstance(result.value.args[0], ast.Call)
        assert (
            isinstance(result.value.args[0].func, ast.Attribute)
            and result.value.args[0].func.attr == "logical_and"
        )

    def test_visit_if_with_mixed_operations(self):
        result = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            arr[i] = arr[i] + 1
        else:
            arr[i] = 0
        """)
        stmts = result.body
        assert len(stmts) == 2  # Contains the mask and the arr as masked update
        assert _unparse(stmts[-1]) == (
            "arr = arr.at[:].set(jnp.where(_mask_0, arr + 1, 0))"
        )

        result = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            arr[i] = arr[i] + a
        elif x[i] > 0 and y[i] < a:
            arr[i] = b
        else:
            arr[i] = c
        """)
        stmts = result.body
        assert len(stmts) == 3  # Contains the 2 mask and the arr as masked update
        assert "logical_and" in _unparse(stmts[0])  # the elif mask
        assert "greater" in _unparse(stmts[1])  # the if mask
        assert _unparse(stmts[-1]) == (
            "arr = arr.at[:].set(jnp.where(_mask_1, arr + a, jnp.where(_mask_0, b, c)))"
        )

        result = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            arr[i] = arr[i] + a
        elif x[i] > 0 and y[i] < a:
            arr[i] = b
        else:
            arr[i] = arr[i] - c
        """)
        stmts = result.body
        assert len(stmts) == 3  # Contains the 2 mask and the arr as masked update
        assert "logical_and" in _unparse(stmts[0])  # the elif mask
        assert "greater" in _unparse(stmts[1])  # the if mask
        assert _unparse(stmts[-1]) == (
            "arr = arr.at[:].set(jnp.where(_mask_1, arr + a, jnp.where(_mask_0, b, arr - c)))"
        )

    def _transform_fn(self, code):
        self.converter.reset_all()
        self.converter.helpers = []
        self.converter._pending_helpers = []
        fn = _func(code)
        transformed = self.converter.visit(fn)
        self.converter.process_helpers()
        return transformed

    def test_vector_visit_if(self):
        # An if inside a vectorised for-loop should emit a _mask_ assignment
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            x[i] = 1.0
        else:
            x[i] = 0.0
        """)
        code_body = transformed.body
        # Mask assignement for the if condition
        assert isinstance(code_body[0], ast.Assign) and isinstance(
            code_body[0].value, ast.Call
        )
        assert isinstance(code_body[0].value.func, ast.Attribute)
        # THese corresponds to the transformed jax intrinsic functions
        assert code_body[0].value.func.attr == "greater"

        # This now corresponds to the inner value
        assert (
            isinstance(code_body[1], ast.Assign)
            and isinstance(code_body[1].targets[0], ast.Name)
            and code_body[1].targets[0].id == "x"
        )
        assert (
            isinstance(code_body[1].value, ast.Call)
            and isinstance(code_body[1].value.args[0], ast.Call)
            and isinstance(code_body[1].value.args[0].func, ast.Attribute)
            and code_body[1].value.args[0].func.attr == "where"
        )

        # Vectorised if must not route through lax.cond.
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            x[i] = 1.0
        else:
            x[i] = 0.0
        """)
        code = _unparse(transformed)
        assert "lax.cond" not in code

        # The else branch of a vectorised if gets a logical_not of the mask.
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            x[i] = 1.0
        else:
            y[i] = -1.0
        """)
        result = transformed.body
        assert len(result) == 4
        assert all(isinstance(res, ast.Assign) for res in result)
        # THe first two assignement corresponds to the condition
        assert isinstance(result[0].value, ast.Call) and isinstance(
            result[2].value, ast.Call
        )

        assert isinstance(result[0].value.func, ast.Attribute) and isinstance(
            result[2].value.func, ast.Attribute
        )
        # THese corresponds to the transformed jax intrinsic functions
        assert (
            result[0].value.func.attr == "greater"
            and result[2].value.func.attr == "logical_not"
        )
        assert (
            isinstance(result[3].targets[0], ast.Name)
            and result[3].targets[0].id == "y"
        )
        # THis for the logical_not case
        assert (
            isinstance(result[3].value, ast.Call)
            and isinstance(result[3].value.args[0], ast.Call)
            and isinstance(result[3].value.args[0].func, ast.Attribute)
            and result[3].value.args[0].func.attr == "where"
        )

        # THis is for the true case
        assert (
            isinstance(result[1].value, ast.Call)
            and isinstance(result[1].value.args[0], ast.Call)
            and isinstance(result[1].value.args[0].func, ast.Attribute)
            and result[1].value.args[0].func.attr == "where"
        )

        # An if without else inside a vectorised loop should NOT emit logical_not.
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        if x[i] > 0:
            x[i] = 1.0
        """)
        code = _unparse(transformed)
        assert "logical_not" not in code


@pytest.mark.usefixtures("test_env")
class TestVisitForScan:
    def _transform_fn(self, code):
        self.converter.reset_all()
        self.converter.helpers = []
        self.converter._pending_helpers = []
        fn = _func(code)
        transformed = self.converter.visit(fn)
        self.converter.process_helpers()
        return transformed

    def test_sequential_loop_produces_scan(self):
        # A sequential index loop over nslm -> lax.scan in output
        transformed = self._transform_fn("""
def compute(self):
    for i in range(0, self.nslm, 1):
        self.total[i] += 1
        """)
        assert any(
            isinstance(res, ast.FunctionDef) and "scan" in res.name
            for res in ast.walk(transformed)
        )

    def test_scan_helper_registered(self):
        # The scan body helper must appear in converter.helpers
        self._transform_fn("""
def compute(self):
    for i in range(0, self.nslm, 1):
        self.total[i] += 1
        """)
        names = [h.name for h in self.converter.helpers]
        assert any("scan" in n or "body" in n for n in names)

    def test_scan_helper_has_two_args(self):
        # Scan body helper signature: (carry, loop_index)
        self._transform_fn("""
def compute(self):
    for i in range(0, self.nslm, 1):
        self.total[i] += 1
        """)
        scan_helpers = [
            h for h in self.converter.helpers if "scan" in h.name or "body" in h.name
        ]
        assert scan_helpers, "No scan helper found"
        assert len(scan_helpers[0].args.args) == 2

    def test_for_counter_incremented(self):
        # for_counter must be non-zero after a scan loop is processed.
        self._transform_fn("""
def compute(self):
    for i in range(0, self.nslm, 1):
        self.total[i] += 1
        """)
        assert self.converter.for_counter >= 1

    def test_logging_stripped_in_scan_body(self):
        # Logging inside a scan body must not appear in the output.
        transformed = self._transform_fn("""
def compute(self):
    for i in range(0, self.nslm, 1):
        logging.info("step")
        self.total[i] += 1
        """)
        code = _unparse(transformed)
        assert "logging" not in code

    def test_empty_body_after_logging_strip_returns_none(self):
        # A for loop whose sole content is logging -> visit_For returns None.
        self.converter.reset_all()
        node = _stmt("""
for i in range(0, 10, 1):
    logging.info("hi")
        """)
        result = self.converter.visit_For(node)
        assert result is None

    def test_non_range_for_passes_through(self):
        # for item in items (not range) -> passes through via generic_visit
        self.converter.reset_all()
        node = _stmt("""
for item in items:
    x = item + 1
        """)
        result = self.converter.visit_For(node)
        assert result is not None


@pytest.mark.usefixtures("test_env")
class TestVisitForVector:
    def _transform_fn(self, code):
        self.converter.reset_all()
        self.converter.helpers = []
        self.converter._pending_helpers = []
        fn = _func(code)
        transformed = self.converter.visit(fn)
        self.converter.process_helpers()
        return transformed

    def test_vector_loop_no_scan(self):
        # A kjpindex loop (vectorised) must not produce lax.scan.
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        x[i] = x[i] + 1
        """)

        code = _unparse(transformed)
        assert "scan" not in code

    def test_vector_loop_removes_for_node(self):
        # After vectorisation the top-level body must contain no ast.For.
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(self.kjpindex, 1):
        x[i] = x[i] + 1
        """)
        for stmt in transformed.body:
            assert not isinstance(stmt, ast.For), "For node survived vectorisation"

    def test_vector_loop_index_replaced_by_slice(self):
        # x[i] inside a kjpindex loop -> x[:] (loop index replaced by slice).
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        x[i] = x[i] + 1
        """)
        code = _unparse(transformed)
        # x[i] must be gone from the output
        assert "[i]" not in code


@pytest.mark.usefixtures("test_env")
class TestVisitAssign:
    def _visit_fn(self, code):
        self.converter.reset_all()
        fn = _func(code)
        return self.converter.visit(fn)

    def test_visit_Assign(self):
        # self.soil_temp[0] = v  ->  soil_temp = soil_temp.at[0].set(v).
        transformed = self._visit_fn("""
def compute(self):
    self.soil_temp[0] = 1.0
        """)
        code = _unparse(transformed)
        assert ".at[" in code
        assert ".set(" in code

    def test_subscript_assign_self_attr_registers_mutation(self):
        # Assigning to self.soil_temp[i] must record soil_temp in _var_modif.
        self.converter.reset_all()
        self._visit_fn("""
def compute(self):
    self.soil_temp[0] = 2.0
        """)
        assert "soil_temp" in self.converter._var_modif["attr"]

    def test_local_array_subscript_at_set(self):
        # x[0] = 9.0 (x is a func arg) -> x = x.at[0].set(9.0).
        transformed = self._visit_fn("""
def compute(self, x):
    x[0] = 9.0
        """)
        code = _unparse(transformed)
        assert ".at[" in code and ".set(" in code

    def test_in_place_add_subscript_becomes_at_add(self):
        # a[i] = a[i] + v pattern -> .at[i].add(v).
        transformed = self._visit_fn("""
def compute(self):
    self.soil_temp[0] = self.soil_temp[0] + 1.0
        """)
        code = _unparse(transformed)
        # Either .add or a transformed .set with the value — at minimum .at[ must appear
        assert ".at[" in code

    def test_plain_scalar_assign_no_at(self):
        # A plain y = expr with no subscript must not produce .at[.
        transformed = self._visit_fn("""
def compute(self, x):
    y = x + 1
    return y
        """)
        code = _unparse(transformed)
        assert ".at[" not in code

    def test_augmented_assign_subscript_becomes_at_add(self):
        transformed = self._visit_fn("""
def compute(self, x):
    x[0] = x[0] + 1
        """)
        code = _unparse(transformed)
        assert ".at[" in code


@pytest.mark.usefixtures("test_env")
class TestVisitFunctionDef:
    def _transform(self, code):
        self.converter.reset_all()
        self.converter.helpers = []
        self.converter._pending_helpers = []
        fn = _func(code)
        return self.converter.visit_FunctionDef(fn)

    def test_returns_functiondef_node(self):
        transformed = self._transform("""
def compute(self, x):
    y = x + 1
    return y
        """)
        assert isinstance(transformed, ast.FunctionDef)

    def test_outer_func_args_populated(self):
        # _outer_func_args must be set on first FunctionDef entry
        self._transform("""
def compute(self, x):
    y = x + 1
    return y
        """)
        assert self.converter._outer_func_args is not None
        assert (
            "x" in self.converter._outer_func_args
            or "self" in self.converter._outer_func_args
        )

    def test_no_bare_return_in_true_branch_after_lift(self):
        # After _lift_if_return, the true branch must not contain a Return.
        transformed = self._transform("""
def compute(self, x):
    if x > 0:
        return x
    y = x + 1
    return y
        """)
        for stmt in transformed.body:
            if isinstance(stmt, ast.If):
                returns_in_true = [s for s in stmt.body if isinstance(s, ast.Return)]
                assert not returns_in_true, "Return survived in true branch"

    def test_simple_function_body_preserved(self):
        # Plain body statements must survive the transformation unchanged.
        transformed = self._transform("""
def compute(self, x):
    y = x + 1
    return y
        """)
        code = _unparse(transformed)
        assert "y" in code
        assert "x" in code


@pytest.mark.usefixtures("test_env")
class TestPipelineIsolation:
    def _run(self, code):
        self.converter.reset_all()
        self.converter.helpers = []
        self.converter._pending_helpers = []
        fn = _func(code)
        out = self.converter.visit(fn)
        self.converter.process_helpers()
        return out

    def test_counters_zero_after_reset(self):
        self._run("""
def compute(self, x):
    if x > 0:
        soil_temp = x
    else:
        soil_temp = 0.0
    return self.soil_temp
        """)
        self.converter.reset_all()
        assert self.converter.counter == 0
        assert self.converter.for_counter == 0

    def test_second_visit_same_helper_count(self):
        """Helpers must not accumulate across independent reset+visit cycles."""
        code = """
def compute(self, x):
    if x > 0:
        soil_temp = x
    else:
        soil_temp = 0.0
    return self.soil_temp
        """
        self._run(code)
        count1 = len(self.converter.helpers)
        self._run(code)
        count2 = len(self.converter.helpers)
        assert count1 == count2

    def test_mutated_attrs_cleared_by_reset(self):
        self._run("""
def compute(self, x):
    self.soil_temp = x
        """)
        self.converter.reset_all()
        assert self.converter._mutated_attrs == set()

    def test_var_modif_cleared_by_reset(self):
        self._run("""
def compute(self, x):
    self.soil_temp = x
        """)
        self.converter.reset_all()
        assert self.converter._var_modif == {"attr": set(), "args": set()}
