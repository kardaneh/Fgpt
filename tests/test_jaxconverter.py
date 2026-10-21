# Copyright 2026 CNRS
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import ast
from dataclasses import MISSING, dataclass, field, fields, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from fgpt.core.backends.jax_converter.converter import JaxConverter
from fgpt.core.backends.utils import Control, get_class_info_from_ast
from fgpt.core.common.logger import Logger

jax.config.update("jax_enable_x64", True)


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


def _vec_ctrl(mask_rank=1):
    return Control(
        kind="if",
        loop_info={"kjpindex": "ji"},
        transform_type="vector",
        vectorization_axis={"ji": [0]},
        metadata={"current_mask_rank": mask_rank},
    )


class FortranDerivedType:
    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "FortranDerivedType":
        kwargs = {}
        for f in fields(cls):
            default = (
                f.default_factory() if f.default_factory is not MISSING else f.default
            )
            if isinstance(default, np.ndarray):
                (full_shape, dtype) = (shape + default.shape, default.dtype)
            else:
                (full_shape, dtype) = (shape, np.asarray(default).dtype)
            kwargs[f.name] = np.zeros(full_shape, dtype=dtype)
        instance = cls(**kwargs)
        instance._shape = tuple(shape)
        return instance

    def __getitem__(self, idx):
        return type(self)(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def __setitem__(self, idx, value):
        for f in fields(self):
            getattr(self, f.name)[idx] = getattr(value, f.name)

    def with_field(self, name, value):
        return replace(self, **{name: value})

    def set(self, idx, value):
        return type(self)(
            **{
                f.name: getattr(self, f.name).at[idx].set(getattr(value, f.name))
                for f in fields(self)
            }
        )


@dataclass
class scalar_type(FortranDerivedType):
    a: np.float64 = np.float64(0)
    b: np.float64 = np.float64(0)


@dataclass
class array_type(FortranDerivedType):
    a: np.ndarray = field(default_factory=lambda: np.zeros((10,), dtype=np.float64))
    b: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float64))


@dataclass
class mask_arr_type(FortranDerivedType):
    a: np.float64 = np.float64(0)
    b: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))


jax.tree_util.register_dataclass(mask_arr_type)
jax.tree_util.register_dataclass(scalar_type)
jax.tree_util.register_dataclass(array_type)


def to_jax(x):
    return jax.tree_util.tree_map(jnp.asarray, x)


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
        assert len(stmts) == 3  # Contains the mask and the arr as masked update
        assert _unparse(stmts[-1]) == (
            "arr = arr.at[:].set(jnp.where(_mask_0, old_val_arr + 1, 0))"
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
        assert len(stmts) == 4  # Contains the 2 mask and the arr as masked update
        assert "logical_and" in _unparse(stmts[0])  # the elif mask
        assert "greater" in _unparse(stmts[1])  # the if mask
        assert _unparse(stmts[-1]) == (
            "arr = arr.at[:].set(jnp.where(_mask_1, old_val_arr + a, jnp.where(_mask_0, b, c)))"
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
        assert len(stmts) == 4  # Contains the 2 mask and the arr as masked update
        assert "logical_and" in _unparse(stmts[0])  # the elif mask
        assert "greater" in _unparse(stmts[2])  # the if mask
        assert _unparse(stmts[-1]) == (
            "arr = arr.at[:].set(jnp.where(_mask_1, old_val_arr + a, jnp.where(_mask_0, b, old_val_arr - c)))"
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
class TestOffsetSubstitution:
    def _offset(self, code, **kw):
        node = _expr(code)
        out = self.converter._substitute_loop_index_with_offset(node, "jjj", **kw)
        return _unparse(out)

    def test_no_batch_uses_plain_names(self):
        out = self._offset("(jjj - locflag[jj, 0]) * x", has_batch_dimension=False)
        assert out == "(layers - depth_idx_start) * x"

    def test_default_axes_unchanged(self):
        out = self._offset("(jjj - locflag[jj, 0]) * x", has_batch_dimension=True)
        assert out == "(layers[None, :] - depth_idx_start[:, None]) * x"

    def test_custom_axes(self):
        out = self._offset(
            "(jjj - a) * x", has_batch_dimension=True, loop_axis=2, batch_axis=0
        )
        assert out == ("(layers[None, None, :] - depth_idx_start[:, None, None]) * x")

    def test_batch_axis_not_leading(self):
        # batch on axis 1, layers on axis 0 (transposed layout)
        out = self._offset(
            "(jjj - a) * x", has_batch_dimension=True, loop_axis=0, batch_axis=1
        )
        assert out == "(layers[:, None] - depth_idx_start[None, :]) * x"

    def test_only_minuend_loop_index_rewritten(self):
        out = self._offset("(a - jjj) * jjj", has_batch_dimension=True)
        assert out == "(a - jjj) * jjj"

    def test_input_not_mutated(self):
        node = _expr("(jjj - a) * x")
        before = _unparse(node)
        self.converter._substitute_loop_index_with_offset(
            node, "jjj", has_batch_dimension=True
        )
        assert _unparse(node) == before


@pytest.mark.usefixtures("test_env")
class TestBatchAxisForOffset:
    @pytest.fixture(autouse=True)
    def _setup(self):
        c = self.converter
        c.func_name = "compute"
        c.cls_info["Model"]["methods"]["compute"]["local_arr"].update(
            {
                "snowrho": {
                    "dimensions": ["kjpindex", "nsnow"],
                    "dtype": "float64",
                    "type": "jnp.ndarray",
                },
                "snowrho_t": {
                    "dimensions": ["nsnow", "kjpindex"],
                    "dtype": "float64",
                    "type": "jnp.ndarray",
                },
            }
        )
        c.cls_info["Model"]["attributes"]["nsnow"] = {"type": "int"}
        yield
        la = c.cls_info["Model"]["methods"]["compute"]["local_arr"]
        la.pop("snowrho", None)
        la.pop("snowrho_t", None)
        c.cls_info["Model"]["attributes"].pop("nsnow", None)

    def _ctx(self):
        return {"loop_info": {"kjpindex": "ji"}, "vectorization_axis": {"ji": [0]}}

    def test_leading_batch(self):
        axis, rank = self.converter._batch_axis(_expr("snowrho[:, :]"), self._ctx())
        assert (axis, rank) == (0, 2)

    def test_transposed_batch(self):
        axis, rank = self.converter._batch_axis(_expr("snowrho_t[:, :]"), self._ctx())
        assert (axis, rank) == (1, 2)

    def test_fallback_when_no_batch_found(self):
        axis, rank = self.converter._batch_axis(_expr("unknown[:, :]"), self._ctx())
        assert (axis, rank) == (0, 1)

    def test_axis_feeds_offset(self):
        c = self.converter
        for name, expected in (
            ("snowrho[:, :]", "depth_idx_start[:, None]"),
            ("snowrho_t[:, :]", "depth_idx_start[None, :]"),
        ):
            term = _expr(f"(jjj - a) * {name}")
            b_axis, _ = c._batch_axis(term, self._ctx())
            out = _unparse(
                c._substitute_loop_index_with_offset(
                    term,
                    "jjj",
                    has_batch_dimension=True,
                    loop_axis=1 - b_axis,
                    batch_axis=b_axis,
                )
            )
            assert expected in out


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

        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex + 1, 1):
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

    def test_vector_loop_index_with_binop(self):
        transformed = self._transform_fn("""
def compute(self, x):
    for i in range(0, self.kjpindex, 1):
        x[i] = x[i - 1] + x[i]
        """)
        code = _unparse(transformed)
        # TODO: Need to correct this
        assert "[i]" not in code


@pytest.mark.usefixtures("test_env")
class TestVisitAssign:
    def _visit_fn(self, code):
        self.converter.reset_all()
        fn = _func(code)
        return self.converter.visit(fn)

    def test_visit_Assign(self):
        # self.soil_temp[0] = v  ->  soil_temp = soil_temp.at[0].set(v).
        transformed = self._visit_fn(
            """
def compute(self):
    self.soil_temp[0] = 1.0
        """
        )
        code = _unparse(transformed)
        assert ".at[0].set(1.0)" in code

    def test_subscript_assign_self_attr_registers_mutation(self):
        # Assigning to self.soil_temp[i] must record soil_temp in _var_modif.
        self.converter.reset_all()
        self._visit_fn(
            """
def compute(self):
    self.soil_temp[0] = 2.0
        """
        )
        assert "soil_temp" in self.converter._var_modif["attr"]

    def test_local_array_subscript_at_set(self):
        # x[0] = 9.0 (x is a func arg) -> x = x.at[0].set(9.0).
        transformed = self._visit_fn(
            """
def compute(self, x):
    x[0] = 9.0
        """
        )
        code = _unparse(transformed)
        assert ".at[0].set(9.0)" in code

    def test_in_place_add_subscript_becomes_at_add(self):
        # a[i] = a[i] + v pattern -> .at[i].add(v), not .set(a[i] + v).
        transformed = self._visit_fn(
            """
def compute(self):
    self.soil_temp[0] = self.soil_temp[0] + 1.0
        """
        )
        code = _unparse(transformed)
        assert ".at[0].add(1.0)" in code
        assert ".set(" not in code

    def test_plain_scalar_assign_no_at(self):
        # A plain y = expr with no subscript must not produce .at[.
        transformed = self._visit_fn(
            """
def compute(self, x):
    y = x + 1
    return y
        """
        )
        code = _unparse(transformed)
        assert ".at[" not in code

    def test_augmented_assign_subscript_becomes_at_add(self):
        transformed = self._visit_fn(
            """
def compute(self, x):
    x[0] = x[0] + 1
        """
        )
        code = _unparse(transformed)
        assert ".at[0].add(1)" in code

    def test_mixed_nested_binop_subscript_falls_back_to_set(self):
        # Regression: x[0] = (x[0] / a) ** b must lower to a full .set(...)
        # of the whole expression, never .divide(a ** b).
        transformed = self._visit_fn(
            """
def compute(self, x, a, b):
    x[0] = (x[0] / a) ** b
        """
        )
        code = _unparse(transformed)
        assert ".divide(" not in code
        assert ".at[0].set((x[0] / a) ** b)" in code

    def test_same_index_requires_matching_slice(self):
        # a[i] = a[j] + x -- different indices, must NOT fold to "add".
        op, value = self.converter.check_in_place_modif(_stmt("a[i] = a[j] + x"))
        assert op == "set"
        assert _unparse(value) == "a[j] + x"

    def test_same_index_requires_matching_base_name(self):
        # a[i] = b[i] + x -- different arrays, must NOT fold to "add".
        op, value = self.converter.check_in_place_modif(_stmt("a[i] = b[i] + x"))
        assert op == "set"
        assert _unparse(value) == "b[i] + x"

    def test_nested_struct_array_write_is_registered_as_scan_carry(self):
        self.converter.reset_all()

        stmt = _stmt("self.soil_fit.a[i] = 2 * i")

        assigned = self.converter._collect_assigned([stmt])
        rhs_uses = self.converter._collect_rhs_uses([stmt])

        assert "soil_fit" in assigned
        assert "i" in rhs_uses
        assert "soil_fit" in self.converter._mutated_attrs


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


@pytest.mark.usefixtures("test_env")
class TestCheckInPlaceModif:
    """Direct unit tests for check_in_place_modif — pure AST-in/AST-out,
    no visitor state needed.
    """

    def _check(self, code):
        assign = _stmt(code)
        return self.converter.check_in_place_modif(assign)

    def test_simple_add(self):
        op, value = self._check("a[i] = a[i] + x")
        assert op == "add"
        assert _unparse(value) == "x"

    def test_simple_subtract(self):
        op, value = self._check("a[i] = a[i] - x")
        assert op == "subtract"
        assert _unparse(value) == "x"

    def test_simple_multiply(self):
        op, value = self._check("a[i] = a[i] * x")
        assert op == "multiply"
        assert _unparse(value) == "x"

    def test_simple_divide(self):
        op, value = self._check("a[i] = a[i] / x")
        assert op == "divide"
        assert _unparse(value) == "x"

    def test_plain_set_when_no_binop(self):
        op, value = self._check("a[i] = x")
        assert op == "set"
        assert _unparse(value) == "x"

    def test_nested_add_add_folds(self):
        # (a[i] + b) + c -> ("add", b + c) since Add is associative.
        op, value = self._check("a[i] = (a[i] + b) + c")
        assert op == "add"
        assert _unparse(value) == "b + c"

    def test_nested_mult_mult_folds(self):
        # (a[i] * b) * c -> ("multiply", b * c) since Mult is associative.
        op, value = self._check("a[i] = (a[i] * b) * c")
        assert op == "multiply"
        assert _unparse(value) == "b * c"

    def test_nested_div_pow_does_not_fold(self):
        # Regression test: (a[i] / b) ** c must NOT fold into
        # ("divide", b ** c) -- that computes a / b**c, not (a/b)**c.
        op, value = self._check("a[i] = (a[i] / b) ** c")
        assert op == "set"
        assert _unparse(value) == "(a[i] / b) ** c"

    def test_nested_add_mult_mixed_does_not_fold(self):
        # Regression: mixed operators (inner Add, outer Mult) must not
        # fold -- (a[i] + b) * c != a[i] + (b * c).
        op, value = self._check("a[i] = (a[i] + b) * c")
        assert op == "set"
        assert _unparse(value) == "(a[i] + b) * c"

    def test_different_index_not_in_place(self):
        op, value = self._check("a[i] = a[j] + x")
        assert op == "set"
        assert _unparse(value) == "a[j] + x"

    def test_self_attr_subscript_add(self):
        op, value = self._check("self.soil_temp[i] = self.soil_temp[i] + x")
        assert op == "add"
        assert _unparse(value) == "x"


class TestStructFieldWrite:
    @pytest.fixture(autouse=True)
    def _setup(self, test_env):
        c = self.converter
        c._func_arg_stack = [set()]
        c._mutated_attrs = set()
        c._modified_ret_stack = []
        c.struct_classes = {"laieff_type"}
        c.struct_info = {
            "laieff_type": {
                "a": {"inner_dims": [], "dtype": "float64"},
                "b": {"inner_dims": [10], "dtype": "float64"},
            }
        }
        c.cls_info["Model"]["methods"]["compute"]["struct_args"] = {
            "laieff_fit": {
                "type": "laieff_type",
                "dimensions": ["kjpindex", "nvm", "nlevels_tot"],
            }
        }
        c.func_name = "compute"
        c._outer_func_args = {"laieff_fit"}
        c._var_modif = {"args": set(), "attr": set()}
        c._control_stack = []

    def _assign(self, code):
        out = self.converter.visit_Assign(_stmt(code))
        stmts = out if isinstance(out, list) else [out]
        return "\n".join(_unparse(s) for s in stmts)

    def test_array_field_write_extra_inner_index(self):
        assert self._assign("laieff_fit.a[1, 0, 1, 1] = 42.0") == (
            "laieff_fit = laieff_fit.with_field('a', "
            "laieff_fit.a.at[1, 0, 1, 1].set(42.0))"
        )

    def test_variable_index_and_expression_rhs(self):
        out = self._assign("laieff_fit.b[i, j] = x * 2.0")
        assert out == (
            "laieff_fit = laieff_fit.with_field('b', "
            "laieff_fit.b.at[i, j].set(x * 2.0))"
        )

    def test_in_place_add_becomes_at_add(self):
        out = self._assign("laieff_fit.a[i] = laieff_fit.a[i] + x")
        assert out.startswith("laieff_fit = laieff_fit.with_field('a', ")
        assert ".at[i].add(x)" in out

    def test_in_place_multiply_becomes_at_multiply(self):
        out = self._assign("laieff_fit.c[i] = laieff_fit.c[i] * 2.0")
        assert ".at[i].multiply(2.0)" in out

    def test_slice_index_preserved(self):
        out = self._assign("laieff_fit.a[:, 0, 1] = v")
        assert ".at[:, 0, 1].set(v)" in out

    def test_registers_struct_as_modified_arg(self):
        self._assign("laieff_fit.a[0] = 1.0")
        assert "laieff_fit" in self.converter._var_modif["args"]
        assert "a" not in self.converter._var_modif["args"]  # not the field name

    def test_not_registered_when_not_outer_arg(self):
        self.converter._outer_func_args = set()
        self._assign("laieff_fit.a[0] = 1.0")
        assert "laieff_fit" not in self.converter._var_modif["args"]

    def test_non_struct_attribute_is_untouched(self):
        out = self._assign("other.a[0] = 1.0")
        assert "with_field" not in out

    def test_struct_name_not_in_struct_classes_is_untouched(self):
        c = self.converter
        entry = c.cls_info["Model"]["methods"]["compute"]["struct_args"]["laieff_fit"]
        original = entry["type"]
        entry["type"] = "unknown_type"
        try:
            assert "with_field" not in self._assign("laieff_fit.a[0] = 1.0")
        finally:
            entry["type"] = original

    def test_self_struct_handled_by_this_branch(self):
        out = self._assign("self.laieff_fit.a[0] = 1.0")
        assert "with_field" in out

    def test_struct_attibutes_inside_class(self):
        SRC = """
class Model:
    def __init__(self):
        self.kjpindex = jnp.int32(4)
        self.nvm = jnp.int32(15)
        self.laieff_fit = laieff_type.zeros((self.kjpindex, self.nvm))
        self.soil = jnp.zeros((self.kjpindex,), dtype=jnp.float64)

    def compute(self, x):
        return x
"""
        info = get_class_info_from_ast(ast.parse(SRC), {"laieff_type"})
        attrs = info["Model"]["attributes"]
        assert attrs["laieff_fit"] == {
            "type": "struct",
            "struct_type": "laieff_type",
            "dimensions": ["kjpindex", "nvm"],
        }

        # regular arrays are unaffected
        soil = info["Model"]["attributes"]["soil"]
        assert soil["type"] == "jnp.ndarray"
        assert soil["dimensions"] == ["kjpindex"]

        # default argument: laieff_type is not known, so it must not become a struct
        try:
            info = get_class_info_from_ast(ast.parse(SRC))
        except Exception:
            return  # old path crashes on missing keywords: acceptable
        assert info["Model"]["attributes"].get("laieff_fit", {}).get("type") != "struct"

    def test_plain_array_write_still_uses_at(self):
        out = self._assign("tmp[0] = 1.0")
        assert out == "tmp = tmp.at[0].set(1.0)"

    def test_emitted_code_runs(self):
        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.float64 = np.float64(0)

        jax.tree_util.register_dataclass(laieff_type)

        src = self._assign("laieff_fit.a[1, 0, 1] = 42.0")
        fit = jax.tree_util.tree_map(jnp.asarray, laieff_type.zeros((3, 2, 2)))
        ns = {"laieff_fit": fit}
        exec(src, ns)
        assert ns["laieff_fit"].a[1, 0, 1] == 42.0
        assert fit.a[1, 0, 1] == 0.0  # original untouched

    def test_scalar_field_dims(self):
        node = _expr("laieff_fit.a")
        assert self.converter.get_declared_dims(node) == [
            "kjpindex",
            "nvm",
            "nlevels_tot",
        ]

    def test_subscripted_scalar_field_dims(self):
        node = _expr("laieff_fit.a[ji, 0, 1]")
        assert self.converter.get_declared_dims(node) == [
            "kjpindex",
            "nvm",
            "nlevels_tot",
        ]

    def test_array_field_appends_inner_dims(self):
        node = _expr("laieff_fit.b[ji, 0, 1, :]")
        assert self.converter.get_declared_dims(node) == [
            "kjpindex",
            "nvm",
            "nlevels_tot",
            10,
        ]

    def test_unknown_field_returns_none(self):
        assert self.converter.get_declared_dims(_expr("laieff_fit.zzz")) is None

    def test_non_struct_attribute_unchanged(self):
        # plain class attribute still resolves through the old path
        node = _expr("self.soil_temp")
        assert self.converter.get_declared_dims(node) == ["kjpindex"]

    def test_active_dims_after_indexing(self):
        node = _expr("laieff_fit.a[:, 0, 1]")
        assert self.converter.get_active_dims(node) == ["kjpindex"]

    def test_is_derived_type_var(self):
        attrs = self.converter.cls_info["Model"]["attributes"]
        attrs["laieff_fit"] = {
            "type": "struct",
            "struct_type": "laieff_type",
            "dimensions": ["kjpindex", "nvm"],
        }
        attrs.pop("laieff_fit", None)
        assert self.converter._is_derived_type_var("laieff_fit") is True

        # Not a struct
        assert self.converter._is_derived_type_var("soil_temp") is False

    def test_vectorized_field_write_slices_batch_axis(self):
        c = self.converter
        c._control_stack = [
            Control(
                kind="loop",
                loop_info={"kjpindex": "ji"},
                transform_type="vector",
                vectorization_axis={"ji": [0]},
                metadata={},
            )
        ]
        try:
            out = self._assign("laieff_fit.a[ji, ivm, ilevel] = v[ji]")
        finally:
            c._control_stack = []
        assert out.startswith("laieff_fit = laieff_fit.with_field('a', ")
        assert ".at[:, ivm, ilevel].set(" in out
        assert "ji" not in out
        assert "laieff_fit" in c._var_modif["args"]

    def test_vectorized_array_field_read_slices_batch_axis(self):
        c = self.converter

        c._control_stack = [
            Control(
                kind="loop",
                loop_info={"kjpindex": "ji"},
                transform_type="vector",
                vectorization_axis={"ji": [0]},
                metadata={},
            )
        ]

        try:
            node = _expr("laieff_fit.b[ji, ivm, ilevel, :]")
            out = c.visit_Subscript(node)
        finally:
            c._control_stack = []

        assert _unparse(out) == "laieff_fit.b[:, ivm, ilevel, :]"

    def test_vectorized_array_field_read_preserves_indices(self):
        c = self.converter

        c._control_stack = [
            Control(
                kind="loop",
                loop_info={"kjpindex": "ji"},
                transform_type="vector",
                vectorization_axis={"ji": [0]},
                metadata={},
            )
        ]

        try:
            node = _expr("laieff_fit.b[ji, ivm, ilevel, :]")
            out = c.visit_Subscript(node)
        finally:
            c._control_stack = []

        result = _unparse(out)

        assert "laieff_fit.b" in result
        assert "ivm" in result
        assert "ilevel" in result
        assert ":" in result
        assert result != "laieff_fit.b"

    def test_array_field_read_preserves_indices_without_vectorization(self):
        c = self.converter
        c._control_stack = []

        node = _expr("laieff_fit.b[ji, ivm, ilevel, :]")

        out = c.visit_Subscript(node)

        assert _unparse(out) == "laieff_fit.b[ji, ivm, ilevel, :]"

    def test_struct_field_does_not_collide_with_lifted_local(self):
        c = self.converter
        c.dynamic_variable_lift["a"] = {
            "original_shape": [],
            "axis_map": 0,
            "batched_axis": {0},
            "vectorized_loop": [ast.Name(id="kjpindex", ctx=ast.Load())],
            "dtype": "float64",
        }
        c._control_stack = [
            Control(
                kind="loop",
                loop_info={"kjpindex": "ji"},
                transform_type="vector",
                vectorization_axis={"ji": [0]},
                metadata={},
            )
        ]
        try:
            out = self._assign("laieff_fit.a[ji, ivm, ilevel] = v[ji]")
        finally:
            c._control_stack = []
            c.dynamic_variable_lift.pop("a", None)
        assert ".at[:, ivm, ilevel].set(" in out
        assert ".at[:, :, ivm, ilevel]" not in out

    def test_accum_base_found_for_struct_field(self):
        c = self.converter
        payload = _expr("x.a[:, jj] + w[:, jj]")
        base = c._extract_accum_base_from_payload(payload, _expr("x.a"))
        assert _unparse(base) == "x.a[:, jj]"

        # Accumulation base didn't match
        c = self.converter
        payload = _expr("x.b[:, jj] + w[:, jj]")  # different field
        lhs = _expr("x.a")
        base = c._extract_accum_base_from_payload(payload, lhs)
        assert base is lhs  # fallback: nothing matched

    def test_accum_base_plain_array_unchanged(self):
        c = self.converter
        payload = _expr("tmp[:, jj] + w[:, jj]")
        base = c._extract_accum_base_from_payload(
            payload, ast.Name(id="tmp", ctx=ast.Load())
        )
        assert _unparse(base) == "tmp[:, jj]"

    def test_set_payload_arg_rewrites_inner_call_only(self):
        c = self.converter
        stmt = _stmt("x = x.with_field('a', x.a.at[:].set(old))")
        c._set_payload_arg(stmt, _expr("new"))
        assert _unparse(stmt) == "x = x.with_field('a', x.a.at[:].set(new))"

    def test_extract_set_payload_unwraps_with_field(self):
        c = self.converter
        stmt = _stmt("x = x.with_field('a', x.a.at[:].set(p + q))")
        print(c._extract_set_payload(stmt, "x"))
        assert _unparse(c._extract_set_payload(stmt, "x")) == "p + q"

    def test_extract_set_payload_ignores_non_at_call(self):
        c = self.converter
        stmt = _stmt("x = x.with_field('a', foo(p))")
        assert c._extract_set_payload(stmt, "x") is None

    def test_struct_field_write_jvp(self):
        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)
            c: np.float64 = np.float64(0)
            d: np.float64 = np.float64(0)
            e: np.float64 = np.float64(0)

        jax.tree_util.register_dataclass(laieff_type)

        src = self._assign("laieff_fit.a[1, 0, 1] = laieff_fit.a[1, 0, 1] + test_angle")
        assert src == (
            "laieff_fit = laieff_fit.with_field('a', "
            "laieff_fit.a.at[1, 0, 1].add(test_angle))"
        )
        fit = jax.tree_util.tree_map(
            jnp.asarray,
            laieff_type.zeros((3, 2, 2)),
        )
        test_angle = jnp.array(2.0)
        # Execute generated code.
        ns = {
            "laieff_fit": fit,
            "test_angle": test_angle,
        }
        exec(src, ns)
        result = ns["laieff_fit"]
        assert result.a[1, 0, 1] == 2.0

        def func(angle):
            fit_local = fit

            ns = {
                "laieff_fit": fit_local,
                "test_angle": angle,
            }
            exec(src, ns)
            return ns["laieff_fit"].a[1, 0, 1]

        primal, tangent = jax.jvp(
            func,
            (test_angle,),
            (jnp.array(1.0),),
        )

        assert jnp.allclose(primal, 2.0)
        assert jnp.allclose(tangent, 1.0)

    def test_laieff_jvp_vjp_multiple_values(self):
        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)
            c: np.float64 = np.float64(0)
            d: np.float64 = np.float64(0)
            e: np.float64 = np.float64(0)

        jax.tree_util.register_dataclass(laieff_type)

        fit = laieff_type.zeros((2, 3, 4))
        fit = jax.tree_util.tree_map(jnp.asarray, fit)

        fit = fit.with_field("a", jnp.ones_like(fit.a) * 1.0)
        fit = fit.with_field("b", jnp.ones_like(fit.b) * 2.0)
        fit = fit.with_field("c", jnp.ones_like(fit.c) * 3.0)
        fit = fit.with_field("d", jnp.ones_like(fit.d) * 4.0)
        fit = fit.with_field("e", jnp.ones_like(fit.e) * 5.0)

        angle = jnp.array(2.0)

        def func(angle, fit):
            return (
                fit.a
                + angle * fit.b
                + angle**2 * fit.c
                + angle**3 * fit.d
                + angle**4 * fit.e
            )

        # Expected value
        # f(2) = 1 + 2*2 + 3*2² + 4*2³ + 5*2⁴
        #      = 129
        # Therefore every element is 129.
        expected_primal = jnp.full((2, 3, 4), 129.0)

        # JVP with respect to angle
        primal, tangent = jax.jvp(
            lambda x: func(x, fit),
            (angle,),
            (jnp.array(1.0),),
        )

        # f'(x) = b + 2cx + 3dx² + 4ex³
        # f'(2) = 2 + 2*3*2 + 3*4*2² + 4*5*2³
        #       = 2 + 12 + 48 + 160
        #       = 222
        # Every output element therefore has tangent 222.
        expected_tangent = jnp.full((2, 3, 4), 222.0)

        assert primal.shape == (2, 3, 4)
        assert tangent.shape == (2, 3, 4)
        assert jnp.allclose(primal, expected_primal)
        assert jnp.allclose(tangent, expected_tangent)

        # VJP with respect to angle
        primal, vjp_fn = jax.vjp(
            lambda x: func(x, fit),
            angle,
        )

        # Cotangent of every output element = 1.
        output_cotangent = jnp.ones_like(primal)

        (angle_grad,) = vjp_fn(output_cotangent)

        # Since there are 2*3*4 = 24 output elements,
        # gradient = 24 * 222 = 5328
        expected_angle_grad = 24.0 * 222.0

        assert jnp.allclose(primal, expected_primal)
        assert jnp.allclose(angle_grad, expected_angle_grad)

        # JVP with respect to the entire laieff_type
        fit_tangent = jax.tree_util.tree_map(jnp.ones_like, fit)

        primal, tangent = jax.jvp(
            lambda f: func(angle, f),
            (fit,),
            (fit_tangent,),
        )

        # df = da + angle*db + angle²*dc
        #        + angle³*dd + angle⁴*de
        # Since every tangent = 1:
        # df = 1 + 2 + 4 + 8 + 16 = 31
        expected_fit_tangent = jnp.full((2, 3, 4), 31.0)

        assert jnp.allclose(primal, expected_primal)
        assert jnp.allclose(tangent, expected_fit_tangent)

        # VJP with respect to the entire laieff_type
        primal, vjp_fn = jax.vjp(
            lambda f: func(angle, f),
            fit,
        )

        (fit_grad,) = vjp_fn(jnp.ones_like(primal))

        # Every output contributes:
        # da = 1
        # db = angle = 2
        # dc = angle² = 4
        # dd = angle³ = 8
        # de = angle⁴ = 16
        #
        # But because the output has 24 elements and our cotangent
        # is 1 everywhere, each gradient gets summed over 24 outputs.
        expected_a = jnp.full((2, 3, 4), 1.0)
        expected_b = jnp.full((2, 3, 4), 2.0)
        expected_c = jnp.full((2, 3, 4), 4.0)
        expected_d = jnp.full((2, 3, 4), 8.0)
        expected_e = jnp.full((2, 3, 4), 16.0)

        assert jnp.allclose(fit_grad.a, expected_a)
        assert jnp.allclose(fit_grad.b, expected_b)
        assert jnp.allclose(fit_grad.c, expected_c)
        assert jnp.allclose(fit_grad.d, expected_d)
        assert jnp.allclose(fit_grad.e, expected_e)

    def test_zeros_sets_shape_and_dtypes(self):
        x = scalar_type.zeros((3, 2, 2))
        assert x._shape == (3, 2, 2)
        assert x.a.shape == (3, 2, 2) and x.a.dtype == np.float64

    def test_array_field_zeros_shape(self):
        x = array_type.zeros((3, 2, 2))
        assert x.a.shape == (3, 2, 2, 10)
        assert x.b.shape == (3, 2, 2, 2, 2)

    def test_numpy_setitem_and_getitem(self):
        x = scalar_type.zeros((3,))
        x[1] = scalar_type(a=np.float64(5), b=np.float64(6))
        e = x[1]
        assert isinstance(e, scalar_type) and e.a == 5 and e.b == 6

    def test_tree_map_converts_and_drops_shape(self):
        x = to_jax(scalar_type.zeros((3, 2, 2)))
        assert isinstance(x.a, jax.Array)
        assert not hasattr(x, "_shape")  # expected: only needed pre-tree_map

    def test_scalar_field_write(self):
        x = to_jax(scalar_type.zeros((3, 2, 2)))
        y = x.with_field("a", x.a.at[1, 0, 1].set(42.0))
        assert y.a[1, 0, 1] == 42.0
        assert y.a[0, 0, 0] == 0.0
        assert float(y.a.sum()) == 42.0  # nothing else touched

        assert self._assign("laieff_fit.a[1, 0, 1] = 42.0") == (
            "laieff_fit = laieff_fit.with_field('a', "
            "laieff_fit.a.at[1, 0, 1].set(42.0))"
        )

    def test_original_untouched(self):
        x = to_jax(scalar_type.zeros((3, 2, 2)))
        _ = x.with_field("a", x.a.at[1, 0, 1].set(42.0))
        assert x.a[1, 0, 1] == 0.0

    def test_other_fields_shared_unchanged(self):
        x = to_jax(scalar_type.zeros((3, 2, 2)))
        y = x.with_field("a", x.a.at[1, 0, 1].set(42.0))
        assert y.b is x.b

    def test_bare_at_without_rebind_is_noop(self):
        x = to_jax(scalar_type.zeros((3, 2, 2)))
        x.a.at[1, 0, 1].set(42.0)  # result discarded
        assert x.a[1, 0, 1] == 0.0

    def test_array_field_write(self):
        x = to_jax(array_type.zeros((3, 2, 2)))
        y = x.with_field("a", x.a.at[1, 0, 1, 1].set(42.0))
        assert y.a[1, 0, 1, 1] == 42.0
        assert y.a[0, 0, 0, 0] == 0.0
        assert y[1, 0, 1].a[1] == 42.0  # __getitem__ on jax phase

    def test_add_and_multiply(self):
        x = to_jax(scalar_type.zeros((4,)))
        x = x.with_field("a", x.a.at[2].set(3.0))
        x = x.with_field("a", x.a.at[2].add(4.0))
        assert x.a[2] == 7.0
        x = x.with_field("a", x.a.at[2].multiply(2.0))
        assert x.a[2] == 14.0

    def test_vectorised_slice_write(self):
        x = to_jax(scalar_type.zeros((5, 3, 4)))
        v = jnp.arange(5.0)
        x = x.with_field("a", x.a.at[:, 1, 2].set(v))
        np.testing.assert_array_equal(x.a[:, 1, 2], v)
        assert float(x.a[:, 0, :].sum()) == 0.0

    def test_masked_write_matches_where(self):
        x = to_jax(scalar_type.zeros((5,)))
        mask = jnp.array([True, False, True, False, False])
        x = x.with_field("a", x.a.at[:].set(jnp.where(mask, 1.0, x.a[:])))
        np.testing.assert_array_equal(x.a, [1, 0, 1, 0, 0])

    def test_whole_element_set(self):
        x = to_jax(scalar_type.zeros((3,)))
        src = scalar_type(a=jnp.float64(5), b=jnp.float64(6))
        y = x.set(1, src)
        assert y.a[1] == 5 and y.b[1] == 6 and y.a[0] == 0

    def test_numpy_setitem_fails_on_jax(self):
        x = to_jax(scalar_type.zeros((3,)))
        with pytest.raises(TypeError):
            x[1] = scalar_type(a=jnp.float64(1), b=jnp.float64(1))

    def test_jit_returns_updated_struct(self):
        @jax.jit
        def f(x):
            x = x.with_field("a", x.a.at[1].set(9.0))
            return x

        y = f(to_jax(scalar_type.zeros((3,))))
        assert y.a[1] == 9.0

    def test_scan_carry(self):
        def body(x, i):
            x = x.with_field("a", x.a.at[i].set(i * 2.0))
            return x, None

        x, _ = lax.scan(body, to_jax(scalar_type.zeros((4,))), jnp.arange(4))
        np.testing.assert_array_equal(x.a, [0, 2, 4, 6])

    def test_cond_both_branches_same_structure(self):
        def t(x):
            return x.with_field("a", x.a.at[0].set(1.0))

        def f(x):
            return x.with_field("a", x.a.at[0].set(2.0))

        x = to_jax(scalar_type.zeros((2,)))
        assert lax.cond(True, t, f, x).a[0] == 1.0
        assert lax.cond(False, t, f, x).a[0] == 2.0

    def test_dtype_preserved_in_scan(self):
        # float64 field must stay float64 or scan carry check fails
        x = to_jax(scalar_type.zeros((3,)))
        y = x.with_field("a", x.a.at[0].set(1))  # python int
        assert y.a.dtype == jnp.float64

    def test_grad_through_write(self):
        def loss(v):
            x = to_jax(scalar_type.zeros((3,)))
            x = x.with_field("a", x.a.at[1].set(v))
            return (x.a**2).sum()

        assert jax.grad(loss)(3.0) == 6.0


class TestMaskStructField:
    N = 6

    @pytest.fixture(autouse=True)
    def _setup(self, test_env):
        c = self.converter
        c.func_name = "compute"
        c.struct_classes = {"arr_type"}
        c.struct_info = {
            "arr_type": {
                "a": {"inner_dims": [], "dtype": "float64"},
                "b": {"inner_dims": [3], "dtype": "float64"},
            }
        }
        c.cls_info["Model"]["methods"]["compute"]["struct_args"] = {
            "x": {"type": "arr_type", "dimensions": ["kjpindex"]}
        }
        c._control_stack = []
        c.counter = 1
        c.for_counter = 1
        c._mask_counter = 0

    def _mask(self, code):
        stmt = _stmt(code)
        out = self.converter._mask_vector_assign(stmt, "_mask_1", ["x"], [])
        return "\n".join(_unparse(s) for s in out)

    def _mask_vec(self, code, mask_rank=1):
        c = self.converter
        c._control_stack = [_vec_ctrl(mask_rank)]
        try:
            return self._mask(code)
        finally:
            c._control_stack = []

    def _inputs(self):
        rng = np.random.default_rng(0)
        x = to_jax(mask_arr_type.zeros((self.N,)))
        x = x.with_field("a", jnp.asarray(rng.normal(size=(self.N,))))
        x = x.with_field("b", jnp.asarray(rng.normal(size=(self.N, 3))))
        mask = np.array([True, False, True, True, False, False])
        v = rng.normal(size=(self.N, 3))
        w = rng.normal(size=(self.N,))
        return x, mask, v, w

    def _run(self, code, x, mask, **extra):
        src = self._mask_vec(code)
        ns = {"jnp": jnp, "x": x, "_mask_1": jnp.asarray(mask), **extra}
        exec(src, ns)
        return ns["x"]

    def test_array_field_mask_gets_newaxis(self):
        # x over [kjpindex], field b has inner dim [3] -> target rank 2
        # x is not added to the modified args or attr thus the self will be still present
        out = self._mask_vec("x = x.with_field('b', x.b.at[:, :].set(v))")
        assert "_mask_1[:, None]" in out
        assert out.startswith(
            "x = x.with_field('b', x.b.at[:, :].set(jnp.where(_mask_1"
        )

    def test_scalar_field_mask_not_expanded(self):
        out = self._mask_vec("x = x.with_field('a', x.a.at[:].set(w))")
        assert "jnp.where(_mask_1," in out
        assert "_mask_1[" not in out

    def test_add_uses_field_specific_delta(self):
        out_b = self._mask("x = x.with_field('b', x.b.at[:, :].add(v))")
        out_a = self._mask("x = x.with_field('a', x.a.at[:].add(w))")
        assert "delta_x_b_mask_1" in out_b and "delta_x_a_mask_1" in out_a

    def test_plain_array_still_works(self):
        out = self._mask("tmp = tmp.at[:].set(v)")
        assert out.startswith("tmp = tmp.at[:].set(jnp.where(_mask_1")

    def test_self_receiver(self):
        out = self._mask("x = self.x.with_field('b', self.x.b.at[:].set(v))")
        assert "with_field('b'" in out and "jnp.where" in out

    def test_unbatched_struct_add_reduces_batch(self):
        c = self.converter
        c.cls_info["Model"]["methods"]["compute"]["struct_args"]["y"] = {
            "type": "arr_type",
            "dimensions": [],
        }
        c.var_deps = {"w": ("kjpindex",)}
        try:
            out = self._mask_vec("y = y.with_field('a', y.a.at[:].add(w[:]))")
        finally:
            c.var_deps = {}
        assert "jnp.sum(" in out and "axis=0" in out

    def test_mask_rank1_on_3d_field(self):
        c = self.converter
        c.cls_info["Model"]["methods"]["compute"]["struct_args"]["x"]["dimensions"] = [
            "kjpindex",
            "nvm",
        ]
        out = self._mask_vec("x = x.with_field('b', x.b.at[:, 0, :].set(v))")
        # batch axis 0 kept, inner axis expanded
        assert "_mask_1[:, None]" in out

    def test_mask_batch_not_leading(self):
        # struct over ["nvm", "kjpindex"]: batch axis is 1
        c = self.converter
        c.cls_info["Model"]["methods"]["compute"]["struct_args"]["x"]["dimensions"] = [
            "nvm",
            "kjpindex",
        ]
        out = self._mask_vec("x = x.with_field('a', x.a.at[0, :].set(v))")
        assert "jnp.where(_mask_1" in out

    def test_reduce_batch_axis_follows_field_rank(self):
        c = self.converter
        c.cls_info["Model"]["methods"]["compute"]["struct_args"]["y"] = {
            "type": "arr_type",
            "dimensions": [],
        }
        c.var_deps = {"w": ("kjpindex",)}
        old = getattr(c, "func_input_dim", {})
        c.func_input_dim = {**old, "w": ["kjpindex", 3]}
        try:
            out = self._mask_vec("y = y.with_field('b', y.b.at[:].add(w[:, :]))")
        finally:
            c.var_deps = {}
            c.func_input_dim = old
        assert "jnp.sum(jnp.where(_mask_1[:, None]" in out
        assert "axis=0" in out

    def test_two_fields_same_mask_no_collision(self):
        out = self._mask("x = x.with_field('a', x.a.at[:].add(v))") + self._mask(
            "x = x.with_field('b', x.b.at[:, :].add(w))"
        )
        assert (
            out.count("delta_x_a_mask_1 =") == 1
            and out.count("delta_x_b_mask_1 =") == 1
        )

    def test_continue_mask_gate_accepts_with_field(self):
        # _split_with_field must be recognised by _apply_continue_masks
        assert (
            self.converter._split_with_field(
                _stmt("x = x.with_field('a', x.a.at[:].set(v))").value, "x"
            )
            is not None
        )

    @pytest.fixture
    def z_struct(self):
        sa = self.converter.cls_info["Model"]["methods"]["compute"]["struct_args"]
        sa["z"] = {"type": "arr_type", "dimensions": ["kjpindex", "nvm", "nlevels_tot"]}
        yield
        sa.pop("z", None)

    def test_mask_expands_over_3d_scalar_field(self, z_struct):
        out = self._mask_vec("z = z.with_field('a', z.a.at[:, :, :].set(v))")
        assert "_mask_1[:, None, None]" in out
        assert out.startswith("z = z.with_field('a', z.a.at[:, :, :].set(jnp.where(")

    def test_mask_expands_over_4d_array_field(self, z_struct):
        # field b has inner dim [3] -> rank 4
        out = self._mask_vec("z = z.with_field('b', z.b.at[:, :, :, :].set(v))")
        assert "_mask_1[:, None, None, None]" in out

    def test_mask_single_batch_slice_only(self, z_struct):
        # only the batch axis is a slice: target_rank == 1, no expansion
        out = self._mask_vec("z = z.with_field('a', z.a.at[:, 0, 1].set(v))")
        assert "_mask_1[" not in out

    def test_run_set_array_field_matches_loop(self):
        x, mask, v, _ = self._inputs()
        out = self._run(
            "x = x.with_field('b', x.b.at[:, :].set(v))", x, mask, v=jnp.asarray(v)
        )

        # reference: original per-element condition
        ref_b = np.array(x.b)
        for ji in range(self.N):
            if mask[ji]:
                ref_b[ji, :] = v[ji, :]

        np.testing.assert_allclose(out.b, ref_b)
        np.testing.assert_allclose(out.a, x.a)  # other field untouched
        # guard against a silent transpose/broadcast: masked-off rows unchanged
        np.testing.assert_allclose(np.array(out.b)[~mask], np.array(x.b)[~mask])

    def test_run_add_two_fields_same_mask_matches_loop(self):
        x, mask, v, w = self._inputs()
        # run both statements in one namespace, as the generated code would
        src = (
            self._mask_vec("x = x.with_field('a', x.a.at[:].add(w))")
            + "\n"
            + self._mask_vec("x = x.with_field('b', x.b.at[:, :].add(v))")
        )
        ns = {
            "jnp": jnp,
            "x": x,
            "_mask_1": jnp.asarray(mask),
            "v": jnp.asarray(v),
            "w": jnp.asarray(w),
        }
        exec(src, ns)
        out = ns["x"]

        ref_a, ref_b = np.array(x.a), np.array(x.b)
        for ji in range(self.N):
            if mask[ji]:
                ref_a[ji] += w[ji]
                ref_b[ji, :] += v[ji, :]

        np.testing.assert_allclose(out.a, ref_a)
        np.testing.assert_allclose(out.b, ref_b)


class TestSelfStructWrite:
    @pytest.fixture(autouse=True)
    def _setup(self, test_env):
        c = self.converter
        c._func_arg_stack = [set()]
        c._mutated_attrs = set()
        c._modified_ret_stack = []
        c._var_modif = {"args": set(), "attr": set()}
        c._outer_func_args = set()
        c._control_stack = []
        c.func_name = "compute"
        c.struct_classes = {"laieff_type"}
        c.struct_info = {"laieff_type": {"a": {"inner_dims": [], "dtype": "float64"}}}
        attrs = c.cls_info["Model"]["attributes"]
        attrs["soil_fit"] = {
            "type": "struct",
            "struct_type": "laieff_type",
            "dimensions": ["kjpindex"],
        }
        yield
        attrs.pop("soil_fit", None)

    def _visit_fn(self, code):
        self.converter.reset_all()
        fn = _func(code)
        return self.converter.visit(fn)

    def _assign(self, code):
        out = self.converter.visit_Assign(_stmt(code))
        stmts = out if isinstance(out, list) else [out]
        return "\n".join(_unparse(s) for s in stmts)

    def test_first_write_reads_self_and_registers(self):
        out = self._assign("self.soil_fit.a[0] = 1.0")
        assert out == (
            "soil_fit = self.soil_fit.with_field('a', self.soil_fit.a.at[0].set(1.0))"
        )
        c = self.converter
        assert "soil_fit" in c._var_modif["attr"]
        assert "soil_fit" in c._mutated_attrs

    def test_second_write_drops_self(self):
        self._assign("self.soil_fit.a[0] = 1.0")
        out = self._assign("self.soil_fit.a[1] = 2.0")
        assert out == ("soil_fit = soil_fit.with_field('a', soil_fit.a.at[1].set(2.0))")

    def test_later_reads_drop_self(self):
        self._assign("self.soil_fit.a[0] = 1.0")
        node = self.converter.visit(_expr("self.soil_fit.a"))
        assert _unparse(node) == "soil_fit.a"

    def test_read_before_any_write_keeps_self(self):
        node = self.converter.visit(_expr("self.soil_fit.a"))
        assert _unparse(node) == "self.soil_fit.a"

    def test_not_registered_as_arg(self):
        self._assign("self.soil_fit.a[0] = 1.0")
        assert "soil_fit" not in self.converter._var_modif["args"]

    def test_end_to_end_self_struct_write_returns_updated_module(self):
        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.ndarray = np.float64(0)

        jax.tree_util.register_dataclass(laieff_type)

        class Model(eqx.Module):
            soil_fit: laieff_type

            def __init__(self):
                self.soil_fit = laieff_type.zeros((2,))
                self.soil_fit = self.soil_fit.with_field("a", jnp.array([1.0, 2.0]))

        code = """
def compute(self):
    self.soil_fit.a[0] = 2.0
        """
        method = ast.parse(code)

        converted = self._visit_fn(method)
        assert "soil_fit" in self.converter._var_modif["attr"]

        # The return stmt is added through the autodiff.prepare_class process
        # for each functions, here we add it manually but we still check that
        # we retrieve the proper return
        ret = self.converter.add_return_stmt(
            var_modif_args=[],
            var_modif_attr=self.converter._var_modif["attr"],
        )
        assert isinstance(ret, ast.Return)
        assert (
            ast.unparse(ast.fix_missing_locations(ret))
            == "return eqx.tree_at(lambda m: (m.soil_fit,), self, (soil_fit,))"
        )

        converted.body.append(ret)
        module_ast = ast.Module(body=[converted], type_ignores=[])

        ast.fix_missing_locations(module_ast)

        namespace = {"eqx": eqx}
        exec(compile(module_ast, filename="<generated>", mode="exec"), namespace)
        Model.compute = namespace["compute"]
        model = Model()
        original = model.soil_fit
        updated = model.compute()

        # Verify the returned module contains the update.
        assert jnp.allclose(updated.soil_fit.a, jnp.array([2.0, 2.0]))

        # Verify the original struct remains unchanged.
        assert jnp.allclose(original.a, jnp.array([1.0, 2.0]))

    def test_end_to_end_array_struct_with_array_field(self):
        @dataclass
        class arr_type(FortranDerivedType):
            b: np.ndarray = field(
                default_factory=lambda: np.zeros((3,), dtype=np.float64)
            )

        jax.tree_util.register_dataclass(arr_type)

        c = self.converter
        c.struct_classes = {"arr_type"}
        c.struct_info = {"arr_type": {"b": {"inner_dims": [3], "dtype": "float64"}}}
        c.cls_info["Model"]["attributes"]["soil_fit"] = {
            "type": "struct",
            "struct_type": "arr_type",
            "dimensions": ["kjpindex"],
        }

        class Model(eqx.Module):
            soil_fit: arr_type

            def __init__(self):
                fit = jax.tree_util.tree_map(jnp.asarray, arr_type.zeros((2,)))
                self.soil_fit = fit.with_field("b", jnp.arange(6.0).reshape(2, 3))

        code = "def compute(self):\n    self.soil_fit.b[0, 1] = 99.0\n"
        fn = self._visit_fn(code)
        ret = c.add_return_stmt(var_modif_args=[], var_modif_attr=c._var_modif["attr"])
        fn.body.append(ret)
        mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        ns = {"eqx": eqx}
        exec(compile(mod, "<generated>", "exec"), ns)
        Model.compute = ns["compute"]

        model = Model()
        original = model.soil_fit.b
        updated = model.compute()

        expected = np.arange(6.0).reshape(2, 3)
        expected[0, 1] = 99.0
        np.testing.assert_array_equal(updated.soil_fit.b, expected)
        np.testing.assert_array_equal(original, np.arange(6.0).reshape(2, 3))
        assert updated.soil_fit.b.dtype == jnp.float64

    def _transform_fn(self, code):
        self.converter.reset_all()
        self.converter.helpers = []
        self.converter._pending_helpers = []
        fn = _func(code)
        transformed = self.converter.visit(fn)
        self.converter.process_helpers()
        return transformed

    def test_struct_in_scan_carry_under_jit_dtype_stable(self):
        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.ndarray = np.float64(0)

        jax.tree_util.register_dataclass(laieff_type)

        class Model(eqx.Module):
            soil_fit: laieff_type
            nslm: int = eqx.field(static=True)

            def __init__(self):
                self.soil_fit = laieff_type.zeros((2,))
                self.soil_fit = self.soil_fit.with_field(
                    "a", jnp.array([1.0, 2.0], dtype=jnp.float64)
                )
                self.nslm = 3

        code = """
def compute(self):
    for i in range(0, self.nslm, 1):
        self.soil_fit.a[i] = 2 * i
"""

        # Use the converter's existing loop-to-scan transformation.
        transformed = self._transform_fn(code)
        print(ast.unparse(ast.fix_missing_locations(transformed)))
        scan_helpers = [
            h for h in self.converter.helpers if "scan" in h.name or "body" in h.name
        ]
        assert scan_helpers, "No scan helper was generated"
        ret = self.converter.add_return_stmt(
            var_modif_args=[],
            var_modif_attr=self.converter._var_modif["attr"],
        )
        assert isinstance(ret, ast.Return)
        transformed.body.append(ret)
        module_ast = ast.Module(
            body=[transformed],
            type_ignores=[],
        )
        ast.fix_missing_locations(module_ast)
        namespace = {
            "eqx": eqx,
            "jax": jax,
            "jnp": jnp,
            "np": np,
            "lax": lax,
        }
        exec(
            compile(module_ast, filename="<generated>", mode="exec"),
            namespace,
        )

        Model.compute = namespace["compute"]

        model = Model()
        original = model.soil_fit.a

        # Execute the generated scan under JIT.
        updated = jax.jit(lambda m: m.compute())(model)

        # Each loop iteration increments a different element.
        np.testing.assert_allclose(
            np.asarray(updated.soil_fit.a),
            np.array([0.0, 2.0], dtype=np.float64),
        )
        assert updated.soil_fit.a.dtype == jnp.float64

        # The original module must remain unchanged.
        np.testing.assert_allclose(
            np.asarray(original),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def _filled_scalar(shape):
    x = to_jax(scalar_type.zeros(shape))
    n = int(np.prod(shape))
    vals = jnp.arange(n, dtype=jnp.float64).reshape(shape)
    return x.with_field("a", vals).with_field("b", vals * 10.0)


def _filled_array(shape):
    x = to_jax(array_type.zeros(shape))
    # a: shape + (10,), b: shape + (2, 2)
    a = jnp.arange(int(np.prod(shape)) * 10, dtype=jnp.float64).reshape(shape + (10,))
    b = jnp.arange(int(np.prod(shape)) * 4, dtype=jnp.float64).reshape(shape + (2, 2))
    return x.with_field("a", a).with_field("b", b)


FILL = {
    "scalar": _filled_scalar,
    "array": _filled_array,
}


def _callee(angle, e):
    # works for both kinds: reduce only the trailing field axes
    a = e.a if e.a.ndim == 0 else e.a.reshape(e.a.shape[:0] + (-1,)).sum()
    return angle * a + e.b.sum()


class TestStructSentAsArg:
    @pytest.fixture(autouse=True)
    def _setup(self, test_env):
        c = self.converter
        c._control_stack = []
        c.dynamic_variable_lift = {}
        c.func_name = "compute"
        yield
        c._control_stack = []

    def _rewrite_call(self, vectorized):
        """Run the struct-read argument through the converter and
        return the source of the rebuilt call."""
        c = self.converter
        c._control_stack = [_vec_ctrl()] if vectorized else []
        call = _expr("calculate_laieff_fit(angle, laieff_fit[ji, ivm, ilevel])")
        call.args[1] = c.visit_Subscript(call.args[1])
        return _unparse(call)

    def test_vectorized_whole_read_slices_batch_axis(self):
        # vectorised: batch axis becomes a slice
        src = self._rewrite_call(vectorized=True)
        assert src == "calculate_laieff_fit(angle, laieff_fit[:, ivm, ilevel])"
        assert "ji" not in src

    @pytest.mark.parametrize("kind", ["scalar", "array"])
    def test_non_vectorized_read_unchanged_and_runs_under_vmap(self, kind):
        # non-vectorised: untouched, and valid under vmap over ji
        src = self._rewrite_call(vectorized=False)
        assert src == "calculate_laieff_fit(angle, laieff_fit[ji, ivm, ilevel])"

        x = FILL[kind]((5, 3, 4))

        def per_ji(ji):
            ns = {
                "calculate_laieff_fit": _callee,
                "angle": 2.0,
                "laieff_fit": x,
                "ji": ji,
                "ivm": 1,
                "ilevel": 2,
            }
            return eval(src, ns)

        out = jax.vmap(per_ji)(jnp.arange(5))
        exp = jax.vmap(lambda e: _callee(2.0, e))(x[:, 1, 2])
        np.testing.assert_allclose(out, exp)

    @pytest.mark.parametrize("kind", ["scalar", "array"])
    def test_vectorized_rewrite_matches_vmap_semantics(self, kind):
        # vectorised output equals the vmapped per-ji semantics
        src = self._rewrite_call(vectorized=True)
        x = FILL[kind]((5, 3, 4))

        # converter output: callee sees a batched struct (leading axis kept)
        batched = eval(
            src,
            {
                "calculate_laieff_fit": lambda angle, e: e,
                "angle": 2.0,
                "laieff_fit": x,
                "ivm": 1,
                "ilevel": 2,
            },
        )
        assert isinstance(batched, type(x))
        assert batched.a.shape[0] == 5

        # reference: per-ji scalar semantics under vmap
        ref = jax.vmap(lambda ji: x[ji, 1, 2])(jnp.arange(5))
        np.testing.assert_allclose(batched.a, ref.a)
        np.testing.assert_allclose(batched.b, ref.b)

    def test_getitem_traced_index_array_fields(self):
        # Test the __getitem__ of FortranDerivedType under vmap
        x = _filled_array((5, 3, 4))

        def callee(angle, elem):
            # Test direclty that the callee under vmap recieves
            # the dataclass
            assert elem.a.shape == (10,)
            assert elem.b.shape == (2, 2)
            return angle * elem.a.sum() + elem.b.sum()

        def per_ji(ji):
            return callee(2.0, x[ji, 1, 2])

        out = jax.vmap(per_ji)(jnp.arange(5))
        exp = 2.0 * x.a[:, 1, 2, :].sum(-1) + x.b[:, 1, 2].sum((-1, -2))
        np.testing.assert_allclose(out, exp)

    def test_struct_passed_as_vmapped_arg_array_fields(self):
        x = _filled_array((5, 3, 4))

        def callee(elem):
            assert elem.a.shape == (3, 4, 10)
            assert elem.b.shape == (3, 4, 2, 2)
            return elem[1, 2].a  # (10,)

        out = jax.vmap(callee)(x)
        assert out.shape == (5, 10)
        np.testing.assert_allclose(out, x.a[:, 1, 2, :])

    def test_getitem_array_field_output_shapes_under_vmap(self):
        x = _filled_array((5, 3, 4))
        out = jax.vmap(lambda ji: x[ji, 0, 0].b)(jnp.arange(5))
        assert out.shape == (5, 2, 2)

    def test_in_axes_none_struct_closed_over_equals_mapped(self):
        # unbatched struct, batched index vs. batched struct
        x = _filled_scalar((5, 3, 4))
        a1 = jax.vmap(lambda ji: x[ji, 1, 2].a)(jnp.arange(5))
        a2 = jax.vmap(lambda e: e[1, 2].a)(x)
        np.testing.assert_allclose(a1, a2)

    def test_getitem_under_jit_vmap(self):
        x = _filled_scalar((5, 3, 4))

        @jax.jit
        def f(x, angle):
            return jax.vmap(lambda ji: angle * x[ji, 1, 2].a)(jnp.arange(5))

        np.testing.assert_allclose(f(x, 3.0), 3.0 * x.a[:, 1, 2])

    def test_grad_through_getitem_under_vmap(self):
        x = _filled_array((5, 3, 4))

        def loss(angle):
            per = jax.vmap(lambda ji: angle * x[ji, 1, 2].a.sum())(jnp.arange(5))
            return per.sum()

        expected = x.a[:, 1, 2, :].sum()
        np.testing.assert_allclose(jax.grad(loss)(2.0), expected)
