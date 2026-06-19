import ast
import pytest
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from logger import Logger
from jax_converter.converter import JaxConverter

def _make_cls_info():
    """Minimal but representative cls_info for a single class `Model`."""
    return {
        'Model': {
            'attributes': {
                'kjpindex': {'type': 'int'},
                'nnobio': {'type': 'int'},
                'soil_temp': {'type': 'jnp.ndarray', 'dimensions': ['kjpindex'], 'dtype': 'float64'},
                'soil_moist': {'type': 'jnp.ndarray', 'dimensions': ['kjpindex', 'nnobio'], 'dtype': 'float64'},
                'mask': {'type': 'jnp.ndarray', 'dimensions': ['kjpindex'], 'dtype': 'bool'},
            },
            'methods': {
                'compute': {
                    'args': ['self', 'x'],
                    'local_arr': {
                        'tmp': {'dimensions': ['kjpindex'], 'dtype': 'float64', 'type': 'jnp.ndarray'},
                    },
                },
                'helper': {
                    'args': ['self', 'y'],
                    'local_arr': {},
                },
            },
        }
    }


@pytest.fixture(scope='class')
def test_env(request):
    converter = JaxConverter(
        cls_info=_make_cls_info(),
        logger=Logger(),
        mode='jax',
    )
    converter.func_name = 'compute'
    converter.func_input_dim = {'x': ['kjpindex']}
    converter.call_edge = {'compute': [], 'helper': []}
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


@pytest.mark.usefixtures('test_env')
class TestJaxConverter:

    def test_scope_push_pop(self):
        # A variable added in a pushed scope is not visible after pop
        self.converter._push_scope()
        self.converter._add_local('local_var', ('kjpindex',))
        assert self.converter._is_local('local_var') is True
        self.converter._pop_scope()
        assert self.converter._is_local('local_var') is False

    def test_scope_is_local(self):
        # A variable defined in an outer scope is visible from a nested scope
        self.converter._push_scope()
        self.converter._add_local('outer_var', ())
        self.converter._push_scope()
        assert self.converter._is_local('outer_var') is True
        self.converter._pop_scope()
        self.converter._pop_scope()

        # Unknown case 
        assert self.converter._is_local('totally_unknown_name') is False

    def test_fresh_names(self):
        start = self.converter.counter
        true_name, false_name = self.converter._fresh_names()
        assert true_name == f'_if_true_{start}'
        assert false_name == f'_if_false_{start}'
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
        stmt = _stmt('x = 1')
        assert self.converter._is_logging_call(stmt) is False

    def test_collect_loop_vars(self):
        # Simple target 
        stmts = _parse('for i in range(10):\n    pass').body
        result = self.converter._collect_loop_vars(stmts)
        assert result == {'i'}

        # Tuple target
        stmts = _parse('for i, j in pairs:\n    pass').body
        result = self.converter._collect_loop_vars(stmts)
        assert result == {'i', 'j'}

        # No loop 
        stmts = _parse('x = 1\ny = 2').body
        result = self.converter._collect_loop_vars(stmts)
        assert result == set()

    def test_first_reads(self):
        # Simple case 
        stmts = _parse('y = x + 1').body
        result = self.converter._first_reads(stmts)
        assert 'x' in result

        # `y` is only ever stored to, never loaded, so it should not appear
        stmts = _parse('y = 1').body
        result = self.converter._first_reads(stmts)
        assert 'y' not in result

        # a[i] = a[i] + 1 -> `a` must be recorded as a LOAD (used before written)
        stmts = _parse('a[i] = a[i] + 1').body
        result = self.converter._first_reads(stmts)
        assert 'a' in result

        # First_reads doesn't retrieve the loop index
        stmts = _parse('for i in range(10):\n    y = i + 1').body
        result = self.converter._first_reads(stmts)
        assert 'i' not in result

        # Attribute 
        stmts = _parse('y = self.kjpindex + 1').body
        result = self.converter._first_reads(stmts)
        assert 'self.kjpindex' in result

    def test_collect_rhs_uses(self):
        # Simple case 
        stmts = _parse('y = a + b').body
        result = self.converter._collect_rhs_uses(stmts)
        assert result == {'a', 'b'}

        # The assigned target `y` should not appear in RHS uses
        stmts = _parse('y = a + b').body
        result = self.converter._collect_rhs_uses(stmts)
        assert 'y' not in result

        # Subscripts 
        stmts = _parse('y = a[i]').body
        result = self.converter._collect_rhs_uses(stmts)
        assert 'a' in result
        assert 'i' in result

    def test_collect_assigned(self):
        # Plain name 
        stmts = _parse('x = 1').body
        result = self.converter._collect_assigned(stmts)
        assert 'x' in result

        # Attribute mutated 
        self.converter._mutated_attrs = set()
        stmts = _parse('self.soil_temp = 1').body
        result = self.converter._collect_assigned(stmts)
        assert 'soil_temp' in result
        assert 'soil_temp' in self.converter._mutated_attrs

        stmts = _parse('x += 1').body
        result = self.converter._collect_assigned(stmts)
        assert 'x' in result

        stmts = _parse('a[i] = 1').body
        result = self.converter._collect_assigned(stmts)
        assert 'a' in result

        # NO duplicates 
        stmts = _parse('x = 1\nx = 2').body
        result = self.converter._collect_assigned(stmts)
        assert result.count('x') == 1

    def test_subscript_uses_loop_vars(self):
        node = _expr('a[i]')
        assert self.converter._subscript_uses_loop_vars(node, ['i']) is True

        # No loop index usage inside subscript 
        node = _expr('a[j]')
        assert self.converter._subscript_uses_loop_vars(node, ['i']) is False

        node = _expr('a[i, k]')
        assert self.converter._subscript_uses_loop_vars(node, ['i']) is True

        node = _expr('f(a[i])')
        assert self.converter._subscript_uses_loop_vars(node, ['i']) is True

    def test_to_arg(self):
        # strip 'self' prefix 
        assert self.converter.to_arg('self.kjpindex') == 'kjpindex'

        assert self.converter.to_arg('kjpindex') == 'kjpindex'

    def test_make_operand(self):
        # NOt mutated attributes 
        self.converter._mutated_attrs = set()
        result = self.converter.make_operand('self.kjpindex')
        assert isinstance(result, ast.Attribute)
        assert result.attr == 'kjpindex'

        # mutated attributes 
        self.converter._mutated_attrs.add('self.kjpindex')
        result = self.converter.make_operand('kjpindex')
        assert isinstance(result, ast.Name)
        assert result.id == 'kjpindex'

        result = self.converter.make_operand('local_var')
        assert isinstance(result, ast.Name)
        assert result.id == 'local_var'

    def test_replace_returns(self):
        # strips return
        stmts = _parse('return 1').body
        result = self.converter._replace_returns(stmts)
        assert result == []

        stmts = _parse('x = 1\nreturn x').body
        result = self.converter._replace_returns(stmts)
        assert len(result) == 1
        assert isinstance(result[0], ast.Assign)

    def test_lift_if_return(self):
        # Merge branches 
        code = (
            'def f(self):\n'
            '    if cond:\n'
            '        x = 1\n'
            '        return x\n'
            '    y = 2\n'
        )
        fn = _func(code)
        result = self.converter._lift_if_return(fn)
        # body should now contain a single If with both branches and no Return
        assert len(result.body) == 1
        assert isinstance(result.body[0], ast.If)
        assert not any(isinstance(s, ast.Return) for s in ast.walk(result.body[0]))

        # No return operation 
        code = 'def f(self):\n    if cond:\n        x = 1\n    y = 2\n'
        fn = _func(code)
        result = self.converter._lift_if_return(fn)
        assert len(result.body) == 2

    def test_is_pure_select(self):
        # Condition true for batching branches 
        code = 'if cond:\n    x = 1\nelse:\n    x = 2\n'
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is True

        # Condition false in the true branch 
        code = 'if cond:\n    x = f(1)\nelse:\n    x = 2\n'
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is False

        # Mismatched branches 
        code = 'if cond:\n    x = 1\nelse:\n    y = 2\n'
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is False

        # Subscript targets 
        code = 'if cond:\n    a[i] = 1\nelse:\n    a[i] = 2\n'
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is False

        # A single if-branch with no else still qualifies; the missing
        # else branch is handled by the fallback logic in _handle_scalar_select
        code = 'if cond:\n    x = 1\n'
        node = _stmt(code)
        assert self.converter._is_pure_select_cond(node) is True

    def test_transform_compare(self):
        # Simple compare 
        node = _expr('a > b')
        result = self.converter._transform_compare(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'greater'

        # Chained compare 
        node = _expr('a < b < c')
        result = self.converter._transform_compare(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'logical_and'

        node = _expr('a in b')
        with pytest.raises(NotImplementedError):
            self.converter._transform_compare(node)

    def test_transform_boolop(self):
        node = _expr('a and b')
        result = self.converter._transform_boolop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'logical_and'

        node = _expr('a or b')
        result = self.converter._transform_boolop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'logical_or'

    def test_transform_if_test(self):
        node = _expr('not a')
        result = self.converter._transform_if_test(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'logical_not'

    def test_transform_binop(self):
        node = _expr('a | b')
        result = self.converter._transform_binop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'logical_or'

        node = _expr('a & b')
        result = self.converter._transform_binop(node)
        assert isinstance(result, ast.Call)
        assert result.func.attr == 'logical_and'

        node = _expr('a + b')
        result = self.converter._transform_binop(node)
        assert isinstance(result, ast.BinOp)
        assert isinstance(result.op, ast.Add)

    def test_check_if(self):
        # Array true case 
        node = _expr('self.soil_temp')
        assert self.converter.check_if_array(node) is True

        node = _expr('self.kjpindex')
        assert self.converter.check_if_array(node) is False

        # For function input array 
        node = _expr('x')
        assert self.converter.check_if_array(node) is True

        node = _expr('totally_unknown')
        assert self.converter.check_if_array(node) is False

        node = _expr('self.soil_moist')
        assert self.converter.check_if_array(node, required_dims=['nnobio']) is True
        assert self.converter.check_if_array(node, required_dims=['unrelated_dim']) is False

    def test_is_arr_at_op_call(self):
        node = _expr('arr.at[0].set(1)')
        assert self.converter.is_arr_at_op_call(node, 'arr') is True  

        # Local variable form: arr.at[0].set(1)
        node = _expr('soil_temp.at[0].set(1)')
        assert self.converter.is_arr_at_op_call(node, 'soil_temp') is True

        # self.attr.at[...] does NOT match — is_arr_at_op_call expects a
        # bare local-variable base, not a self.attr base.
        node = _expr('self.soil_temp.at[0].set(1)')
        assert self.converter.is_arr_at_op_call(node, 'soil_temp') is False

        node = _expr('self.soil_temp.at[0].set(1)')
        assert self.converter.is_arr_at_op_call(node, 'other_name') is False

        node = _expr('f(x)')
        assert self.converter.is_arr_at_op_call(node, 'x') is False

    def test_strip_any(self):
        node = _expr('(a < b).any()')
        result = self.converter.strip_any(node)
        assert isinstance(result, ast.Compare)

        node = _expr('f(x)')
        result = self.converter.strip_any(node)
        assert result is node

    def test_set_working_function(self):
        # Update class data 
        self.converter.set_working_function('helper', {'y': ['kjpindex']}, {'helper': []})
        assert self.converter.func_name == 'helper'
        assert self.converter.func_input_dim == {'y': ['kjpindex']}
        # restore for other tests in the class
        self.converter.set_working_function('compute', {'x': ['kjpindex']}, {'compute': [], 'helper': []})

    def test_reset_all(self):
        self.converter._mutated_attrs = {'soil_temp'}
        self.converter.var_state = {'tmp': ('stateful', None)}
        self.converter.dynamic_variable_lift = {'tmp': {}}
        self.converter.reset_all()
        assert self.converter._mutated_attrs == set()
        assert self.converter.var_state == {}
        assert self.converter.dynamic_variable_lift == {}
        assert self.converter.for_counter == 0
        assert self.converter.counter == 0