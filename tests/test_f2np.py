import ast
from unittest.mock import MagicMock

import pytest
from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

from fgpt.core.common.logger import Logger
from fgpt.core.frontend.processor import Processor
from fgpt.core.transpiler.f2np import F2NP


@pytest.fixture(scope="class")
def test_env(request):
    f2np = F2NP(extractor=MagicMock())
    request.cls.f2np = f2np
    yield


def normalize_ast(node):
    """
    Remove non-semantic attributes like line numbers.
    """
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, None)
    return node


def ast_to_dict(node):
    """
    Convert AST into a nested dict for stable comparison.
    """
    if isinstance(node, ast.AST):
        return {
            "_type": type(node).__name__,
            **{field: ast_to_dict(value) for field, value in ast.iter_fields(node)},
        }
    elif isinstance(node, list):
        return [ast_to_dict(x) for x in node]
    else:
        return node


def assert_ast_equal(actual, expected):
    """
    Compare two ASTs structurally.
    """
    actual = normalize_ast(actual)
    expected = normalize_ast(expected)

    actual_dict = ast_to_dict(actual)
    expected_dict = ast_to_dict(expected)

    assert actual_dict == expected_dict


@pytest.mark.usefixtures("test_env")
class TestF2NP:
    def parse(self, code):
        return Processor(logger=Logger()).parse_fortran_string(code)

    def get_stmt(self, tree, node_type):
        return walk(tree, node_type)[0]

    def parse_and_get(self, code, node_type):
        tree = self.parse(code)
        return self.get_stmt(tree, node_type)

    def test_handle_subroutine_stmt_ast(self):
        code = """
        subroutine compute_sum(a, b, result)
        end subroutine compute_sum
        """
        stmt = self.parse_and_get(code, F23.Subroutine_Stmt)
        func_def = self.f2np.handle_subroutine_stmt(stmt)

        assert isinstance(func_def, ast.FunctionDef)
        assert func_def.name == "compute_sum"
        assert [arg.arg for arg in func_def.args.args] == ["a", "b", "result"]

    def test_handle_call_stmt_ast(self):
        code = """
        subroutine test_call()
            call compute_sum(a, b)
        end subroutine test_call
        """

        stmt = self.parse_and_get(code, F23.Call_Stmt)
        self.f2np.extractor.allowed_external_subroutines = []

        result = self.f2np.handle_call_stmt(stmt)

        assert isinstance(result, ast.Expr)
        assert isinstance(result.value, ast.Call)
        assert result.value.func.id == "compute_sum"
        assert len(result.value.args) == 2

        # Call statements that are meant as print statements
        code = """
        subroutine test_call()
            call xios_orchidee_send_field('field_name')
        end subroutine test_call
        """

        stmt = self.parse_and_get(code, F23.Call_Stmt)
        self.f2np.extractor.allowed_external_subroutines = ["xios_orchidee_send_field"]

        result = self.f2np.handle_call_stmt(stmt)

        assert isinstance(result, ast.Expr)
        assert isinstance(result.value, ast.Call)

        # logging.info(...)
        assert isinstance(result.value.func, ast.Attribute)
        assert result.value.func.attr == "info"

        # Check message content
        assert "INFO: xios_orchidee_send_field:" in result.value.args[0].value

    def test_handle_type_decl_explicit_shape_ast(self):
        code = """
        subroutine test()
            real :: arr(10, 20)
        end subroutine test
        """

        stmt = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        result = self.f2np.handle_type_declaration_stmt(stmt)

        assert isinstance(result, ast.Assign)

        # arr = np.zeros(...)
        assert result.targets[0].id == "arr"
        assert isinstance(result.value, ast.Call)
        assert result.value.func.attr == "zeros"

        # shape check
        shape = result.value.args[0]
        assert isinstance(shape, ast.Tuple)
        assert len(shape.elts) == 2

    def test_handle_where_stmt_ast(self):
        code = """
        subroutine test_where()
            real :: a(10), b(10)
            where (a > 0)
                b = a
            end where
        end subroutine test_where
        """

        stmt = self.parse_and_get(code, F23.Where_Construct_Stmt)
        result = self.f2np.handle_where_stmt(stmt)

        # Check that we got an ast.If
        assert isinstance(result, ast.If)

        # Check test part is ast.Call
        assert isinstance(result.test, ast.Call)
        assert result.test.func.attr == "any"

        # Check mask assignment exists
        assert isinstance(result.body[0], ast.Assign)
        assert result.body[0].targets[0].id == "mask"

    def test_handle_do_stmt(self):
        code = """
        subroutine simple_do()
            integer :: i
            do i = 1, 10
                ! loop body
            end do
        end subroutine simple_do
        """
        stmt = self.parse_and_get(code, F23.Loop_Control)
        result = self.f2np.handle_do_stmt(stmt)

        # Check we get an ast.For
        assert isinstance(result, ast.For)
        assert result.target.id == "i"
        assert isinstance(result.iter, ast.Call)
        assert result.iter.func.id == "range"
        # start = 0 (1-1), end = 10
        assert (
            isinstance(result.iter.args[0], ast.Constant)
            and result.iter.args[0].value == 0
        )
        assert (
            isinstance(result.iter.args[1], ast.Constant)
            and result.iter.args[1].value == 10
        )
        assert (
            isinstance(result.iter.args[2], ast.Constant)
            and result.iter.args[2].value == 1
        )

        # For loop with stride
        code = """
        subroutine do_stride()
            integer :: i
            do i = 1, 10, 2
                ! loop body
            end do
        end subroutine do_stride
        """
        stmt = self.parse_and_get(code, F23.Loop_Control)

        result = self.f2np.handle_do_stmt(stmt)

        assert (
            isinstance(result.iter.args[2], ast.Constant)
            and result.iter.args[2].value == 2
        )

        # For loop with negative stride
        code = """
        subroutine do_neg()
            integer :: i
            do i = 10, 1, -1
                ! loop body
            end do
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Loop_Control)

        result = self.f2np.handle_do_stmt(stmt)

        # stride negative
        stride_arg = result.iter.args[2]
        # If negative literal, it will be a UnaryOp
        if isinstance(stride_arg, ast.UnaryOp) and isinstance(stride_arg.op, ast.USub):
            assert (
                isinstance(stride_arg.operand, ast.Constant)
                and stride_arg.operand.value == 1
            )
        else:
            assert isinstance(stride_arg, ast.Constant) and stride_arg.value == -1

        # While do loop
        code = """
        subroutine do_neg()
            integer :: i
            do while (i < 10)
                i = i + 1
            end do
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Loop_Control)

        result = self.f2np.handle_do_stmt(stmt)
        assert isinstance(result, ast.While)

    def test_handle_if_condition(self):
        # Test if condition simple
        code = """
        subroutine do_neg()
            integer :: i
            IF (flag) THEN
                ! if body
            END IF
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.If_Then_Stmt)
        result = self.f2np.handle_if_condition(stmt)
        assert isinstance(result, ast.If)
        assert isinstance(result.test, ast.Name)
        assert result.test.id == "flag"
        assert result.body == []
        assert result.orelse == []

        # Test if condition with comparaison
        code = """
        subroutine do_neg()
            integer :: i
            IF (i .GT. 0) THEN
                ! if body
            END IF
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.If_Then_Stmt)
        result = self.f2np.handle_if_condition(stmt)
        assert isinstance(result, ast.If)
        assert isinstance(result.test, ast.Compare)

        # Test if condition with complex comparaison
        code = """
        subroutine do_neg()
            integer :: i
            IF (i .GT. 0 .AND. j .LT. 10) THEN
                ! if body
            END IF
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.If_Then_Stmt)
        result = self.f2np.handle_if_condition(stmt)
        assert isinstance(result, ast.If)
        assert isinstance(result.test, ast.BoolOp)

        # Exception handling
        with pytest.raises(ValueError):
            self.f2np.handle_if_condition(None)

    def test_handle_print_stmt(self):
        code = """
        subroutine do_neg()
            PRINT *, "Hello World"
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)

        result = self.f2np.handle_print_stmt(stmt)
        assert isinstance(result, ast.Expr)
        assert isinstance(result.value, ast.Call)
        # The logging method should be 'info'
        assert result.value.func.attr == "info"

        # Formatted value print: print(f'{}')
        code = """
        subroutine do_neg()
            PRINT *, var
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)

        result = self.f2np.handle_print_stmt(stmt)
        assert isinstance(result, ast.Expr)
        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "info"
        assert isinstance(call.args[0], ast.JoinedStr)
        # Should contain the variable as FormattedValue
        assert any(
            isinstance(v, ast.FormattedValue) and v.value.id == "var"
            for v in call.args[0].values
        )

        # mixed printing values
        code = """
        subroutine do_neg()
            PRINT *, "Value:", var
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)

        result = self.f2np.handle_print_stmt(stmt)
        assert isinstance(result, ast.Expr)
        call = result.value
        assert isinstance(call, ast.Call)
        joined_values = call.args[0].values
        assert any(
            isinstance(v, ast.Constant) and v.value == "Value:" for v in joined_values
        )
        assert any(
            isinstance(v, ast.FormattedValue) and v.value.id == "var"
            for v in joined_values
        )

    def test_handle_intrinsic_function_reference(self):
        # Simple maximum function with multiple values
        code = """
        subroutine do_neg()
            m1 = MAX(5, 6, 7)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Intrinsic_Function_Reference)
        result = self.f2np.handle_intrinsic_function_reference(stmt)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "maximum"  # np.maximum

        # Now we need to check the internal function since in python np.maximum takes only two elements and not n thus
        # in this we have np.maximum(np.maximum(5,6), 7)
        assert isinstance(result.args[0], ast.Call)
        internal_max = result.args[0]
        assert internal_max.func.attr == "maximum"
        assert isinstance(result.args[1], ast.Constant) and result.args[1].value == 7

        # Instrinsic funciton EPSILON
        code = """
        subroutine do_neg()
            m1 = EPSILON(2.0)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Intrinsic_Function_Reference)
        result = self.f2np.handle_intrinsic_function_reference(stmt)
        # Should return ast.Attribute for np.finfo(np.float64).eps
        assert isinstance(result, ast.Attribute)
        assert result.attr == "eps"

        # With keyword argument
        code = """
        subroutine do_neg()
            m1 = COS(X=value)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Intrinsic_Function_Reference)
        result = self.f2np.handle_intrinsic_function_reference(stmt)
        assert isinstance(result, ast.Call)
        assert result.func.attr == "cos"

    def test_handle_real_literal_constant(self):
        code = """
        subroutine do_neg()
            m1 = 3.14
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Real_Literal_Constant)
        result = self.f2np.handle_real_literal_constant(stmt)
        assert isinstance(result, ast.Constant)
        assert result.value == 3.14

    def test_handle_part_ref(self):
        code = """
        subroutine do_neg()
            b = a(2)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        # Without this array will be confused for a function
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        result = self.f2np.handle_part_ref(stmt)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Constant)
        assert result.slice.value == 2

        # Test slice index with Fortran array slicing where it can start from n to -n
        code = """
        subroutine do_neg()
            b = a(1:5)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        result = self.f2np.handle_part_ref(stmt)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Slice)
        # lower and upper
        # Since the lower values started at 1 and in python we start at everything at 0 thus the lower slice is None
        # this is only applied to the lower bound values
        assert result.slice.lower is None
        assert isinstance(result.slice.upper, ast.Constant)
        assert result.slice.upper.value == 5

        # Multiple slice
        code = """
        subroutine do_neg()
            b = a(1:5, 2:4)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        result = self.f2np.handle_part_ref(stmt)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Tuple)
        assert all(isinstance(s, ast.Slice) for s in result.slice.elts)

        # Function reference thus not present inside the all_array_info
        code = """
        subroutine do_neg()
            b = f(x)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        result = self.f2np.handle_part_ref(stmt)
        assert isinstance(result, ast.Call)
        assert result.func.id == "f"

        # Full slice
        code = """
        subroutine do_neg()
            b = a(:)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        result = self.f2np.handle_part_ref(stmt)
        assert isinstance(result.slice, ast.Slice)
        assert result.slice.lower is None
        assert result.slice.upper is None

    def test_handle_level_4expr(self):
        code = """
        subroutine do_neg()
            b = a .EQ. b
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Level_4_Expr)
        result = self.f2np.handle_level_4expr(stmt)
        assert isinstance(result, ast.Compare)
        assert isinstance(result.ops[0], ast.Eq)
        assert result.left.id == "a"
        assert result.comparators[0].id == "b"

        # Lower than comparator
        code = """
        subroutine do_neg()
            b = x .LT. 10
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Level_4_Expr)
        result = self.f2np.handle_level_4expr(stmt)
        assert isinstance(result.ops[0], ast.Lt)
        assert result.left.id == "x"
        assert result.comparators[0].value == 10

        # Test unknown operator
        code = """
        subroutine do_neg()
            b = a /= b
        end subroutine do_neg
        """
        # `/=`` doesn't exist within the conditional or the replacement dict
        stmt = self.parse_and_get(code, F23.Level_4_Expr)
        with pytest.raises(KeyError):
            self.f2np.handle_level_4expr(stmt)

    def test_handle_OR_AND_Operand(self):
        code = """
        subroutine do_neg()
            b = a .AND. b
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Or_Operand)
        result = self.f2np.handle_OR_AND_Operand(stmt)
        assert isinstance(result, ast.BoolOp)
        assert isinstance(result.op, ast.And)
        assert result.values[0].id == "a"
        assert result.values[1].id == "b"

        # Simple or
        code = """
        subroutine do_neg()
            b = x .OR. y
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Equiv_Operand)
        result = self.f2np.handle_OR_AND_Operand(stmt)
        assert isinstance(result, ast.BoolOp)
        assert isinstance(result.op, ast.Or)
        assert result.values[0].id == "x"
        assert result.values[1].id == "y"

        # Unary test case of not freeze_cwr
        code = """
        subroutine do_neg()
            b = .NOT. flag
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.And_Operand)
        result = self.f2np.handle_OR_AND_Operand(stmt)
        assert isinstance(result, ast.UnaryOp)
        assert isinstance(result.op, ast.Not)
        assert result.operand.id == "flag"

        # bitwise and
        code = """
        subroutine do_neg()
            b = (a(:) == 1) .AND. (b(:) == 2)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Or_Operand)
        self.f2np.extractor.all_array_info = {"test": {"a": [], "b": []}}
        result = self.f2np.handle_OR_AND_Operand(stmt)
        assert isinstance(result, ast.BinOp)
        assert isinstance(result.op, ast.BitAnd)
        assert isinstance(result.left, ast.Compare)
        assert isinstance(result.right, ast.Compare)

    def test_handle_assignment(self):
        # SIMPLE assignement
        code = """
        subroutine test_assign()
            a = b
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        result = self.f2np.handle_assignment(stmt)

        assert isinstance(result, ast.Assign)
        assert isinstance(result.targets[0], ast.Name)
        assert result.targets[0].id == "a"
        assert isinstance(result.value, ast.Name)
        assert result.value.id == "b"

        # Assignement for arrays
        code = """
        subroutine test_assign()
            a(:, :) = zero
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        self.f2np.extractor.all_array_info = {"test": {"a": [], "b": []}}
        result = self.f2np.handle_assignment(stmt)

        assert isinstance(result, ast.Assign)
        # LHS should be Subscript with Tuple of Slices
        lhs = result.targets[0]
        assert isinstance(lhs, ast.Subscript)
        assert isinstance(lhs.slice, ast.Tuple)
        for elt in lhs.slice.elts:
            assert isinstance(elt, ast.Slice)
        # RHS should be Name
        assert isinstance(result.value, ast.Name)
        assert result.value.id == "zero"

        # assignement with constant
        code = """
        subroutine test_assign()
            a = 1.0
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        result = self.f2np.handle_assignment(stmt)

        assert isinstance(result, ast.Assign)
        assert isinstance(result.value, ast.Constant)
        assert result.value.value == 1.0

        # assignement with intrinsic function
        code = """
        subroutine test_assign()
            a = SUM(b)
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        result = self.f2np.handle_assignment(stmt)

        assert isinstance(result, ast.Assign)
        assert isinstance(result.value, ast.Call)
        assert isinstance(result.value.func, ast.Attribute)  # np.sum
        assert result.value.func.attr == "sum"

    def test_handle_expr(self):
        # Test paranthesis
        code = """
        subroutine test_assign()
            a = (2 + 4)
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Parenthesis)
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.BinOp)
        assert isinstance(result.left, ast.Constant) and isinstance(
            result.right, ast.Constant
        )

        # Test binary expression
        code = """
        subroutine test_assign()
            jsl = nslm + 1
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Level_2_Expr)
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.BinOp)
        assert isinstance(result.op, ast.Add)
        assert isinstance(result.left, ast.Name)
        assert isinstance(result.right, ast.Constant)

        # Test int_literal constant
        code = """
        subroutine test_assign()
            jsl = 42
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Int_Literal_Constant)
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.Constant)
        assert result.value == 42

        # Test logical value
        code = """
        subroutine test_assign()
            jsl = .TRUE.
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Logical_Literal_Constant)
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.Constant)
        assert result.value is True

        # Test name
        code = """
        subroutine test_assign()
            jsl = nslm
        end subroutine test_assign
        """
        tree = self.parse(code)
        stmt = walk(walk(tree, F23.Assignment_Stmt), F23.Name)[1]
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.Name)
        assert result.id == "nslm"

    def test_recursive_ast(self):
        # Simple case
        code = """
        subroutine add_numbers(a, b, result)
            implicit none
            real :: a, b, result
            result = a + b

        end subroutine add_numbers
        """
        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)
        assert isinstance(result[0], ast.FunctionDef)
        assert result[0].name == "add_numbers"
        assert (
            len(result[0].body) == 1
        )  # Here the real :: a,b, result are not transformed since they are function args neither implicit none
        assert isinstance(result[0].body[0], ast.Assign)

        # Complex case
        code = """
        subroutine complex_logic(arr, n, threshold, result)
            implicit none
            integer :: n, i, j
            real :: arr(n), threshold, result
            real :: temp

            result = 0.0

            do i = 1, n
                if (arr(i) > threshold) then

                    temp = arr(i)

                    do j = 1, i
                        if (mod(j, 2) == 0) then
                            temp = temp + j
                        else
                            if (temp > 100.0) then
                                call log_value(temp)
                            else
                                temp = temp - j
                            end if
                        end if
                    end do

                    result = result + temp

                else if (arr(i) == threshold) then

                    call handle_equal(arr(i))

                else

                    result = result - arr(i)

                end if
            end do

            if (result > 1000.0) then
                call finalize_result(result)
            end if

        end subroutine complex_logic
        """
        self.f2np.extractor.all_array_info = {"test": {"arr": []}}
        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)
        func = result[0]
        assert isinstance(func, ast.FunctionDef)
        assert func.name == "complex_logic"
        assert [arg.arg for arg in func.args.args] == [
            "arr",
            "n",
            "threshold",
            "result",
        ]
        # Structural checks
        assert any(isinstance(node, ast.For) for node in ast.walk(func))
        assert any(isinstance(node, ast.If) for node in ast.walk(func))
        # Calls
        calls = [node for node in ast.walk(func) if isinstance(node, ast.Call)]
        call_names = [
            c.func.id if isinstance(c.func, ast.Name) else None for c in calls
        ]

        assert "log_value" in call_names
        assert "handle_equal" in call_names
        assert "finalize_result" in call_names

        if_nodes = [node for node in ast.walk(func) if isinstance(node, ast.If)]
        assert len(if_nodes) >= 3  # ensures nesting was preserved
        # print(ast.unparse(ast.fix_missing_locations(func)))
        code_out = ast.unparse(ast.fix_missing_locations(func))
        assert "result =" in code_out
        expected_output = """
def complex_logic(arr, n, threshold, result):
    result = 0.0
    for i in range(0, n, 1):
        if arr[i] > threshold:
            temp = arr[i]
            for j in range(0, i, 1):
                if np.mod(j, 2) == 0:
                    temp = temp + j
                elif temp > 100.0:
                    log_value(temp)
                else:
                    temp = temp - j
            result = result + temp
        elif arr[i] == threshold:
            handle_equal(arr[i])
        else:
            result = result - arr[i]
    if result > 1000.0:
        finalize_result(result)
        """
        expected_ast = ast.parse(expected_output).body[0]
        assert_ast_equal(func, expected_ast)

        # Complex edge case
        code = """
        subroutine extreme_case(x, y, n)
            implicit none
            integer :: n, i
            real :: x(n), y, acc

            acc = 0.0

            do i = 1, n
                if (x(i) > 0.0) then
                    if (x(i) > y) then
                        call process_high(x(i))
                        acc = acc + x(i)
                    else
                        if (mod(i, 3) == 0) then
                            call process_mod(i)
                        else
                            acc = acc - x(i)
                        end if
                    end if
                else
                    if (x(i) < -y) then
                        call process_negative(x(i))
                    end if
                end if
            end do

            call finalize(acc)

        end subroutine extreme_case
        """
        self.f2np.extractor.all_array_info = {"test": {"x": []}}
        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)
        func = result[0]
        assert isinstance(func, ast.FunctionDef)
        assert func.name == "extreme_case"
        assert [arg.arg for arg in func.args.args] == ["x", "y", "n"]
        # Structural checks
        assert any(isinstance(node, ast.For) for node in ast.walk(func))
        assert any(isinstance(node, ast.If) for node in ast.walk(func))
        # Calls
        calls = [node for node in ast.walk(func) if isinstance(node, ast.Call)]
        call_names = [
            c.func.id if isinstance(c.func, ast.Name) else None for c in calls
        ]
        assert "process_high" in call_names
        assert "process_negative" in call_names
        assert "process_mod" in call_names
        assert "finalize" in call_names

        # print(ast.unparse(ast.fix_missing_locations(func)))
        if_nodes = [node for node in ast.walk(func) if isinstance(node, ast.If)]
        assert len(if_nodes) == 4

        expected_output = """
def extreme_case(x, y, n):
    acc = 0.0
    for i in range(0, n, 1):
        if x[i] > 0.0:
            if x[i] > y:
                process_high(x[i])
                acc = acc + x[i]
            elif np.mod(i, 3) == 0:
                process_mod(i)
            else:
                acc = acc - x[i]
        elif x[i] < -y:
            process_negative(x[i])
    finalize(acc)
        """
        expected_ast = ast.parse(expected_output).body[0]
        assert_ast_equal(func, expected_ast)
