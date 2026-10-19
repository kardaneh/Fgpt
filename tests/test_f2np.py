import ast
import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from unittest.mock import MagicMock

import numpy as np
import pytest
from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

from fgpt.core.common.logger import Logger
from fgpt.core.frontend.processor import Processor
from fgpt.core.transpiler.f2np import F2NP
from fgpt.core.transpiler.intrinsic import build_fortran_reshape_helper


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


# PEP 695 type parameters, added to the AST in Python 3.12. ast.parse() sets this
# field on function and class nodes; nodes we build by hand never do, so a parsed
# AST and a constructed one compare unequal on 3.12+ despite being equivalent.
IGNORED_FIELDS = frozenset({"type_params"})


def ast_to_dict(node):
    """
    Convert AST into a nested dict for stable comparison.
    """
    if isinstance(node, ast.AST):
        return {
            "_type": type(node).__name__,
            **{
                field: ast_to_dict(value)
                for field, value in ast.iter_fields(node)
                if field not in IGNORED_FIELDS
            },
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

        # Regression test: uneven interleaving of string literals and values
        # (4 strings, 3 values) must not misalign the f-string.
        code = """
        subroutine do_neg()
            print *, "Data processed successfully!", &
                                " a=", a, " b=", b, " c=", c
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        call = result.value
        joined = call.args[0]
        assert isinstance(joined, ast.JoinedStr)

        code_out = ast.unparse(ast.fix_missing_locations(call))
        assert (
            code_out
            == "logging.info(f'Data processed successfully! a={a} b={b} c={c}')"
        )

        # Bare consecutive values (no label strings between them) still get
        # a readable separator.
        code = """
        subroutine do_neg()
            print *, a, b, c
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        code_out = ast.unparse(ast.fix_missing_locations(result.value))
        assert code_out == "logging.info(f'{a}, {b}, {c}')"

        code = """
        subroutine do_neg()
            print '(A,ES12.4,A,I0,A,I0,A)', "Data reset to ", value, &
            " over [", lo, ",", hi, "]"
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        code_out = ast.unparse(ast.fix_missing_locations(result.value))
        assert code_out == "logging.info(f'Data reset to {value} over [{lo},{hi}]')"

    def test_resolve_var_dtype_finds_global_declaration(self):
        code = """
        module m
            real :: gvar
            integer :: gcount
        contains
            subroutine t()
            end subroutine t
        end module m
        """
        tree = self.parse(code)
        self.f2np.func_name = "t"
        self.f2np.extractor.var_dummy = {"t": []}
        self.f2np.extractor.dec_global = {
            "t": {
                "gvar": [
                    d
                    for d in walk(tree, F23.Type_Declaration_Stmt)
                    if d.tostr().startswith("REAL")
                ],
                "gcount": [
                    d
                    for d in walk(tree, F23.Type_Declaration_Stmt)
                    if d.tostr().startswith("INTEGER")
                ],
            }
        }

        assert self.f2np._resolve_var_dtype("gvar") == ("np", "float64")
        assert self.f2np._resolve_var_dtype("gcount") == ("np", "int32")

    def test_resolve_var_dtype_prefers_dummy_over_global(self):
        # A dummy arg shadowing a global name of a different type should
        # resolve via var_dummy, not accidentally fall through to dec_global.
        code = """
        subroutine t(x)
            integer, intent(in) :: x
        end subroutine t
        """
        decl = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        self.f2np.func_name = "t"
        self.f2np.extractor.var_dummy = {"t": [decl]}
        self.f2np.extractor.dec_global = {"t": {"x": []}}  # no matching entity here

        assert self.f2np._resolve_var_dtype("x") == ("np", "int32")

    def test_resolve_var_dtype_unresolvable_name_defaults_to_float64(self):
        self.f2np.func_name = "t"
        self.f2np.extractor.var_dummy = {"t": []}
        self.f2np.extractor.dec_global = {"t": {}}

        assert self.f2np._resolve_var_dtype("unknown_local") == ("np", "float64")

    def test_is_supported_print_target(self):
        # PRINT * -> supported (list-directed)
        stmt = self.parse_and_get(
            'subroutine t()\nprint *, "hello"\nend subroutine t\n', F23.Print_Stmt
        )
        assert self.f2np._is_supported_print_target(stmt) is True

        # PRINT is always supported (no unit specifier exists in Fortran's
        # PRINT), regardless of its format
        for code in (
            'subroutine t()\nprint *, "hello"\nend subroutine t\n',
            "subroutine t()\nprint '(A)', \"hello\"\nend subroutine t\n",
        ):
            stmt = self.parse_and_get(code, F23.Print_Stmt)
            assert self.f2np._is_supported_print_target(stmt) is True

        # WRITE(*,*) / WRITE(6,...) / WRITE(numout,...) -> supported (stdout)
        for code in (
            'subroutine t()\nwrite(*,*) "hello"\nend subroutine t\n',
            'subroutine t()\nwrite(6,*) "hello"\nend subroutine t\n',
            'subroutine t()\nwrite(numout,*) "hello"\nend subroutine t\n',
            'subroutine t()\nwrite(unit=6, fmt=*) "hello"\nend subroutine t\n',
            'subroutine t()\nwrite(numout,100) "hello"\nend subroutine t\n',
            "subroutine t()\nwrite(6,'(A)') \"hello\"\nend subroutine t\n",
        ):
            stmt = self.parse_and_get(code, F23.Write_Stmt)
            assert self.f2np._is_supported_print_target(stmt) is True

        # WRITE(1363,*) -> unsupported (a genuine file unit)
        stmt = self.parse_and_get(
            'subroutine t()\nwrite(1363,*) "hello"\nend subroutine t\n', F23.Write_Stmt
        )
        assert self.f2np._is_supported_print_target(stmt) is False

        # WRITE(*,*) -> supported
        stmt = self.parse_and_get(
            'subroutine t()\nwrite(*,*) "hello"\nend subroutine t\n', F23.Write_Stmt
        )
        assert self.f2np._is_supported_print_target(stmt) is True

        # WRITE(6,*) -> supported (unit 6 is conventional stdout)
        stmt = self.parse_and_get(
            'subroutine t()\nwrite(6,*) "hello"\nend subroutine t\n', F23.Write_Stmt
        )
        assert self.f2np._is_supported_print_target(stmt) is True

        # WRITE(numout,*) -> supported (variable assumed to alias stdout)
        stmt = self.parse_and_get(
            'subroutine t()\nwrite(numout,*) "hello"\nend subroutine t\n',
            F23.Write_Stmt,
        )
        assert self.f2np._is_supported_print_target(stmt) is True

        # WRITE(UNIT=6, FMT=*) -> supported (explicit keyword form)
        stmt = self.parse_and_get(
            'subroutine t()\nwrite(unit=6, fmt=*) "hello"\nend subroutine t\n',
            F23.Write_Stmt,
        )
        assert self.f2np._is_supported_print_target(stmt) is True

        # WRITE(1363,*) -> unsupported (a genuine file unit)
        stmt = self.parse_and_get(
            'subroutine t()\nwrite(1363,*) "hello"\nend subroutine t\n', F23.Write_Stmt
        )
        assert self.f2np._is_supported_print_target(stmt) is False

    def test_recursive_ast_write_stmt_targets(self):
        # WRITE(6,*) and WRITE(*,*) should lower to logging.info, same as PRINT
        for unit in ("6", "*"):
            code = f"""
            subroutine test_write()
                write({unit},*) "hello", var
            end subroutine test_write
            """
            stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
            _, _, result = self.f2np.recursive_ast(stmt)
            func = result[0]
            assert len(func.body) == 1
            call = func.body[0].value
            assert isinstance(call, ast.Call)
            assert call.func.attr == "info"

        # A genuine file unit (not 6, not stdout) must still be rejected
        code = """
        subroutine test_write_file()
            write(1363,*) "hello", var
        end subroutine test_write_file
        """
        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        with pytest.raises(NotImplementedError):
            self.f2np.recursive_ast(stmt)

    def test_handle_intrinsic_function_reference_trim(self):
        code = """
        subroutine do_neg()
            m1 = TRIM(name)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Intrinsic_Function_Reference)
        result = self.f2np.handle_intrinsic_function_reference(stmt)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Attribute)
        assert result.func.attr == "strip"
        assert (
            isinstance(result.func.value, ast.Name) and result.func.value.id == "name"
        )
        assert result.args == []

        call_code = ast.unparse(ast.fix_missing_locations(result))
        assert call_code == "name.strip()"
        assert eval(call_code, {"name": "hello   "}) == "hello"

    def test_handle_print_stmt_with_trim(self):
        code = """
        subroutine do_neg()
            PRINT *, TRIM(name)
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        assert isinstance(result, ast.Expr)
        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "info"
        assert isinstance(call.args[0], ast.JoinedStr)

        formatted = call.args[0].values[0]
        assert isinstance(formatted, ast.FormattedValue)
        inner_call = formatted.value
        assert isinstance(inner_call, ast.Call)
        assert inner_call.func.attr == "strip"
        assert inner_call.func.value.id == "name"

        code = """
        subroutine do_neg()
            PRINT *, "Value:", TRIM(var)
        end subroutine do_neg
        """

        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        assert isinstance(result, ast.Expr)
        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "info"
        assert isinstance(call.args[0], ast.JoinedStr)
        joined_values = call.args[0].values
        assert any(
            isinstance(v, ast.Constant) and v.value == "Value:" for v in joined_values
        )
        assert any(
            isinstance(v, ast.FormattedValue)
            and isinstance(v.value, ast.Call)
            and v.value.func.attr == "strip"
            for v in joined_values
        )

        # Regression test: TRIM(x) // "literal" as a single print item requires
        # handle_expr to support Level_3_Expr (Fortran's `//` concatenation).
        code = """
        subroutine do_neg()
            print '(A,A,A,A)', "Data written to file: ", trim(filename), &
                    "  (binary: ", trim(binname)//")"
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        call = result.value
        joined = call.args[0]
        assert isinstance(joined, ast.JoinedStr)

        code_out = ast.unparse(ast.fix_missing_locations(call))
        assert code_out == (
            "logging.info(f\"Data written to file: {filename.strip()}  (binary: {binname.strip() + ')'}\")"
        )

        code = """
        subroutine do_neg()
            b = trim(binname)//")"
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Level_3_Expr)
        result = self.f2np.handle_expr(stmt)

        assert isinstance(result, ast.BinOp)
        assert isinstance(result.op, ast.Add)
        assert isinstance(result.left, ast.Call)
        assert result.left.func.attr == "strip"
        assert result.left.func.value.id == "binname"
        assert isinstance(result.right, ast.Constant)
        assert result.right.value == ")"

    def test_handle_print_with_implied_do(self):
        code = """
        subroutine do_test()
            print *, (a(i), i=1,5)
        end subroutine do_test
        """

        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)
        assert isinstance(result, ast.Expr)

        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "info"

        assert isinstance(call.args[0], ast.Call)
        assert isinstance(call.args[0].args[0], ast.Call)
        assert (
            isinstance(call.args[0].args[0].func, ast.Name)
            and call.args[0].args[0].func.id == "map"
        )

        code = """
        subroutine do_test()
            print *, (2*i, i=1,5)
        end subroutine do_test
        """

        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        assert isinstance(result, ast.Expr)

        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "info"

        assert isinstance(call.args[0], ast.Call)
        assert isinstance(call.args[0].args[0], ast.Call)
        assert (
            isinstance(call.args[0].args[0].func, ast.Name)
            and call.args[0].args[0].func.id == "map"
        )

        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert (
            code_out
            == "logging.info(' '.join(map(str, [2 * i for i in range(0, 5, 1)])))"
        )

        code = """
        subroutine do_test()
            print *, "Values:", (a(i), i=1,5)
        end subroutine do_test
        """

        stmt = self.parse_and_get(code, F23.Print_Stmt)
        result = self.f2np.handle_print_stmt(stmt)
        assert isinstance(result, ast.Expr)

        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.attr == "info"
        joined = call.args[0]
        assert isinstance(joined, ast.JoinedStr)

        assert any(
            isinstance(v, ast.Constant) and v.value == "Values:" for v in joined.values
        )

    def test_handle_write_with_implied_do(self, caplog):
        code = """
        subroutine do_test()
            write(*, *) (a(i), i=1,5)
        end subroutine do_test
        """

        stmt = self.parse_and_get(code, F23.Write_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        assert isinstance(result, ast.Expr)

        call = result.value
        assert isinstance(call, ast.Call)

        # Check the generated join(map(...)) structure
        assert isinstance(call.args[0], ast.Call)
        assert isinstance(call.args[0].args[0], ast.Call)

        assert (
            isinstance(call.args[0].args[0].func, ast.Name)
            and call.args[0].args[0].func.id == "map"
        )

        code = """
        subroutine do_test()
            write(*, *) (2*i, i=1,5)
        end subroutine do_test
        """

        stmt = self.parse_and_get(code, F23.Write_Stmt)
        result = self.f2np.handle_print_stmt(stmt)

        assert isinstance(result, ast.Expr)

        call = result.value
        assert isinstance(call, ast.Call)

        assert isinstance(call.args[0], ast.Call)
        assert isinstance(call.args[0].args[0], ast.Call)
        assert (
            isinstance(call.args[0].args[0].func, ast.Name)
            and call.args[0].args[0].func.id == "map"
        )

        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert code_out == (
            "logging.info(' '.join(map(str, [2 * i for i in range(0, 5, 1)])))"
        )

        code = """
        subroutine do_test()
            write(*, *) "Values:", (2*i, i=1,5)
        end subroutine do_test
        """

        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        func = self.f2np.recursive_ast(walk(stmt, F23.Subroutine_Subprogram)[0])[2][0]
        result = func.body[0]

        assert isinstance(result, ast.Expr)

        call = result.value
        assert isinstance(call, ast.Call)

        # We expect the final output to be a formatted string
        joined = call.args[0]
        assert isinstance(joined, ast.JoinedStr)

        assert any(
            isinstance(v, ast.Constant) and v.value == "Values:" for v in joined.values
        )
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np, "logging": logging}

        exec(
            compile(code_out, "<emitted>", "exec"),
            namespace,
        )

        with caplog.at_level(logging.INFO):
            namespace["do_test"]()

        output = caplog.text
        assert "Values:[0, 2, 4, 6, 8]" in output

    def test_inq_file_exists(self, tmp_path):
        code = """
        logical function inq_file_exists(filename)
            implicit none
            character(len=*), intent(in) :: filename
            logical :: there

            inquire(file=filename, exist=there)
            inq_file_exists = there
        end function inq_file_exists
        """
        tree = self.parse(code)
        func_subprogram = self.get_stmt(tree, F23.Function_Subprogram)
        _, _, result = self.f2np.recursive_ast(func_subprogram)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"os": os}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        existing = tmp_path / "present.txt"
        existing.write_text("x")
        assert namespace["inq_file_exists"](str(existing)) is True
        assert namespace["inq_file_exists"](str(tmp_path / "absent.txt")) is False

    def test_inq_file_size(self, tmp_path):
        code = """
        integer function inq_file_size(filename)
            implicit none
            character(len=*), intent(in) :: filename
            integer :: sz
            logical :: there

            inquire(file=filename, exist=there, size=sz)
            if (.not. there) then
                inq_file_size = 0
            else
                inq_file_size = sz
            end if
        end function inq_file_size
        """
        tree = self.parse(code)
        func_subprogram = self.get_stmt(tree, F23.Function_Subprogram)
        _, _, result = self.f2np.recursive_ast(func_subprogram)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"os": os}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        existing = tmp_path / "present.txt"
        existing.write_text("hello world")  # 11 bytes
        assert namespace["inq_file_size"](str(existing)) == 11
        assert namespace["inq_file_size"](str(tmp_path / "absent.txt")) == 0

    def test_handle_inquire_stmt_requires_file_spec(self):
        code = """
        subroutine t(u, there)
            integer :: u
            logical :: there
            inquire(unit=u, exist=there)
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Inquire_Stmt)
        with pytest.raises(NotImplementedError):
            self.f2np.handle_inquire_stmt(stmt)

    def test_handle_inquire_stmt_rejects_unsupported_specifiers(self):
        code = """
        subroutine t(filename, is_opened)
            character(len=*) :: filename
            logical :: is_opened
            inquire(file=filename, opened=is_opened)
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Inquire_Stmt)
        with pytest.raises(NotImplementedError):
            self.f2np.handle_inquire_stmt(stmt)

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
        assert isinstance(result.slice.lower, ast.Constant)
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

        # Vector subscipts B = A((/1, 3, 5/))
        # meaning pick out elements 1, 3, and 5 of A
        code = """
        subroutine do_neg()
            b = a((/1, 3, 5/))
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        result = self.f2np.handle_part_ref(stmt)

        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.List)
        assert [e.value for e in result.slice.elts] == [0, 2, 4]  # shifted -1

        # Bracket syntax equivalent
        code = """
        subroutine do_neg()
            b = a([2, 4])
        end subroutine do_neg
        """
        stmt = self.parse_and_get(code, F23.Part_Ref)
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        result = self.f2np.handle_part_ref(stmt)

        assert [e.value for e in result.slice.elts] == [1, 3]

        expr = "a((/1, 3, 5/))"
        code = (
            "subroutine t(a, b)\n"
            "  real :: a(6), b(3)\n"
            f"  b = {expr}\n"
            "end subroutine t\n"
        )
        tree = Processor(logger=Logger()).parse_fortran_string(code)
        self.f2np.extractor.all_array_info = {"test": {"a": []}}
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {
            "np": np,
        }
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        a = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
        b = np.zeros(3)
        b = namespace["t"](a, b)

        # Fortran indices 1,3,5 -> values 10, 30, 50
        assert list(b) == [10.0, 30.0, 50.0]

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

        # Not equal
        code = """
        subroutine do_neg()
            b = a /= b
        end subroutine do_neg
        """
        # `/=`` doesn't exist within the conditional or the replacement dict
        stmt = self.parse_and_get(code, F23.Level_4_Expr)
        result = self.f2np.handle_level_4expr(stmt)

        assert isinstance(result.ops[0], ast.NotEq)
        assert result.left.id == "a"
        assert (
            isinstance(result.comparators[0], ast.Name)
            and result.comparators[0].id == "b"
        )

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

        # Array constructor:
        # Basic multi-element literal: (/2, 3/)
        code = """
        subroutine test_assign()
            b = RESHAPE(a, (/2, 3/))
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Array_Constructor)
        result = self.f2np.handle_expr(stmt)

        assert isinstance(result, ast.List)
        assert len(result.elts) == 2
        assert all(isinstance(e, ast.Constant) for e in result.elts)
        assert [e.value for e in result.elts] == [2, 3]

        # Single-element literal: (/5/)
        code = """
        subroutine test_assign()
            b = RESHAPE(a, (/5/))
        end subroutine test_assign
        """
        # (/5/) is still considered as array_constructor
        stmt = self.parse_and_get(code, F23.Array_Constructor)
        result = self.f2np.handle_expr(stmt)

        assert isinstance(result, ast.List)
        assert len(result.elts) == 1
        assert result.elts[0].value == 5

        # Literal used as SOURCE (not just SHAPE)
        code = """
        subroutine test_assign()
            b = RESHAPE((/1, 2, 3, 4, 5, 6/), shp)
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Array_Constructor)
        result = self.f2np.handle_expr(stmt)

        assert isinstance(result, ast.List)
        assert [e.value for e in result.elts] == [1, 2, 3, 4, 5, 6]

        # Array constructor brackets[] instead of (//) style
        code = """
        subroutine test_assign()
            b = RESHAPE(a, [2, 3])
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Array_Constructor)
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.List)
        assert [e.value for e in result.elts] == [2, 3]

        # Works fine for the expressions in the array constuctor
        code = """
        subroutine test_assign()
            b = RESHAPE(a, (/n, m + 1/))
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Array_Constructor)
        result = self.f2np.handle_expr(stmt)
        assert isinstance(result, ast.List)
        assert len(result.elts) == 2
        assert isinstance(result.elts[0], ast.Name) and result.elts[0].id == "n"
        assert isinstance(result.elts[1], ast.BinOp)  # m + 1

        code = """
        subroutine test_assign()
            b = RESHAPE((/1.0, 2.0, 3.0, 4.0, 5.0, 6.0/), SHP)
        end subroutine test_assign
        """
        stmt = self.parse_and_get(code, F23.Array_Constructor)
        result = self.f2np.handle_expr(stmt)
        assert all(
            isinstance(e, ast.Constant) and isinstance(e.value, float)
            for e in result.elts
        )

        stmt = self.parse_and_get(
            "subroutine t()\narr = (/ (i, i=1,10) /)\nend subroutine t\n",
            F23.Array_Constructor,
        )
        result = self.f2np.handle_expr(stmt)

        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert isinstance(result, ast.ListComp)
        assert code_out == "[i for i in range(0, 10, 1)]"
        assert eval(code_out)

        stmt = self.parse_and_get(
            "subroutine t()\narr = (/ (2*i-1, i=1,n,2) /)\nend subroutine t\n",
            F23.Array_Constructor,
        )
        result = self.f2np.handle_expr(stmt)

        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert isinstance(result, ast.ListComp)
        assert isinstance(result.elt, ast.BinOp)
        assert code_out == "[2 * i - 1 for i in range(0, n, 2)]"
        assert eval(code_out, {"n": 10})  # passe for n = 10

        self.f2np.extractor.all_array_info = {"test": {"matrix": []}}
        stmt = self.parse_and_get(
            "subroutine t()\narr = (/ ((matrix(i,j), j=1,3), i=1,2) /)\nend subroutine t\n",
            F23.Array_Constructor,
        )
        result = self.f2np.handle_expr(stmt)
        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert isinstance(result, ast.ListComp)
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # matrix[i][j], 0-based
        assert eval(code_out, {"matrix": matrix})

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

        code_out = ast.unparse(ast.fix_missing_locations(func))
        assert "result =" in code_out
        expected_output = """
def complex_logic(arr, n, threshold, result):
    result = 0.0
    for i in range(0, n, 1):
        if arr[i] > threshold:
            temp = arr[i]
            for j in range(0, i + 1, 1):
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
        assert_ast_equal(func, ast.fix_missing_locations(expected_ast))

        # Outer dependant loop
        code = """
        subroutine do_neg()
            integer :: i, j
            do i = 1, 10
                do j = 1, i
                    arr(i) = x(j)
                end do
            end do
        end subroutine do_neg
        """
        self.f2np.extractor.all_array_info = {"do_neg": {"arr": [], "x": []}}
        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)
        func = result[0]
        assert isinstance(func, ast.FunctionDef)
        assert func.name == "do_neg"

        for_loops = [node for node in ast.walk(func) if isinstance(node, ast.For)]
        assert len(for_loops) == 2
        # since in fortran, the inner loop just the outer loop variable as the stop
        # thus we need to have it at + 1 due to the python exclusive rule
        assert (
            ast.unparse(ast.fix_missing_locations(for_loops[0].iter))
            == "range(0, 10, 1)"
        )
        assert (
            ast.unparse(ast.fix_missing_locations(for_loops[1].iter))
            == "range(0, i + 1, 1)"
        )

        code = """
        subroutine do_neg()
            integer :: i, j
            do i = 1, 10
                do j = i, 1
                    arr(i) = x(j)
                end do
            end do
        end subroutine do_neg
        """
        self.f2np.extractor.all_array_info = {"do_neg": {"arr": [], "x": []}}
        stmt = self.parse_and_get(code, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)
        func = result[0]
        assert isinstance(func, ast.FunctionDef)
        assert func.name == "do_neg"

        for_loops = [node for node in ast.walk(func) if isinstance(node, ast.For)]
        assert len(for_loops) == 2
        # since in fortran, the inner loop just the outer loop variable as the start
        # meaning they have the same iteration value thus the outer loop runs on the values of the inner
        assert (
            ast.unparse(ast.fix_missing_locations(for_loops[0].iter))
            == "range(0, 10, 1)"
        )
        assert (
            ast.unparse(ast.fix_missing_locations(for_loops[1].iter))
            == "range(i, 1, 1)"
        )

    def test_extract_dtype_derived_type(self):
        code = """
        subroutine test()
            TYPE(laieff_type) :: laieff_fit
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        idx, attr = self.f2np._extract_dtype(stmt)

        # Sentinel: idx is None for derived types, attr carries the type name
        assert idx is None
        assert attr == "laieff_type"

    def test_extract_dtype_still_handles_intrinsics(self):
        # Regression check: derived-type branch shouldn't break intrinsics
        code = """
        subroutine test()
            real :: x
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        idx, attr = self.f2np._extract_dtype(stmt)

        assert idx == "np"
        assert attr == "float64"

    def test_handle_type_declaration_stmt_derived_type_no_init(self):
        code = """
        subroutine test()
            TYPE(laieff_type) :: laieff_fit
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        result = self.f2np.handle_type_declaration_stmt(stmt)

        assert isinstance(result, ast.Assign)
        assert result.targets[0].id == "laieff_fit"

        # laieff_fit = LaieffType() (or laieff_type(), depending on casing convention)
        assert isinstance(result.value, ast.Call)
        assert isinstance(result.value.func, ast.Name)
        assert result.value.func.id == "laieff_type"
        assert result.value.args == []
        assert result.value.keywords == []

    def test_handle_type_declaration_stmt_derived_type_with_structure_constructor(self):
        code = """
        subroutine test()
            TYPE(laieff_type) :: laieff_fit = laieff_type(1.0, 2.0)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        result = self.f2np.handle_type_declaration_stmt(stmt)

        assert isinstance(result, ast.Assign)
        assert result.targets[0].id == "laieff_fit"
        assert isinstance(result.value, ast.Call)
        assert result.value.func.id == "laieff_type"
        assert len(result.value.args) == 2
        assert all(isinstance(a, ast.Constant) for a in result.value.args)
        assert [a.value for a in result.value.args] == [1.0, 2.0]

    def test_handle_structure_constructor_positional(self):
        code = """
        subroutine test()
            laieff_fit = laieff_type(1.0, arr_vals)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Structure_Constructor)
        result = self.f2np.handle_structure_constructor(stmt)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Name)
        assert result.func.id == "laieff_type"

        assert len(result.args) == 2
        assert isinstance(result.args[0], ast.Constant) and result.args[0].value == 1.0
        assert isinstance(result.args[1], ast.Name) and result.args[1].id == "arr_vals"
        assert result.keywords == []

    def test_handle_structure_constructor_keyword(self):
        code = """
        subroutine test()
            laieff_fit = laieff_type(comp1=1.0, comp2=arr_vals)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Structure_Constructor)
        result = self.f2np.handle_structure_constructor(stmt)

        assert isinstance(result, ast.Call)
        assert result.args == []
        assert len(result.keywords) == 2

        kw_names = [kw.arg for kw in result.keywords]
        assert kw_names == ["comp1", "comp2"]
        assert isinstance(result.keywords[0].value, ast.Constant)
        assert result.keywords[0].value.value == 1.0
        assert isinstance(result.keywords[1].value, ast.Name)
        assert result.keywords[1].value.id == "arr_vals"

    def test_handle_structure_constructor_mixed_positional_and_keyword(self):
        code = """
        subroutine test()
            laieff_fit = laieff_type(1.0, comp2=arr_vals)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Structure_Constructor)
        result = self.f2np.handle_structure_constructor(stmt)

        assert len(result.args) == 1
        assert result.args[0].value == 1.0
        assert len(result.keywords) == 1
        assert result.keywords[0].arg == "comp2"

    def test_handle_structure_constructor_empty(self):
        code = """
        subroutine test()
            laieff_fit = laieff_type()
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Structure_Constructor)
        result = self.f2np.handle_structure_constructor(stmt)

        assert isinstance(result, ast.Call)
        assert result.func.id == "laieff_type"
        assert result.args == []
        assert result.keywords == []

    def test_handle_structure_constructor_via_handle_expr(self):
        # Ensure handle_expr dispatches to handle_structure_constructor
        code = """
        subroutine test()
            laieff_fit = laieff_type(1.0, 2.0)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Structure_Constructor)
        result = self.f2np.handle_expr(stmt)

        assert isinstance(result, ast.Call)
        assert result.func.id == "laieff_type"

    def test_handle_assignment_rhs_structure_constructor(self):
        # RHS of a plain assignment can be a structure constructor
        code = """
        subroutine test()
            laieff_fit = laieff_type(1.0, 2.0)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        result = self.f2np.handle_assignment(stmt)

        assert isinstance(result, ast.Assign)
        assert result.targets[0].id == "laieff_fit"
        assert isinstance(result.value, ast.Call)
        assert result.value.func.id == "laieff_type"

    def test_handle_structure_constructor_as_call_argument(self):
        code = """
        subroutine test()
            call foo(laieff_type(1.0, 2.0))
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Call_Stmt)
        self.f2np.extractor.allowed_external_subroutines = []
        result = self.f2np.handle_call_stmt(stmt)

        assert isinstance(result, ast.Expr)
        call = result.value
        assert isinstance(call, ast.Call)
        assert call.func.id == "foo"
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Call)
        assert call.args[0].func.id == "laieff_type"

    def test_handle_data_ref_simple_component(self):
        code = """
        subroutine test()
            b = laieff_fit%foo
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Data_Ref)
        result = self.f2np.handle_data_ref(stmt)

        assert isinstance(result, ast.Attribute)
        assert result.attr == "foo"
        assert isinstance(result.value, ast.Name)
        assert result.value.id == "laieff_fit"
        assert isinstance(result.ctx, ast.Load)

    def test_handle_data_ref_array_component_index(self):
        code = """
        subroutine test()
            b = laieff_fit%arr(2)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Data_Ref)
        result = self.f2np.handle_data_ref(stmt)

        assert isinstance(result, ast.Subscript)
        assert isinstance(result.value, ast.Attribute)
        assert result.value.attr == "arr"
        assert result.value.value.id == "laieff_fit"

        # Fortran index 2 -> Python index 1
        assert isinstance(result.slice, ast.Constant)
        assert result.slice.value == 2

    def test_handle_data_ref_array_component_slice(self):
        code = """
        subroutine test()
            b = laieff_fit%arr(1:5)
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Data_Ref)
        result = self.f2np.handle_data_ref(stmt)

        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Slice)
        assert isinstance(result.slice.lower, ast.Constant)
        assert result.slice.upper.value == 5

    def test_handle_data_ref_via_handle_expr(self):
        code = """
        subroutine test()
            b = laieff_fit%foo
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Data_Ref)
        result = self.f2np.handle_expr(stmt)

        assert isinstance(result, ast.Attribute)
        assert result.attr == "foo"

    def test_handle_data_ref_array_of_structs_scalar_field(self):
        code = "subroutine test()\n b = laieff_fit(ipts,ivm,ilevel)%a\n end subroutine test"
        stmt = self.parse_and_get(code, F23.Data_Ref)
        result = self.f2np.handle_data_ref(stmt)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.value, ast.Attribute)
        assert result.value.attr == "a"
        assert result.value.value.id == "laieff_fit"
        assert isinstance(result.slice, ast.Tuple)
        assert len(result.slice.elts) == 3
        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert code_out == "laieff_fit.a[ipts, ivm, ilevel]"

    def test_handle_data_ref_array_of_structs_array_field(self):
        code = "subroutine test()\n b = soil_profile(ipts)%temp(ilevel)\n end subroutine test"
        stmt = self.parse_and_get(code, F23.Data_Ref)
        result = self.f2np.handle_data_ref(stmt)
        assert isinstance(result, ast.Subscript)
        assert result.value.attr == "temp"
        assert isinstance(result.slice, ast.Tuple)
        assert (
            len(result.slice.elts) == 2
        )  # outer idx + inner idx concatenated ipts, ilevel
        code_out = ast.unparse(ast.fix_missing_locations(result))
        assert code_out == "soil_profile.temp[ipts, ilevel]"

    def test_handle_data_ref_array_of_structs_scalar_field_runtime_write(self):
        class FortranDerivedType:
            @classmethod
            def zeros(cls, shape):
                kwargs = {}
                for f in fields(cls):
                    default = (
                        f.default_factory()
                        if f.default_factory is not MISSING
                        else f.default
                    )
                    if isinstance(default, np.ndarray):
                        full_shape, dtype = shape + default.shape, default.dtype
                    else:
                        full_shape, dtype = shape, np.asarray(default).dtype
                    kwargs[f.name] = np.zeros(full_shape, dtype=dtype)
                return cls(**kwargs)

            def __getitem__(self, idx):
                return type(self)(
                    **{f.name: getattr(self, f.name)[idx] for f in fields(self)}
                )

        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)

        # Build the write statement through the real pipeline, exactly as the
        # transpiler would emit it for `laieff_fit(ipts,ivm,ilevel) % a = 42.0`.
        code = """
        subroutine test()
            laieff_fit(1,0,1)%a = 42.0
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        self.f2np.extractor.all_array_info = {"test": {"laieff_fit": []}}
        assign = self.f2np.handle_assignment(stmt)
        src = ast.unparse(ast.fix_missing_locations(assign))
        assert src == "laieff_fit.a[1, 0, 1] = 42.0"

        laieff_fit = laieff_type.zeros((3, 2, 2))
        ns = {"laieff_fit": laieff_fit}
        exec(compile(src, "<emitted>", "exec"), ns)

        # The write landed on the real object, not a copy from __getitem__.
        assert laieff_fit.a[1, 0, 1] == 42.0
        assert laieff_fit.a[0, 0, 0] == 0.0  # untouched cells unaffected

        # Sanity: __getitem__ itself still works fine for the *other* use case
        # (passing one element by value into a function)
        elem = laieff_fit[1, 0, 1]
        assert isinstance(elem, laieff_type)
        assert elem.a == 42.0

    def test_handle_data_ref_array_of_structs_array_field_runtime_write(self):
        class FortranDerivedType:
            @classmethod
            def zeros(cls, shape):
                kwargs = {}
                for f in fields(cls):
                    default = (
                        f.default_factory()
                        if f.default_factory is not MISSING
                        else f.default
                    )
                    if isinstance(default, np.ndarray):
                        full_shape, dtype = shape + default.shape, default.dtype
                    else:
                        full_shape, dtype = shape, np.asarray(default).dtype
                    kwargs[f.name] = np.zeros(full_shape, dtype=dtype)
                instance = cls(**kwargs)
                instance._shape = tuple(shape)
                return instance

            def __getitem__(self, idx):
                return type(self)(
                    **{f.name: getattr(self, f.name)[idx] for f in fields(self)}
                )

        @dataclass
        class laieff_type(FortranDerivedType):
            a: np.ndarray = field(
                default_factory=lambda: np.zeros((10,), dtype=np.float64)
            )
            b: np.ndarray = field(
                default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
            )

        code = """
        subroutine test()
            laieff_fit(1,0,1)%a(1) = 42.0
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        self.f2np.extractor.all_array_info = {"test": {"laieff_fit": []}}
        assign = self.f2np.handle_assignment(stmt)
        src = ast.unparse(ast.fix_missing_locations(assign))
        assert src == "laieff_fit.a[1, 0, 1, 1] = 42.0"

        laieff_fit = laieff_type.zeros((3, 2, 2))
        ns = {"laieff_fit": laieff_fit}
        exec(compile(src, "<emitted>", "exec"), ns)

        # The write landed on the real object, not a copy from __getitem__.
        assert laieff_fit.a[1, 0, 1, 1] == 42.0
        assert laieff_fit.a[0, 0, 0, 0] == 0.0  # untouched cells unaffected

        elem = laieff_fit[1, 0, 1]
        assert isinstance(elem, laieff_type)
        assert elem.a[1] == 42.0

    def test_handle_assignment_lhs_data_ref(self):
        # laieff_fit%foo = 1.0 -> LHS must carry Store context
        code = """
        subroutine test()
            laieff_fit%foo = 1.0
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        result = self.f2np.handle_assignment(stmt)

        assert isinstance(result, ast.Assign)
        lhs = result.targets[0]
        assert isinstance(lhs, ast.Attribute)
        assert lhs.attr == "foo"
        assert isinstance(lhs.ctx, ast.Store)

        assert isinstance(result.value, ast.Constant)
        assert result.value.value == 1.0

    def test_handle_assignment_lhs_data_ref_array_component(self):
        code = """
        subroutine test()
            laieff_fit%arr(2) = 1.0
        end subroutine test
        """
        stmt = self.parse_and_get(code, F23.Assignment_Stmt)
        result = self.f2np.handle_assignment(stmt)

        lhs = result.targets[0]
        assert isinstance(lhs, ast.Subscript)
        assert isinstance(lhs.ctx, ast.Store)
        assert isinstance(lhs.value, ast.Attribute)
        assert lhs.value.attr == "arr"

    def test_build_dataclass_fields_scalar_intrinsic(self):
        code = """
        module test_mod
            TYPE :: laieff_type
                REAL :: comp1
            END TYPE laieff_type
        end module test_mod
        """
        decl = self.parse_and_get(code, F23.Data_Component_Def_Stmt)
        fields = self.f2np._build_dataclass_fields(decl)

        assert len(fields) == 1
        field = fields[0]
        assert isinstance(field, ast.AnnAssign)
        assert field.target.id == "comp1"
        assert isinstance(field.annotation, ast.Attribute)
        assert field.annotation.attr == "float64"

        # scalar default: np.float64(0)
        assert isinstance(field.value, ast.Call)
        assert field.value.func.attr == "float64"
        assert field.value.args[0].value == 0

    def test_build_dataclass_fields_array_intrinsic(self):
        code = """
        module test_mod
            TYPE :: laieff_type
                REAL :: arr(10)
            END TYPE laieff_type
        end module test_mod
        """
        decl = self.parse_and_get(code, F23.Data_Component_Def_Stmt)
        fields = self.f2np._build_dataclass_fields(decl)

        assert len(fields) == 1
        field = fields[0]
        assert field.target.id == "arr"
        assert field.annotation.attr == "ndarray"

        # array default: field(default_factory=lambda: np.zeros((10,), dtype=np.float64))
        assert isinstance(field.value, ast.Call)
        assert field.value.func.id == "field"
        assert field.value.keywords[0].arg == "default_factory"
        assert isinstance(field.value.keywords[0].value, ast.Lambda)

        lambda_body = field.value.keywords[0].value.body
        assert isinstance(lambda_body, ast.Call)
        assert lambda_body.func.attr == "zeros"

        code = """
        module test_mod
            TYPE :: profile_type
                REAL(KIND = r_std), DIMENSION(nlevels_tot) :: temp
                REAL(KIND = r_std), DIMENSION(nlevels_tot) :: dens
            END TYPE profile_type
        end module test_mod
        """

        decl = self.parse_and_get(code, F23.Data_Component_Def_Stmt)
        fields = self.f2np._build_dataclass_fields(decl)

        assert len(fields) == 1
        field = fields[0]
        assert field.target.id == "temp"
        assert field.annotation.attr == "ndarray"

        assert isinstance(field.value, ast.Call)
        assert field.value.func.id == "field"
        assert field.value.keywords[0].arg == "default_factory"
        assert isinstance(field.value.keywords[0].value, ast.Lambda)

        lambda_body = field.value.keywords[0].value.body
        assert isinstance(lambda_body, ast.Call)
        assert lambda_body.func.attr == "zeros"

    def test_build_dataclass_fields_nested_derived_type(self):
        code = """
        module test_mod
            TYPE :: outer_type
                TYPE(inner_type) :: comp
            END TYPE outer_type
        end module test_mod
        """
        decl = self.parse_and_get(code, F23.Data_Component_Def_Stmt)
        fields = self.f2np._build_dataclass_fields(decl)

        assert len(fields) == 1
        field = fields[0]
        assert field.target.id == "comp"
        assert field.annotation.id == "inner_type"

        # nested default: comp = inner_type()
        assert isinstance(field.value, ast.Call)
        assert field.value.func.id == "inner_type"
        assert field.value.args == []
        assert field.value.keywords == []

    def test_build_dataclass_fields_multiple_components_same_stmt(self):
        code = """
        module test_mod
            TYPE :: laieff_type
                REAL :: comp1, comp2
            END TYPE laieff_type
        end module test_mod
        """
        decl = self.parse_and_get(code, F23.Data_Component_Def_Stmt)
        fields = self.f2np._build_dataclass_fields(decl)

        assert len(fields) == 2
        assert [f.target.id for f in fields] == ["comp1", "comp2"]
        assert all(isinstance(f, ast.AnnAssign) for f in fields)

    def test_build_dataclass_mixed_components(self):
        code = """
        module test_mod
            type :: laieff_type
                real :: comp1
                real(kind=r_std), dimension(nlevels_tot) :: temp
                integer :: A
            end type laieff_type
        end module test_mod
        """

        decl = self.parse(code)
        fields = []
        for decl in walk(decl, F23.Data_Component_Def_Stmt | F23.Type_Declaration_Stmt):
            fields.extend(self.f2np._build_dataclass_fields(decl))

        assert len(fields) == 3
        assert all(isinstance(f, ast.AnnAssign) for f in fields)
        expected = ast.parse(
            "temp: np.ndarray = field(default_factory=lambda: np.zeros((nlevels_tot,), dtype=np.float64))"
        ).body[0]
        assert ast.dump(fields[1]) == ast.dump(expected)

        # Mixed components with and inner derived type
        code = """
        module test_mod
            type :: laieff_type
                real :: comp1
                real(kind=r_std), dimension(nlevels_tot) :: temp
                integer :: A
                TYPE(inner_type) :: comp
            end type laieff_type
        end module test_mod
        """

        decl = self.parse(code)
        fields = []
        for decl in walk(decl, F23.Data_Component_Def_Stmt | F23.Type_Declaration_Stmt):
            fields.extend(self.f2np._build_dataclass_fields(decl))

        assert len(fields) == 4
        assert all(isinstance(f, ast.AnnAssign) for f in fields)
        assert (
            isinstance(fields[-1].annotation, ast.Name)
            and fields[-1].annotation.id == "inner_type"
        )

    def test_recursive_ast_function_without_result_suffix(self):
        code = """
        REAL(KIND=r_std) FUNCTION calculate_laieff_fit(test_angle, laieff_fit)
            REAL(KIND=r_std), INTENT(IN) :: test_angle
            TYPE(laieff_type), INTENT(IN) :: laieff_fit
            calculate_laieff_fit = laieff_fit % a + test_angle * laieff_fit % b
        END FUNCTION calculate_laieff_fit
        """
        stmt = self.parse_and_get(code, F23.Function_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)

        func = result[0]
        assert isinstance(func, ast.FunctionDef)
        assert func.name == "calculate_laieff_fit"

        # Implicit result variable: function name doubles as return value
        return_nodes = [n for n in func.body if isinstance(n, ast.Return)]
        assert len(return_nodes) == 1
        assert return_nodes[0].value.id == "calculate_laieff_fit"

    def test_recursive_ast_function_with_result_suffix_still_works(self):
        # Regression guard: explicit RESULT(...) must still take precedence
        code = """
        REAL(KIND=r_std) FUNCTION compute(x) RESULT(y)
            REAL(KIND=r_std), INTENT(IN) :: x
            y = x * 2.0
        END FUNCTION compute
        """
        stmt = self.parse_and_get(code, F23.Function_Subprogram)
        _, _, result = self.f2np.recursive_ast(stmt)

        func = result[0]
        return_nodes = [n for n in func.body if isinstance(n, ast.Return)]
        assert len(return_nodes) == 1
        assert return_nodes[0].value.id == "y"

    def test_recursive_ast_prunes_empty_nested_if(self):
        code = """
        subroutine test_empty_if(almaoutput, impaze, hist2_id)
        LOGICAL :: almaoutput, impaze
        INTEGER :: hist2_id

        IF (.NOT. impaze) THEN
        END IF

        IF (almaoutput) THEN
          IF (hist2_id > 0) THEN
          END IF
        ELSE
          IF (.NOT. impaze) THEN
            IF (hist2_id > 0) THEN
            END IF
          END IF
        END IF

        end subroutine test_empty_if
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, module_stack = self.f2np.recursive_ast(subroutine)

        func_def = module_stack[-1]
        assert isinstance(func_def, ast.FunctionDef)

        # Every IF in the source, at every nesting level, was empty in
        # both branches -- all of them (including the outer IF/ELSE,
        # once its only content -- the inner empty IFs -- collapses)
        # should be pruned away entirely.
        assert not any(isinstance(n, ast.If) for n in ast.walk(func_def))

    def test_recursive_ast_keeps_partially_empty_if(self):
        code = """
        subroutine test_partial_if(almaoutput, hist2_id, numout)
        LOGICAL :: almaoutput
        INTEGER :: hist2_id, numout

        IF (almaoutput) THEN
        ELSE
          numout = hist2_id
        END IF

        end subroutine test_partial_if
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, module_stack = self.f2np.recursive_ast(subroutine)

        func_def = module_stack[-1]
        if_nodes = [n for n in ast.walk(func_def) if isinstance(n, ast.If)]

        # The IF/ELSE has real content in `orelse`, so it must survive --
        # but its empty `body` needs a `pass` inserted since Python
        # doesn't allow an empty if-body even when else has content.
        assert len(if_nodes) == 1
        assert len(if_nodes[0].body) == 1
        assert isinstance(if_nodes[0].body[0], ast.Pass)
        assert len(if_nodes[0].orelse) == 1
        assert isinstance(if_nodes[0].orelse[0], ast.Assign)


@pytest.mark.usefixtures("test_env")
class TestF2NPIntrinsicLowering:
    """
    End-to-end checks on the code emitted for Fortran intrinsics.

    ``tests/test_intrinsic.py`` only exercises ``normalize_intrinsic_call``,
    i.e. the normalized argument dictionary. These tests go one step further
    and assert on the NumPy call that is actually emitted, then execute it,
    which is what catches arguments emitted as keywords that NumPy only
    accepts positionally.
    """

    # Test out if the reshape helper flag is True when PAD and/or ORDER is
    # present
    def test_reshape_sets_helper_flag_only_when_needed(self):
        assert self.f2np.needs_fortran_reshape_helper is False

        self.unparse_intrinsic("RESHAPE(A, SHP)")
        assert self.f2np.needs_fortran_reshape_helper is False

        self.unparse_intrinsic("RESHAPE(A, SHP, PAD=B)")
        assert self.f2np.needs_fortran_reshape_helper is True

    def unparse_intrinsic(self, expr):
        code = (
            "subroutine t(a, b, shp, m, n, k)\n"
            "  real :: a(6), b(6)\n"
            "  integer :: shp(2), n, k\n"
            "  logical :: m(6)\n"
            f"  b = {expr}\n"
            "end subroutine t\n"
        )
        tree = Processor(logger=Logger()).parse_fortran_string(code)
        node = walk(tree, F23.Intrinsic_Function_Reference)[0]
        return ast.unparse(self.f2np.handle_intrinsic_function_reference(node))

    @pytest.mark.parametrize(
        "fortran, expected",
        [
            # np.matmul / np.dot are ufuncs: arguments are positional-only.
            ("MATMUL(A, B)", "np.matmul(A, B)"),
            ("DOT_PRODUCT(A, B)", "np.dot(A, B)"),
            # np.reshape made `a` positional-only and dropped `newshape`
            # in favour of `shape` (deprecated in NumPy 2.1).
            ("RESHAPE(A, SHP)", "np.reshape(A, SHP, order='F')"),
            ("RESHAPE(SOURCE=A, SHAPE=SHP)", "np.reshape(A, SHP, order='F')"),
            # Reductions keep passing DIM/MASK by keyword.
            ("SUM(A, DIM=1)", "np.sum(A, axis=0)"),
            ("SUM(A, MASK=M)", "np.sum(A, where=M)"),
            ("MAXVAL(A)", "np.max(A)"),
            ("SQRT(A)", "np.sqrt(A)"),
            # np.reshape with inline literals
            ("RESHAPE(A, (/2, 3/))", "np.reshape(A, [2, 3], order='F')"),
            ("RESHAPE(A, [2, 3])", "np.reshape(A, [2, 3], order='F')"),
            ("RESHAPE(A, (/n, k + 1/))", "np.reshape(A, [n, k + 1], order='F')"),
            (
                "RESHAPE((/1.0, 2.0, 3.0, 4.0, 5.0, 6.0/), SHP)",
                "np.reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], SHP, order='F')",
            ),
            # np.reshape with PAD and ORDER correction
            ("RESHAPE(A, SHP)", "np.reshape(A, SHP, order='F')"),
            ("RESHAPE(A, SHP, PAD=B)", "fortran_reshape(A, SHP, pad=B)"),
            (
                "RESHAPE(A, SHP, ORDER=(/2, 1/))",
                "fortran_reshape(A, SHP, order=[2, 1])",
            ),
            (
                "RESHAPE(A, SHP, PAD=B, ORDER=(/2, 1/))",
                "fortran_reshape(A, SHP, pad=B, order=[2, 1])",
            ),
            (
                "RESHAPE((/1,2,3,4,5,6/), (/3,5/), (/-1,-2/), (/2,1/))",
                "fortran_reshape([1, 2, 3, 4, 5, 6], [3, 5], pad=[-1, -2], order=[2, 1])",
            ),
        ],
    )
    def test_emitted_call_matches_numpy_signature(self, fortran, expected):
        assert self.unparse_intrinsic(fortran) == expected

    @pytest.mark.parametrize(
        "fortran",
        [
            "MATMUL(A, B)",
            "DOT_PRODUCT(A, B)",
            "RESHAPE(A, SHP)",
            "SUM(A, DIM=1)",
            "SUM(A, MASK=M)",
            "PRODUCT(A)",
            "MAXVAL(A)",
            "MINVAL(A)",
            "MAXLOC(A)",
            "MINLOC(A)",
            "SQRT(A)",
            "MIN(A, B)",
            "MAX(A, B)",
            "RESHAPE(A, (/2, 3/))",
            "RESHAPE(A, [2, 3])",
            "RESHAPE(A, (/n, k + 1/))",
            "RESHAPE((/1.0, 2.0, 3.0, 4.0, 5.0, 6.0/), SHP)",
        ],
    )
    def test_emitted_call_is_executable(self, fortran):
        """The emitted NumPy call must not raise (e.g. TypeError on keywords)."""
        namespace = {
            "np": np,
            "A": np.arange(6.0),
            "B": np.arange(6.0) + 1.0,
            "SHP": (2, 3),
            "M": np.arange(6.0) > 2.0,
            "n": 2,
            "k": 2,
        }
        eval(compile(self.unparse_intrinsic(fortran), "<emitted>", "eval"), namespace)

    def test_reshape_with_order_executable(self):
        code = (
            "subroutine t(a, shp, ord)\n"
            "  real :: a(6), b(2,3)\n"
            "  integer :: shp(2), ord(2)\n"
            "  b = RESHAPE(a, shp, ORDER=ord)\n"
            "end subroutine t\n"
        )
        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))
        assert self.f2np.needs_fortran_reshape_helper is True

        # Simulate what Transformer's module-assembly step will do: splice
        # the helper's FunctionDef in ahead of the subroutine and exec both
        # together, rather than importing fortran_reshape from a package.
        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        a = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        b = np.zeros((2, 3))
        b = namespace["t"](a, (2, 3), [2, 1])

        # Matches the manually-worked Fortran RESHAPE(A,[2,3],ORDER=[2,1])
        # example: fills column-major per ORDER, then restores declared shape.
        assert b.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_reshape_with_pad_executable(self):
        code = (
            "subroutine t(a, shp, p)\n"
            "  real :: a(4), b(2,3), p(2)\n"
            "  integer :: shp(2)\n"
            "  b = RESHAPE(a, shp, PAD=p)\n"
            "end subroutine t\n"
        )
        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))

        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        a = np.array([1, 2, 3, 4], dtype=np.float64)
        p = np.array([99, 100], dtype=np.float64)
        b = np.zeros((2, 3))
        b = namespace["t"](a, (2, 3), p)

        # 6 elements needed, only 4 in SOURCE -> pad cycles [99,100] for the
        # remaining 2, then fills column-major into (2,3).
        expected_flat = np.array([1, 2, 3, 4, 99, 100], dtype=np.float64)
        assert b.tolist() == expected_flat.reshape((2, 3), order="F").tolist()

    def test_reshape_pad_and_order_combined_executable(self):
        # Taken from the example: https://www.ibm.com/docs/fr/xl-fortran-aix/16.1.0?topic=procedures-reshapesource-shape-pad-order
        code = (
            "subroutine t()\n"
            "  real :: b(3,5)\n"
            "  b = RESHAPE((/1,2,3,4,5,6/), (/3,5/), (/-1,-2/), (/2,1/))\n"
            "end subroutine t\n"
        )
        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))

        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        b = np.zeros((3, 5))
        b = namespace["t"]()

        expected = namespace["fortran_reshape"](
            [1, 2, 3, 4, 5, 6], [3, 5], pad=[-1, -2], order=[2, 1]
        )
        assert b.tolist() == expected.tolist()

    def test_reshape_2d_source_executable(self):
        code = (
            "subroutine t(a)\n"
            "  real :: a(2,3), b(3,2)\n"
            "  b = RESHAPE(a, (/3,2/))\n"
            "end subroutine t\n"
        )

        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))

        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        # Fortran arrays are column-major
        a = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float64)

        b = namespace["t"](a)

        expected = np.reshape(a, (3, 2), order="F")
        assert b.tolist() == expected.tolist()

    def test_reshape_3d_source_executable(self):
        code = (
            "subroutine t(a)\n"
            "  real :: a(2,2,3), b(3,2,2)\n"
            "  b = RESHAPE(a, (/3,2,2/))\n"
            "end subroutine t\n"
        )

        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))

        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        a = np.arange(1, 13, dtype=np.float64).reshape((2, 2, 3), order="F")

        b = namespace["t"](a)

        expected = np.reshape(a, (3, 2, 2), order="F")
        assert b.tolist() == expected.tolist()

    def test_reshape_2d_to_2d_with_pad_and_order_executable(self):
        code = (
            "subroutine t(a)\n"
            "  real :: a(2,2), b(3,3)\n"
            "  b = RESHAPE(a, (/3,3/), (/99,100/), (/2,1/))\n"
            "end subroutine t\n"
        )

        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))

        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        # Fortran column-major 2D input
        a = np.array([[1, 3], [2, 4]], dtype=np.float64)

        b = namespace["t"](a)
        expected = namespace["fortran_reshape"](
            a,
            [3, 3],
            pad=[99, 100],
            order=[2, 1],
        )
        assert b.tolist() == expected.tolist()

    def test_reshape_3d_to_3d_with_pad_and_order_executable(self):
        code = (
            "subroutine t(a)\n"
            "  real :: a(2,2,2), b(3,2,2)\n"
            "  b = RESHAPE(a, (/3,2,2/), (/99,100/), (/3,1,2/))\n"
            "end subroutine t\n"
        )

        tree = Processor(logger=Logger()).parse_fortran_string(code)
        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]
        func.body.append(ast.Return(value=ast.Name(id="b", ctx=ast.Load())))

        helper_def = build_fortran_reshape_helper()
        module = ast.Module(body=[helper_def, func], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {"np": np}
        exec(compile(module, "<emitted>", "exec"), namespace)

        # 8-element 3D Fortran-style array
        a = np.arange(1, 9, dtype=np.float64).reshape((2, 2, 2), order="F")

        b = namespace["t"](a)

        expected = namespace["fortran_reshape"](
            a,
            [3, 2, 2],
            pad=[99, 100],
            order=[3, 1, 2],
        )

        assert b.tolist() == expected.tolist()
