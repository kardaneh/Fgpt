import ast
import logging
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk
from scipy.io import FortranFile

from fgpt.core.common.logger import Logger
from fgpt.core.frontend.processor import Processor
from fgpt.core.transpiler.f2np import F2NP
from fgpt.core.transpiler.fortran_io import (
    IOAccess,
    IOForm,
    IOUnit,
    build_fortan_derived_type_array_read,
    build_fortan_derived_type_scalar_read,
)


@pytest.fixture(scope="class")
def test_env(request):
    f2np = F2NP(extractor=MagicMock())
    request.cls.f2np = f2np
    yield


class FakeFFile:
    """
    Stand-in for scipy.io.FortranFile: each call to read_reals/read_ints
    pops the next pre-baked record off a queue, so tests can control
    exactly what bytes each read call "sees" without touching disk.
    """

    def __init__(self, records):
        self._records = list(records)

    def read_reals(self, dtype):
        return np.asarray(self._records.pop(0), dtype=dtype)

    def read_ints(self, dtype):
        return np.asarray(self._records.pop(0), dtype=dtype)

    def read_record(self, dtype):
        flat = self._records.pop(0)
        per_record_size = sum(
            int(np.prod(dtype.fields[name][0].shape))
            if dtype.fields[name][0].shape
            else 1
            for name in dtype.names
        )

        n = len(flat) // per_record_size
        rec = np.zeros(n, dtype=dtype)
        pos = 0
        for k in range(n):
            for name in dtype.names:
                field_dtype = dtype.fields[name][0]
                size = int(np.prod(field_dtype.shape)) if field_dtype.shape else 1
                chunk = flat[pos : pos + size]
                rec[k][name] = (
                    chunk[0]
                    if size == 1
                    else np.array(chunk).reshape(field_dtype.shape)
                )
                pos += size
        return rec


class FortranDerivedType:
    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "FortranDerivedType":
        kwargs = {}
        for f in fields(cls):
            default = (
                f.default_factory() if f.default_factory is not MISSING else f.default
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
        return type(self)(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def __setitem__(self, idx, value):
        for f in fields(self):
            getattr(self, f.name)[idx] = getattr(value, f.name)


class TestReadDataclassScalar:
    def test_mixed_scalar_fields(self):
        # a,b (float) share a dtype -> one read_reals call; c (int) -> separate read_ints call
        @dataclass
        class ScalarMixed:
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)
            c: np.int32 = np.int32(0)

        ff = FakeFFile([[1.5, 2.5, 7]])
        obj = ScalarMixed()
        dataclass_scalar_helper = build_fortan_derived_type_scalar_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_scalar_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_scalar"](ff, obj)

        assert float(obj.a) == 1.5
        assert float(obj.b) == 2.5
        assert int(obj.c) == 7
        assert isinstance(obj.a, np.float64)
        assert isinstance(obj.c, np.int32)

    def test_scalar_and_array_field_same_dtype_combine(self):
        # a (float scalar) + d (float array) share a dtype -> ONE combined read
        @dataclass
        class ScalarWithArrayField:
            a: np.float64 = np.float64(0)
            d: np.ndarray = field(
                default_factory=lambda: np.zeros((3,), dtype=np.float64)
            )

        ff = FakeFFile([[10.0, 1.0, 2.0, 3.0]])
        obj = ScalarWithArrayField()
        dataclass_scalar_helper = build_fortan_derived_type_scalar_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_scalar_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_scalar"](ff, obj)

        assert float(obj.a) == 10.0
        assert obj.d.tolist() == [1.0, 2.0, 3.0]

        # a (float scalar) + d (2D float array)
        @dataclass
        class ScalarWithArrayField:
            a: np.float64 = np.float64(0)
            d: np.ndarray = field(
                default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
            )

        ff = FakeFFile([[10.0, 1.0, 2.0, 3.0, 4.0]])
        obj = ScalarWithArrayField()

        dataclass_scalar_helper = build_fortan_derived_type_scalar_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_scalar_helper)
        ast.fix_missing_locations(module)

        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)

        namespace["read_fortran_struct_scalar"](ff, obj)

        assert float(obj.a) == 10.0
        assert obj.d.shape == (2, 2)

        expected_d = np.array(
            [1.0, 2.0, 3.0, 4.0],
            dtype=np.float64,
        ).reshape((2, 2), order="F")

        assert np.array_equal(obj.d, expected_d)

    def test_all_fields_same_dtype_single_read(self):
        # every field shares a dtype -> exactly one read call total
        @dataclass
        class AllFloat:
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)
            c: np.float64 = np.float64(0)

        ff = FakeFFile([[1.0, 2.0, 3.0]])
        obj = AllFloat()
        dataclass_scalar_helper = build_fortan_derived_type_scalar_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_scalar_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_scalar"](ff, obj)

        assert (float(obj.a), float(obj.b), float(obj.c)) == (1.0, 2.0, 3.0)
        assert ff._records == []  # confirms only one read call was made

    def test_alternating_dtypes_never_grouped(self):
        # float, int, float -> three separate reads, none combined
        @dataclass
        class Alternating:
            a: np.float64 = np.float64(0)
            b: np.int32 = np.int32(0)
            c: np.float64 = np.float64(0)

        ff = FakeFFile([[1.5, 9, 2.5]])
        obj = Alternating()
        dataclass_scalar_helper = build_fortan_derived_type_scalar_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_scalar_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_scalar"](ff, obj)

        assert float(obj.a) == 1.5
        assert int(obj.b) == 9
        assert float(obj.c) == 2.5

    def test_single_field(self):
        @dataclass
        class SingleField:
            a: np.float64 = np.float64(0)

        ff = FakeFFile([[42.0]])
        obj = SingleField()
        dataclass_scalar_helper = build_fortan_derived_type_scalar_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_scalar_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_scalar"](ff, obj)

        assert float(obj.a) == 42.0


class TestReadDataclassArrayOfStructs:
    def test_mixed_scalar_fields(self):
        @dataclass
        class AoSMixed(FortranDerivedType):
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)
            c: np.int32 = np.int32(0)

        outer_shape = (2, 3)
        n = 6
        a_vals = list(range(n))
        b_vals = list(range(100, 100 + n))
        c_vals = list(range(200, 200 + n))

        flat = []
        for k in range(n):
            flat += [a_vals[k], b_vals[k], c_vals[k]]

        ff = FakeFFile([flat])
        obj = AoSMixed.zeros(outer_shape)
        dataclass_array_helper = build_fortan_derived_type_array_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_array_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_array"](ff, obj)

        expected_a = np.array(a_vals, dtype=np.float64).reshape(outer_shape, order="F")
        expected_b = np.array(b_vals, dtype=np.float64).reshape(outer_shape, order="F")
        expected_c = np.array(c_vals, dtype=np.int32).reshape(outer_shape, order="F")

        assert np.array_equal(obj.a, expected_a)
        assert np.array_equal(obj.b, expected_b)
        assert np.array_equal(obj.c, expected_c)

    def test_mixed_array_scalar_fields(self):
        @dataclass
        class AoSWithArrayField(FortranDerivedType):
            a: np.float64 = np.float64(0)
            d: np.ndarray = field(
                default_factory=lambda: np.zeros((3,), dtype=np.float64)
            )

        outer_shape = (2, 2)
        a_vals = [10, 20, 30, 40]
        d_vals_per_elem = [
            [100, 101, 102],
            [110, 111, 112],
            [120, 121, 122],
            [130, 131, 132],
        ]

        # One Fortran record: for each of the 4 struct elements, write a's
        # single value followed by d's 3 values, true per-element
        # interleaving, matching what a genuine WRITE(unit) of an
        # array-of-derived-types with a mixed scalar+array field produces.
        flat = []
        for k in range(4):
            flat += [a_vals[k]] + d_vals_per_elem[k]

        ff = FakeFFile([flat])
        obj = AoSWithArrayField.zeros(outer_shape)
        dataclass_array_helper = build_fortan_derived_type_array_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_array_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_array"](ff, obj)

        expected_a = np.array(a_vals, dtype=np.float64).reshape(outer_shape, order="F")
        flat_d = np.array(
            d_vals_per_elem, dtype=np.float64
        )  # shape (4, 3): row k = element k's 3 values
        expected_d = flat_d.reshape((3,) + outer_shape, order="F")
        # The actual destination is:
        #     outer_shape + (3,)
        # so transpose the expected data to match it.
        expected_d = expected_d.transpose(1, 2, 0)

        assert obj.a.shape == outer_shape
        assert obj.d.shape == outer_shape + (3,)
        assert np.array_equal(obj.a, expected_a)
        assert np.array_equal(obj.d, expected_d)

    def test_3d_outer_shape(self):
        @dataclass
        class AoS3D(FortranDerivedType):
            a: np.float64 = np.float64(0)

        outer_shape = (2, 2, 2)
        n = 8
        a_vals = list(range(n))

        ff = FakeFFile([a_vals])
        obj = AoS3D.zeros(outer_shape)
        dataclass_array_helper = build_fortan_derived_type_array_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_array_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_array"](ff, obj)

        expected_a = np.array(a_vals, dtype=np.float64).reshape(outer_shape, order="F")
        assert np.array_equal(obj.a, expected_a)

    def test_getitem_still_materializes_scalar_struct_after_read(self):
        # Regression guard: reading via the array path shouldn't break the
        # separate __getitem__ contract used for by-value function calls.
        @dataclass
        class AoSMixed(FortranDerivedType):
            a: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)

        outer_shape = (2,)
        ff = FakeFFile([[1.0, 10.0, 2.0, 20.0]])
        obj = AoSMixed.zeros(outer_shape)
        dataclass_array_helper = build_fortan_derived_type_array_read()
        module = ast.Module(body=[], type_ignores=[])
        module.body.extend(dataclass_array_helper)
        ast.fix_missing_locations(module)
        namespace = {"np": np, "fields": fields}
        exec(compile(module, "<emitted>", "exec"), namespace)
        namespace["read_fortran_struct_array"](ff, obj)

        elem = obj[0]
        assert isinstance(elem, AoSMixed)
        assert is_dataclass(elem)
        assert elem.a == 1.0
        assert elem.b == 10.0


@pytest.mark.usefixtures("test_env")
class TestF2NPIO:
    def parse(self, code):
        return Processor(logger=Logger()).parse_fortran_string(code)

    def get_stmt(self, tree, node_type):
        return walk(tree, node_type)[0]

    def parse_and_get(self, code, node_type):
        tree = self.parse(code)
        return self.get_stmt(tree, node_type)

    def test_demo_stdout(self, caplog):
        source = """
        subroutine demo_stdout(arr, size)
            implicit none
            integer, intent(in) :: size
            real,    intent(in) :: arr(size)
            integer :: i

            print *, "--- print * (list-directed) ---"
            do i = 1, size
              print *, "Element ", i, " = ", arr(i)
            end do

            print *, "--- write(*,*) (list-directed) ---"
            write(*, *) "Array contents:"
            do i = 1, size
              write(*, *) " Element ", i, " : ", arr(i)
            end do

            print *, "--- write(*,'(...)') (explicit format) ---"
            do i = 1, size
              write(*, '(A,I4,A,F12.6)') " Element ", i, " : ", arr(i)
            end do
        end subroutine demo_stdout
        """

        tree = self.parse_and_get(source, F23.Subroutine_Subprogram)

        self.f2np.extractor.all_array_info = {"test": {"arr": []}}

        func = self.f2np.recursive_ast(walk(tree, F23.Subroutine_Subprogram)[0])[2][0]

        code_out = ast.unparse(ast.fix_missing_locations(func))

        namespace = {
            "np": np,
            "logging": logging,
        }

        exec(
            compile(code_out, "<emitted>", "exec"),
            namespace,
        )

        arr = np.array(
            [1.5, 2.5, 3.5],
            dtype=np.float32,
        )

        with caplog.at_level(logging.INFO):
            namespace["demo_stdout"](arr, len(arr))

        output = caplog.text

        assert "--- print * (list-directed) ---" in output
        assert "--- write(*,*) (list-directed) ---" in output
        assert "--- write(*,'(...)') (explicit format) ---" in output

        assert "Element" in output
        assert "1.5" in output
        assert "2.5" in output
        assert "3.5" in output

    def test_resolve_var_dtype_character(self):
        code = """
        subroutine t(label)
            character(len=12) :: label
        end subroutine t
        """
        decl = self.parse_and_get(code, F23.Type_Declaration_Stmt)
        self.f2np.func_name = "t"
        self.f2np.extractor.var_dummy = {"t": [decl]}
        self.f2np.extractor.dec_global = {"t": {}}

        assert self.f2np._resolve_var_dtype("label") == ("char", "12")

    def test_handle_open_stmt_formatted_replace(self):
        code = """
        subroutine t(filename, unit)
            character(len=*) :: filename
            integer :: unit
            open(newunit=unit, file=filename, status="replace")
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Open_Stmt)
        result = self.f2np.handle_open_stmt(stmt)

        assert isinstance(result, ast.Assign)
        assert result.targets[0].id == "unit"
        assert result.value.func.id == "open"
        assert result.value.args[1].value == "w"
        assert "unit" in self.f2np.io_units
        assert self.f2np.io_units["unit"].form == IOForm.FORMATTED

    def test_handle_open_stmt_with_iostat_wraps_try(self):
        code = """
        subroutine t(filename, unit, ios)
            character(len=*) :: filename
            integer :: unit, ios
            open(newunit=unit, file=filename, status="old", iostat=ios)
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Open_Stmt)
        result = self.f2np.handle_open_stmt(stmt)
        assert isinstance(result, ast.Try)
        assert result.body[0].value.args[1].value == "r"
        assert result.body[1].targets[0].id == "ios"
        assert result.body[1].value.value == 0
        assert result.handlers[0].body[0].value.value == 1

    def test_write_to_open_unit_is_file_write_not_stdout(self):
        code = """
        subroutine t(unit, x)
            integer :: unit
            real :: x
            open(newunit=unit, file="f.txt", status="replace")
            write(unit, *) x
        end subroutine t
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        write_call = func.body[1].value
        assert isinstance(write_call, ast.Call)
        assert write_call.func.attr == "write"
        assert write_call.func.value.id == "unit"

    def test_write_and_stdout(self, tmp_path, caplog):
        code = """
        subroutine t(x, filename)
            implicit none

            integer :: unit
            real :: x
            integer :: unit
            character(len=*), intent(in) :: filename
            open(newunit=unit, file=filename, status="replace")

            write(numout, *) "x: ", x
            write(unit, *) x

            close(unit)
        end subroutine t
        """

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        code_out = ast.unparse(ast.fix_missing_locations(func))

        namespace = {
            "np": np,
            "logging": logging,
        }

        exec(
            compile(code_out, "<emitted>", "exec"),
            namespace,
        )

        f = tmp_path / "out_stdout.txt"

        with caplog.at_level(logging.INFO):
            namespace["t"](3, str(f))

        assert isinstance(func.body[1], ast.Expr)
        logging_call = func.body[1]
        assert isinstance(logging_call.value, ast.Call)
        assert isinstance(logging_call.value.func, ast.Attribute)
        assert (
            isinstance(logging_call.value.func.value, ast.Name)
            and logging_call.value.func.value.id == "logging"
        )
        assert f.read_text().split() == ["3"]

        output = caplog.text
        assert "x: " in output

    def test_close_stmt(self):
        code = """
        subroutine t(unit)
            integer :: unit
            close(unit)
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Close_Stmt)
        self.f2np.io_units["unit"] = IOUnit(
            unit_name="unit", file_expr=ast.Constant("f")
        )
        result = self.f2np.handle_close_stmt(stmt)

        assert isinstance(result, ast.Expr)
        assert result.value.func.attr == "close"
        assert "unit" not in self.f2np.io_units

    def test_read_stmt_with_iostat(self):
        code = """
        subroutine t(unit, ios, x)
            integer :: unit, ios
            real :: x
            read(unit, *, iostat=ios) x
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Read_Stmt)
        self.f2np.io_units["unit"] = IOUnit(
            unit_name="unit", file_expr=ast.Constant("f")
        )
        result = self.f2np.handle_read_stmt(stmt)

        assert isinstance(result, ast.Try)
        assert result.body[0].value.func.id == "float"
        assert result.body[1].targets[0].id == "ios"

    def test_stop_stmt_with_message(self):
        code = """
        subroutine t(ios)
            integer :: ios
            if (ios /= 0) stop "load_formatted: cannot open file"
        end subroutine t
        """
        stmt = self.parse_and_get(code, F23.Stop_Stmt)
        result = self.f2np.handle_stop_stmt(stmt)

        assert isinstance(result, ast.Raise)
        assert result.exc.func.id == "RuntimeError"
        assert result.exc.args[0].value == "load_formatted: cannot open file"

    def test_dump_load_formatted_executes(self, tmp_path):
        code = """
        subroutine dump_formatted(arr, size, filename)
            integer, intent(in) :: size
            real, intent(in) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, i
            open(newunit=unit, file=filename, status='replace')
            do i = 1, size
                write(unit, *) arr(i)
            end do
            close(unit)
        end subroutine dump_formatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out.txt"
        namespace["dump_formatted"]([1.5, 2.5, 3.5], 3, str(f))

        assert f.read_text().split() == ["1.5", "2.5", "3.5"]

        code = """
        subroutine load_formatted(arr, size, filename)
            implicit none
            integer, intent(in)  :: size
            real,    intent(out) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, i, ios

            open(newunit=unit, file=filename, status='old', iostat=ios)
            if (ios /= 0) stop "load_formatted: cannot open file"
            do i = 1, size
            read(unit, *, iostat=ios) arr(i)
            if (ios /= 0) stop "load_formatted: read error"
            end do
            close(unit)
        end subroutine load_formatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out.txt"
        arr = np.zeros((3,), dtype=np.float64)
        namespace["load_formatted"](arr, 3, str(f))

        assert arr.tolist() == [1.5, 2.5, 3.5]

        # Dump formatted_fmt
        code = """
        subroutine dump_formatted_fmt(arr, size, filename)
            implicit none
            integer, intent(in)  :: size
            real,    intent(in)  :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, i

            open(newunit=unit, file=filename, status='replace')
            do i = 1, size
            write(unit, '(ES24.16)') arr(i)
            end do
            close(unit)
        end subroutine dump_formatted_fmt
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out_fmt.txt"
        namespace["dump_formatted_fmt"]([1.5, 2.5, 3.5], 3, str(f))

        assert f.read_text().split() == ["1.5", "2.5", "3.5"]

        code = """
        subroutine load_formatted(arr, size, filename)
            implicit none
            integer, intent(in)  :: size
            real,    intent(out) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, i, ios

            open(newunit=unit, file=filename, status='old', iostat=ios)
            if (ios /= 0) stop "load_formatted: cannot open file"
            do i = 1, size
            read(unit, *, iostat=ios) arr(i)
            if (ios /= 0) stop "load_formatted: read error"
            end do
            close(unit)
        end subroutine load_formatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)
        arr = np.zeros((3,), dtype=np.float64)
        namespace["load_formatted"](arr, 3, str(f))

        assert arr.tolist() == [1.5, 2.5, 3.5]

    def test_dump_load_unformatted_executes(self, tmp_path):
        code = """
        subroutine dump_unformatted(arr, size, filename)
            implicit none
            integer, intent(in)  :: size
            real,    intent(in)  :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit

            open(newunit=unit, file=filename, status='replace', form='unformatted')
            write(unit) size
            write(unit) arr
            close(unit)
        end subroutine dump_unformatted
        """
        tree = self.parse(code)
        decls = walk(tree, F23.Type_Declaration_Stmt)

        self.f2np.func_name = "dump_unformatted"
        self.f2np.extractor.var_dummy = {"dump_unformatted": decls}

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"FortranFile": FortranFile, "np": np}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out_bin.dat"
        namespace["dump_unformatted"]([1.5, 2.5, 3.5], 3, str(f))

        # Check that the file was actually created.
        assert f.exists()
        with FortranFile(f, "r") as infile:
            size = infile.read_ints(np.int32)[0]
            arr = infile.read_reals(np.float64)

        assert size == 3
        np.testing.assert_allclose(arr, [1.5, 2.5, 3.5])

        code = """
        subroutine load_unformatted(arr, size, filename)
            implicit none
            integer, intent(in)  :: size
            real,    intent(out) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, nread, ios

            open(newunit=unit, file=filename, status='old', &
                form='unformatted', iostat=ios)
            if (ios /= 0) stop "load_unformatted: cannot open file"
            read(unit) nread
            if (nread /= size) stop "load_unformatted: size mismatch"
            read(unit) arr
            close(unit)
        end subroutine load_unformatted
        """

        tree = self.parse(code)
        decl = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.func_name = "load_unformatted"
        self.f2np.extractor.all_array_info = {"load_unformatted": {"arr": [3]}}
        self.f2np.extractor.var_local = {"load_unformatted": decl}

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"FortranFile": FortranFile, "np": np}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out_bin.dat"
        arr = np.zeros((3,), dtype=np.float64)
        namespace["load_unformatted"](arr, 3, str(f))
        assert arr.tolist() == [1.5, 2.5, 3.5]

    def test_append_formatted(self, tmp_path):
        code = """
        subroutine append_formatted(arr, size, filename)
            implicit none
            integer, intent(in)  :: size
            real,    intent(in)  :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, i

            open(newunit=unit, file=filename, status='old', position='append')
            do i = 1, size
            write(unit, *) arr(i)
            end do
            close(unit)
        end subroutine append_formatted
        """

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out_app_fmt.txt"

        # The file must already exist because the Fortran OPEN uses
        # status='old'.
        f.write_text("10.0\n20.0\n")

        # Append new values.
        namespace["append_formatted"](
            [1.5, 2.5, 3.5],
            3,
            str(f),
        )

        # Existing content must remain, with the new values appended.
        assert f.read_text().split() == [
            "10.0",
            "20.0",
            "1.5",
            "2.5",
            "3.5",
        ]

    def test_dump_with_header(self, tmp_path):
        code = """
        subroutine dump_with_header(arr, size, filename, header)
            implicit none
            integer, intent(in)  :: size
            real,    intent(in)  :: arr(size)
            character(len=*), intent(in) :: filename
            character(len=*), intent(in) :: header
            integer :: unit, i

            open(newunit=unit, file=filename, status='replace')
            write(unit, '(A)') "# " // trim(header)
            do i = 1, size
            write(unit, '(ES24.16)') arr(i)
            end do
            close(unit)
        end subroutine dump_with_header
        """

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out_dump_with_header.txt"
        namespace["dump_with_header"](
            [1.5, 2.5, 3.5], 3, str(f), "generated by io_generic.f90"
        )
        lines = f.read_text().splitlines()

        assert lines[0] == "# generated by io_generic.f90"
        assert len(lines) == 4

        values = [float(line) for line in lines[1:]]

        assert np.allclose(values, [1.5, 2.5, 3.5])

    def test_stream_formatted_executes(self, tmp_path):
        code = """
        subroutine stream_formatted(arr, size, filename)
            implicit none
            integer, intent(in) :: size
            real, intent(in) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, i

            open(newunit=unit, file=filename, access='stream', form='formatted')

            do i = 1, size
                write(unit, *) arr(i)
            end do

            close(unit)
        end subroutine stream_formatted
        """

        tree = self.parse(code)
        decls = walk(tree, F23.Type_Declaration_Stmt)

        self.f2np.func_name = "stream_formatted"
        self.f2np.extractor.var_dummy = {"stream_formatted": decls}

        tree = self.parse(code)
        subroutine = self.get_stmt(
            tree,
            F23.Subroutine_Subprogram,
        )

        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "stream_formatted.txt"
        namespace["stream_formatted"]([1.5, 2.5, 3.5], 3, str(f))

        assert f.read_text().split() == ["1.5", "2.5", "3.5"]

    def test_stream_unformatted_executes(self, tmp_path):
        code = """
        subroutine stream_unformatted(arr, size, filename)
            implicit none
            integer, intent(in) :: size
            real, intent(in) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit

            open(newunit=unit, file=filename, access='stream', form='unformatted')
            write(unit) size
            write(unit) arr

            close(unit)
        end subroutine stream_unformatted
        """

        tree = self.parse(code)
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.func_name = "stream_unformatted"
        self.f2np.extractor.var_dummy = {"stream_unformatted": decls}

        tree = self.parse(code)
        subroutine = self.get_stmt(
            tree,
            F23.Subroutine_Subprogram,
        )
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        # Byte stream file adding
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "stream_unformatted.bin"
        namespace["stream_unformatted"](
            [1.5, 2.5, 3.5],
            3,
            str(f),
        )

        with open(f, "rb") as fh:
            size_bytes = fh.read(4)
            arr_bytes = fh.read()

        size = np.frombuffer(size_bytes, dtype=np.int32)[0]

        arr = np.frombuffer(arr_bytes, dtype=np.float64)

        assert size == 3
        assert np.array_equal(arr, np.array([1.5, 2.5, 3.5]))

    def test_load_stream_formatted(self, tmp_path):
        code = """
        subroutine load_stream_formatted(arr, size, filename)
            implicit none
            integer, intent(in) :: size
            real, intent(out) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, nread, i

            open(newunit=unit, file=filename, access='stream', form='formatted', status='old')

            read(unit, *) nread
            if (nread /= size) then
                stop "size mismatch"
            end if

            do i = 1, size
                read(unit, *) arr(i)
            end do

            close(unit)
        end subroutine load_stream_formatted
        """

        tree = self.parse(code)
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.func_name = "load_stream_formatted"
        self.f2np.extractor.var_dummy = {
            "load_stream_formatted": [
                decls[0],  # size
                decls[1],  # arr
                decls[2],  # filename
            ]
        }
        self.f2np.extractor.var_local = {
            "load_stream_formatted": [
                decls[3],  # unit, nread, i
            ]
        }

        tree = self.parse(code)
        subroutine = self.get_stmt(
            tree,
            F23.Subroutine_Subprogram,
        )

        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        code_out = ast.unparse(ast.fix_missing_locations(func))

        filename = tmp_path / "input.dat"
        filename.write_text("3\n1.5\n2.5\n3.5\n")

        namespace = {"np": np}
        exec(
            compile(code_out, "<emitted>", "exec"),
            namespace,
        )

        arr = np.empty(3, dtype=np.float64)
        namespace["load_stream_formatted"](arr, 3, str(filename))

        assert np.allclose(arr.tolist(), [1.5, 2.5, 3.5])

    def test_load_stream_unformatted(self, tmp_path):
        code = """
        subroutine load_stream_unformatted(arr, size, filename)
            implicit none
            integer, intent(in) :: size
            real, intent(out) :: arr(size)
            character(len=*), intent(in) :: filename
            integer :: unit, nread

            open(newunit=unit, file=filename, access='stream', form='unformatted', status='old')

            read(unit) nread
            if (nread /= size) then
                stop "size mismatch"
            end if

            read(unit) arr

            close(unit)
        end subroutine load_stream_unformatted
        """

        tree = self.parse(code)
        decls = walk(tree, F23.Type_Declaration_Stmt)

        self.f2np.func_name = "load_stream_unformatted"
        self.f2np.extractor.var_dummy = {
            "load_stream_unformatted": [
                decls[0],  # size
                decls[1],  # arr
                decls[2],  # filename
            ]
        }
        self.f2np.extractor.var_local = {
            "load_stream_unformatted": [
                decls[3],  # unit, nread
            ]
        }
        self.f2np.extractor.all_array_info = {"load_stream_unformatted": {"arr": [3]}}
        tree = self.parse(code)
        subroutine = self.get_stmt(
            tree,
            F23.Subroutine_Subprogram,
        )

        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))

        filename = tmp_path / "input.bin"
        # Stream-unformatted data has NO Fortran record markers.
        size = np.array([3], dtype=np.int32)
        values = np.array(
            [1.5, 2.5, 3.5],
            dtype=np.float64,
        )

        with open(filename, "wb") as f:
            f.write(size.tobytes())
            f.write(values.tobytes())

        namespace = {"np": np}
        exec(
            compile(code_out, "<emitted>", "exec"),
            namespace,
        )

        arr = np.empty(3, dtype=np.float64)
        namespace["load_stream_unformatted"](
            arr,
            3,
            str(filename),
        )

        assert np.allclose(arr.tolist(), [1.5, 2.5, 3.5])

    def test_file_write_with_implied_do_produces_one_line(self, tmp_path):
        code = """
        subroutine dump_multicol(mat, nrow, ncol, filename)
            integer, intent(in) :: nrow, ncol
            real, intent(in) :: mat(nrow,ncol)
            character(len=*), intent(in) :: filename
            integer :: unit, i, j
            open(newunit=unit, file=filename, status='replace')
            do i = 1, nrow
                write(unit, '(*(ES24.16,1X))') (mat(i,j), j = 1, ncol)
            end do
            close(unit)
        end subroutine dump_multicol
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.extractor.all_array_info = {"dump_multicol": {"mat": []}}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        f = tmp_path / "out.txt"
        mat = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
        namespace["dump_multicol"](mat, 2, 3, str(f))

        lines = f.read_text().splitlines()
        assert len(lines) == 2
        values = [[float(x) for x in line.split()] for line in lines]
        np.testing.assert_allclose(values, mat, rtol=1e-6, atol=1e-6)

        code = """
        subroutine load_multicol(mat, nrow, ncol, filename)
            implicit none
            integer, intent(in)  :: nrow, ncol
            real,    intent(out) :: mat(nrow,ncol)
            character(len=*), intent(in) :: filename
            integer :: unit, i, ios

            open(newunit=unit, file=filename, status='old', iostat=ios)
            if (ios /= 0) stop "load_multicol: cannot open file"
            do i = 1, nrow
                read(unit, *, iostat=ios) mat(i,1:ncol)
                if (ios /= 0) stop "load_multicol: read error"
            end do
            close(unit)
        end subroutine load_multicol
        """

        class MockIndexCorrection(ast.NodeTransformer):
            def visit_Slice(self, node):
                node = self.generic_visit(node)

                # Mimic the external Fortran -> Python lower-bound correction:
                # 1:ncol -> 0:ncol
                if isinstance(node.lower, ast.Constant) and node.lower.value == 1:
                    node.lower = ast.Constant(value=0)

                return node

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)

        self.f2np.extractor.all_array_info = {"load_multicol": {"mat": []}}

        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        func = MockIndexCorrection().visit(func)
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np}
        exec(compile(code_out, "<emitted>", "exec"), namespace)
        loaded = np.empty((2, 3), dtype=np.float64)
        namespace["load_multicol"](loaded, 2, 3, str(f))

        # Verify that the file was read correctly.
        np.testing.assert_allclose(
            loaded,
            mat,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_char_unformatted_write(self):
        result = self.f2np._build_char_unformatted_write(
            "unit",
            ast.Name(id="label", ctx=ast.Load()),
            12,
            IOUnit(
                unit_name="unit",
                file_expr=ast.Constant("f"),
                access=IOAccess.SEQUENTIAL,
            ),
        )
        code = ast.unparse(ast.fix_missing_locations(result))
        assert (
            code
            == "unit.write_record(np.array(label.encode('ascii').ljust(12), dtype='S12'))"
        )

    def test_char_unformatted_write_stream(self):
        result = self.f2np._build_char_unformatted_write(
            "unit",
            ast.Name(id="label", ctx=ast.Load()),
            12,
            IOUnit(
                unit_name="unit", file_expr=ast.Constant("f"), access=IOAccess.STREAM
            ),
        )
        code = ast.unparse(ast.fix_missing_locations(result))
        assert code == "unit.write(label.encode('ascii').ljust(12))"

    def test_char_unformatted_executes(self, tmp_path):
        write_stmt = self.f2np._build_char_unformatted_write(
            "unit",
            ast.Name(id="label", ctx=ast.Load()),
            12,
            IOUnit(
                unit_name="unit",
                file_expr=ast.Constant("f"),
                access=IOAccess.SEQUENTIAL,
            ),
        )
        read_stmt = self.f2np._build_char_unformatted_read(
            "unit",
            12,
            IOUnit(
                unit_name="unit",
                file_expr=ast.Constant("f"),
                access=IOAccess.SEQUENTIAL,
            ),
            ast.Name(id="out", ctx=ast.Load()),
        )

        f = tmp_path / "chars.bin"
        unit = FortranFile(str(f), "w")
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[write_stmt], type_ignores=[])
                ),
                "<w>",
                "exec",
            ),
            {"unit": unit, "label": "hello", "np": np},
        )
        unit.close()

        unit2 = FortranFile(str(f), "r")
        ns = {"unit": unit2, "np": np}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[read_stmt], type_ignores=[])
                ),
                "<r>",
                "exec",
            ),
            ns,
        )
        unit2.close()

        assert ns["out"] == "hello"

    def test_char_unformatted_stream_executes(self, tmp_path):
        write_stmt = self.f2np._build_char_unformatted_write(
            "unit",
            ast.Name(id="label", ctx=ast.Load()),
            12,
            IOUnit(
                unit_name="unit", file_expr=ast.Constant("f"), access=IOAccess.STREAM
            ),
        )
        read_stmt = self.f2np._build_char_unformatted_read(
            "unit",
            12,
            IOUnit(
                unit_name="unit", file_expr=ast.Constant("f"), access=IOAccess.STREAM
            ),
            ast.Name(id="out", ctx=ast.Load()),
        )

        f = tmp_path / "chars.dat"
        unit = open(f, "wb")
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[write_stmt], type_ignores=[])
                ),
                "<w>",
                "exec",
            ),
            {"unit": unit, "label": "hello"},
        )
        unit.close()

        unit2 = open(f, "rb")
        ns = {"unit": unit2}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[read_stmt], type_ignores=[])
                ),
                "<r>",
                "exec",
            ),
            ns,
        )
        unit2.close()

        assert ns["out"] == "hello"

    def test_bool_dump_load_formatted(self, tmp_path):
        code = """
        subroutine dump_logical_formatted(arr, value, size, filename)
            implicit none

            integer, intent(in) :: size
            logical, intent(in) :: arr(size)
            logical, intent(in) :: value
            character(len=*), intent(in) :: filename

            integer :: unit, i

            open(newunit=unit, file=filename, status='replace', &
                form='formatted')

            write(unit, *) value
            do i = 1, size
                write(unit, *) arr(i)
            end do

            close(unit)
        end subroutine dump_logical_formatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "dump_logical_formatted"
        self.f2np.extractor.all_array_info = {"dump_logical_formatted": {"arr": [3]}}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)
        f = tmp_path / "out_bool.txt"
        namespace["dump_logical_formatted"]([False, True, True], False, 3, str(f))
        content = f.read_text().split()

        assert content[0] == "False"
        assert content[1:] == ["False", "True", "True"]

        code = """
        subroutine load_logical_formatted(arr, value, size, filename)
            implicit none

            integer, intent(in) :: size
            logical, intent(out) :: arr(size)
            logical, intent(in) :: value
            character(len=*), intent(in) :: filename

            integer :: unit, i, ios

            open(newunit=unit, file=filename, status='old', &
                form='formatted', iostat=ios)

            if (ios /= 0) stop "load_logical_formatted: cannot open file"
            read(unit, *) value
            do i = 1, size
                read(unit, *, iostat=ios) arr(i)

                if (ios /= 0) stop "load_logical_formatted: read error"
            end do

            close(unit)
        end subroutine load_logical_formatted
        """

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "load_logical_formatted"
        self.f2np.extractor.all_array_info = {"load_logical_formatted": {"arr": [3]}}
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.extractor.var_dummy = {"load_logical_formatted": decls}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        # This is usually done by the Transformer class
        func.body.append(ast.Return(value=ast.Name(id="value", ctx=ast.Load())))
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        arr = np.zeros((3,), dtype=np.bool)
        value = True  # will be changed to False
        value = namespace["load_logical_formatted"](arr, value, 3, str(f))
        assert value is False
        assert arr.tolist() == [False, True, True]

    def test_bool_dump_load_unformatted(self, tmp_path):
        code = """
        subroutine dump_logical_unformatted(arr, value, size, filename)
            implicit none

            integer, intent(in) :: size
            logical, intent(in) :: arr(size)
            logical, intent(in) :: value
            character(len=*), intent(in) :: filename

            integer :: unit, i

            open(newunit=unit, file=filename, status='replace', &
                form='unformatted')

            write(unit, *) value
            write(unit, *) size
            write(unit, *) arr

            close(unit)
        end subroutine dump_logical_unformatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "dump_logical_unformatted"
        self.f2np.extractor.all_array_info = {"dump_logical_unformatted": {"arr": [3]}}
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.extractor.var_dummy = {"dump_logical_unformatted": decls}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np, "FortranFile": FortranFile}
        exec(compile(code_out, "<emitted>", "exec"), namespace)
        f = tmp_path / "out_bin.dat"
        namespace["dump_logical_unformatted"]([False, True, True], False, 3, str(f))

        # Check that the file was actually created.
        assert f.exists()
        with FortranFile(f, "r") as infile:
            value = np.bool(infile.read_ints(np.bool)[0])
            size = infile.read_ints(np.int32)[0]
            arr = infile.read_ints(np.bool)
        assert value is np.False_
        assert size == 3
        assert arr.tolist() == [False, True, True]

        code = """
        subroutine load_logical_unformatted(arr, value, size, filename)
            implicit none

            integer, intent(in) :: size
            logical, intent(out) :: arr(size)
            logical, intent(in) :: value
            character(len=*), intent(in) :: filename

            integer :: unit, nread, ios

            open(newunit=unit, file=filename, status='old', &
                form='unformatted', iostat=ios)

            if (ios /= 0) stop "load_logical_unformatted: cannot open file"

            read(unit, *) value
            read(unit, iostat=ios) nread
            if (ios /= 0) stop "load_logical_unformatted: read error"

            if (nread /= size) stop "load_logical_unformatted: size mismatch"

            read(unit, iostat=ios) arr
            if (ios /= 0) stop "load_logical_unformatted: read error"

            close(unit)
        end subroutine load_logical_unformatted
        """

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "load_logical_unformatted"
        self.f2np.extractor.all_array_info = {"load_logical_unformatted": {"arr": [3]}}
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.extractor.var_dummy = {"load_logical_unformatted": decls}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        func.body.append(ast.Return(value=ast.Name(id="value", ctx=ast.Load())))
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np, "FortranFile": FortranFile}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        arr = np.zeros((3,), dtype=np.bool)
        value = True  # will be changed to False
        value = namespace["load_logical_unformatted"](arr, value, 3, str(f))

        assert value is np.False_
        assert size == 3
        assert arr.tolist() == [False, True, True]

    def test_mixed_dump_load_formatted(self, tmp_path):
        code = """
        subroutine dump_mixed_formatted(values, flags, size, name, filename)
            implicit none

            integer, intent(in) :: size
            real,    intent(in) :: values(size)
            logical, intent(in) :: flags(size)
            character(len=5), intent(in) :: name
            character(len=*), intent(in) :: filename

            integer :: unit, i

            open(newunit=unit, file=filename, status='replace', &
                form='formatted')

            write(unit, *) size
            do i = 1, size
                write(unit, *) flags(i)
            end do
            do i = 1, size
                write(unit, *) values(i)
            end do
            write(unit, *) name

            close(unit)
        end subroutine dump_mixed_formatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "dump_mixed_formatted"
        self.f2np.extractor.all_array_info = {
            "dump_mixed_formatted": {"values": [3], "flags": [3]}
        }
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}
        exec(compile(code_out, "<emitted>", "exec"), namespace)
        f = tmp_path / "out_bool.txt"
        namespace["dump_mixed_formatted"](
            [1.5, 2.5, 3.5], [False, True, True], 3, "HELLO", str(f)
        )
        content = f.read_text().split()

        assert content[0] == "3"
        assert content[1:4] == ["False", "True", "True"]
        assert content[4:7] == ["1.5", "2.5", "3.5"]
        assert content[-1] == "HELLO"
        code = """
        subroutine load_mixed_formatted(values, flags, size, name, filename)
            implicit none

            integer, intent(in) :: size
            real,    intent(out) :: values(size)
            logical, intent(out) :: flags(size)
            character(len=5), intent(in) :: name
            character(len=*), intent(in) :: filename

            integer :: unit, i, ios

            open(newunit=unit, file=filename, status='old', &
                form='formatted', iostat=ios)

            if (ios /= 0) stop "load_mixed_formatted: cannot open file"
            read(unit, *) size
            do i = 1, size
                read(unit, *, iostat=ios) flags(i)

                if (ios /= 0) stop "load_mixed_formatted: read error for flags"
            end do
            do i = 1, size
                read(unit, *, iostat=ios) values(i)

                if (ios /= 0) stop "load_mixed_formatted: read error for values"
            end do
            read(unit, *) name

            close(unit)
        end subroutine load_mixed_formatted
        """

        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "load_mixed_formatted"
        self.f2np.extractor.all_array_info = {
            "load_mixed_formatted": {"values": [3], "flags": [3]}
        }
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.extractor.var_dummy = {"load_mixed_formatted": decls}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        # This is usually done by the Transformer class
        func.body.append(
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Name(id="size", ctx=ast.Load()),
                        ast.Name(id="name", ctx=ast.Load()),
                    ]
                )
            )
        )
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {}

        exec(compile(code_out, "<emitted>", "exec"), namespace)
        flags = np.zeros((3,), dtype=np.bool)
        values = np.zeros((3,), dtype=np.float64)
        size, name = namespace["load_mixed_formatted"](
            values, flags, 3, "HELLO", str(f)
        )

        assert size == 3
        assert name == "HELLO"
        assert values.tolist() == [1.5, 2.5, 3.5]
        assert flags.tolist() == [False, True, True]

    def test_mixed_dump_load_unformatted(self, tmp_path):
        code = """
        subroutine dump_mixed_unformatted(values, flags, size, name, filename)
            implicit none

            integer, intent(in) :: size
            real,    intent(in) :: values(size)
            logical, intent(in) :: flags(size)
            character(len=5), intent(in) :: name
            character(len=*), intent(in) :: filename

            integer :: unit

            open(newunit=unit, file=filename, status='replace', &
                form='unformatted')

            write(unit) size
            write(unit) values
            write(unit) flags
            write(unit) name

            close(unit)
        end subroutine dump_mixed_unformatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "dump_mixed_unformatted"
        self.f2np.extractor.all_array_info = {
            "dump_mixed_unformatted": {"values": [3], "flags": [3]}
        }
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.extractor.var_dummy = {"dump_mixed_unformatted": decls}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]

        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np, "FortranFile": FortranFile}
        exec(compile(code_out, "<emitted>", "exec"), namespace)
        f = tmp_path / "out_bin.dat"
        namespace["dump_mixed_unformatted"](
            [1.5, 2.5, 3.5], [False, True, True], 3, "HELLO", str(f)
        )

        assert f.exists()
        with FortranFile(f, "r") as infile:
            size = infile.read_ints(np.int32)[0]
            values = infile.read_reals(np.float64)
            flags = infile.read_ints(np.bool)
            name = infile.read_record("S5")[0].decode("ascii").rstrip()
        assert values.tolist() == [1.5, 2.5, 3.5]
        assert size == 3
        assert flags.tolist() == [False, True, True]
        assert name == "HELLO"

        code = """
        subroutine load_mixed_unformatted(values, flags, size, name, filename)
            implicit none

            integer, intent(in) :: size
            real,    intent(out) :: values(size)
            logical, intent(out) :: flags(size)
            character(len=5), intent(in) :: name
            character(len=*), intent(in) :: filename

            integer :: unit, nread, ios

            open(newunit=unit, file=filename, status='old', &
                form='unformatted', iostat=ios)

            if (ios /= 0) stop "load_mixed_unformatted: cannot open file"

            read(unit, iostat=ios) nread
            if (ios /= 0) stop "load_mixed_unformatted: read error"

            if (nread /= size) stop "load_mixed_unformatted: size mismatch"

            read(unit, iostat=ios) values
            if (ios /= 0) stop "load_mixed_unformatted: values read error"

            read(unit, iostat=ios) flags
            if (ios /= 0) stop "load_mixed_unformatted: flags read error"

            read(unit, *) name

            close(unit)
        end subroutine load_mixed_unformatted
        """
        tree = self.parse(code)
        subroutine = self.get_stmt(tree, F23.Subroutine_Subprogram)
        self.f2np.func_name = "load_mixed_unformatted"
        self.f2np.extractor.all_array_info = {
            "load_mixed_unformatted": {"values": [3], "flags": [3]}
        }
        decls = walk(tree, F23.Type_Declaration_Stmt)
        self.f2np.extractor.var_dummy = {"load_mixed_unformatted": decls}
        _, _, result = self.f2np.recursive_ast(subroutine)
        func = result[0]
        func.body.append(ast.Return(value=ast.Name(id="name", ctx=ast.Load())))
        code_out = ast.unparse(ast.fix_missing_locations(func))
        namespace = {"np": np, "FortranFile": FortranFile}
        exec(compile(code_out, "<emitted>", "exec"), namespace)

        values = np.zeros((3,), dtype=np.float64)
        flags = np.zeros((3,), dtype=np.bool)
        name = namespace["load_mixed_unformatted"](values, flags, 3, "TREES", str(f))

        assert values.tolist() == [1.5, 2.5, 3.5]
        assert flags.tolist() == [False, True, True]
        assert name == "HELLO"
