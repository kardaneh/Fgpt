from collections import deque

import pytest
from fparser.two import Fortran2003 as F23

from fgpt.core.common.logger import Logger
from fgpt.core.passes import TapenadePass


@pytest.fixture(scope="class")
def tapenade_pass(request):
    all_array_info = {
        "array1": [
            {"dim_str": "1", "dim_end": "10"},
            {"dim_str": "1", "dim_end": "20"},
            {"dim_str": "1", "dim_end": "30"},
        ],
        "array2": [
            {"dim_str": "1", "dim_end": "10"},
            {"dim_str": "1", "dim_end": "20"},
            {"dim_str": "1", "dim_end": "30"},
        ],
        "a": [
            {"dim_str": "1", "dim_end": "10"},
            {"dim_str": "1", "dim_end": "20"},
            {"dim_str": "1", "dim_end": "30"},
        ],
    }

    request.cls.tapenade = TapenadePass(
        logger=Logger(),
        all_array_info=all_array_info,
    )


@pytest.mark.usefixtures("tapenade_pass")
class TestTapenadePass:
    def test_extract_reduction_chain_no_intrinsic(self):
        stmt = F23.Assignment_Stmt("result = array1")

        assert self.tapenade.extract_reduction_chain(stmt) == {}

    def test_extract_reduction_chain_single_reduction(self):
        stmt = F23.Assignment_Stmt("result = SUM(array1, Dim=2)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["intrinsic"] == "SUM"
        assert info["dim"] == ["2"]
        assert info["original_dims"] == ["2"]
        assert info["root_array"] == "array1"
        assert info["num_reductions"] == 1
        assert info["final_rank"] == 2

    def test_extract_reduction_chain_nested_same_dim(self):
        stmt = F23.Assignment_Stmt("result = SUM(SUM(array1, Dim=2), Dim=2)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["dim"] == ["2", "2"]
        assert info["original_dims"] == ["2", "3"]
        assert info["num_reductions"] == 2
        assert info["final_rank"] == 1

    def test_extract_reduction_chain_nested_different_dims(self):
        stmt = F23.Assignment_Stmt("result = SUM(SUM(array1, Dim=2), Dim=1)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["dim"] == ["2", "1"]
        assert info["original_dims"] == ["2", "1"]
        assert info["num_reductions"] == 2
        assert info["final_rank"] == 1

    def test_extract_reduction_chain_three_nested_reductions(self):
        stmt = F23.Assignment_Stmt(
            "result = SUM(SUM(SUM(array1, Dim=2), Dim=1), Dim=1)"
        )

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["dim"] == ["2", "1", "1"]
        assert info["original_dims"] == ["2", "1", "3"]
        assert info["num_reductions"] == 3
        assert info["final_rank"] == 0

    def test_extract_reduction_chain_all_dimensions(self):
        stmt = F23.Assignment_Stmt("result = SUM(array1)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["dim"] == ["ALL"]
        assert info["original_dims"] == ["ALL"]
        assert info["num_reductions"] == 1
        assert info["final_rank"] == 0

    def test_extract_reduction_chain_dim_without_keyword(self):
        stmt = F23.Assignment_Stmt("result = SUM(array1, 2)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["dim"] == ["2"]
        assert info["original_dims"] == ["2"]
        assert info["final_rank"] == 2

    def test_extract_reduction_chain_dim_variable(self):
        stmt = F23.Assignment_Stmt("result = SUM(array1, dim_var)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["dim"] == ["dim_var"]
        assert info["original_dims"] == ["dim_var"]
        assert info["final_rank"] == 3

    def test_extract_reduction_chain_nested_tapenade_expression(self):
        stmt = F23.Assignment_Stmt(
            "result1 = Maxloc(MINLOC(SUM(ABS(cons - array2(:, :, :)), Dim=2), Dim=2))"
        )

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert info["intrinsic"] == "MAXLOC"
        assert info["dim"] == ["2", "2", "ALL"]
        assert info["original_dims"] == ["2", "3", "ALL"]
        assert info["num_reductions"] == 3
        assert info["final_rank"] == 0
        assert info["root_array"] == "array2"

    def test_extract_reduction_chain_chain(self):
        stmt = F23.Assignment_Stmt("result = SUM(SUM(array1, Dim=2), Dim=2)")

        result = self.tapenade.extract_reduction_chain(stmt)
        info = next(iter(result.values()))

        assert len(info["chain"]) == 2

        first = info["chain"][0]
        assert first["intrinsic"] == "SUM"
        assert first["dim"] == "2"
        assert first["original_dim"] == 2
        assert first["rank_before"] == 3
        assert first["rank_after"] == 2
        assert first["dim_mapping_before"] == [1, 2, 3]
        assert first["dim_mapping_after"] == [1, 3]

        second = info["chain"][1]
        assert second["intrinsic"] == "SUM"
        assert second["dim"] == "2"
        assert second["original_dim"] == 3
        assert second["rank_before"] == 2
        assert second["rank_after"] == 1
        assert second["dim_mapping_before"] == [1, 3]
        assert second["dim_mapping_after"] == [1]

    def test_process_call_stmt_single_isize(self):
        call_stmt = F23.Call_Stmt(
            "CALL pushreal8array(array1(:, :, :), isize3ofarray1)"
        )

        result = self.tapenade.process_call_stmt(call_stmt)
        expected = "CALL pushreal8array(array1(:, :, :), 30)"

        assert result.tostr() == expected

    def test_process_call_stmt_isize_with_lb(self):
        # Create array info with non-1 lower bound
        self.tapenade.all_array_info["array3"] = [
            {"dim_str": "2", "dim_end": "10"},
        ]
        call_stmt = F23.Call_Stmt("CALL pushreal8array(array3(:), isize1ofarray3)")

        result = self.tapenade.process_call_stmt(call_stmt)
        expected = "CALL pushreal8array(array3(:), (10) - (2) + 1)"

        assert result.tostr() == expected

    def test_process_call_stmt_multiplication(self):
        call_stmt = F23.Call_Stmt(
            "CALL pushreal8array(array1(ji, :, :), isize2ofarray1*isize3ofarray1)"
        )

        result = self.tapenade.process_call_stmt(call_stmt)
        expected = "CALL pushreal8array(array1(ji, :, :), 30 * 20)"

        assert result.tostr() == expected

    def test_process_call_stmt_nested_multiplication(self):
        call_stmt = F23.Call_Stmt(
            "CALL pushreal8array(array1, isize1ofarray1*isize2ofarray1*isize3ofarray1)"
        )

        result = self.tapenade.process_call_stmt(call_stmt)
        expected = "CALL pushreal8array(array1, 30 * 20 * 10)"

        assert result.tostr() == expected

    def test_process_call_stmt_name_array(self):
        call_stmt = F23.Call_Stmt("CALL pushreal8array(array1, isize1ofarray1)")

        result = self.tapenade.process_call_stmt(call_stmt)
        expected = "CALL pushreal8array(array1, 10)"

        assert result.tostr() == expected

    def test_get_array_info_from_original(self):
        result = self.tapenade.get_array_info("array1")
        assert result is not None
        assert result[0]["dim_str"] == "1"
        assert result[0]["dim_end"] == "10"

    def test_get_array_info_not_found(self):
        result = self.tapenade.get_array_info("nonexistent")
        assert result is None

    def test_set_array_info_module_level(self):
        self.tapenade.current_subroutine = None
        self.tapenade.set_array_info("module_array", [{"dim_str": "1", "dim_end": "5"}])
        assert "module_array" in self.tapenade.all_array_info_module_level
        assert (
            self.tapenade.all_array_info_module_level["module_array"][0]["dim_end"]
            == "5"
        )

    def test_set_array_info_subroutine_level(self):
        self.tapenade.current_subroutine = "test_sub"
        self.tapenade.set_array_info("sub_array", [{"dim_str": "1", "dim_end": "15"}])
        assert "sub_array" in self.tapenade.all_array_info_subroutine_level["test_sub"]
        assert (
            self.tapenade.all_array_info_subroutine_level["test_sub"]["sub_array"][0][
                "dim_end"
            ]
            == "15"
        )
        self.tapenade.current_subroutine = None

    def test_collect_all_arrays_part_ref_with_colon(self):
        stmt = F23.Assignment_Stmt("result = array1(:, :, :)")
        arrays_queue = deque()
        seen = set()

        self.tapenade.collect_all_arrays(stmt, arrays_queue, seen)

        assert len(arrays_queue) == 1
        assert isinstance(arrays_queue[0], F23.Part_Ref)

    def test_collect_all_arrays_part_ref_without_colon(self):
        stmt = F23.Assignment_Stmt("result = array1(1, 2, 3)")
        arrays_queue = deque()
        seen = set()

        self.tapenade.collect_all_arrays(stmt, arrays_queue, seen)

        assert len(arrays_queue) == 0

    def test_collect_all_arrays_name(self):
        stmt = F23.Assignment_Stmt("result = array1")
        arrays_queue = deque()
        seen = set()

        self.tapenade.collect_all_arrays(stmt, arrays_queue, seen)

        assert len(arrays_queue) == 1
        assert arrays_queue[0].tostr() == "array1"

    def test_collect_all_arrays_name_already_seen(self):
        stmt = F23.Assignment_Stmt("result = array1 + array1")
        arrays_queue = deque()
        seen = {"array1"}

        self.tapenade.collect_all_arrays(stmt, arrays_queue, seen)

        assert len(arrays_queue) == 0

    def test_process_array_shape_simple(self):
        arrays_queue = deque()
        arrays_queue.append(F23.Part_Ref("array1(:, :, :)"))

        declared_shape = [
            {"dim_str": "1", "dim_end": "isize1ofarray1"},
            {"dim_str": "1", "dim_end": "isize2ofarray1"},
            {"dim_str": "1", "dim_end": "isize3ofarray1"},
        ]

        result = self.tapenade.process_array_shape(
            arrays_queue, declared_shape, lhs_array_name="new_array"
        )

        assert result == ["1:10", "1:20", "1:30"]

    def test_process_array_shape_with_reduction(self):
        stmt = F23.Assignment_Stmt("result = SUM(array1(:, :, :), Dim=2)")
        reduction_info = self.tapenade.extract_reduction_chain(stmt)

        arrays_queue = deque()
        arrays_queue.append(F23.Part_Ref("array1(:, :, :)"))

        declared_shape = [
            {"dim_str": "1", "dim_end": "isize1ofarray1"},
            {"dim_str": "1", "dim_end": "isize2ofarray1"},
            {"dim_str": "1", "dim_end": "isize3ofarray1"},
        ]

        result = self.tapenade.process_array_shape(
            arrays_queue, declared_shape, reduction_info, lhs_array_name="new_array"
        )

        assert result == ["1:10", "1:30"]

    def test_process_array_shape_empty_result(self):
        arrays_queue = deque()

        declared_shape = [
            {"dim_str": "1", "dim_end": "?"},
        ]

        result = self.tapenade.process_array_shape(
            arrays_queue, declared_shape, lhs_array_name="new_array"
        )

        assert result == []

    def test_process_array_shape_with_base_array(self):
        self.tapenade.all_array_info["base_array"] = [
            {"dim_str": "1", "dim_end": "5"},
            {"dim_str": "1", "dim_end": "10"},
        ]

        arrays_queue = deque()
        arrays_queue.append(F23.Part_Ref("base_array(:, :)"))

        declared_shape = [
            {"dim_str": "1", "dim_end": "?"},
            {"dim_str": "1", "dim_end": "?"},
        ]

        result = self.tapenade.process_array_shape(
            arrays_queue,
            declared_shape,
            lhs_array_name="base_arrayd",
        )

        assert result == ["1:5", "1:10"]

    def test_process_array_shape_with_lhs_base_variant(self):
        self.tapenade.all_array_info["base_array"] = [
            {"dim_str": "1", "dim_end": "10"},
            {"dim_str": "1", "dim_end": "20"},
            {"dim_str": "1", "dim_end": "30"},
        ]

        arrays_queue = deque()
        arrays_queue.append(F23.Part_Ref("base_array(:, :, :)"))

        declared_shape = [
            {"dim_str": "1", "dim_end": "?"},
            {"dim_str": "1", "dim_end": "?"},
            {"dim_str": "1", "dim_end": "?"},
        ]

        result = self.tapenade.process_array_shape(
            arrays_queue, declared_shape, lhs_array_name="base_arrayb"
        )

        assert result == ["1:10", "1:20", "1:30"]

    def test_check_tapenade_isize_no_isize(self):
        code = """
        SUBROUTINE test()
        REAL, DIMENSION(10, 20) :: array
        END SUBROUTINE test
        """
        tree = self.tapenade.processor.parse_fortran_string(code)
        self.tapenade.current_subroutine = "test"
        self.tapenade.check_tapenade_isize(tree)

        assert len(self.tapenade.array_shape_not_defined) == 0
        assert "array" in self.tapenade.all_array_info_subroutine_level["test"]

    def test_check_tapenade_isize_with_isize(self):
        code = """
        SUBROUTINE test()
        REAL, DIMENSION(isize1ofarray, isize2ofarray) :: array
        END SUBROUTINE test
        """
        tree = self.tapenade.processor.parse_fortran_string(code)

        self.tapenade.current_subroutine = "test"
        self.tapenade.check_tapenade_isize(tree)

        assert "array" in self.tapenade.array_shape_not_defined
        assert (
            self.tapenade.array_shape_not_defined["array"][0]["dim_end"]
            == "isize1ofarray"
        )

    def test_check_tapenade_isize_allocatable_in_subroutine_raises(self):
        code = """
        SUBROUTINE test()
        REAL, DIMENSION(:, :), ALLOCATABLE :: alloc_array
        END SUBROUTINE test
        """
        tree = self.tapenade.processor.parse_fortran_string(code)

        self.tapenade.current_subroutine = "test"
        with pytest.raises(
            AssertionError,
            match="ALLOCATABLE array 'alloc_array' found in subroutine 'test'.",
        ):
            self.tapenade.check_tapenade_isize(tree)

    def test_get_root_array_name_from_name(self):
        node = F23.Name("array1")
        result = self.tapenade._get_root_array_name(node)
        assert result == "array1"

    def test_get_root_array_name_from_part_ref(self):
        node = F23.Part_Ref("array1(:, :, :)")
        result = self.tapenade._get_root_array_name(node)
        assert result == "array1"

    def test_integration_full_clean_with_subroutine(self, tmp_path):
        code = """
        MODULE test_module
        IMPLICIT NONE
        REAL, DIMENSION(10, 20, 30) :: array1
        CONTAINS
        SUBROUTINE test_sub()
            IMPLICIT NONE
            REAL, DIMENSION(isize1ofarray1, isize2ofarray1, isize3ofarray1) :: local_array
            local_array = array1
        END SUBROUTINE test_sub
        END MODULE test_module
        """

        tree = self.tapenade.processor.parse_fortran_string(code)
        self.tapenade.clean_tapenade_statements(tree)
        assert True
