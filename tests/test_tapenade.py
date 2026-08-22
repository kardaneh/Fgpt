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
    # ------------------------------------------------------------------
    # extract_reduction_chain
    # ------------------------------------------------------------------

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
