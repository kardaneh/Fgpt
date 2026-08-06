import ast

import pytest

from fgpt.core.transpiler.intrinsic import (
    intrinsic_signatures,
    normalize_intrinsic_call,
)


class TestIntrinsic:
    def test_sum(self):
        sig = intrinsic_signatures["SUM"]

        result = normalize_intrinsic_call(
            sig, positional_args=["A", ast.Constant(value=2)], keyword_args={}
        )

        assert result["array"] == "A"
        assert result["axis"].value == 1  # dim_to_axis(2)
        assert result["where"] is None

        # Keyword override
        sig = intrinsic_signatures["SUM"]

        result = normalize_intrinsic_call(
            sig, positional_args=["A"], keyword_args={"dim": ast.Constant(value=3)}
        )

        assert result["array"] == "A"
        assert result["axis"].value == 2  # 3 - 1

    def test_product(self):
        sig = intrinsic_signatures["PRODUCT"]

        result = normalize_intrinsic_call(
            sig, positional_args=["X", ast.Constant(value=2)], keyword_args={}
        )

        assert result["array"] == "X"
        assert result["axis"].value == 1
        assert result["where"] is None

    def test_max_varargs(self):
        sig = intrinsic_signatures["MAX"]
        # Here we are not testing the axis thus we can send it as int
        result = normalize_intrinsic_call(
            sig, positional_args=[1, 5, 3], keyword_args={}
        )

        assert result["values"] == [1, 5, 3]

    def test_min_varargs(self):
        sig = intrinsic_signatures["MIN"]

        result = normalize_intrinsic_call(sig, positional_args=[10, 2], keyword_args={})

        assert result["values"] == [10, 2]

    def test_sqrt(self):
        sig = intrinsic_signatures["SQRT"]

        result = normalize_intrinsic_call(sig, positional_args=["X"], keyword_args={})

        assert result["array"] == "X"

    def test_reshape_defaults(self):
        sig = intrinsic_signatures["RESHAPE"]

        result = normalize_intrinsic_call(
            sig, positional_args=["A", (2, 2)], keyword_args={}
        )

        assert result["source"] == "A"
        # `shape`, not `newshape`: NumPy 2.1 deprecated the `newshape` keyword.
        assert result["shape"] == (2, 2)

        # positional args given for the whole intrinsic function
        result = normalize_intrinsic_call(
            sig, positional_args=["A", (2, 2), (-1, -2), (2, 1)], keyword_args={}
        )
        assert result["source"] == "A"
        assert result["shape"] == (2, 2)
        assert result["pad"] == (-1, -2)
        assert result["order"] == (2, 1)

        # positional and keyword arguments
        result = normalize_intrinsic_call(
            sig,
            positional_args=["A", (2, 2)],
            keyword_args={"pad": (-1, -2), "order": (2, 1)},
        )
        assert result["source"] == "A"
        assert result["shape"] == (2, 2)
        assert result["pad"] == (-1, -2)
        assert result["order"] == (2, 1)

    def test_matmul(self):
        sig = intrinsic_signatures["MATMUL"]

        result = normalize_intrinsic_call(
            sig, positional_args=["A", "B"], keyword_args={}
        )

        assert result["a"] == "A"
        assert result["b"] == "B"

    def test_dot_product(self):
        sig = intrinsic_signatures["DOT_PRODUCT"]

        result = normalize_intrinsic_call(
            sig, positional_args=["x", "y"], keyword_args={}
        )

        assert result["a"] == "x"
        assert result["b"] == "y"

    def test_missing_required_arg_raises(self):
        sig = intrinsic_signatures["SQRT"]

        with pytest.raises(ValueError):
            normalize_intrinsic_call(sig, [], {})
