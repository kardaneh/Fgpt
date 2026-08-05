import ast

import pytest

from fgpt.core.common.utils import AdjustIndices, ReplaceGlobals


def _array_info():
    return {
        "arr": [
            {"dim_str": "1"},
        ],
        "arr0": [
            {"dim_str": "0"},
        ],
        "arr5": [
            {"dim_str": "5"},
        ],
        "mat": [
            {"dim_str": "1"},
            {"dim_str": "1"},
        ],
    }


def _make_cls_info():
    return {
        "Model": {
            "self": {
                "attributes": {
                    "temp": {},
                    "soil": {},
                    "mask": {},
                },
                "methods": [
                    "compute",
                    "helper",
                ],
                "instances": {
                    "forcing": {
                        "class_name": ast.Name(id="forcing", ctx=ast.Load()),
                        "attributes": {
                            "rain": {},
                            "wind": {},
                        },
                        "methods": ["update"],
                    }
                },
            }
        }
    }


def _cls_attributes():
    return {
        "attributes": {
            "nbeg": (1,),
            "nsoil": (5,),
        },
        "instances": {},
    }


def assert_ast_equal(a, b):
    assert ast.dump(a, include_attributes=False) == ast.dump(
        b, include_attributes=False
    )


@pytest.fixture(scope="class")
def test_env(request):
    transformer = AdjustIndices(
        conv_vars={"i", "j", "k"},
        array_info=_array_info(),
        cls_attributes=_cls_attributes(),
    )
    request.cls.transformer = transformer

    replacer = ReplaceGlobals(cls_info=_make_cls_info())
    request.cls.replacer = replacer

    yield


def _expr(code: str):
    return ast.parse(code).body[0].value


def _stmt(code: str):
    return ast.parse(code).body[0]


@pytest.mark.usefixtures("test_env")
class TestAdjustIndices:
    def test_adjust_index(self):
        # Name
        node = ast.Name(id="idx", ctx=ast.Load())
        result = self.transformer._adjust_index(node)

        assert isinstance(result, ast.BinOp)
        assert isinstance(result.op, ast.Sub)
        assert result.right.value == 1

        # Conv var
        node = ast.Name(id="i", ctx=ast.Load())
        result = self.transformer._adjust_index(node)
        assert result is node

        # Constant
        result = self.transformer._adjust_index(ast.Constant(value=5))
        assert result.value == 4

        # Constant 0
        result = self.transformer._adjust_index(ast.Constant(value=0))
        assert result.value == 0

    def test_apply_offset_conv_var(self):
        node = ast.Name(id="i", ctx=ast.Load())
        result = self.transformer._apply_offset_if_convvar(node, offset=-4)
        assert isinstance(result, ast.BinOp)
        assert result.right.value == -4

        # Non conv var
        node = ast.Name(id="idx", ctx=ast.Load())
        result = self.transformer._apply_offset_if_convvar(node, offset=-4)
        assert result is node

    def test_visit_Subscript(self):
        node = _expr("arr[idx]")
        result = self.transformer.visit_Subscript(node)
        assert isinstance(result.slice, ast.BinOp)
        assert isinstance(result.slice.op, ast.Sub)

        # Conventional variable
        node = _expr("arr0[i]")
        result = self.transformer.visit_Subscript(node)
        assert isinstance(result.slice, ast.BinOp)
        assert isinstance(result.slice.op, ast.Add)

        # Offset
        node = _expr("arr5[i]")
        result = self.transformer.visit_Subscript(node)
        assert isinstance(result.slice, ast.BinOp)
        assert result.slice.right.value == -4

    def test_visit_Assign(self):
        node = _stmt("i = idx")
        result = self.transformer.visit_Assign(node)
        assert isinstance(result.value, ast.BinOp)
        assert isinstance(result.value.op, ast.Sub)

        # Compare assign
        node = _stmt("mask = a == b")
        self.transformer.visit_Assign(node)
        assert "mask" in self.transformer.adjusted_vars

    def test_For(self):
        # Un used for loop
        node = _stmt("for i in range(0, 10):\n    pass")
        result = self.transformer.visit_For(node)
        assert result.target.id == "_"

    def test_extract_loop_vars_tuple(self):
        target = ast.parse("(i, j)").body[0].value
        vars_ = self.transformer._extract_loop_vars(target)
        assert set(vars_) == {"i", "j"}

    def test_visit_If(self):
        node = _stmt("if a > b:\n    pass")
        result = self.transformer.visit_If(node)
        assert result is None

    def test_adjusted_vars(self):
        self.transformer.adjusted_vars.add("idx")

        node = _stmt("if idx == limit:\n    x = 1")
        result = self.transformer.visit_If(node)
        comp = result.test.comparators[0]
        assert isinstance(comp, ast.BinOp)

    def test_argmin_offset(self):
        node = _expr("arr.argmin(arr[i])")
        result = self.transformer.visit_Call(node)
        assert isinstance(result, ast.BinOp)

    def test_regular_call(self):
        node = _expr("foo(x)")
        result = self.transformer.visit_Call(node)
        assert isinstance(result, ast.Call)


@pytest.mark.usefixtures("test_env")
class TestReplaceGlobals:
    def test_get_attr_node(self):
        # Local attribute
        result = self.replacer.get_attr_node("temp")
        expected = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="temp",
            ctx=ast.Load(),
        )
        assert_ast_equal(result, expected)

        # Composed attribute
        result = self.replacer.get_attr_node("rain")
        expected = ast.Attribute(
            value=ast.Name(
                id="forcing",
                ctx=ast.Load(),
            ),
            attr="rain",
            ctx=ast.Load(),
        )
        assert_ast_equal(result, expected)

        # Unknown node
        assert self.replacer.get_attr_node("unknown") is None

    def test_visit_Name(self):
        node = ast.Name(
            id="temp",
            ctx=ast.Load(),
        )
        result = self.replacer.visit_Name(node)
        expected = ast.Attribute(
            value=ast.Name(
                id="self",
                ctx=ast.Load(),
            ),
            attr="temp",
            ctx=ast.Load(),
        )
        assert_ast_equal(result, expected)

        # No replacement
        node = ast.Name(
            id="x",
            ctx=ast.Load(),
        )
        result = self.replacer.visit_Name(node)
        assert isinstance(result, ast.Name)
        assert result.id == "x"

    def test_visit_Assign(self):
        # ASSIGN target replaced
        node = _stmt("temp = value")
        result = self.replacer.visit_Assign(node)
        target = result.targets[0]
        assert isinstance(target, ast.Attribute)
        assert target.attr == "temp"

        # subscript
        node = _stmt("soil[i] = value")
        result = self.replacer.visit_Assign(node)
        target = result.targets[0]
        assert isinstance(target.value, ast.Attribute)
        assert target.value.attr == "soil"

    def test_visit_Call(self):
        node = _expr("compute(x)")
        result = self.replacer.visit_Call(node)
        expected = ast.Attribute(
            value=ast.Name(
                id="self",
                ctx=ast.Load(),
            ),
            attr="compute",
            ctx=ast.Load(),
        )
        assert_ast_equal(
            result.func,
            expected,
        )

        # Composed call
        node = _expr("update(x)")
        result = self.replacer.visit_Call(node)
        expected = ast.Attribute(
            value=ast.Name(
                id="forcing",
                ctx=ast.Load(),
            ),
            attr="update",
            ctx=ast.Load(),
        )
        assert_ast_equal(
            result.func,
            expected,
        )

    def test_logging_fstring(self):
        node = _expr('logging.info(f"{temp}")')
        result = self.replacer.visit_Call(node)
        joined = result.args[0]
        formatted = joined.values[0]
        assert isinstance(
            formatted.value,
            ast.Attribute,
        )
        assert formatted.value.attr == "temp"

    def test_visit_For(self):
        # For range arg
        node = _stmt("for i in range(temp):\n    pass")
        result = self.replacer.visit_For(node)
        arg = result.iter.args[0]
        assert isinstance(
            arg,
            ast.Attribute,
        )
        assert arg.attr == "temp"

    def test_visit_If(self):
        # Logical condition
        node = _stmt("if mask:\n    pass")
        result = self.replacer.visit_If(node)
        assert isinstance(
            result.test,
            ast.Attribute,
        )
        assert result.test.attr == "mask"

        # Compare condition
        node = _stmt("if temp > rain:\n    pass")
        result = self.replacer.visit_If(node)
        assert isinstance(
            result.test.left,
            ast.Attribute,
        )
        assert isinstance(
            result.test.comparators[0],
            ast.Attribute,
        )

    def test_replace_compare(self):
        compare = _expr("temp > rain")
        self.replacer._replace_compare(compare)
        assert isinstance(
            compare.left,
            ast.Attribute,
        )
        assert isinstance(
            compare.comparators[0],
            ast.Attribute,
        )

    def test_visit_BinOp(self):
        node = _expr("temp + rain")
        result = self.replacer.visit_BinOp(node)
        assert isinstance(
            result.left,
            ast.Attribute,
        )
        assert isinstance(
            result.right,
            ast.Attribute,
        )
        # Nested binop
        node = _expr("temp + (rain * soil)")
        result = self.replacer.visit_BinOp(node)
        assert isinstance(
            result.left,
            ast.Attribute,
        )
        assert isinstance(
            result.right.left,
            ast.Attribute,
        )
        assert isinstance(
            result.right.right,
            ast.Attribute,
        )

    def test_local_scope(self):
        self.replacer._local_scope.add("temp")
        node = ast.Name(
            id="temp",
            ctx=ast.Load(),
        )
        result = self.replacer.visit_Name(node)
        assert isinstance(result, ast.Name)
        assert result.id == "temp"
        self.replacer._local_scope.clear()
