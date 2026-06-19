import ast
import pytest
from unittest.mock import MagicMock
import sys, os
from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from logger import Logger
from processor import Processor
from transformer import Transformer

@pytest.fixture(scope="class")
def test_env(request):

    transformer = Transformer(benchmark_dir="", isolator=MagicMock(), extractor=MagicMock(), ignore_case=None, config_path="template.yaml")
    transformer.benchmark_dir = os.path.join('.', 'examples')

    request.cls.transformer = transformer
    yield

@pytest.mark.usefixtures("test_env")
class TestTransformer:

    def test_read_file_ast(self):
        code = "x = np.int32(0)"
        tree = ast.parse(code)
        assign_nodes = [node for node in tree.body if isinstance(node, ast.Assign)]
        # Test read script creation for scalar values 
        result = self.transformer.read_file_ast(assign_nodes)
        assert len(result) == 1
        assert isinstance(result[0], ast.Assign)

        # Test empty ast for read_file_ast
        result = self.transformer.read_file_ast([])
        assert result == []

    # Test dependant variables retrieval
    def test_init_dependant_variables(self):
        code = """
a = 1
b = 2
        """
        read_ast = ast.parse(code)

        # fake dependency: c depends on a and b
        self.transformer.dependant_variables = {
            "c": ["a", "b"]
        }

        c_assign = ast.parse("c = a + b").body[0]

        updated_body = self.transformer.init_dependant_variables(
            read_ast,
            [c_assign]
        )

        code_out = ast.unparse(ast.Module(body=updated_body, type_ignores=[]))

        assert "c = a + b" in code_out
        assert code_out.index("b = 2") < code_out.index("c = a + b")

    # Test loop variable collection
    def test_collect_loop_variables(self):
        code = """
for i in range(10):
    pass

for j, k in [(1,2)]:
    pass
    """
        func = ast.parse(code)

        self.transformer.extractor.loop_dict = {
            "test": {"loop1": ["i"], "loop2": ["j", "k"]}
        }

        vars_collected = self.transformer._collect_loop_variables(func, "test")

        assert "i" in vars_collected
        assert "j" in vars_collected
        assert "k" in vars_collected

    # Test class template retrieval
    def test_prepare_read_code_global(self):
        code = "x = np.int32(0)"
        assign_nodes = [ast.parse(code).body[0]]

        # THese are for the "For" loop filling
        self.transformer.variable_order = ["x"]
        self.transformer.scalar = ["x"]

        result = self.transformer.prepare_read_code_for_global_template(
            assign_nodes,
            subroutine_key="test_sub"
        )

        assert isinstance(result, ast.Module)
    
    def test_create_instances(self):
        class_code = "class Hydrol_soil: pass"
        tree = ast.parse(class_code)

        # Test instance creation: hs = Hydrol_soil()
        class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        result = self.transformer.create_instances(class_nodes)
        
        assert len(result) == 1
        assert isinstance(result[0], ast.Assign)

        assign = result[0]
        assert isinstance(assign.targets[0], ast.Name)
        assert assign.value.func.id == "Hydrol_soil"

        # Test instance creation for object class inside another class 
        result = self.transformer.create_instances(class_nodes, self_mode=True)
        target = result[0].targets[0]
        assert isinstance(target, ast.Attribute)
        assert target.value.id == "self"

        # TEST for the create instance edge case where we have an invalid node
        result = self.transformer.create_instances([ast.Assign()])
        assert result is None

    def test_create_cls_info(self):
        # Create cls info has the class name as the primary key and the values are set inside the instance name or self of wiht all the attributes inside 
        code = """
class A: pass
class B: pass
"""
        tree = ast.parse(code)

        cls_info, imports, instances = self.transformer.create_cls_info(
            out_module=tree,
            subroutine_key="test_sub"
        )

        assert len(instances) == 2
        assert len(cls_info) == 2

        # Test create_cls_info with self_mode = True, means that the instance name will be that of 'self' instead of the class instance
        code = "class A: pass"
        tree = ast.parse(code)

        cls_info, _, instances = self.transformer.create_cls_info(
            out_module=tree,
            subroutine_key="test_sub",
            self_mode=True
        )
        assert len(instances) == 1
        assert 'self' in list(cls_info['A'].keys())
       
    def test_add_instance(self):
        # Test add instance with instance and method call obj = A(), obj.declaration()
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        instance_node = ast.Assign(
            targets=[ast.Name(id="obj", ctx=ast.Store())],
            value=ast.Call(func=ast.Name(id="A", ctx=ast.Load()), args=[], keywords=[])
        )

        self.transformer.add_instance(
            idx=0,
            instance_node=instance_node,
            cls_info={'A': {'obj': {'attributes': {}, 'methods': {"declaration": ast.FunctionDef(
                name="declaration",
                args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[],
                decorator_list=[]
            )}}}},
            functions_def=func,
            method_names=["declaration"]
        )

        assert len(func.body) == 2  # instance + call
        assert isinstance(func.body[0], ast.Assign)
        assert isinstance(func.body[1], ast.Expr) and isinstance(func.body[1].value.func, ast.Attribute)  # call type of obj.declaration
        assert func.body[1].value.func.attr == "declaration"

        # Test add instance for when we have instance node but the class doesn't have the method in question
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        self.transformer.add_instance(
            idx=0,
            instance_node=instance_node,
            cls_info={},
            functions_def=func,
            method_names=["declaration"]
        )

        assert len(func.body) == 1 # If the method in question is not present inside the cls_info not the instance is added
        
        # Test add instance for an invalid values
        with pytest.raises(TypeError):
            self.transformer.add_instance(
                idx=0,
                instance_node=ast.Assign(),
                cls_info={},
                function_def=None,
                method_names=[]
            )

    def test_resolve_insert_index_bounds(self):
        # Resolves index bounds for the adding instances 
        func = ast.parse("""
def f():
    a = 1
    b = 2
        """).body[0]

        idx = self.transformer._resolve_insert_index(10, func)

        assert isinstance(idx, int)
        assert 0 <= idx <= len(func.body)
        
        # Test index bounds insertion for negative values 
        func = ast.parse("def f(): pass").body[0]
        idx = self.transformer._resolve_insert_index(-1, func)
        assert idx >= 0

    def test_adjust_for_dependencies(self):
        # dependencie injection after the from 0 to 1 due to the presence of call function
        code = """
a = 1
b = x()
    """
        func = ast.FunctionDef(
            name="f",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.parse(code).body,
            decorator_list=[]
        )

        idx = self.transformer._adjust_for_dependencies(0, func)
        assert idx == 1

        # No change in dependency adjustement
        func = ast.parse("""
def f():
    a = 1
    b = 2
        """).body[0]

        idx = self.transformer._adjust_for_dependencies(0, func)
        assert idx == 0
        
        # Dependency adjustement after the function call 
        func = ast.parse("""
def f():
    a = 1
    b = x()
        """).body[0]

        idx = self.transformer._adjust_for_dependencies(2, func)

        assert idx == 2  # no shift because idx >= i
    
    def test_get_instance_info_name(self):
        # Retrieve instance name
        node = ast.Assign(
            targets=[ast.Name(id="obj", ctx=ast.Store())],
            value=ast.Call(func=ast.Name(id="A", ctx=ast.Load()), args=[], keywords=[])
        )

        name, ref = self.transformer._get_instance_info(node)
        assert name == "obj"
        assert isinstance(ref, ast.Name)

    def test_create_call_statements(self):
        # Test create call statement with no return 
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        result = self.transformer.create_call_statements(func)

        assert isinstance(result, ast.Expr)
        assert isinstance(result.value, ast.Call)
        assert result.value.func.id == "foo"

        # Test for create call statement with return statement
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Return(value=ast.Name(id="b", ctx=ast.Load()))],
            decorator_list=[]
        )

        result = self.transformer.create_call_statements(func)
        assert isinstance(result, ast.Assign)  # uses _build_assignment

        # Test create call statement for method call obj.method()
        func = ast.FunctionDef(
            name="bar",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[],
            decorator_list=[]
        )

        result = self.transformer.create_call_statements(func, instance="obj")

        assert isinstance(result.value.func, ast.Attribute)
        assert result.value.func.attr == "bar"

        # Call statements with args
        func = ast.parse("""
def foo(a, b): pass
        """).body[0]

        result = self.transformer.create_call_statements(func)
        assert isinstance(result.value, ast.Call)

        # Test edge case, when we have a funciton that doesn't have a `self` arg
        func = ast.FunctionDef(
            name="bar",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[],
            decorator_list=[]
        )
        with pytest.raises(ValueError):
            result = self.transformer.create_call_statements(func, instance="obj")

    def test_build_args(self):
        # Test helper function to build args for the call statement
        func = ast.FunctionDef(
            name="f",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="a"), ast.arg(arg="self"), ast.arg(arg="b")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[],
            decorator_list=[]
        )

        args = self.transformer._build_args(func)

        assert len(args) == 2
        assert all(isinstance(a, ast.Name) for a in args)
    
    # Test assignement creation helper 
    def test_build_assignment_tuple(self):
        return_node = ast.Return(
            value=ast.Tuple(elts=[ast.Name(id="a"), ast.Name(id="b")])
        )

        call = ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[], keywords=[])

        result = self.transformer._build_assignment(return_node, call)

        assert isinstance(result.targets[0], ast.Tuple)

    def test_convert_specification(self):
        # Test for scalar declaration
        code = "integer, parameter :: i_std = 4"
        decl = Processor(logger=Logger()).parse_fortran_statement(code)

        self.transformer.isolator.processor.remove_intent_and_save = MagicMock(return_value = [decl])
        result = self.transformer.convert_SPECIFICATION_PART([[decl]])

        assert isinstance(result, list)
        assert isinstance(result[0], ast.Assign)

        # Test for scalar condition without parameter condition 
        code = "real :: a = 10."
        decl = Processor(logger=Logger()).parse_fortran_statement(code)
        self.transformer.isolator.processor.remove_intent_and_save = MagicMock(return_value = [decl])

        result = self.transformer.convert_SPECIFICATION_PART([[decl]])
        assert isinstance(result[0], ast.Assign)
        assert isinstance(result[0].value, ast.Call) and isinstance(result[0].value.func, ast.Attribute)
        assert result[0].value.func.attr == 'float64'

        # Now check for array 
        code = "REAL(KIND = r_std), ALLOCATABLE, DIMENSION(kjpindex, nstm) :: ae_ns"
        decl = Processor(logger=Logger()).parse_fortran_statement(code)
        self.transformer.isolator.processor.remove_intent_and_save = MagicMock(return_value = [decl])

        result = self.transformer.convert_SPECIFICATION_PART([[decl]])
        assert isinstance(result[0], ast.Assign)
        assert isinstance(result[0].value, ast.Call) and isinstance(result[0].value.func, ast.Attribute)
        assert result[0].value.func.attr == 'zeros'
        # check the dimension of the created array

        assert isinstance(result[0].value.args[0], ast.Tuple) and len(result[0].value.args[0].elts) == 2
        assert len(result[0].value.keywords) == 1 and isinstance(result[0].value.keywords[0], ast.keyword)
        assert isinstance(result[0].value.keywords[0].value, ast.Attribute) and result[0].value.keywords[0].value.attr == 'float64'
        
        # Test in the case of the convertion is for a function
        self.transformer._contains_function = MagicMock(return_value=True)
        result = self.transformer.convert_SPECIFICATION_PART(["function foo"])
        assert result == []

    def test_pre_init_variables(self):
        # Assign 
        code = """
a = 1
b = 2
    """
        tree = ast.parse(code)
        collected = []
        def fake_collect(target):
            if isinstance(target, ast.Name):
                collected.append(target.id)
        self.transformer._collect_target_names = fake_collect
        self.transformer.pre_init_variables(tree)
        assert set(collected) == {"a", "b"}

        # Annassign 
        code = """
x: int = 5
    """
        tree = ast.parse(code)
        collected = []
        self.transformer._collect_target_names = fake_collect
        self.transformer.pre_init_variables(tree)
        assert collected == ["x"]

        # Empty variables
        tree = ast.parse("")
        self.transformer._collect_target_names = MagicMock()
        self.transformer.pre_init_variables(tree)
        assert self.transformer.pre_init == []

        # Exception handling for the helper function
        tree = ast.parse("a = 1")
        def broken_collect(target):
            raise RuntimeError("fail")
        self.transformer._collect_target_names = broken_collect
        with pytest.raises(RuntimeError):
            self.transformer.pre_init_variables(tree)
    
    def test_search_dependant_variables(self):
        # Search for dependant variables 
        code = """
        integer :: n
        integer :: m 
        real :: arr(n, m)
        """

        tree = Processor(logger=Logger()).parse_fortran_statement(code)
        decls = list(walk(tree, F23.Type_Declaration_Stmt))

        self.transformer.pre_init = set()
        self.transformer._preprocess_declarations = MagicMock(side_effect=lambda x: x)
        self.transformer.search_dependant_variables(decls)

        assert set(self.transformer.dependant_variables["arr"]) == {"n", "m"}
        
        # Multiple depdencies 
        code = """
        integer :: x
        integer :: y
        integer :: z
        real :: a(x, y)
        real :: b(z)
        """

        tree = Processor(logger=Logger()).parse_fortran_statement(code)
        decls = list(walk(tree, F23.Type_Declaration_Stmt))

        self.transformer.pre_init = set()
        self.transformer._preprocess_declarations = MagicMock(side_effect=lambda x: x)
        self.transformer.search_dependant_variables(decls)

        result = self.transformer.dependant_variables
        assert set(result["a"]) == {"x", "y"}
        assert set(result["b"]) == {"z"}

        # No dependant variables  
        self.transformer._build_symbol_table = MagicMock(return_value={})
        self.transformer._preprocess_declarations = MagicMock(side_effect=lambda x: x)
        self.transformer._is_array_declaration = MagicMock(return_value=False)
        self.transformer.search_dependant_variables(decls)
        assert self.transformer.dependant_variables == {}

        # Exception handling 
        self.transformer._build_symbol_table = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            self.transformer.search_dependant_variables([object()])

    def test_get_read_func(self):
        # TEST read function 
        assert self.transformer._get_read_func("float64") == "read_reals"
        assert self.transformer._get_read_func("int32") == "read_ints"
    
    def test_get_imports_from_specs(self):
        # Test import from specification
        specs = [("mod", ["A", "B"])]

        result = self.transformer.get_imports_from_specs(specs)

        assert len(result) == 1
        assert isinstance(result[0], ast.ImportFrom)

    def test_separate_scalar_with_input(self, monkeypatch):
        # Declaration stmts sent as inputs 
        self.transformer.global_state = False

        code = """
        integer :: a
        integer :: b = 5
        integer, dimension(10) :: c
        integer :: d(20)
        logical :: flag
        real, intent(in) :: x
        real :: y = 3.14
        """

        tree = Processor(logger=Logger()).parse_fortran_statement(code)
        decls = list(walk(tree, F23.Type_Declaration_Stmt))

        self.transformer.separate_scalar(
            subroutine_key="test",
            dec_stmts=decls
        )
        assert set(self.transformer.scalar) == {"a", "flag", "x"}

        # Declaration stmts coming from extractor.dec_global thus global state
        dec_global = {}

        for decl in decls:
            for entity in walk(decl, F23.Entity_Decl):
                name = entity.children[0].string
                dec_global[name] = [decl]
        self.transformer.global_state = True
        self.transformer.variable_order = ["a", "b", "flag", "x"]

        self.transformer.extractor.dec_global = {
            "test": dec_global
        }

        self.transformer.separate_scalar("test")
        assert set(self.transformer.scalar) == {"a", "flag", "x"}

        # Test the case of unknown intent 
        self.transformer.global_state = False
        code = """
        integer :: x
        """

        tree = Processor(logger=Logger()).parse_fortran_statement(code)
        decls = list(walk(tree, F23.Type_Declaration_Stmt))
        self.transformer.extractor.var_dummy = {
            "test": decls
        }

        self.transformer.pre_init = set()
        self.transformer.separate_scalar("test")

        assert self.transformer.scalar == []
    
    def test_insert_at(self):
        # Insert imports after existing import 
        module = ast.Module(
            body=[ast.Import(names=[ast.alias(name="sys")])],
            type_ignores=[]
        )
        node = ast.Import(names=[ast.alias(name="os")])
        self.transformer.insert_at(None, node, module)
        assert isinstance(module.body[1], ast.Import)
        assert len(module.body) == 2

        # Insert function inside module
        module = ast.Module(body=[], type_ignores=[])
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Pass()],
            decorator_list=[]
        )

        self.transformer.insert_at(None, func, module)
        assert isinstance(module.body[0], ast.FunctionDef)

        # Insert function inside class 
        class_node = ast.ClassDef(
            name="MyClass",
            bases=[],
            keywords=[],
            body=[],
            decorator_list=[]
        )
        module = ast.Module(body=[class_node], type_ignores=[])
        method = ast.FunctionDef(
            name="method",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Pass()],
            decorator_list=[]
        )

        self.transformer.insert_at(None, method, module)
        assert isinstance(class_node.body[0], ast.FunctionDef)

        # Insert class after imports 
        module = ast.Module(
            body=[ast.Import(names=[ast.alias(name="sys")])],
            type_ignores=[]
        )
        class_node = ast.ClassDef(
            name="MyClass",
            bases=[],
            keywords=[],
            body=[],
            decorator_list=[]
        )

        self.transformer.insert_at(None, class_node, module)
        assert isinstance(module.body[0], (ast.Import, ast.ImportFrom))
        assert isinstance(module.body[1], ast.ClassDef)

        # Insert assign in function
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        module = ast.Module(body=[func], type_ignores=[])
        assign = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Constant(value=1)
        )

        self.transformer.insert_at(None, assign, module, method_name="foo")
        assert isinstance(func.body[0], ast.Assign)

        # Insert with index 
        module = ast.Module(body=[], type_ignores=[])
        func1 = ast.FunctionDef(
            name="a",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Pass()],
            decorator_list=[]
        )
        func2 = ast.FunctionDef(
            name="b",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Pass()],
            decorator_list=[]
        )
        module.body.append(func1)
        self.transformer.insert_at(0, func2, module)
        assert module.body[0].name == "b"

        # Unsupported index
        module = ast.Module(body=[], type_ignores=[])
        node = ast.If(test=ast.Constant(value=True), body=[], orelse=[])
        with pytest.raises(TypeError):
            self.transformer.insert_at(None, node, module)

    def make_assign(self, name, expr):
        return ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.parse(expr).body[0].value
        )
    
    def test_insert_all_assign_nodes(self):
        # Basic testing of inserting assign nodes with respect to their dependency
        self.transformer.global_state = False
        # a depends on b thus b must come first
        assign_nodes = [
            self.make_assign("a", "b + 1"),
            self.make_assign("b", "1")
        ]

        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        module = ast.Module(body=[func], type_ignores=[])
        self.transformer.insert_all_assign_nodes(assign_nodes, module, method_name="foo")
        names = [stmt.targets[0].id for stmt in func.body]
        assert names == ["b", "a"]

        # Testing with no dependcies 
        self.transformer.global_state = False
        assign_nodes = [
            self.make_assign("a", "1"),
            self.make_assign("b", "2"),
        ]

        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        module = ast.Module(body=[func], type_ignores=[])
        self.transformer.insert_all_assign_nodes(assign_nodes, module, method_name="foo")
        names = {stmt.targets[0].id for stmt in func.body}
        assert names == {"a", "b"}

        # Testing with global_state at True with newly variables placed first and then declared
        self.transformer.global_state = True
        self.transformer.variable_order = ["a"]

        self.transformer.dependant_variables = {}

        assign_nodes = [
            self.make_assign("a", "1"),  
            self.make_assign("b", "2"),     
        ]

        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        module = ast.Module(body=[func], type_ignores=[])
        self.transformer.insert_all_assign_nodes(assign_nodes, module, method_name="foo")
        names = [stmt.targets[0].id for stmt in func.body]
        # b first (new), then a (declared)
        assert names == ["b", "a"]

        # Testing with global_state True and skipping dependencies since the b is dependant on another variable 
        self.transformer.global_state = True
        self.transformer.variable_order = ["a", "b"]
        # b is dependency-managed -> should be skipped
        self.transformer.dependant_variables = {"b": ["x"]}
        assign_nodes = [
            self.make_assign("a", "1"),
            self.make_assign("b", "2"),
        ]

        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        module = ast.Module(body=[func], type_ignores=[])
        self.transformer.insert_all_assign_nodes(assign_nodes, module, method_name="foo")
        names = [stmt.targets[0].id for stmt in func.body]
        assert names == ["a"]  # b skipped

        # Inserting with dependency chain
        self.transformer.global_state = False
        assign_nodes = [
            self.make_assign("c", "b + 1"),
            self.make_assign("b", "a + 1"),
            self.make_assign("a", "1"),
        ]

        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )

        module = ast.Module(body=[func], type_ignores=[])
        self.transformer.insert_all_assign_nodes(assign_nodes, module, method_name="foo")
        names = [stmt.targets[0].id for stmt in func.body]
        assert names == ["a", "b", "c"]

        # Testing with attributes inside a function with no dependencies  
        self.transformer.global_state = False
        node = ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="x", ctx=ast.Store())],
            value=ast.Constant(value=1)
        )
        func = ast.FunctionDef(
            name="foo",
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[],
            decorator_list=[]
        )
        module = ast.Module(body=[func], type_ignores=[])
        self.transformer.insert_all_assign_nodes([node], module, method_name="foo")
        assert isinstance(func.body[0], ast.Assign)

        # Exception handling 
        self.transformer.global_state = False
        bad_node = ast.Assign(targets=[], value=ast.Constant(value=1))
        module = ast.Module(body=[], type_ignores=[])
        with pytest.raises(ValueError):
            self.transformer.insert_all_assign_nodes([bad_node], module, method_name="foo")