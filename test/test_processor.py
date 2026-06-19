import pytest
import tempfile
import shutil
import sys, os
from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from logger import Logger
from processor import Processor

@pytest.fixture(scope="class")
def test_env(request):
    test_dir = tempfile.mkdtemp()

    processor = Processor(logger=Logger())
    processor.benchmark_dir = os.path.join('.', 'examples')

    # Create test Fortran files
    simple_fortran = os.path.join(test_dir, "simple.f90")
    with open(simple_fortran, "w") as f:
        f.write("""
        program simple
        integer :: a = 10
        end program simple
        """)

    complex_fortran = os.path.join(test_dir, "complex.f90")
    with open(complex_fortran, "w") as f:
        f.write("""
        module test_mod
        implicit none
        contains
        subroutine test_sub(a, b)
        integer, intent(in) :: a
        integer, intent(out) :: b
        b = a * 2
        end subroutine test_sub
        end module test_mod
        """)

    benchmark_subdir = os.path.join(processor.benchmark_dir, "test_sub")
    os.makedirs(benchmark_subdir, exist_ok=True)

    # attach to class if using class-based tests
    request.cls.test_dir = test_dir
    request.cls.processor = processor
    request.cls.simple_fortran = simple_fortran
    request.cls.complex_fortran = complex_fortran
    request.cls.benchmark_subdir = benchmark_subdir

    yield # https://stackoverflow.com/questions/26405380/how-do-i-correctly-setup-and-teardown-for-my-pytest-class-with-tests

    # teardown 
    shutil.rmtree(test_dir)
    shutil.rmtree(processor.benchmark_dir)

@pytest.mark.usefixtures("test_env")
class TestProcessor:

    def test_parse_fortran_file(self):
        # Test parsing a simple Fortran file
        parse_tree = self.processor.parse_fortran_file(self.simple_fortran)
        assert isinstance(parse_tree, F23.Program)
            
        # Test parsing a complex Fortran file
        parse_tree = self.processor.parse_fortran_file(self.complex_fortran)
        assert isinstance(parse_tree.children[1], F23.Module)
            
        # Test invalid file
        with pytest.raises(Exception):
            self.processor.parse_fortran_file("nonexistent.f90")

    def test_parse_fortran_string(self):
        # Test parsing a simple program
        code = "program test\ninteger :: a\nend program test"
        parse_tree = self.processor.parse_fortran_string(code)
        assert isinstance(parse_tree, F23.Program)
        
        # Test parsing a module
        code = "module test\nimplicit none\nend module test"
        parse_tree = self.processor.parse_fortran_string(code)
        assert isinstance(parse_tree.children[0], F23.Module)
        
        # Test invalid code
        with pytest.raises(Exception):
            self.processor.parse_fortran_string("invalid fortran code")
        
    def test_parse_fortran_statement(self):
        # Test assignment statement
        stmt = "a = 10"
        node = self.processor.parse_fortran_statement(stmt)
        assert isinstance(node.children[0], F23.Assignment_Stmt)
        
        # Test if statement
        stmt = "if (a > 0) then\nb = 10\nend if"
        node = self.processor.parse_fortran_statement(stmt)
        assert isinstance(node.children[0], F23.If_Construct)
        
        # Test invalid statement
        with pytest.raises(Exception):
            self.processor.parse_fortran_statement("invalid statement")

    def test_parse_fortran_comment(self):
        # Test comment parsing
        comment = "! This is a comment"
        node = self.processor.parse_fortran_comment(comment)
        assert node.tostr().strip() == "! This is a comment"
        
        # Test comment with program context
        comment = "! Module comment"
        node = self.processor.parse_fortran_comment(comment)
        assert node.tostr().strip() == "! Module comment"

    def test_find_enclosing_parent(self):
        # Initialize processor with mock logger
        # Test 1: Basic assignment statement parent finding
        assignment_stmt = F23.Assignment_Stmt("zwholdmax(ji, :) = snow3lhold_1d(snowrho(ji, :), snowdz(ji, :))")

        part_ref_node = assignment_stmt.children[-1]  # Part_Ref for snow3lhold_1d
        name_node = part_ref_node.children[0]  # Name('snow3lhold_1d')

        # Manually set up parent relationships
        part_ref_node.parent = assignment_stmt
        name_node.parent = part_ref_node

        # Test finding Assignment_Stmt parent from Name node
        result = self.processor.find_enclosing_parent(name_node, F23.Assignment_Stmt)
        assert result is not None, "Should find Assignment_Stmt parent"
        assert isinstance(result, F23.Assignment_Stmt), "Result should be Assignment_Stmt"
        assert result == assignment_stmt, "Should return the exact assignment statement"

        # Test 2: Finding Part_Ref parent from Name node
        result = self.processor.find_enclosing_parent(name_node, F23.Part_Ref)
        assert result is not None, "Should find Part_Ref parent"
        assert isinstance(result, F23.Part_Ref), "Result should be Part_Ref"
        assert result == part_ref_node, "Should return the exact Part_Ref node"

        # Test 3: Non-existent parent type
        result = self.processor.find_enclosing_parent(name_node, F23.Subroutine_Subprogram)
        assert result is None, "Should return None for non-existent parent type"

        # Test 4: Starting node is already the target type
        result = self.processor.find_enclosing_parent(assignment_stmt, F23.Assignment_Stmt)
        assert result is not None, "Should return the node itself when it matches target type"
        assert result == assignment_stmt, "Should return the same assignment statement"

        # Test 5: Null node input
        print("Test 5: Null node input")
        result = self.processor.find_enclosing_parent(None, F23.Assignment_Stmt)
        assert result is None, "Should return None for None input"

        # Test 6: Complex hierarchy with subroutine
        subroutine = self.processor.parse_fortran_string("subroutine test()\n  zwholdmax(ji, :) = snow3lhold_1d(snowrho(ji, :), snowdz(ji, :))\nend subroutine")
        assignment_stmt = walk(subroutine, F23.Assignment_Stmt)[0]
        part_ref_node = assignment_stmt.children[-1]  # Part_Ref for snow3lhold_1d
        name_node = part_ref_node.children[0]  # Name('snow3lhold_1d')

        result = self.processor.find_enclosing_parent(name_node, F23.Subroutine_Subprogram)
        assert result is not None,  "Should find Subroutine_Subprogram parent"
        assert isinstance(result, F23.Subroutine_Subprogram), "Result should be Subroutine_Subprogram"

    def test_initiate_empty_routine(self):
        # Test creating a simple subroutine
        subroutine_node = self.processor.initiate_empty_routine("test_sub")
        assert isinstance(subroutine_node, F23.Subroutine_Subprogram)
        assert "read_dummy" in subroutine_node.tostr()

    def test_separate_entity_declarations(self):
        # Test separating multiple declarations
        decl = "integer :: a, b, c"
        parsed_decl = self.processor.parse_fortran_statement(decl)
        separated = self.processor.separate_entity_declarations(parsed_decl.children[0])
        assert len(separated) == 3
        assert isinstance(separated[0], F23.Type_Declaration_Stmt)
        
        # Test single declaration
        decl = "real :: x"
        parsed_decl = self.processor.parse_fortran_statement(decl)
        separated = self.processor.separate_entity_declarations(parsed_decl.children[0])
        assert len(separated) == 1

    def test_add_entity_to_declaration(self):
        # Test adding entities to declaration
        decl = "real :: a, b"
        parsed_decl = self.processor.parse_fortran_statement(decl)
        modified = self.processor.add_entity_to_declaration(parsed_decl.children[0], ["a"])
        assert "a_copy" in modified.tostr()
        
        # Test with multiple variables
        modified = self.processor.add_entity_to_declaration(parsed_decl.children[0], ["a", "b"])
        assert "a_copy", modified.tostr()
        assert "b_copy", modified.tostr()

    def test_separate_entity_allocation(self):
        # Test separating allocation statements
        alloc = "allocate(a(10), b(20), stat=ierr)"
        parsed_alloc = self.processor.parse_fortran_statement(alloc)
        separated = self.processor.separate_entity_allocation(parsed_alloc.children[0])
        assert len(separated) == 2
        assert isinstance(separated[0], F23.Allocate_Stmt)
        
        # Test with stat option
        assert F23.Alloc_Opt_List("STAT = ierr").tostr() in separated[0].tostr()
    
    def test_add_entity_to_allocation(self):
        # Test adding entities to allocation
        alloc = "allocate(a(10))"
        parsed_alloc = self.processor.parse_fortran_statement(alloc)
        modified = self.processor.add_entity_to_allocation(parsed_alloc.children[0], ["a"], openacc=True)
        assert len(modified) == 2
        assert isinstance(modified[0].children[0], F23.If_Construct)
        assert "a_copy" in modified[1].tostr()

    def test_map_declaration(self):
        # Test mapping implicit to explicit declaration
        implicit = "real, dimension(:) :: b"
        explicit = "real, dimension(10) :: a"
        
        parsed_implicit = self.processor.parse_fortran_statement(implicit)
        parsed_explicit = self.processor.parse_fortran_statement(explicit)
        
        mapped = self.processor.map_declaration(parsed_implicit, parsed_explicit)
        assert F23.Dimension_Attr_Spec("DIMENSION(10)").tostr() in mapped.tostr()
        
        # Test with direct dimensions
        mapped = self.processor.map_declaration(parsed_implicit, dimensions="20")
        assert F23.Dimension_Attr_Spec("DIMENSION(20)").tostr() in mapped.tostr()
    
    def test_combine_allocate_declaration(self):
        # Test combining allocation and declaration
        decl = "real, allocatable :: a(:)"
        alloc = "allocate(a(10))"
        
        parsed_decl = self.processor.parse_fortran_statement(decl)
        parsed_alloc = self.processor.parse_fortran_statement(alloc)
        
        combined = self.processor.combine_allocate_declaration([parsed_decl.children[0], parsed_alloc.children[0]])
        assert F23.Dimension_Attr_Spec("DIMENSION(10)").tostr() in combined.tostr()
        assert F23.Attr_Spec_List("allocatable").tostr() not in combined.tostr()

    def test_remove_intent_and_save(self):
        # Test removing intent and save attributes
        decl = "real, intent(in), save :: a"
        parsed_decl = self.processor.parse_fortran_statement(decl)
        cleaned = self.processor.remove_intent_and_save([parsed_decl.children[0]])
        assert len(cleaned) == 1
        assert F23.Intent_Attr_Spec("intent(in)").tostr() not in cleaned[0].tostr()
        assert F23.Attr_Spec("save").tostr() not in cleaned[0].tostr()

    def test_out_module_fortran(self):
        # Test generating module code
        module_tree = self.processor.out_module_fortran("test_sub")
        assert isinstance(module_tree.children[1], F23.Module)
        assert "module_global" in module_tree.tostr()
        #self.assertIn("declaration_initialization", module_tree.tostr())

    def test_check_point(self):
        # Test creating check point for real scalar
        checkpoint = self.processor.check_point("a", "a_copy", ["REAL"])
        assert isinstance(checkpoint.children[0], F23.If_Construct)
        assert ".EQ." in checkpoint.tostr()
        
        # Test for logical array
        checkpoint = self.processor.check_point("flags", "flags_copy", ["LOGICAL", "DIMENSION"])
        assert isinstance(checkpoint.children[0], F23.If_Construct)
        assert ".EQV.", checkpoint.tostr()

    def test_out_main_fortran(self):
        # Test generating main program
        main_tree = self.processor.out_main_fortran()
        assert isinstance(main_tree, F23.Program)
        assert F23.Program_Stmt("program main").tostr() in main_tree.tostr()
        assert F23.Contains_Stmt("contains").tostr() in main_tree.tostr()

    def test_write_fortran_code_to_file(self):
        # Test writing code to file
        test_file = os.path.join(self.test_dir, "test_output.f90")
        code = "program test\ninteger :: a\nend program test"
        parse_tree = self.processor.parse_fortran_string(code)
        
        self.processor.write_fortran_code_to_file(parse_tree, test_file)
        assert os.path.exists(test_file)
        
        with open(test_file, "r") as f:
            content = f.read()
            assert F23.Program_Stmt("program test").tostr() in content
    
    def test_create_call_stmt(self):
        # Test creating call statement
        code = "subroutine test(a)\ninteger :: a\nend subroutine test"
        subroutine_tree = self.processor.parse_fortran_string(code)
        call_stmt = self.processor.create_call_stmt(subroutine_tree)
        
        assert isinstance(call_stmt, F23.Call_Stmt)
        assert call_stmt.tostr() == F23.Call_Stmt("CALL test(a)").tostr()

    def test_process_queue(self):
        # Test processing declaration queue
        declarations = [
            self.processor.parse_fortran_statement("real :: b").children[0],
            self.processor.parse_fortran_statement("integer, parameter :: a = 10").children[0],
            self.processor.parse_fortran_statement("real, allocatable :: d(:)").children[0],
            self.processor.parse_fortran_statement("integer, parameter :: c").children[0]
        ]
        
        processed = self.processor.process_queue(declarations)
        assert len(processed) == 4
        # Check that parameter without initialization comes first
        assert F23.Type_Declaration_Stmt("integer, parameter :: a = 10").tostr() == processed[0].tostr()
        # Check that allocatable comes last
        assert F23.Type_Declaration_Stmt("real, allocatable :: d(:)").tostr() == processed[-1].tostr()

    def test_compile_and_run(self):
        # This test would actually compile code - we'll just test the directory handling
        # In a real scenario, you'd need a proper build environment
        original_cwd = os.getcwd()
        test_dir = os.path.join(self.test_dir, "compile_test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a simple Fortran file
        test_file = os.path.join(test_dir, "test.f90")
        with open(test_file, "w") as f:
            f.write("program test\ninteger :: a\nend program test")
            
        result = self.processor.compile_and_run(original_cwd, test_dir)
        assert result == 0
        os.chdir(original_cwd)
        shutil.rmtree(test_dir)
