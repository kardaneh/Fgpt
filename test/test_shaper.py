import pytest
import tempfile
import shutil
import sys, os
from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from shaper import Shaper
from logger import Logger
from processor import Processor

@pytest.fixture(scope="class")
def test_env(request):
    test_dir = tempfile.mkdtemp()

    # Create test Fortran files with 3-level deep module hierarchy
    level3_module = os.path.join(test_dir, "level3_mod.f90")

    with open(level3_module, "w") as f:
        f.write("""
        module level3_mod
        use level2_mod
        implicit none

        contains

        subroutine level3_sub(arr)
        real, dimension(40,30), intent(inout) :: arr
        call level2_sub(arr)
        end subroutine level3_sub

        end module level3_mod
        """)
    
    level2_module = os.path.join(test_dir, "level2_mod.f90")
    with open(level2_module, "w") as f:
        f.write("""
        module level2_mod
        use level1_mod
        implicit none
        real, dimension(30, 40) :: explicit_array_2d
                
        contains
                
        subroutine level2_sub(data_in)
        real, intent(inout) :: data_in(:,:)
        call level1_sub(data_in)
        end subroutine level2_sub
                
        end module level2_mod
        """)
    
    level1_module = os.path.join(test_dir, "level1_mod.f90")
    with open(level1_module, "w") as f:
        f.write("""
        module level1_mod
        implicit none
        real, dimension(10, 20) :: explicit_array_1d
                
        contains
                
        subroutine level1_sub(input_array)
        real, intent(inout) :: input_array(:,:)
        call main_caller(input_array)
        end subroutine level1_sub
                
        subroutine main_caller(inout_array)
        real, intent(inout) :: inout_array(:,:)
        inout_array = inout_array + 2. 
        end subroutine main_caller
                
        end module level1_mod
        """)

    processor = Processor(logger=Logger())
    level1_tree = processor.parse_fortran_file(level1_module)

    parsed_modules = {
            "level1_mod": level1_tree
        }
    
    module_path = {
        "level1_mod": level1_module
        }
    
    request.cls.test_dir = test_dir
    request.cls.level3_module = level3_module
    request.cls.processor = processor
    request.cls.level2_module = level2_module
    request.cls.level1_module = level1_module
    request.cls.level1_tree = level1_tree
    request.cls.parsed_modules = parsed_modules
    request.cls.module_path = module_path

    yield

    shutil.rmtree(test_dir)

@pytest.fixture(autouse=True)
def setup_method(request):
    cls = request.cls

    cls.dummy_arg_list = defaultdict(list)
    cls.actual_arg_spec_list = defaultdict(list)
    cls.call_subroutines = defaultdict(list)

    call_stmt = walk(cls.level1_tree, F23.Call_Stmt)[0]

    processor = Processor(logger=Logger())
    cls.call_subroutines["main_caller"].append(call_stmt)
    cls.actual_arg_spec_list["main_caller"].append(["input_array"])
    cls.dummy_arg_list["main_caller"] = ["inout_array"]
    cls.dummy_arg_list["level1_sub"] = ["input_array"]

    cls.shaper_level1 = Shaper(
        cls.test_dir,
        cls.parsed_modules,
        cls.module_path,
        cls.dummy_arg_list,
        cls.actual_arg_spec_list,
        cls.call_subroutines,
        logger=Logger()
    )

@pytest.mark.usefixtures("test_env")
class TestShaper:
    
    def test_3_level_deep_shape_resolution(self):
        # Level 3: Start with implicit array in level3_sub
        implicit_decl_level3 = "real, intent(inout) :: inout_array(:,:)"
        parsed_implicit_level3 = F23.Type_Declaration_Stmt(implicit_decl_level3)
        # Shape resolution should go through the call chain:
        shaped_level3 = self.shaper_level1.shaper_subroutine(parsed_implicit_level3, "main_caller")
        assert shaped_level3 is not None
        assert isinstance(shaped_level3, F23.Type_Declaration_Stmt)
        shaped_str = walk(shaped_level3, F23.Explicit_Shape_Spec_List)[0].tostr()
        assert '40, 30' in shaped_str

    def test_find_fortran_files_subroutine(self):

        # Test finding level3_sub from level2_mod perspective
        self.shaper_level1.current_module_imp = "level1_mod"
        self.shaper_level1.find_fortran_files_subroutine("level1_sub")

        # Verify the subroutine was found and call info was populated
        assert "level1_sub" in  self.shaper_level1.actual_arg_spec_list
        assert "level1_sub" in self.shaper_level1.call_subroutines
        #self.assertEqual(len(self.shaper_level2.call_subroutines["level3_sub"]), 1)

    def test_find_enclosing_subroutine(self):

        # Parse a subroutine with a call statement
        sub_code = """
        subroutine test_enclosing()
        integer :: x
        call some_sub(x)
        end subroutine test_enclosing
        """
        sub_tree = Processor(logger=Logger()).parse_fortran_string(sub_code)

        # Get the call statement node
        call_node = walk(sub_tree, F23.Call_Stmt)[0]

        # Find the enclosing subroutine
        enclosing = self.shaper_level1.find_enclosing_subroutine(call_node)

        assert enclosing is not None
        assert isinstance(enclosing, F23.Subroutine_Subprogram)
        assert walk(enclosing, F23.Name)[0].string == "test_enclosing"
    
    def test_complex_shape_scenarios(self):
        """Test more complex shape resolution scenarios"""

        # Create a module with mixed explicit and implicit shapes
        mixed_module = os.path.join(self.test_dir, "mixed_mod.f90")
        with open(mixed_module, "w") as f:
            f.write("""
            module mixed_mod
            implicit none
            real, dimension(15, 25, 35) :: multi_dim_array

            contains

            subroutine complex_sub(data)
            real, intent(inout) :: data(:,:,:)  ! 3D implicit shape
            data = data * 2.0
            end subroutine complex_sub

            subroutine intermediate_sub(arr)
            real, intent(inout) :: arr(:,:,:)   ! Also implicit
            call complex_sub(arr)
            end subroutine intermediate_sub

            subroutine starter()
            real :: my_array(15, 25, 35)
            call intermediate_sub(my_array)
            end subroutine starter

            end module mixed_mod
            """)

        # Parse and register module
        processor = self.processor 
        mixed_tree = processor.parse_fortran_file(mixed_module)
        self.parsed_modules["mixed_mod"] = mixed_tree

        # Update call information
        call_stmt = walk(mixed_tree, F23.Call_Stmt)[1]
        self.call_subroutines["intermediate_sub"].append(call_stmt)
        self.actual_arg_spec_list["intermediate_sub"].append(["my_array"])
        self.dummy_arg_list["intermediate_sub"] = ["arr"]

        call_stmt2 = walk(mixed_tree, F23.Call_Stmt)[0]
        self.call_subroutines["complex_sub"].append(call_stmt2)
        self.actual_arg_spec_list["complex_sub"].append(["arr"])
        self.dummy_arg_list["complex_sub"] = ["data"]

        # Test 3D shape resolution
        implicit_decl_3d = "real, intent(inout) :: data(:,:,:)"
        parsed_implicit_3d = F23.Type_Declaration_Stmt(implicit_decl_3d)

        mixed_shaper = Shaper(
            self.test_dir,
            self.parsed_modules,
            self.module_path,
            self.dummy_arg_list,
            self.actual_arg_spec_list,
            self.call_subroutines,
            logger=Logger()
        )

        shaped_3d = mixed_shaper.shaper_subroutine(parsed_implicit_3d, "complex_sub")
        assert shaped_3d is not None
        shaped_str_3d = walk(shaped_3d, F23.Explicit_Shape_Spec_List)[0].tostr()
        assert '15, 25, 35' in shaped_str_3d