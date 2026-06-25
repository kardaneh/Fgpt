import os
import shutil
import tempfile

import pytest
from fparser.two import Fortran2003 as F23

from fgpt.logger import Logger
from fgpt.navigator import Navigator
from fgpt.processor import Processor


@pytest.fixture(scope="class")
def test_env(request):
    test_dir = tempfile.mkdtemp()

    simple_module = os.path.join(test_dir, "simple_mod.f90")
    with open(simple_module, "w") as f:
        f.write("""
        module simple_mod
        implicit none
        integer, parameter :: global_param = 42
        real, dimension(:), allocatable :: global_array
        contains

        subroutine test_sub(a, b, n)
            integer, intent(in) :: a
            integer, intent(in) :: n
            real, dimension(n), intent(out) :: b
            integer :: i
            real :: local_scalar
            local_scalar = real(a) * 2.0
            if (.not. allocated(global_array)) then
                allocate(global_array(5))
                global_array = [1.0, 2.0, 3.0, 4.0, 5.0]
            end if
            do i = 1, n
                b(i) = local_scalar + real(i) + global_param + global_array(mod(i-1, 5) + 1)
            end do

        end subroutine test_sub
        end module simple_mod
        """)

    dependent_module = os.path.join(test_dir, "dependent_mod.f90")
    with open(dependent_module, "w") as f:
        f.write("""
        module dependent_mod
        use simple_mod
        implicit none
        integer :: dependent_var

        contains

        subroutine dependent_sub(x, m)
        integer, intent(in) :: m
        real, dimension(m),intent(inout) :: x
        dependent_var = 5
        call test_sub(dependent_var, x, m)
        global_array =  [10.0, 20.0, 30.0, 40.0, 50.0]
        end subroutine dependent_sub

        end module dependent_mod
        """)

    processor = Processor(logger=Logger())
    simple_tree = processor.parse_fortran_file(simple_module)
    dependent_tree = processor.parse_fortran_file(dependent_module)
    parsed_modules = {"simple_mod": simple_tree, "dependent_mod": dependent_tree}
    module_path = {"simple_mod": simple_module, "dependent_mod": dependent_module}

    request.cls.test_dir = test_dir
    request.cls.simple_module = simple_module
    request.cls.dependent_module = dependent_module
    request.cls.simple_tree = simple_tree
    request.cls.dependent_tree = dependent_tree
    request.cls.parsed_modules = parsed_modules
    request.cls.module_path = module_path

    yield

    shutil.rmtree(test_dir)


@pytest.fixture(autouse=True)
def setup_method(request):
    cls = request.cls

    cls.simple_navigator = Navigator(
        cls.test_dir, cls.simple_tree, cls.parsed_modules, cls.module_path, Logger()
    )
    cls.dependent_navigator = Navigator(
        cls.test_dir, cls.dependent_tree, cls.parsed_modules, cls.module_path, Logger()
    )


@pytest.mark.usefixtures("test_env")
class TestNavigator:
    def test_initialization(self):
        # Test that initialization sets up all attributes correctly
        assert self.simple_navigator.module_dir_sc == self.test_dir
        assert isinstance(self.simple_navigator.module_tree_sc.children[1], F23.Module)
        assert self.simple_navigator.var_declaration == []
        assert self.simple_navigator.var_initial == []
        assert self.simple_navigator.return_key_sc is False
        assert self.simple_navigator.visited_modules_sc == set()
        assert self.simple_navigator.child_modules_sc == set()
        assert self.simple_navigator.module_set_sc == set()
        assert len(self.simple_navigator.queue_sc) == 0
        assert self.simple_navigator.full_scout is False

    def test_find_variable_in_module(self):
        # Test finding a variable in the current module
        self.simple_navigator.variable_name_sc = "global_param"
        self.simple_navigator.find_variable_in_module()

        assert self.simple_navigator.return_key_sc is True
        assert len(self.simple_navigator.var_declaration) == 1
        assert isinstance(
            self.simple_navigator.var_declaration[0], F23.Type_Declaration_Stmt
        )

        # Test finding an array
        self.simple_navigator.variable_name_sc = "global_array"
        self.simple_navigator.var_declaration = []
        self.simple_navigator.return_key_sc = False

        self.simple_navigator.find_variable_in_module()

        assert self.simple_navigator.return_key_sc is True
        assert len(self.simple_navigator.var_declaration) == 2

        type_decl_found = any(
            isinstance(stmt, F23.Type_Declaration_Stmt)
            for stmt in self.simple_navigator.var_declaration
        )
        allocate_found = any(
            isinstance(stmt, F23.Allocate_Stmt)
            for stmt in self.simple_navigator.var_declaration
        )

        assert allocate_found, "Allocate_Stmt not found in the parse tree"
        assert type_decl_found, "Type_Declaration_Stmt not found in the parse tree"

    def test_find_external_subroutine_in_module(self):
        # Test finding a subroutine in the current module
        self.simple_navigator.variable_name_sc = "test_sub"
        self.simple_navigator.find_external_subroutine_in_module()

        assert self.simple_navigator.return_key_sc is True
        assert len(self.simple_navigator.var_declaration) == 1
        assert isinstance(self.simple_navigator.var_declaration[0], F23.Use_Stmt)

    def test_add_modules_to_queue(self):
        # Test adding modules to the queue from USE statements
        self.dependent_navigator.add_modules_to_queue()

        # Verify the queue was populated correctly
        assert len(self.dependent_navigator.queue_sc) == 1
        assert len(self.dependent_navigator.module_set_sc) == 1
        assert "simple_mod" in self.dependent_navigator.module_set_sc

    def test_variable_finder(self):
        # Test finding a variable that requires module traversal
        self.dependent_navigator.variable_finder("global_param")

        # Verify the variable was found through module dependencies
        assert self.dependent_navigator.return_key_sc is True
        assert len(self.dependent_navigator.var_declaration) == 1
        assert (
            len(self.dependent_navigator.visited_modules_sc) == 2
        )  # dependent_mod and simple_mod

        # Test finding an array
        self.dependent_navigator.var_declaration = []
        self.dependent_navigator.return_key_sc = False

        self.dependent_navigator.variable_finder("global_array")

        assert self.dependent_navigator.return_key_sc is True
        assert len(self.dependent_navigator.var_declaration) == 2

        type_decl_found = any(
            isinstance(stmt, F23.Type_Declaration_Stmt)
            for stmt in self.dependent_navigator.var_declaration
        )
        allocate_found = any(
            isinstance(stmt, F23.Allocate_Stmt)
            for stmt in self.dependent_navigator.var_declaration
        )

        assert allocate_found, "Allocate_Stmt not found in the parse tree"
        assert type_decl_found, "Type_Declaration_Stmt not found in the parse tree"

    def test_external_subroutine_finder(self):
        # Test finding an external subroutine that requires module traversal
        self.dependent_navigator.external_subroutine_finder("test_sub")

        # Verify the subroutine was found through module dependencies
        assert self.dependent_navigator.return_key_sc is True
        assert len(self.dependent_navigator.var_declaration) == 1
        assert (
            len(self.dependent_navigator.visited_modules_sc) == 2
        )  # dependent_mod and simple_mod

    def test_find_var_in_child_modules(self):
        # Test the module traversal logic for variables
        self.dependent_navigator.variable_name_sc = "global_param"
        self.dependent_navigator.module_set_sc.add("dependent_mod")
        self.dependent_navigator.child_modules_sc.add("dependent_mod")
        self.dependent_navigator.visited_modules_sc.add("dependent_mod")

        self.dependent_navigator.add_modules_to_queue()

        # Verify the variable is found in child modules
        self.dependent_navigator.find_var_in_child_modules(key="variable")

        assert self.dependent_navigator.return_key_sc is True
        assert len(self.dependent_navigator.var_declaration) == 1

    def test_error_handling(self):
        self.simple_navigator.variable_finder("nonexistent_var")
        assert self.simple_navigator.return_key_sc is False

        self.simple_navigator.external_subroutine_finder("nonexistent_sub")
        assert self.simple_navigator.return_key_sc is False
