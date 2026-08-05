import os
import shutil
import tempfile
from collections import defaultdict

import pytest
from fparser.two import Fortran2003 as F23

from fgpt.core.common.logger import Logger
from fgpt.core.frontend.extractor import Extractor
from fgpt.core.frontend.processor import Processor


@pytest.fixture(scope="class")
def test_env(request):
    # Create temporary directory
    test_dir = tempfile.mkdtemp()

    simple_module = os.path.join(test_dir, "simple_mod.f90")
    with open(simple_module, "w") as f:
        f.write("""
        module simple_mod
        implicit none
        integer, parameter :: i_std = 4

        contains

        subroutine test_sub(a, b)
        integer, intent(in) :: a
        real, intent(out) :: b(:)
        integer :: i
        do i = 1, size(b)
            b(i) = a * 2.0
        end do
        end subroutine test_sub

        end module simple_mod
        """)

    complex_module = os.path.join(test_dir, "complex_mod.f90")
    with open(complex_module, "w") as f:
        f.write("""
        module complex_mod
        use simple_mod
        implicit none
        integer, parameter :: n = 10

        contains

        subroutine complex_sub(x, y)
        real, intent(inout) :: x(n)
        real, intent(out) :: y
        integer :: j
        call test_sub(5, x)
        y = sum(x)
        end subroutine complex_sub

        subroutine helper_fn(z, res)
        real, intent(in) :: z(n)
        real, intent(out):: res
        res = sqrt(sum(z**2))
        end subroutine helper_fn

        end module complex_mod
        """)

    processor = Processor(logger=Logger())
    simple_tree = processor.parse_fortran_file(simple_module)
    complex_tree = processor.parse_fortran_file(complex_module)

    request.cls.test_dir = test_dir
    request.cls.simple_module = simple_module
    request.cls.complex_module = complex_module
    request.cls.processor = processor
    request.cls.simple_tree = simple_tree
    request.cls.complex_tree = complex_tree

    yield

    shutil.rmtree(test_dir)


@pytest.mark.usefixtures("test_env")
class TestExtractor:
    def setup_method(self):
        self.simple_extractor = Extractor(
            self.test_dir, self.simple_tree, logger=Logger()
        )
        self.complex_extractor = Extractor(
            self.test_dir, self.complex_tree, logger=Logger()
        )

    def test_simple_extractor_init(self):
        assert self.simple_extractor is not None

    def test_complex_extractor_init(self):
        assert self.complex_extractor is not None

    def test_initialization(self):
        # Test that initialization sets up all attributes correctly
        assert self.simple_extractor.module_dir == self.test_dir
        assert isinstance(self.simple_extractor.module_tree.children[1], F23.Module)
        assert isinstance(self.simple_extractor.subroutines, defaultdict)
        assert isinstance(self.simple_extractor.dummy_arg_list, defaultdict)
        assert self.simple_extractor.exclude == {
            "kjpindex",
            "nslm",
            "nstm",
            "nvm",
            "nice",
            "ncirc",
            "nsnow",
            "DIM",
            "dim",
            "MASK",
            "next_calc_loop",
        }

    def test_extract_loop_vect(self):
        extractor = Extractor(self.test_dir, self.simple_tree, logger=Logger())

        extractor.find_subroutines()
        sub_key = "test_sub"
        sub_tree = extractor.subroutines[sub_key]

        # Non-vector loop
        extractor.extract_loop_vect(sub_key, sub_tree)
        assert extractor.loop_vect[sub_key] is None

        # Vector loop case
        code = """
        subroutine vect_sub(a)
        integer, intent(in) :: a
        real :: b(10)
        integer :: i
        do i = 1, kjpindex
            b(i) = a * 2.0
        end do
        end subroutine vect_sub
        """

        parser = Processor(logger=Logger())
        sub_tree = parser.parse_fortran_string(code)

        extractor.extract_loop_vect("vect_sub", sub_tree)
        assert extractor.loop_vect["vect_sub"] is not None

    def test_find_subroutines(self):
        extractor = Extractor(self.test_dir, self.simple_tree, logger=Logger())
        extractor.find_subroutines()

        assert extractor.subroutine_keys_all == {"test_sub"}
        assert extractor.subroutine_keys_ncl == {"test_sub"}

        complex_extractor = Extractor(self.test_dir, self.complex_tree, logger=Logger())
        complex_extractor.find_subroutines()

        assert complex_extractor.subroutine_keys_all == {
            "test_sub",
            "complex_sub",
            "helper_fn",
        }

        assert set(complex_extractor.call_within_sub["complex_sub"].keys()) == {
            "test_sub"
        }

    def test_extract_intent_clean_subroutine(self):
        complex_extractor = Extractor(self.test_dir, self.complex_tree, logger=Logger())
        simple_extractor = Extractor(self.test_dir, self.simple_tree, logger=Logger())

        complex_extractor.find_subroutines()
        simple_extractor.find_subroutines()

        called_key = "test_sub"
        called_tree = simple_extractor.subroutines[called_key]

        simple_extractor.find_variables(called_tree, called_key)
        simple_extractor.extract_intent(called_key, called_tree)

        assert simple_extractor.general_usage_dict[called_key]["a"] == "IN"

        simple_extractor.clean_subroutine(called_key, called_tree)

        assert simple_extractor.general_usage_dict[called_key]["b"] == "INOUT"

    def test_add_intent(self):
        decl = "real :: a(10)"
        parsed_decl = Processor(logger=Logger()).parse_fortran_statement(decl)

        new_decl = Extractor(
            self.test_dir, self.simple_tree, logger=Logger()
        ).add_intent(parsed_decl.children[0], "in")
        assert "INTENT(IN)" in new_decl.tostr()

        new_decl = Extractor(
            self.test_dir, self.simple_tree, logger=Logger()
        ).add_intent(parsed_decl.children[0], "out")
        assert "INTENT(OUT)" in new_decl.tostr()

    def test_find_global_variables(self):
        extractor = Extractor(self.test_dir, self.complex_tree, logger=Logger())

        extractor.find_subroutines()

        sub_key = "complex_sub"
        sub_tree = extractor.subroutines[sub_key]

        extractor.find_variables(sub_tree, sub_key)

        extractor.find_global_variables(
            self.test_dir,
            self.complex_tree,
            extractor.var_global[sub_key],
            sub_key,
        )

        assert "test_sub" in extractor.dec_global[sub_key]

    def test_extract_array_info(self):
        extractor = Extractor(self.test_dir, self.complex_tree, logger=Logger())

        extractor.find_subroutines()

        sub_key = "complex_sub"
        sub_tree = extractor.subroutines[sub_key]

        extractor.find_variables(sub_tree, sub_key)

        extractor.find_global_variables(
            self.test_dir,
            self.complex_tree,
            extractor.var_global[sub_key],
            sub_key,
        )

        extractor.extract_all_array_info(
            extractor.dec_global[sub_key],
            extractor.var_dummy[sub_key],
            sub_key,
        )

        assert "x" in extractor.all_array_info[sub_key]
        assert len(extractor.all_array_info[sub_key]["x"]) == 1

    def test_process_declaration_variables(self):
        extractor = Extractor(self.test_dir, self.simple_tree, logger=Logger())

        extractor.find_subroutines()

        sub_key = "test_sub"
        sub_tree = extractor.subroutines[sub_key]

        extractor.find_variables(sub_tree, sub_key)

        extractor.process_declaration_variables(
            extractor.var_dummy[sub_key],
            sub_key,
        )

        assert extractor.scalar_variables[sub_key] == [F23.Name("a")]
        assert extractor.shapes_variables[sub_key] == [F23.Name("n")]
