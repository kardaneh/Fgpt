import argparse
import os
import shutil
from collections import defaultdict, deque

from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

from fgpt.extractor import Extractor
from fgpt.logger import Logger
from fgpt.processor import Processor
from fgpt.transformer import Transformer


class Isolator:
    """
    The Isolator class is designed to extract and prepare a Fortran procedure (e.g., subroutine or function)
    so that it can be compiled and executed independently of the rest of the original codebase.

    This process is particularly useful for:
    - Isolated testing or debugging of specific Fortran routines
    - Simplified transformation, such as source-to-source translation (e.g., to Python)
    - Generating standalone reproducible test cases from large code bases

    Functionality:
    --------------
    - Identifies and loads the specified Fortran module (`target_module`)
    - Parses the source file into an abstract syntax tree (AST) using `fparser`
    - Stores both original and parsed versions of the module
    - Sets up paths for temporary isolated compilation and execution
    - Prepares metadata about subroutine calls and dependencies

    Parameters
    ----------
    rest_of_path : str
        The relative path to the directory containing the target Fortran module.
    target_module : str
        The name of the module to be isolated (without `.f90`).
    work : str
        The working directory root (typically an environment variable like `$works`).

    Attributes
    ----------
    module_file_sp : str
        Path to the original or copied source file of the target module.
    module_tree_sp : Fortran2008.Program
        AST representation of the module using `fparser`.
    module_tree_cp : Fortran2008.Program
        A re-parsed version of the module tree for structural transformation.
    module_string : str
        Stringified source code of the module (used for copying or rewriting).
    target_module_dir : str
        Output directory for the isolated module artifacts.
    child_procedure_call : dict
        Tracks subroutine calls made within other subroutines.
    child_error_flag : dict
        Tracks compilation or transformation errors during isolation.

    Notes
    -----
    The isolation workflow is designed to be extendable. This class can later be
    used in conjunction with automatic input generation, test harness creation,
    or source-to-source translation routines.
    """

    def __init__(
        self,
        rest_of_path="modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/",
        target_module="hydrol",
        work=os.getenv("works"),
        openacc=False,
        tapenade=False,
        f2py=False,
    ):
        self.logger = Logger(console_output=True, file_output=True, record=True)
        self.logger.show_header("Isolator")
        self.processor = Processor(logger=self.logger)
        self.rest_of_path = rest_of_path
        self.target_module = target_module
        self.scratch_dir = work
        self.module_dir_sp = os.path.join(self.scratch_dir, self.rest_of_path)
        self.path_to_target = os.path.join(
            self.module_dir_sp, f"{self.target_module}.f90"
        )
        path_to_original = os.path.join(
            self.module_dir_sp, f"{self.target_module}_org.fgpt"
        )
        if os.path.exists(path_to_original):
            file_to_parse = path_to_original
        else:
            shutil.copy(self.path_to_target, path_to_original)
            file_to_parse = self.path_to_target
        self.module_tree_cp = self.processor.parse_fortran_file(file_to_parse)
        self.target_module_dir = os.path.join(
            os.getcwd(), self.target_module.split(".")[0]
        )
        self.openacc = openacc
        self.f2py = f2py
        self.tapenade = tapenade
        self.working_subroutines = defaultdict()
        self.isolated_subroutines = set()

        self.isolate_procedure = self.logger.log_event("Isolated Procedure")(
            self.isolate_procedure
        )

    def create_target_directory(self):
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        os.makedirs(self.target_module_dir)

    def collect_all_subroutines(self, cls, prodecure_key):
        collected = set()
        queue = deque([prodecure_key])

        while queue:
            current_key = queue.popleft()
            if current_key not in collected:
                collected.add(current_key)
                if current_key in cls.call_within_sub:
                    for child_procedure in cls.call_within_sub[current_key].keys():
                        if child_procedure not in collected:
                            queue.append(child_procedure)
        return collected

    def isolate_procedure(
        self, cls, parent_procedure, child_procedure, transformer=None
    ):
        call_statements = cls.call_within_sub[parent_procedure][child_procedure]

        for i, call_stmt in enumerate(call_statements):
            self.logger.info(f"  Call site {i + 1}: {call_stmt.tostr()}")

        procedure_tree = cls.subroutines[child_procedure]

        if isinstance(procedure_tree, F23.Subroutine_Subprogram):
            procedure_type = "subroutine"
        elif isinstance(procedure_tree, F23.Function_Subprogram):
            procedure_type = "function"
        else:
            raise ValueError(f"Unknown procedure type: {type(procedure_tree)}")

        cls.find_variables(procedure_tree, child_procedure, parent_procedure)

        calls = cls.call_within_sub[child_procedure]
        cls.var_global[child_procedure] = [
            name
            for name in cls.var_global[child_procedure]
            if name.tostr() not in calls
        ]

        if cls.var_global[child_procedure]:
            cls.find_global_variables(
                self.module_dir_sp,
                self.module_tree_cp,
                cls.var_global[child_procedure],
                child_procedure,
            )

        if cls.var_dummy[child_procedure]:
            cls.process_declaration_variables(
                cls.var_dummy[child_procedure], child_procedure
            )

        if cls.dec_global[child_procedure]:
            for key in cls.dec_global[child_procedure].keys():
                cls.process_declaration_variables(
                    cls.dec_global[child_procedure][key], child_procedure
                )

        scalar_names = {var.string for var in cls.scalar_variables[child_procedure]}
        global_names = {var.string for var in cls.var_global[child_procedure]}
        shape_to_search = [
            var
            for var in cls.shapes_variables[child_procedure]
            if var.string not in scalar_names and var.string not in global_names
        ]

        # this is for fully in/out procedure
        # shape_to_read = [
        #         var for var in cls.shapes_variables[child_procedure]
        #         if var.string in scalar_names and var.string not in global_names
        #         ]

        if shape_to_search:
            cls.find_global_variables(
                self.module_dir_sp,
                self.module_tree_cp,
                shape_to_search,
                child_procedure,
            )
            cls.var_global[child_procedure].extend(shape_to_search)

        cls.extract_all_array_info(
            cls.dec_global[child_procedure],
            cls.var_dummy[child_procedure],
            child_procedure,
        )

        nested_procedures = cls.call_within_sub[child_procedure].keys()

        if nested_procedures:
            self.logger.info(
                f"Found {len(nested_procedures)} nested procedure(s) in '{child_procedure}':"
            )
            for i, grand_child_procedure in enumerate(nested_procedures, 1):
                self.logger.info(f"   {i}. {grand_child_procedure}")
                if grand_child_procedure not in self.isolated_subroutines:
                    self.logger.info(
                        f"🔄 Call recursively for '{grand_child_procedure}' from parent '{child_procedure}'"
                    )
                    self.isolate_procedure(
                        cls,
                        child_procedure,
                        grand_child_procedure,
                        transformer=transformer,
                    )
                    self.collect_global_vars_decl(
                        cls.dec_global[grand_child_procedure],
                        cls.dec_global[child_procedure],
                    )
                else:
                    self.logger.info(
                        f"⏭️  Skipping '{grand_child_procedure}' - already isolated"
                    )
                    self.collect_global_vars_decl(
                        cls.dec_global[grand_child_procedure],
                        cls.dec_global[child_procedure],
                    )
        else:
            self.logger.info(
                f" No nested procedures found in '{child_procedure}' - proceeding to complete isolation"
            )

        cls.extract_intent(child_procedure, procedure_tree, calls)
        cls.clean_subroutine(child_procedure, procedure_tree)
        cls.extract_local_in_variables(child_procedure, procedure_tree)
        cls.extract_modified_variables(child_procedure, procedure_tree)

        assert os.path.exists(self.processor.benchmark_dir), (
            "benchmark directory does not exist!"
        )
        sub_dir = os.path.join(self.processor.benchmark_dir, child_procedure)
        os.makedirs(sub_dir, exist_ok=True)

        subroutine_dir = os.path.join(self.target_module_dir, child_procedure)
        os.makedirs(subroutine_dir)
        self.logger.info(f"📁 Created parent function directory: {subroutine_dir}")

        self.input_dict = cls.organize_code_components(
            child_procedure, cls.dec_global[child_procedure], openacc=self.openacc
        )

        subroutine_tree_cp = self.processor.parse_fortran_string(
            procedure_tree.tofortran()
        )
        self.processor.remove_io_statements(subroutine_tree_cp)
        self.working_subroutines[child_procedure] = subroutine_tree_cp

        sub_trees = []
        for sub_name in self.collect_all_subroutines(cls, child_procedure):
            sub_trees.append(self.working_subroutines[sub_name])

        self.processor.update_global_module(
            self.input_dict,
            subroutine_dir,
            child_procedure,
            procedure_tree,
            sub_trees,
            auto_diff=self.tapenade,
        )
        if transformer is not None:
            self.logger.info("Transforming global module to Python...")
            out_module = transformer.update_global_python(
                subroutine_key=child_procedure
            )
            transformer.transfer_to_pyfile(
                out_module, child_procedure, folder_name=self.target_module
            )

        arg_list = ", ".join([name for name in cls.dummy_arg_list[child_procedure]])

        if procedure_type == "subroutine":
            call_stmt_org = F23.Call_Stmt(f"CALL {child_procedure}({arg_list})")
        elif procedure_type == "function":
            call_stmt_org = F23.Assignment_Stmt(
                f"{cls.func_result[child_procedure]} = {child_procedure}({arg_list})"
            )
        else:
            raise ValueError(f"Unsupported procedure type: {procedure_type}")

        call_stmts = [call_stmt_org]

        dec_dummy = defaultdict(lambda: defaultdict(list))
        for decleration in cls.var_dummy[child_procedure]:
            decleration_name, decleration_list = (
                self.processor.break_allocatable_declaration(decleration)
            )
            dec_dummy[child_procedure][decleration_name] = decleration_list

        self.input_dict = cls.organize_code_components(
            child_procedure, dec_dummy[child_procedure], openacc=self.openacc
        )
        self.processor.update_main_program(
            input_dict=self.input_dict,
            call_stmts=call_stmts,
            var_modif=cls.var_modif_info[child_procedure],
            subroutine_dir=subroutine_dir,
            subroutine_name=child_procedure,
            procedure_tree=procedure_tree,
            openacc=self.openacc,
            dummy_add_decl=None,
            error_flag=None,
            acc_data_copyin=None,
        )
        if transformer is not None:
            self.logger.info("Transforming main program to Python...")
            main_tree = transformer.update_main_python(
                out_module=out_module, subroutine_key=child_procedure
            )
            transformer.transfer_to_pyfile(
                main_tree,
                child_procedure,
                folder_name=self.target_module,
                python_file_type="main",
            )

        error_status = self.processor.compile_and_run(
            os.getcwd(), subroutine_dir, auto_diff=self.tapenade
        )
        assert error_status == 0, (
            "Error: Compilation failed or main_program not generated."
        )

        if transformer is not None:
            transformer.run_python_scripts(os.getcwd(), subroutine_dir)

        write_module_tree = procedure_tree.get_root()
        write_module_name = (
            walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        )
        assert write_module_name in cls.module_path, (
            f"Module '{write_module_name}' not found in module_path. Available modules: {list(cls.module_path.keys())}"
        )
        self.processor.write_fortran_code_to_file(
            write_module_tree, cls.module_path[write_module_name]
        )

        write_module_tree = call_statements[0].get_root()
        write_module_name = (
            walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        )
        assert write_module_name in cls.module_path, (
            f"Module '{write_module_name}' not found in module_path."
        )

        self.processor.write_fortran_code_to_file(
            write_module_tree, cls.module_path[write_module_name]
        )
        self.isolated_subroutines.add(child_procedure)

    def collect_global_vars_decl(self, in_dict, out_dict):
        for child_key, child_value in in_dict.items():
            if child_key not in out_dict:
                out_dict[child_key] = child_value

    def process_subroutines(
        self, parent_subroutine="hydrol_main", target_subroutines=["hydrol_soil"]
    ):
        self.logger.start_task(
            "Procedure Isolation/Transformation/Automatic Differentiation",
            description="Isolation, Transformation, and Automatic Differentiation of procedures through FGPT",
            target_module=self.target_module,
        )

        cls = Extractor(self.module_dir_sp, self.module_tree_cp, self.logger)
        cls.module_path[self.target_module] = self.path_to_target
        cls.parsed_modules[self.target_module] = self.module_tree_cp
        cls.find_subroutines()

        if self.f2py:
            self.logger.info(
                "Initializing Transformer for Python conversion using f2np..."
            )
            transpy = Transformer(
                "benchmark",
                self,
                cls,
                None,
                config_path="template.yaml",
                logger=self.logger,
            )
        else:
            self.logger.info("Skipping Transformer initialization as f2np is disabled.")
            transpy = None

        for child_procedure in target_subroutines:
            self.logger.info(
                f"  Isolating target subroutine: '{child_procedure}' (called from '{parent_subroutine}')"
            )
            self.isolate_procedure(
                cls, parent_subroutine, child_procedure, transformer=transpy
            )

    def run(self, parent_subroutine="hydrol_main", target_subroutines=None):
        self.logger.info(
            f"Starting isolation for target subroutines: {target_subroutines} from parent subroutine: {parent_subroutine}"
        )
        self.create_target_directory()
        self.process_subroutines(
            parent_subroutine=parent_subroutine, target_subroutines=target_subroutines
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Fortran procedure isolator for FGPT")

    parser.add_argument(
        "--rest_of_path",
        type=str,
        required=True,
        help="Relative path to the directory containing the target Fortran module",
    )

    parser.add_argument(
        "--target_module",
        type=str,
        required=True,
        help="Name of the module to be isolated (without .f90)",
    )

    parser.add_argument(
        "--work",
        type=str,
        required=True,
        help="Working directory root (typically environment variable like $works)",
    )

    parser.add_argument(
        "--parent_subroutine",
        type=str,
        default="hydrol_main",
        help="Name of the parent subroutine containing target subroutines",
    )

    parser.add_argument(
        "--target_subroutines",
        type=str,
        nargs="+",
        default=[
            "hydrol_alma",
            "hydrol_vegupd",
            "hydrol_canop",
            "hydrol_flood",
            "hydrol_hydraulic_arch_tuzet_calc",
            "hydrol_soil",
            "explicitsnow_main",
        ],
        help="List of subroutines to isolate",
    )

    parser.add_argument(
        "--openacc",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable OpenACC support (True/False)",
    )

    parser.add_argument(
        "--f2py",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable f2py Python conversion (True/False)",
    )

    parser.add_argument(
        "--tapenade",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable Tapenade auto-differentiation (True/False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    isolator = Isolator(
        rest_of_path=args.rest_of_path,
        target_module=args.target_module,
        work=args.work,
        openacc=args.openacc,
        tapenade=args.tapenade,
        f2py=args.f2py,
    )

    isolator.run(
        parent_subroutine=args.parent_subroutine,
        target_subroutines=args.target_subroutines,
    )
