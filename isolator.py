import logging
import os
import shutil
from collections import deque, defaultdict
from processor import Processor
from extractor import Extractor
from modifier import Modifier
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from fparser.two import Fortran2008 as F28

class Isolator:
    """
    The Isolator class is designed to extract and prepare a Fortran procedure (e.g., subroutine or function)
    so that it can be compiled and executed independently of the rest of the original codebase.

    This process is particularly useful for:
    - Isolated testing or debugging of specific Fortran routines
    - Simplified transformation, such as source-to-source translation (e.g., to Python)
    - Generating standalone reproducible test cases from large code bases

    Functionality:
    -------------
    - Identifies and loads the specified Fortran module (`target_module`)
    - Parses the source file into an abstract syntax tree (AST) using `fparser`
    - Stores both original and parsed versions of the module
    - Sets up paths for temporary isolated compilation and execution
    - Prepares metadata about subroutine calls and dependencies

    Initialization Parameters:
    --------------------------
    rest_of_path : str
        The relative path to the directory containing the target Fortran module.
    target_module : str
        The name of the module to be isolated (without `.f90`).
    work : str
        The working directory root (typically an environment variable like `$works`).

    Attributes:
    -----------
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

    Notes:
    ------
    The isolation workflow is designed to be extendable. This class can later be
    used in conjunction with automatic input generation, test harness creation,
    or source-to-source translation routines.
    """
    def __init__(self, rest_of_path, target_module, work, openacc):
        self.module_global_file = "module_global.f90"
        self.main_program_file = "main.f90"
        self.rest_of_path = rest_of_path
        self.target_module = target_module
        self.scratch_dir = work
        self.module_dir_sp = os.path.join(self.scratch_dir, self.rest_of_path)
        self.path_to_target = os.path.join(self.module_dir_sp, f"{self.target_module}.f90")
        self.path_to_original = os.path.join(self.module_dir_sp, f"{self.target_module}_org.f90")
        if os.path.exists(self.path_to_original):
            self.module_file_sp =  self.path_to_original
        else:
            shutil.copy(self.path_to_target, self.path_to_original)
            self.module_file_sp =  self.path_to_target
        self.processor = Processor()
        self.module_tree_sp = self.processor.parse_fortran_file(self.module_file_sp)
        self.module_string = self.module_tree_sp.tostr()
        self.module_tree_cp = self.processor.parse_fortran_string(self.module_string)
        self.target_module_dir = os.path.join(os.getcwd(), self.target_module.split('.')[0])
        self.openacc = openacc
        self.working_subroutines = defaultdict()
        self.isolated_subroutines = set()
    
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


    def isolate_procedure(self, cls, parent_procedure, child_procedure):
        
        call_statements =  cls.call_within_sub[parent_procedure][child_procedure] 
        
        for i, call_stmt in enumerate(call_statements):
            self.processor.logger.info(f"  Call site {i+1}: {call_stmt.tostr()}")

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
                name for name in cls.var_global[child_procedure]
                if name.tostr() not in calls
                ]

        if cls.var_global[child_procedure]:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, cls.var_global[child_procedure], child_procedure)
        
        if cls.var_dummy[child_procedure]:
            cls.process_declaration_variables(cls.var_dummy[child_procedure], child_procedure)
        
        if cls.dec_global[child_procedure]:
            for key in cls.dec_global[child_procedure].keys():
                cls.process_declaration_variables(cls.dec_global[child_procedure][key], child_procedure)

        scalar_names = {var.string for var in cls.scalar_variables[child_procedure]}
        global_names = {var.string for var in cls.var_global[child_procedure]}
        shape_to_search = [
                var for var in cls.shapes_variables[child_procedure]
                if var.string not in scalar_names and var.string not in global_names
                ]

        shape_to_read = [
                var for var in cls.shapes_variables[child_procedure]
                if var.string in scalar_names and var.string not in global_names
                ]
        print(shape_to_read, '---------------')
        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, shape_to_search, child_procedure)
            cls.var_global[child_procedure].extend(shape_to_search)

        cls.extract_all_array_info(cls.dec_global[child_procedure], cls.var_dummy[child_procedure], child_procedure)

        nested_procedures = cls.call_within_sub[child_procedure].keys()
        
        if nested_procedures:
            self.processor.logger.info(f"Found {len(nested_procedures)} nested procedure(s) in '{child_procedure}':")
            for i, grand_child_procedure in enumerate(nested_procedures, 1):
                self.processor.logger.info(f"   {i}. {grand_child_procedure}")
                if grand_child_procedure not in self.isolated_subroutines:
                    self.processor.logger.info(f"🔄 Call recursively for '{grand_child_procedure}' from parent '{child_procedure}'")
                    self.isolate_procedure(cls, child_procedure, grand_child_procedure)
                    self.collect_global_vars_decl(cls.dec_global[grand_child_procedure], cls.dec_global[child_procedure])
                else:
                    self.processor.logger.info(f"⏭️  Skipping '{grand_child_procedure}' - already isolated")
                    self.collect_global_vars_decl(cls.dec_global[grand_child_procedure], cls.dec_global[child_procedure])
        else:
            self.processor.logger.info(f" No nested procedures found in '{child_procedure}' - proceeding to complete isolation")

        cls.extract_intent(child_procedure, procedure_tree, calls)
        cls.clean_subroutine(child_procedure, procedure_tree)
        cls.extract_local_in_variables(child_procedure, procedure_tree)
        cls.extract_modified_variables(child_procedure, procedure_tree)
        
        assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor.benchmark_dir, child_procedure)
        os.makedirs(sub_dir, exist_ok=True)

        subroutine_dir = os.path.join(self.target_module_dir, child_procedure)
        os.makedirs(subroutine_dir)
        self.processor.logger.info(f"📁 Created parent function directory: {subroutine_dir}")
        
        self.processor.add_declarations(
                cls.dec_global[child_procedure], 
                cls.var_modif_info[child_procedure],
                openacc=self.openacc
                )

        subroutine_tree_cp = self.processor.parse_fortran_string(procedure_tree.tofortran())
        self.processor.remove_io_statements(subroutine_tree_cp)
        self.working_subroutines[child_procedure] = subroutine_tree_cp

        sub_trees = []
        for sub_name in self.collect_all_subroutines(cls, child_procedure):
            sub_trees.append(self.working_subroutines[sub_name])

        #file_path = os.path.join(subroutine_dir, self.module_global_file)
        self.processor.update_global_module(
                cls.dec_global[child_procedure], 
                subroutine_dir,
                child_procedure,
                procedure_tree,
                sub_trees
                )

        arg_list = ', '.join([name for name in cls.dummy_arg_list[child_procedure]])
        
        if procedure_type == "subroutine":
            call_stmt_org = F23.Call_Stmt(f"CALL {child_procedure}({arg_list})")
        elif procedure_type == "function":
            call_stmt_org = F23.Assignment_Stmt(f"{cls.func_result[child_procedure]} = {child_procedure}({arg_list})")
        else:
            raise ValueError(f"Unsupported procedure type: {procedure_type}")
        
        #file_path = os.path.join(subroutine_dir, self.main_program_file)
        call_stmts = [call_stmt_org]

        self.processor.update_main_program(
                custom_dec_inout=cls.var_dummy[child_procedure],
                call_stmts=call_stmts,
                var_modif=cls.var_modif_info[child_procedure],
                subroutine_dir=subroutine_dir,
                subroutine_name=child_procedure,
                dummy_args=cls.dummy_arg_list[child_procedure],
                call_site=call_statements[0],
                openacc=self.openacc,
                dummy_add_decl=None,
                error_flag=None,
                acc_data_copyin=None
                )

        error_status = self.processor.compile_and_run(os.getcwd(), subroutine_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."

        write_module_tree = procedure_tree.get_root()
        write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        assert write_module_name in cls.module_path, \
                f"Module '{write_module_name}' not found in module_path. Available modules: {list(cls.module_path.keys())}"
        self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])

        write_module_tree = call_statements[0].get_root()
        write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        assert write_module_name in cls.module_path, \
            f"Module '{write_module_name}' not found in module_path."

        self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])
        self.isolated_subroutines.add(child_procedure)
        
    def collect_global_vars_decl(self, in_dict, out_dict):
        for child_key, child_value in in_dict.items():
                if child_key not in out_dict:
                    out_dict[child_key] = child_value

    def process_subroutines(self):
        cls = Extractor(self.module_dir_sp, self.module_tree_cp)
        cls.find_subroutines()

        # Process each parent subroutine and its children
        grand_parent_procedure = "hydrol_main"
        parent_procedure = "albedo_surface_main"
        children = ['multilevel_matrix'] #cls.call_within_sub[parent_procedure]
        self.processor.logger.info(f"Processing parent subroutine: '{parent_procedure}' with {len(children)} children")
        # Process each child subroutine of this parent
        for child_procedure in children:
            self.processor.logger.info(f"  Isolating child subroutine: '{child_procedure}' (called from '{parent_procedure}')")
            try:
               # Pass both parent and child to access the specific call sites
                self.isolate_procedure(cls, parent_procedure, child_procedure)
                self.processor.logger.info(f"  Successfully isolated child subroutine: '{child_procedure}'")
            except Exception as e:
                self.processor.logger.error(f"  Failed to isolate child schild_procedureubroutine '{child_procedure}': {e}")
                raise
        #for parent_procedure in ['hydrol_alma', 'hydrol_vegupd','hydrol_canop','hydrol_flood', 'hydrol_hydraulic_arch_tuzet_calc', 'hydrol_soil', 'explicitsnow_main']:
        #    self.isolate_procedure(cls, grand_parent_procedure, parent_procedure)
        
    def run(self):
        self.create_target_directory()
        self.process_subroutines()

if __name__ == "__main__":
    rest_of_path = "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    target_modules = ["hydrol", "explicitsnow", "albedo_surface"]
    target_module =  target_modules[2]
    work = os.getenv("works")
    openacc = False
    isolator = Isolator(rest_of_path, target_module, work, openacc)
    isolator.run()

