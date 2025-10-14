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
    child_subroutine_call : dict
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
        self.child_subroutine_call = defaultdict(list)
        self.child_error_flag = defaultdict(lambda: defaultdict(dict))
        self.openacc = openacc
        self.working_subroutines = defaultdict()
        self.isolated_subroutines = set()
    
    def create_target_directory(self):
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        os.makedirs(self.target_module_dir)

    def collect_all_subroutines(self, cls, subroutine_key):

        collected = []
        queue = deque([subroutine_key])

        while queue:
            current_key = queue.popleft()
            if current_key not in collected:
                collected.append(current_key)
                if current_key in cls.call_within_sub:
                    for child_subroutine in cls.call_within_sub[current_key].keys():
                        if child_subroutine not in collected:
                            queue.append(child_subroutine)
        return collected

    def isolate_procedure_function(self, cls, parent_procedure, child_function):

        call_statements = cls.call_within_sub[parent_procedure][child_function]
        # Log all call sites for this parent-child relationship
        for i, call_stmt in enumerate(call_statements):
            self.processor.logger.info(f"  Call site {i+1}: {call_stmt.tostr()}")

        # Get the parent function tree
        function_tree = cls.subroutines[child_function]
        cls.extract_function_dummy_args(function_tree)
        cls.extract_intent(child_function, function_tree)
        cls.clean_subroutine(child_function, function_tree)

        # Extract variable information for parent function
        cls.find_variables(function_tree, child_function, parent_procedure)
        cls.extract_names(child_function)

        if cls.var_global[child_function]:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, cls.var_global[child_function], child_function)
        if cls.var_dummy[child_function]:
            cls.process_declaration_variables(cls.var_dummy[child_function], child_function)
        if cls.dec_global[child_function]:
            for key in cls.dec_global[child_function].keys():
                cls.process_declaration_variables(cls.dec_global[child_function][key], child_function)

        # Process shape variables
        scalar_names = {var.string for var in cls.scalar_variables[child_function]}
        global_names = {var.string for var in cls.var_global[child_function]}
        shape_to_search = [
                var for var in cls.shapes_variables[child_function]
                if var.string not in scalar_names and var.string not in global_names
                ]
        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, shape_to_search, child_function)
            cls.var_global[child_function].extend(shape_to_search)

        cls.extract_array_info(cls.dec_global[child_function], cls.var_dummy[child_function], child_function)

        # Process nested function calls within the parent function
        for key, values in cls.dec_global[child_function].items():
            if walk(values, F23.Function_Subprogram):
                self.processor.logger.info(
                        f"Calling Function_Subprogram {key} in parent Function_Subprogram {child_function}."
                        )
                self.processor.logger.info(f"Trying to isolate nested function .... {key}")

                assert isinstance(values[1], F23.Function_Subprogram), (
                        f"Expected type 'F23.Function_Subprogram', but got '{type(values[0]).__name__}' instead."
                        )
                assert isinstance(values[0], F23.Name), (
                        f"Expected type 'F23.Name', but got '{type(values[0]).__name__}' instead."
                        )

                nested_function = values[0].tostr()

                for i, call_site in enumerate(cls.call_within_sub[child_function][nested_function], 1):
                    self.processor.logger.info(f"Call site #{i}: {call_site.tostr()}")

                if nested_function in self.isolated_subroutines:
                    self.processor.logger.info(
                            f"Function '{nested_function}' already successfully isolated, skipping..."
                            )
                else:
                    self.isolate_procedure_function(cls, child_function, nested_function)

        child_functions = cls.call_within_sub[child_function].keys()
        if child_functions:
            self.processor.logger.info(f"Processing {len(child_functions)} function calls within parent function {child_function}")
            for child_func in child_functions:
                self.collect_global_vars_decl(cls.dec_global[child_func], cls.dec_global[child_function])

        # Create benchmark directory and generate code
        assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor.benchmark_dir, child_function)
        os.makedirs(sub_dir, exist_ok=True)

        function_dir = os.path.join(self.target_module_dir, child_function)
        os.makedirs(function_dir)
        self.processor.logger.info(f"Created parent function directory: {function_dir}")

        self.processor.add_declarations(
                cls.dec_global[child_function],
                cls.var_modif_info[child_function],
                openacc=self.openacc
                )

        global_file_path = os.path.join(function_dir, self.module_global_file)

        # Clean the function tree and remove I/O statements
        function_tree_cp = self.processor.parse_fortran_string(function_tree.tofortran())
        self.processor.remove_io_statements(function_tree_cp)
        self.working_subroutines[child_function] = function_tree_cp

        # Collect all required subroutines/functions
        sub_trees = []
        for sub_name in self.collect_all_subroutines(cls, child_function):
            sub_trees.append(self.working_subroutines[sub_name])

        # Update global module
        self.processor.update_global_module(
                cls.dec_global[child_function],
                global_file_path,
                child_function,
                function_tree
                )

        # Update main program
        main_file_path = os.path.join(function_dir, self.main_program_file)
        arg_list = ', '.join([name for name in cls.dummy_arg_list[child_function]])
        call_stmt_org = F23.Assignment_Stmt(f"{cls.func_result[child_function]} = {child_function}({arg_list})")

        self.processor.update_main_program(
                custom_dec_inout=cls.var_dummy[child_function],
                custom_subroutine_trees=sub_trees,
                call_stmts=[call_stmt_org],
                var_modif=cls.var_modif_info[child_function],
                file_path=main_file_path,
                subroutine_name=child_function,
                dummy_args=cls.dummy_arg_list[child_function],
                call_site=call_statements[0],
                childs_subroutine_tree=None,
                openacc=self.openacc,
                dummy_add_decl=None,
                error_flag=None,
                acc_data_copyin=None
                )

        # Compile and run
        error_status = self.processor.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."

        # Save modified module files
        write_module_tree = function_tree.get_root()
        write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        assert write_module_name in cls.module_path, \
                f"Module '{write_module_name}' not found in module_path. Available modules: {list(cls.module_path.keys())}"
        self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])

        write_module_tree = call_statements[0].get_root()
        write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        assert write_module_name in cls.module_path, \
            f"Module '{write_module_name}' not found in module_path."
        self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])

        self.isolated_subroutines.add(child_function)

    def isolate_child_subroutine(self, cls, parent_subroutine, child_subroutine, local_var_parent=None):
        
        dummy_as_local = set()
        function_tree = None
        sub_trees = []
        call_statements =  cls.call_within_sub[parent_subroutine][child_subroutine] 
        # Log all call sites for this parent-child relationship
        for i, call_stmt in enumerate(call_statements):
            self.processor.logger.info(f"  Call site {i+1}: {call_stmt.tostr()}")

        if self.openacc:
            if local_var_parent is not None:
                for actual_arg_list in cls.actual_arg_spec_list[child_subroutine]:
                    for iarg, arg in enumerate(actual_arg_list):
                        if arg in local_var_parent:
                            dummy_arg = cls.dummy_arg_list[child_subroutine][iarg]
                            dummy_as_local.add(dummy_arg)
                            self.processor.logger.warning(
                                f"passing local variable '{arg}' as argument '{dummy_arg}' into procedure '{child_subroutine}'!"
                            )

        self.processor.logger.info(f"Trying to isolate a child subroutine .... {child_subroutine}")

        # Extract and process the child subroutine
        subroutine_tree = cls.subroutines[child_subroutine]
        cls.extract_intent(child_subroutine, subroutine_tree)
        cls.clean_subroutine(child_subroutine, subroutine_tree)

        # Extract variable information
        cls.find_variables(subroutine_tree, child_subroutine)
        cls.extract_names(child_subroutine)
        cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, cls.var_global[child_subroutine], child_subroutine)

        # Process declaration variables
        cls.process_declaration_variables(cls.var_dummy[child_subroutine], child_subroutine)
        for key in cls.dec_global[child_subroutine].keys():
            cls.process_declaration_variables(cls.dec_global[child_subroutine][key], child_subroutine)

        # Process shape variables excluding scalars and globals
        #shape_to_search = cls.shapes_variables[child_subroutine] - cls.scalar_variables[child_subroutine] - cls.var_global[child_subroutine]
        scalar_names = {var.string for var in cls.scalar_variables[child_subroutine]}
        global_names = {var.string for var in cls.var_global[child_subroutine]}
        # Filter shapes excluding scalars and globals (all F23.Name nodes)
        shape_to_search = [
                var for var in cls.shapes_variables[child_subroutine]
                if var.string not in scalar_names and var.string not in global_names
                ]
        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, shape_to_search, child_subroutine)
            cls.var_global[child_subroutine].extend(shape_to_search)
            #cls.var_global[child_subroutine].update(shape_to_search)

        # Extract array information
        cls.extract_array_info(cls.dec_global[child_subroutine], cls.var_dummy[child_subroutine], child_subroutine)

        # Process function subprograms within the child subroutine
        for key, values in cls.dec_global[child_subroutine].items():
             if walk(values, F23.Function_Subprogram):
                 self.processor.logger.info(
                         f"Calling Function_Subprogram {key} in Subroutine_Subprogram {child_subroutine}."
                         )
                 self.processor.logger.info(
                         f"Trying to isolate a child function .... {key}"
                         )
                 assert isinstance(values[1], F23.Function_Subprogram), (
                         f"Expected type 'F23.Function_Subprogram', but got '{type(values[0]).__name__}' instead."
                         )
                 assert isinstance(values[0], F23.Name), (
                         f"Expected type 'F23.Name', but got '{type(values[0]).__name__}' instead."
                         )
                 child_function = values[0].tostr()

                 for i, call_site in enumerate(cls.call_within_sub[child_subroutine][child_function], 1):
                     self.processor.logger.info(f"Call site #{i}: {call_site.tostr()}")

                 if child_function in self.isolated_subroutines:
                     self.processor.logger.info(
                             f"Function '{child_function}' already successfully isolated, skipping..."
                             )
                     function_tree = cls.subroutines[child_function]
                 else:
                     self.isolate_procedure_function(cls, child_subroutine, child_function)
                     function_tree = cls.subroutines[child_function]
        
        child_functions = cls.call_within_sub[child_subroutine].keys()
        if child_functions:
            self.processor.logger.info(f"Processing {len(child_functions)} function calls within subroutine {child_subroutine}")
            for child_function in child_functions:
                self.collect_global_vars_decl(cls.dec_global[child_function], cls.dec_global[child_subroutine])


        if dummy_as_local:
            dummy_as_local = dummy_as_local.intersection(cls.all_array_info[child_subroutine].keys())
        if dummy_as_local:
            cls.var_local_names[child_subroutine].update(dummy_as_local)


        modified_subroutine_tree = None
        call_stmt_vec = None
        dummy_add_decl = None
        error_flag = None
        acc_enter_data_copyin = None

        subroutine_tree_cp = self.processor.parse_fortran_string(subroutine_tree.tofortran())
        self.processor.remove_io_statements(subroutine_tree_cp)
        self.working_subroutines[child_subroutine] = subroutine_tree_cp


        if self.openacc:
            cls.extract_loop_vect(child_subroutine, subroutine_tree)
            modifier = Modifier(
                cls.loop_vect[child_subroutine],
                cls.all_array_info[child_subroutine],
                cls.loop_dict,
                cls.var_declared[child_subroutine],
                cls.imp_shape[child_subroutine],
                cls.allowed_external_subroutines,
                cls.var_local_names[child_subroutine]
                )

            working_tree = modifier.replace_gpu_unsupported(working_tree)
            modified_block = modifier.merge_vector_loop(working_tree)

            assert modifier.do_index == 0 and modifier.enddo_index == 0, (
                f"Error: do_index and enddo_index are not reset properly. "
                f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
                )

            modified_subroutine_tree = modifier.add_vector_loop(modified_block)
            call_stmt_vec = modifier.subroutine_call_act_vec
            dummy_add_decl = modifier.dummy_add_decl
            error_flag = modifier.error_flag
            acc_enter_data_copyin = modifier.acc_enter_data_copyin

        if not dummy_as_local:

            assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
            sub_dir = os.path.join(self.processor.benchmark_dir, child_subroutine)
            os.makedirs(sub_dir, exist_ok=True)
            self.processor.logger.info(f"{child_subroutine} directory created inside benchmark: {sub_dir}")

            subroutine_dir = os.path.join(self.target_module_dir, child_subroutine)
            os.makedirs(subroutine_dir)
        
            self.processor.add_declarations(
                    cls.dec_global[child_subroutine], 
                    cls.var_modif_info[child_subroutine],
                    openacc=self.openacc
                    )

            for sub_name in self.collect_all_subroutines(cls, child_subroutine):
                sub_trees.append(self.working_subroutines[sub_name])

            file_path = os.path.join(subroutine_dir, self.module_global_file)
            self.processor.update_global_module(
                    cls.dec_global[child_subroutine], 
                    file_path,
                    child_subroutine,
                    subroutine_tree
                    #call_site=call_statements[0]
                    )
            file_path = os.path.join(subroutine_dir, self.main_program_file)

            arg_list = ', '.join([name for name in cls.dummy_arg_list[child_subroutine]])
            call_stmt_org =  F23.Call_Stmt(f"CALL {child_subroutine}({arg_list})")
            call_stmts = [call_stmt_org]

            if self.openacc:
                sub_trees.append(modified_subroutine_tree)
                call_stmts.append(call_stmt_vec)

            self.processor.update_main_program(
                    custom_dec_inout=cls.var_dummy[child_subroutine],
                    custom_subroutine_trees=sub_trees,
                    call_stmts=call_stmts,
                    var_modif=cls.var_modif_info[child_subroutine],
                    file_path=file_path,
                    subroutine_name=child_subroutine,
                    dummy_args=cls.dummy_arg_list[child_subroutine],
                    call_site=call_statements[0],
                    childs_subroutine_tree=None,
                    openacc=self.openacc,
                    dummy_add_decl=dummy_add_decl,
                    error_flag=error_flag,
                    acc_data_copyin=acc_enter_data_copyin
                    )

            error_status = self.processor.compile_and_run(os.getcwd(), self.target_module_dir)
            assert error_status == 0, "Error: Compilation failed or main_program not generated."

            write_module_tree = subroutine_tree.get_root()
            write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
            assert write_module_name in cls.module_path, \
                    f"Module '{write_module_name}' not found in module_path. Available modules: {list(cls.module_path.keys())}"
            self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])

            write_module_tree = call_statements[0].get_root()
            write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
            assert write_module_name in cls.module_path, \
                f"Module '{write_module_name}' not found in module_path."

            self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])
            self.isolated_subroutines.add(child_subroutine)

        return modified_subroutine_tree, error_flag, function_tree
        

    def isolate_parent_subroutine(self, cls, grand_parent_subroutine, parent_subroutine, subroutines_parent=None):
        assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor.benchmark_dir, parent_subroutine)
        os.makedirs(sub_dir, exist_ok=True)
        self.processor.logger.info(f"{parent_subroutine} directory created inside benchmark: {sub_dir}")

        subroutine_tree = cls.subroutines[parent_subroutine]
        cls.find_variables(subroutine_tree, parent_subroutine)
        cls.extract_names(parent_subroutine)
        queue = deque(cls.call_within_sub[parent_subroutine])

        while queue:

            child_subroutine_key = queue.popleft()
            child_subroutine_tree = cls.subroutines[child_subroutine_key]
            self.child_subroutine_call[parent_subroutine].append(child_subroutine_tree)

            if subroutines_parent is not None:
                for subroutine_key_parent in subroutines_parent:
                    self.child_subroutine_call[subroutine_key_parent].append(child_subroutine_tree)

            if child_subroutine_key not in cls.call_within_sub:
                mod_child_subroutine_tree, error_flag, child_function_tree = self.isolate_child_subroutine(
                        cls, 
                        parent_subroutine, 
                        child_subroutine_key, 
                        cls.var_local_names[parent_subroutine]
                        )
                
                if mod_child_subroutine_tree is not None:
                    self.child_subroutine_call[parent_subroutine].append(mod_child_subroutine_tree)
                
                if error_flag is not None:
                    self.child_error_flag[parent_subroutine][child_subroutine_key] = error_flag

                if child_function_tree is not None:
                    self.child_subroutine_call[parent_subroutine].append(child_function_tree)
                
                self.collect_global_vars_decl(cls.dec_global[child_subroutine_key], cls.dec_global[parent_subroutine])
            else:
                self.parent_subroutine_call.add(parent_subroutine)
                self.isolate_parent_subroutine(cls, parent_subroutine, child_subroutine_key, self.parent_subroutine_call)


        self.processor.logger.info(f"Now, trying to isolate a parent subroutine .... {parent_subroutine}")
        sub_trees = []
    

        call_statements =  cls.call_within_sub[grand_parent_subroutine][parent_subroutine]
        for i, call_stmt in enumerate(call_statements):
            self.processor.logger.info(f"  Call site {i+1}: {call_stmt.tostr()}")

        cls.extract_intent(parent_subroutine, subroutine_tree, cls.call_within_sub[parent_subroutine])
        cls.clean_subroutine(parent_subroutine, subroutine_tree)
        code_string = subroutine_tree.tofortran()
        working_tree = self.processor.parse_fortran_string(code_string)
        #assert working_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'
        cls.find_variables(subroutine_tree, parent_subroutine)
        cls.extract_names(parent_subroutine)
        
        calls = cls.call_within_sub[parent_subroutine]
        declared = set(cls.dec_global[parent_subroutine].keys())
        cls.var_global[parent_subroutine] = [
                name for name in cls.var_global[parent_subroutine]
                if name.string not in calls and name.string not in declared
                ]

        cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, cls.var_global[parent_subroutine], parent_subroutine)

        cls.process_declaration_variables(cls.var_dummy[parent_subroutine], parent_subroutine)
        for key in cls.dec_global[parent_subroutine].keys():
            cls.process_declaration_variables(cls.dec_global[parent_subroutine][key], parent_subroutine)

        scalar_names = {var.string for var in cls.scalar_variables[parent_subroutine]}
        global_names = {var.string for var in cls.var_global[parent_subroutine]}

        shape_to_search = [
                var for var in cls.shapes_variables[parent_subroutine]
                if var.string not in scalar_names and var.string not in global_names
                ]

        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_cp, shape_to_search, parent_subroutine)
            cls.var_global[parent_subroutine].extend(shape_to_search)

        if subroutines_parent is not None:
            for subroutine_key_parent in subroutines_parent:
                self.collect_global_vars_decl(cls.dec_global[parent_subroutine], cls.dec_global[subroutine_key_parent])

        cls.extract_array_info(cls.dec_global[parent_subroutine], cls.var_dummy[parent_subroutine], parent_subroutine)

        # Process function subprograms within the child subroutine
        for key, values in cls.dec_global[parent_subroutine].items():
             if walk(values, F23.Function_Subprogram):
                 self.processor.logger.info(
                         f"Calling Function_Subprogram {key} in PARENT Subroutine_Subprogram {parent_subroutine}."
                         )
                 self.processor.logger.info(
                         f"Trying to isolate a child function .... {key}"
                         )
                 assert isinstance(values[1], F23.Function_Subprogram), (
                         f"Expected type 'F23.Function_Subprogram', but got '{type(values[0]).__name__}' instead."
                         )
                 assert isinstance(values[0], F23.Name), (
                         f"Expected type 'F23.Name', but got '{type(values[0]).__name__}' instead."
                         )
                 child_function = values[0].tostr()

                 for i, call_site in enumerate(cls.call_within_sub[parent_subroutine][child_function], 1):
                     self.processor.logger.info(f"Call site #{i}: {call_site.tostr()}")

                 if child_function in self.isolated_subroutines:
                     self.processor.logger.info(
                             f"Function '{child_function}' already successfully isolated, skipping..."
                             )
                     function_tree = cls.subroutines[child_function]
                 else:
                     self.isolate_procedure_function(cls, parent_subroutine, child_function)
                     function_tree = cls.subroutines[child_function]

        child_functions = cls.call_within_sub[parent_subroutine].keys()
        if child_functions:
            self.processor.logger.info(f"Processing {len(child_functions)} function calls within subroutine {parent_subroutine}")
            for child_function in child_functions:
                self.collect_global_vars_decl(cls.dec_global[child_function], cls.dec_global[parent_subroutine])

        modified_subroutine_tree = None
        call_stmt_vec = None
        dummy_add_decl = None
        error_flag = None
        acc_enter_data_copyin = None

        subroutine_tree_cp = self.processor.parse_fortran_string(subroutine_tree.tofortran())
        self.processor.remove_io_statements(subroutine_tree_cp)
        self.working_subroutines[parent_subroutine] = subroutine_tree_cp

        if self.openacc:
            cls.extract_loop_vect(parent_subroutine, subroutine_tree)
            modifier = Modifier(
                    cls.loop_vect[parent_subroutine],
                    cls.all_array_info[parent_subroutine], 
                    cls.loop_dict, 
                    cls.var_declared[parent_subroutine], 
                    cls.imp_shape[parent_subroutine],
                    cls.allowed_external_subroutines, 
                    cls.var_local_names[parent_subroutine],
                    cls.call_within_sub[parent_subroutine],
                    self.child_error_flag[parent_subroutine]
                    )
            working_tree = modifier.replace_gpu_unsupported(working_tree)
            modified_block = modifier.merge_vector_loop(working_tree)

            assert modifier.do_index == 0 and modifier.enddo_index == 0, (
                    f"Error: do_index and enddo_index are not reset properly. "
                    f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
                    )

            modified_subroutine_tree = modifier.add_vector_loop(modified_block)
            call_stmt_vec = modifier.subroutine_call_act_vec
            dummy_add_decl = modifier.dummy_add_decl
            error_flag = modifier.error_flag
            acc_enter_data_copyin = modifier.acc_enter_data_copyin

        self.processor.add_declarations(
                cls.dec_global[parent_subroutine],
                cls.var_modif_info[parent_subroutine],
                openacc=self.openacc
                )

        for sub_name in self.collect_all_subroutines(cls, parent_subroutine):
            sub_trees.append(self.working_subroutines[sub_name])
       
        subroutine_dir = os.path.join(self.target_module_dir, parent_subroutine)
        os.makedirs(subroutine_dir)

        file_path = os.path.join(subroutine_dir, self.module_global_file)
        self.processor.update_global_module(
                cls.dec_global[parent_subroutine], 
                file_path, 
                parent_subroutine, 
                subroutine_tree
                #call_site=call_statements[0]
                )
        file_path = os.path.join(subroutine_dir, self.main_program_file)


        arg_list = ', '.join([name for name in cls.dummy_arg_list[parent_subroutine]])
        call_stmt_org =  F23.Call_Stmt(f"CALL {parent_subroutine}({arg_list})")
        call_stmts = [call_stmt_org]

        if self.openacc:
            sub_trees.append(modified_subroutine_tree)
            call_stmts.append(call_stmt_vec)

        
        self.processor.update_main_program(
                custom_dec_inout=cls.var_dummy[parent_subroutine],
                custom_subroutine_trees=sub_trees,
                call_stmts=call_stmts,
                var_modif=cls.var_modif_info[parent_subroutine],
                file_path=file_path,
                subroutine_name=parent_subroutine,
                dummy_args=cls.dummy_arg_list[parent_subroutine],
                call_site=call_statements[0],
                childs_subroutine_tree=None,#self.child_subroutine_call[parent_subroutine],
                openacc=self.openacc,
                dummy_add_decl=dummy_add_decl,
                error_flag=error_flag,
                acc_data_copyin=acc_enter_data_copyin
                )

        error_status = self.processor.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."

        write_module_tree = subroutine_tree.get_root()
        write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        assert write_module_name in cls.module_path, \
                f"Module '{write_module_name}' not found in module_path. Available modules: {list(cls.module_path.keys())}"
        self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])

        write_module_tree = call_statements[0].get_root()
        write_module_name = walk(write_module_tree, F23.Module_Stmt)[0].children[1].tostr()
        assert write_module_name in cls.module_path, \
                f"Module '{write_module_name}' not found in module_path."
        self.processor.write_fortran_code_to_file(write_module_tree, cls.module_path[write_module_name])
        self.isolated_subroutines.add(parent_subroutine)
        
    def collect_global_vars_decl(self, in_dict, out_dict):
        for child_key, child_value in in_dict.items():
                if child_key not in out_dict:
                    out_dict[child_key] = child_value

    def process_subroutines(self):
        cls = Extractor(self.module_dir_sp, self.module_tree_cp)
        cls.find_subroutines()
        cls.extract_loop_indices()

        # Log basic information about found subroutines
        self.processor.logger.info(f"Found {len(cls.subroutine_keys_all)} total subroutines: {cls.subroutine_keys_all}")
        self.processor.logger.info(f"Found {len(cls.subroutine_keys_ncl)} subroutines with no internal calls to any subroutine: {cls.subroutine_keys_ncl}")
        self.processor.logger.info(f"Found {len(cls.call_within_sub)} parent subroutines that call other subroutines")

        # Log detailed information about each parent subroutine and its children
        for parent_subroutine in cls.call_within_sub.keys():
            children = cls.call_within_sub[parent_subroutine]
            self.processor.logger.info(f"Parent subroutine '{parent_subroutine}' calls {len(children)} child subroutines:")
            for child_subroutine, call_statements in children.items():
                self.processor.logger.info(f"  - '{child_subroutine}': {len(call_statements)} call site(s)")
                for i, call_stmt in enumerate(call_statements):
                    self.processor.logger.info(f"    Call site {i+1}: {call_stmt.tostr()}")

        # Process each parent subroutine and its children
        grand_parent_subroutine = "hydrol_main"
        parent_subroutine = "explicitsnow_main"
        #children = ['explicitsnow_melt_refrz'] #cls.call_within_sub[parent_subroutine]
        #self.processor.logger.info(f"Processing parent subroutine: '{parent_subroutine}' with {len(children)} children")
        # Process each child subroutine of this parent
        #for child_subroutine in children:
        #    self.processor.logger.info(f"  Isolating child subroutine: '{child_subroutine}' (called from '{parent_subroutine}')")
        #    try:
               # Pass both parent and child to access the specific call sites
        #        self.isolate_child_subroutine(cls, parent_subroutine, child_subroutine)
        #        self.processor.logger.info(f"  Successfully isolated child subroutine: '{child_subroutine}'")
        #    except Exception as e:
        #        self.processor.logger.error(f"  Failed to isolate child schild_subroutineubroutine '{child_subroutine}': {e}")
        #        raise

        for parent_subroutine in ['hydrol_vegupd', 'hydrol_hydraulic_arch_tuzet_calc',  'hydrol_soil', 'explicitsnow_main']:
            self.parent_subroutine_call = set()
            self.isolate_parent_subroutine(cls, grand_parent_subroutine, parent_subroutine)
        
    def run(self):
        self.create_target_directory()
        self.process_subroutines()

if __name__ == "__main__":
    rest_of_path = "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    target_modules = ["hydrol", "explicitsnow"]
    target_module =  target_modules[0]
    work = os.getenv("works")
    openacc = False
    isolator = Isolator(rest_of_path, target_module, work, openacc)
    isolator.run()

