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
    
    def create_target_directory(self):
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        os.makedirs(self.target_module_dir)

    def isolate_child_function(self, cls, function_tree, function_key, parent_subroutine_key):
        """
        Isolate and process a child function found within a parent subroutine.
        
        This method extracts a function from within a subroutine, processes its declarations,
        variables, and dependencies, then generates standalone code for the function that
        can be compiled and executed independently for testing or benchmarking purposes.
        
        Parameters
        ----------
        cls : object
            Class instance containing function metadata and processing methods
        function_values : tuple
            Tuple containing (function_name, function_subprogram)
        parent_subroutine_key : str
            Name of the parent subroutine containing this function
            
        Raises
        ------
        AssertionError
            If function_values does not contain expected types or compilation fails
        NotImplementedError
            If nested functions are found (not supported)
        """ 

        cls.extract_function_dummy_args(function_tree)
        cls.extract_intent(function_key, function_tree)
        cls.clean_subroutine(function_key, function_tree)
        cls.find_function_actual_args(function_key)
        cls.call_within_sub[parent_subroutine_key].add(function_key)

        cls.find_variables(function_tree, function_key, parent_subroutine_key)
        cls.extract_names(function_key)

        if cls.var_global[function_key]:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, cls.var_global[function_key], function_key)
        if cls.var_dummy[function_key]:
            cls.process_declaration_variables(cls.var_dummy[function_key], function_key)
        if cls.dec_global[function_key]:
            for key in cls.dec_global[function_key].keys():
                cls.process_declaration_variables(cls.dec_global[function_key][key], function_key)
        
        shape_to_search = cls.shapes_variables[function_key] - cls.scalar_variables[function_key] - cls.var_global[function_key]
        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, shape_to_search, function_key)
            cls.var_global[function_key].update(shape_to_search)

        cls.extract_array_info(cls.dec_global[function_key], cls.var_dummy[function_key], function_key)

        for key, values in cls.dec_global[function_key].items():
            if walk(values, F23.Function_Subprogram):
                self.processor.logger.error(f"Calling Function_Subprogram {key} in Function_Subprogram {function_key}. Not yet implemented")
                raise NotImplementedError("Nested functions are not supported")

        cls.extract_loop_vect(function_key, function_tree)

        assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor.benchmark_dir, function_key)
        os.makedirs(sub_dir, exist_ok=True)

        function_dir = os.path.join(self.target_module_dir, function_key)
        os.makedirs(function_dir)
        self.processor.logger.info(f"Created function directory: {function_dir}")
        
        self.processor.add_declarations(
                cls.dec_global[function_key], 
                cls.var_modif_info[function_key],
                openacc=self.openacc
                )

        global_file_path = os.path.join(function_dir, self.module_global_file)
        self.processor.update_global_module(
                    cls.dec_global[function_key], 
                    global_file_path, function_key, 
                    self.module_tree_cp
                    )

        main_file_path = os.path.join(function_dir, self.main_program_file)
        arg_list = ', '.join([name for name in cls.dummy_arg_list[function_key]])
        call_stmt_org =  F23.Assignment_Stmt(f"{cls.func_result[function_key]} = {function_key}({arg_list})")

        self.processor.update_main_program(
                custom_dec_inout=cls.var_dummy[function_key],
                custom_subroutine_trees=[function_tree],
                call_stmts=[call_stmt_org],
                var_modif=cls.var_modif_info[function_key],
                file_path=main_file_path,
                subroutine_name=function_key,
                dummy_args=cls.dummy_arg_list[function_key],
                module_tree=self.module_tree_cp,
                childs_subroutine_tree=None,
                openacc=self.openacc,
                dummy_add_decl=None,
                error_flag=None,
                acc_data_copyin=None
                )

        error_status = self.processor.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."
        self.processor.logger.info(f"Successfully isolated and compiled function '{function_key}'")
        self.processor.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target)
        return function_tree

    def isolate_child_subroutine(self, cls, subroutine_key, local_var_parent=None):
        '''assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor.benchmark_dir, subroutine_key)
        os.makedirs(sub_dir, exist_ok=True)
        logging.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")
        '''
        dummy_as_local = set()
        function_tree = None
        if self.openacc:
            if local_var_parent is not None:
                for actual_arg_list in cls.actual_arg_spec_list[subroutine_key]:
                    for iarg, arg in enumerate(actual_arg_list):
                        if arg in local_var_parent:
                            dummy_arg = cls.dummy_arg_list[subroutine_key][iarg]
                            dummy_as_local.add(dummy_arg)
                            self.processor.logger.warning(
                                f"passing local variable '{arg}' as argument '{dummy_arg}' into procedure '{subroutine_key}'!"
                            )

        self.processor.logger.info(f"Trying to isolate a child subroutine .... {subroutine_key}")

        subroutine_tree = cls.subroutines[subroutine_key]
        cls.extract_intent(subroutine_key, subroutine_tree)
        cls.clean_subroutine(subroutine_key, subroutine_tree)
        code_string = subroutine_tree.tofortran()
        working_tree = Processor().parse_fortran_string(code_string)
        #assert working_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'
        cls.find_variables(subroutine_tree, subroutine_key)
        cls.extract_names(subroutine_key)
        cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, cls.var_global[subroutine_key], subroutine_key)

        cls.process_declaration_variables(cls.var_dummy[subroutine_key], subroutine_key)
        for key in cls.dec_global[subroutine_key].keys():
            cls.process_declaration_variables(cls.dec_global[subroutine_key][key], subroutine_key)

        shape_to_search = cls.shapes_variables[subroutine_key] - cls.scalar_variables[subroutine_key] - cls.var_global[subroutine_key]

        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, shape_to_search, subroutine_key)
            cls.var_global[subroutine_key].update(shape_to_search)

        cls.extract_array_info(cls.dec_global[subroutine_key], cls.var_dummy[subroutine_key], subroutine_key)

        for key, values in cls.dec_global[subroutine_key].items():
             if walk(values, F23.Function_Subprogram):
                 self.processor.logger.info(f"Calling Function_Subprogram {key} in Subroutine_Subprogram {subroutine_key}.")
                 self.processor.logger.info(f"Trying to isolate a child function .... {key}")
                 assert isinstance(values[1], F23.Function_Subprogram), f"Expected type 'F23.Function_Subprogram', but got '{type(values[0]).__name__}' instead."
                 assert isinstance(values[0], F23.Name), f"Expected type 'F23.Name', but got '{type(values[0]).__name__}' instead."
                 function_tree_org = values[1]
                 function_key = values[0].tostr()
                 if function_key in cls.subroutines:
                     self.processor.logger.info(f"Function '{function_key}' already successfully isolated, skipping...")
                     function_tree = cls.subroutines[function_key]
                     cls.call_within_sub[subroutine_key].add(function_key)
                 else:
                     function_tree = self.isolate_child_function(cls, function_tree_org, function_key, subroutine_key)
        
        function_keys = cls.call_within_sub[subroutine_key]
        if function_keys:
            self.processor.logger.info(f"Processing {len(function_keys)} function calls within subroutine {subroutine_key}")
            for function_key in cls.call_within_sub[subroutine_key]:
                self.collect_global_vars_decl(cls.dec_global[function_key], cls.dec_global[subroutine_key])


        if dummy_as_local:
            dummy_as_local = dummy_as_local.intersection(cls.all_array_info[subroutine_key].keys())
        if dummy_as_local:
            cls.var_local_names[subroutine_key].update(dummy_as_local)

        cls.extract_loop_vect(subroutine_key, subroutine_tree)

        modified_subroutine_tree = None
        call_stmt_vec = None
        dummy_add_decl = None
        error_flag = None
        acc_enter_data_copyin = None

        if self.openacc:
            modifier = Modifier(
                cls.loop_vect[subroutine_key],
                cls.all_array_info[subroutine_key],
                cls.loop_dict,
                cls.var_declared[subroutine_key],
                cls.imp_shape[subroutine_key],
                cls.allowed_external_subroutines,
                cls.var_local_names[subroutine_key]
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
            sub_dir = os.path.join(self.processor.benchmark_dir, subroutine_key)
            os.makedirs(sub_dir, exist_ok=True)
            self.processor.logger.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")

            subroutine_dir = os.path.join(self.target_module_dir, subroutine_key)
            os.makedirs(subroutine_dir)
        
            self.processor.add_declarations(
                    cls.dec_global[subroutine_key], 
                    cls.var_modif_info[subroutine_key],
                    openacc=self.openacc
                    )

            file_path = os.path.join(subroutine_dir, self.module_global_file)
            self.processor.update_global_module(
                    cls.dec_global[subroutine_key], 
                    file_path,
                    subroutine_key,
                    self.module_tree_cp
                    )
            file_path = os.path.join(subroutine_dir, self.main_program_file)

            sub_trees = [subroutine_tree]
            if function_tree is not None:
                sub_trees.append(function_tree)

            arg_list = ', '.join([name for name in cls.dummy_arg_list[subroutine_key]])
            call_stmt_org =  F23.Call_Stmt(f"CALL {subroutine_key}({arg_list})")
            call_stmts = [call_stmt_org]

            if self.openacc:
                sub_trees.append(modified_subroutine_tree)
                call_stmts.append(call_stmt_vec)

            self.processor.update_main_program(
                    custom_dec_inout=cls.var_dummy[subroutine_key],
                    custom_subroutine_trees=sub_trees,
                    call_stmts=call_stmts,
                    var_modif=cls.var_modif_info[subroutine_key],
                    file_path=file_path,
                    subroutine_name=subroutine_key,
                    dummy_args=cls.dummy_arg_list[subroutine_key],
                    module_tree=self.module_tree_cp,
                    childs_subroutine_tree=None,
                    openacc=self.openacc,
                    dummy_add_decl=dummy_add_decl,
                    error_flag=error_flag,
                    acc_data_copyin=acc_enter_data_copyin
                    )

            error_status = self.processor.compile_and_run(os.getcwd(), self.target_module_dir)
            assert error_status == 0, "Error: Compilation failed or main_program not generated."
            self.processor.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target)
        return modified_subroutine_tree, error_flag
        

    def isolate_parent_subroutine(self, cls, subroutine_key, subroutines_parent=None):
        assert os.path.exists(self.processor.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor.benchmark_dir, subroutine_key)
        os.makedirs(sub_dir, exist_ok=True)
        self.processor.logger.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")

        subroutine_tree = cls.subroutines[subroutine_key]
        cls.find_variables(subroutine_tree, subroutine_key)
        cls.extract_names(subroutine_key)
        queue = deque(cls.call_within_sub[subroutine_key])

        while queue:

            child_subroutine_key = queue.popleft()
            child_subroutine_tree = cls.subroutines[child_subroutine_key]
            self.child_subroutine_call[subroutine_key].append(child_subroutine_tree)

            if subroutines_parent is not None:
                for subroutine_key_parent in subroutines_parent:
                    self.child_subroutine_call[subroutine_key_parent].append(child_subroutine_tree)

            if child_subroutine_key not in cls.call_within_sub:
                mod_child_subroutine_tree, error_flag = self.isolate_child_subroutine(cls, child_subroutine_key, cls.var_local_names[subroutine_key])
                
                if mod_child_subroutine_tree is not None:
                    self.child_subroutine_call[subroutine_key].append(mod_child_subroutine_tree)
                
                if error_flag is not None:
                    self.child_error_flag[subroutine_key][child_subroutine_key] = error_flag
                
                self.collect_global_vars_decl(cls.dec_global[child_subroutine_key], cls.dec_global[subroutine_key])
            else:
                self.parent_subroutine_call.add(subroutine_key)
                self.isolate_parent_subroutine(cls, child_subroutine_key, self.parent_subroutine_call)


        self.processor.logger.info(f"Now, trying to isolate a parent subroutine .... {subroutine_key}")
        subroutine_dir = os.path.join(self.target_module_dir, subroutine_key)
        os.makedirs(subroutine_dir)

        cls.extract_intent(subroutine_key, subroutine_tree, cls.call_within_sub[subroutine_key])
        cls.clean_subroutine(subroutine_key, subroutine_tree)
        code_string = subroutine_tree.tofortran()
        working_tree = Processor().parse_fortran_string(code_string)
        #assert working_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'
        cls.find_variables(subroutine_tree, subroutine_key)
        cls.extract_names(subroutine_key)

        cls.var_global[subroutine_key] = cls.var_global[subroutine_key] - cls.call_within_sub[subroutine_key] - set(cls.dec_global[subroutine_key].keys())
        cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, cls.var_global[subroutine_key], subroutine_key)

        cls.process_declaration_variables(cls.var_dummy[subroutine_key], subroutine_key)
        for key in cls.dec_global[subroutine_key].keys():
            cls.process_declaration_variables(cls.dec_global[subroutine_key][key], subroutine_key)

        shape_to_search = cls.shapes_variables[subroutine_key] - cls.scalar_variables[subroutine_key] - cls.var_global[subroutine_key]

        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, shape_to_search, subroutine_key)
            cls.var_global[subroutine_key].update(shape_to_search)

        if subroutines_parent is not None:
            for subroutine_key_parent in subroutines_parent:
                self.collect_global_vars_decl(cls.dec_global[subroutine_key], cls.dec_global[subroutine_key_parent])

        cls.extract_array_info(cls.dec_global[subroutine_key], cls.var_dummy[subroutine_key], subroutine_key)
        cls.extract_loop_vect(subroutine_key, subroutine_tree)

        modified_subroutine_tree = None
        call_stmt_vec = None
        dummy_add_decl = None
        error_flag = None
        acc_enter_data_copyin = None

        if self.openacc:
            modifier = Modifier(
                    cls.loop_vect[subroutine_key],
                    cls.all_array_info[subroutine_key], 
                    cls.loop_dict, 
                    cls.var_declared[subroutine_key], 
                    cls.imp_shape[subroutine_key],
                    cls.allowed_external_subroutines, 
                    cls.var_local_names[subroutine_key],
                    cls.call_within_sub[subroutine_key],
                    self.child_error_flag[subroutine_key]
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
                cls.dec_global[subroutine_key],
                cls.var_modif_info[subroutine_key],
                openacc=self.openacc
                )
        
        file_path = os.path.join(subroutine_dir, self.module_global_file)
        self.processor.update_global_module(
                cls.dec_global[subroutine_key], 
                file_path, 
                subroutine_key, 
                self.module_tree_cp
                )
        file_path = os.path.join(subroutine_dir, self.main_program_file)

        sub_trees = [subroutine_tree]
        arg_list = ', '.join([name for name in cls.dummy_arg_list[subroutine_key]])
        call_stmt_org =  F23.Call_Stmt(f"CALL {subroutine_key}({arg_list})")
        call_stmts = [call_stmt_org]

        if self.openacc:
            sub_trees.append(modified_subroutine_tree)
            call_stmts.append(call_stmt_vec)

        
        self.processor.update_main_program(
                custom_dec_inout=cls.var_dummy[subroutine_key],
                custom_subroutine_trees=sub_trees,
                call_stmts=call_stmts,
                var_modif=cls.var_modif_info[subroutine_key],
                file_path=file_path,
                subroutine_name=subroutine_key,
                dummy_args=cls.dummy_arg_list[subroutine_key],
                module_tree=self.module_tree_cp,
                childs_subroutine_tree=self.child_subroutine_call[subroutine_key],
                openacc=self.openacc,
                dummy_add_decl=dummy_add_decl,
                error_flag=error_flag,
                acc_data_copyin=acc_enter_data_copyin
                )

        error_status = self.processor.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."
        self.processor.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target) 
        
    def collect_global_vars_decl(self, in_dict, out_dict):
        for child_key, child_value in in_dict.items():
                if child_key not in out_dict:
                    out_dict[child_key] = child_value

    def process_subroutines(self):
        cls = Extractor(self.module_dir_sp, self.module_tree_sp)
        cls.find_subroutines()
        cls.extract_loop_indices()

        #for subroutine in ['hydrol_hydraulic_arch_tuzet_calc',"hydrol_vegupd", 'hydrol_soil']:#cls.subroutine_keys_ncl:
        #    self.parent_subroutine_call = set()
        #    self.isolate_parent_subroutine(cls, subroutine)
        

        
        #subs = {'explicitsnow_age','explicitsnow_compactn','explicitsnow_compactn_up', 'explicitsnow_drift',
        #        'explicitsnow_fall','explicitsnow_gone','explicitsnow_icelevels','explicitsnow_icemelt','explicitsnow_iceprofile',
        #        'explicitsnow_levels','explicitsnow_maxmass','explicitsnow_melt_refrz','explicitsnow_profile','explicitsnow_subli',
        #        'explicitsnow_transf'}
        
        #
        #subs = ['explicitsnow_transf','explicitsnow_subli','explicitsnow_profile','explicitsnow_maxmass','explicitsnow_levels',
        #        'explicitsnow_iceprofile', 'explicitsnow_icemelt','explicitsnow_icelevels','explicitsnow_age', 'explicitsnow_compactn',
        #        'explicitsnow_drift', 'explicitsnow_gone']

        subs = cls.call_within_sub["explicitsnow_main"]
        
        #subs = ['hydrol_diag_soil','hydrol_diag_soil_flux','hydrol_nudge_mc','hydrol_root_profile','hydrol_soil_coef','hydrol_soil_froz',
        #        'hydrol_soil_infilt','hydrol_soil_setup','hydrol_soil_smooth_over_mcs2','hydrol_soil_smooth_under_mcr','hydrol_soil_tridiag','hydrol_split_soil']
        for subroutine in subs:
            self.isolate_child_subroutine(cls, subroutine)
        
    def run(self):
        self.create_target_directory()
        self.process_subroutines()

if __name__ == "__main__":
    rest_of_path = "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    target_modules = ["hydrol", "explicitsnow"]
    target_module =  target_modules[1]
    work = os.getenv("works")
    openacc = False
    isolator = Isolator(rest_of_path, target_module, work, openacc)
    isolator.run()

