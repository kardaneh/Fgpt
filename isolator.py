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
    """
    def __init__(self, rest_of_path, target_module, work):
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
        self.processor_sp = Processor()
        self.module_tree_sp = self.processor_sp.parse_fortran_file(self.module_file_sp)
        self.module_string = self.module_tree_sp.tostr()
        self.module_tree_cp = self.processor_sp.parse_fortran_string(self.module_string)
        self.target_module_dir = os.path.join(os.getcwd(), self.target_module.split('.')[0])
        self.child_subroutine_call = defaultdict(list)
        self.child_error_flag = defaultdict(lambda: defaultdict(dict))
    
    def create_target_directory(self):
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        os.makedirs(self.target_module_dir)

    def isolate_child_function(self, cls, function_values, parent_subroutine_key):
        """
        """
        assert isinstance(function_values[0], F23.Function_Subprogram), \
                f"Expected type 'F23.Function_Subprogram', but got '{type(function_values[0]).__name__}' instead."
        assert isinstance(function_values[1], F23.Program), \
                f"Expected type 'F23.Program', but got '{type(function_values[1]).__name__}' instead."
        assert isinstance(function_values[2], str), \
                f"Expected type 'str', but got '{type(function_values[2]).__name__}' instead."

        function_tree = function_values[0]
        module_tree = function_values[1]
        module_dir = function_values[2]
        function_stmt = walk(function_tree, F23.Function_Stmt)[0]
        for child in function_stmt.children:
            if child is None:
                continue
            if isinstance(child, F23.Name):
                function_key = child.tostr()
                cls.subroutines[function_key] = function_tree
            elif isinstance(child, F23.Dummy_Arg_List):
                arg_list = child
            elif isinstance(child, F23.Suffix):
                cls.func_result[function_key] = child.children[0].tostr()
        assert function_key is not None, f"Unexpected type {function_key} encountered in children."
        if arg_list is not None:
            for child in arg_list.children:
                cls.dummy_arg_list[function_key].append(child.tostr())
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
                raise ValueError(f"Calling Function_Subprogram {key} in Function_Subprogram {function_key}. Not yet implimented")

        raise ValueError(f"Variable {function_key}")

    def isolate_child_subroutine(self, cls, subroutine_key, local_var_parent=None):
        '''assert os.path.exists(self.processor_sp.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor_sp.benchmark_dir, subroutine_key)
        os.makedirs(sub_dir, exist_ok=True)
        logging.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")
        '''
        dummy_as_local = set() 
        if local_var_parent is not None:
            for actual_arg_list in cls.actual_arg_spec_list[subroutine_key]:
                for iarg, arg in enumerate(actual_arg_list):
                    if arg in local_var_parent:
                        dummy_arg = cls.dummy_arg_list[subroutine_key][iarg]
                        dummy_as_local.add(dummy_arg)
                        print(f'\033[38;5;196mWarning: passing local variable "{arg}" as argument "{dummy_arg}" into procedure "{subroutine_key}"!\033[0m')

        print("Trying to isolate a child subroutine .... ", subroutine_key)

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
                 print(f"Calling Function_Subprogram {key} in Subroutine_Subprogram {subroutine_key}.")
                 print("Trying to isolate a child function .... ", key)
                 self.isolate_child_function(cls, values, subroutine_key)


        if dummy_as_local:
            dummy_as_local = dummy_as_local.intersection(cls.all_array_info[subroutine_key].keys())
        if dummy_as_local:
            cls.var_local_names[subroutine_key].update(dummy_as_local)

        cls.extract_loop_vect(subroutine_key, subroutine_tree)

        modifier = Modifier(
                cls.loop_vect[subroutine_key],
                cls.all_array_info[subroutine_key], 
                cls.loop_dict, 
                cls.var_declared[subroutine_key], 
                cls.imp_shape[subroutine_key],
                cls.allowed_external_subroutines, 
                cls.var_local_names[subroutine_key])
        working_tree = modifier.replace_gpu_unsupported(working_tree)
        modified_block = modifier.merge_vector_loop(working_tree)
        
        assert modifier.do_index == 0 and modifier.enddo_index == 0, (
                f"Error: do_index and enddo_index are not reset properly. "
                f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
                )
        
        modified_subroutine_tree = modifier.add_vector_loop(modified_block)

        if not dummy_as_local:

            assert os.path.exists(self.processor_sp.benchmark_dir), "benchmark directory does not exist!"
            sub_dir = os.path.join(self.processor_sp.benchmark_dir, subroutine_key)
            os.makedirs(sub_dir, exist_ok=True)
            logging.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")

            subroutine_dir = os.path.join(self.target_module_dir, subroutine_key)
            os.makedirs(subroutine_dir)

            call_stmt_org = modifier.subroutine_call_act_org
            call_stmt_vec = modifier.subroutine_call_act_vec
        
            self.processor_sp.add_declarations(cls.dec_global[subroutine_key], cls.var_modif_info[subroutine_key])
            file_path = os.path.join(subroutine_dir, self.module_global_file)
            self.processor_sp.update_global_module(cls.dec_global[subroutine_key], file_path, subroutine_key, self.module_tree_cp)
            file_path = os.path.join(subroutine_dir, self.main_program_file)
            sub_trees = [subroutine_tree, modified_subroutine_tree]
            call_stmts = [call_stmt_org, call_stmt_vec]

            self.processor_sp.update_main_program(cls.var_dummy[subroutine_key], sub_trees, call_stmts, \
                    modifier.dummy_add_decl, \
                    modifier.error_flag, \
                    modifier.acc_enter_data_copyin, \
                    cls.var_modif_info[subroutine_key], file_path, subroutine_key, \
                    cls.dummy_arg_list[subroutine_key], self.module_tree_cp)
            error_status = self.processor_sp.compile_and_run(os.getcwd(), self.target_module_dir)
            assert error_status == 0, "Error: Compilation failed or main_program not generated."
            self.processor_sp.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target)
        return modified_subroutine_tree, modifier.error_flag
        

    def isolate_parent_subroutine(self, cls, subroutine_key, subroutines_parent=None):
        assert os.path.exists(self.processor_sp.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor_sp.benchmark_dir, subroutine_key)
        os.makedirs(sub_dir, exist_ok=True)
        logging.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")

        subroutine_tree = cls.subroutines[subroutine_key]
        cls.find_variables(subroutine_tree, subroutine_key)
        cls.extract_names(subroutine_key)

        queue = deque(cls.call_within_sub[subroutine_key])

        while queue:
            child_subroutine_key = queue.popleft()
            child_subroutine_tree = cls.subroutines[child_subroutine_key]
            self.child_subroutine_call[subroutine_key].append(child_subroutine_tree)
            #if subroutines_parent is not None:
            #    for subroutine_key_parent in subroutines_parent:
            #        self.child_subroutine_call[subroutine_key_parent].append(subroutine_tree)
            if child_subroutine_key not in cls.call_within_sub:
                mod_child_subroutine_tree, error_flag = self.isolate_child_subroutine(cls, child_subroutine_key, cls.var_local_names[subroutine_key])
                self.child_subroutine_call[subroutine_key].append(mod_child_subroutine_tree)
                self.child_error_flag[subroutine_key][child_subroutine_key] = error_flag
                self.collect_global_vars_decl(cls.dec_global[child_subroutine_key], cls.dec_global[subroutine_key])
            #else:
            #    self.parent_subroutine_call.add(subroutine_key)
            #    self.isolate_parent_subroutine(cls, child_subroutine_key, self.parent_subroutine_call)


        print("Now, trying to isolate a parent subroutine .... ", subroutine_key)
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

        cls.extract_array_info(cls.dec_global[subroutine_key], cls.var_dummy[subroutine_key], subroutine_key)
        cls.extract_loop_vect(subroutine_key, subroutine_tree)

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
        call_stmt_org = modifier.subroutine_call_act_org
        call_stmt_vec = modifier.subroutine_call_act_vec

        self.processor_sp.add_declarations(cls.dec_global[subroutine_key], cls.var_modif_info[subroutine_key])
        file_path = os.path.join(subroutine_dir, self.module_global_file)
        self.processor_sp.update_global_module(cls.dec_global[subroutine_key], file_path, subroutine_key, self.module_tree_cp)
        file_path = os.path.join(subroutine_dir, self.main_program_file)
        sub_trees = [subroutine_tree, modified_subroutine_tree]
        call_stmts = [call_stmt_org, call_stmt_vec]
        
        self.processor_sp.update_main_program(cls.var_dummy[subroutine_key], sub_trees, call_stmts, \
                modifier.dummy_add_decl,\
                modifier.error_flag,\
                modifier.acc_enter_data_copyin, \
                cls.var_modif_info[subroutine_key], file_path, subroutine_key, \
                cls.dummy_arg_list[subroutine_key], self.module_tree_cp, \
                self.child_subroutine_call[subroutine_key])

        error_status = self.processor_sp.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."
        self.processor_sp.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target) 

    def collect_global_vars_decl(self, in_dict, out_dict):
        for child_key, child_value in in_dict.items():
                if child_key not in out_dict:
                    out_dict[child_key] = child_value

    def process_subroutines(self):
        cls = Extractor(self.module_dir_sp, self.module_tree_sp)
        cls.find_subroutines()
        cls.extract_loop_indices()

        for subroutine in ['hydrol_soil']:#cls.subroutine_keys_ncl:
            self.parent_subroutine_call = set()
            self.isolate_parent_subroutine(cls, subroutine)
        
        '''
        subs = ['hydrol_diag_soil','hydrol_diag_soil_flux','hydrol_nudge_mc','hydrol_root_profile',
                'hydrol_soil_coef','hydrol_soil_froz','hydrol_soil_infilt','hydrol_soil_setup',
                'hydrol_soil_smooth_over_mcs2','hydrol_soil_smooth_under_mcr','hydrol_soil_tridiag',
                'hydrol_split_soil']
        subs = {'explicitsnow_age','explicitsnow_compactn','explicitsnow_compactn_up', 'explicitsnow_drift',
                'explicitsnow_fall','explicitsnow_gone','explicitsnow_icelevels','explicitsnow_icemelt','explicitsnow_iceprofile',
                'explicitsnow_levels','explicitsnow_maxmass','explicitsnow_melt_refrz','explicitsnow_profile','explicitsnow_subli',
                'explicitsnow_transf'}
        '''
        '''subs = ['explicitsnow_transf','explicitsnow_subli','explicitsnow_profile','explicitsnow_maxmass','explicitsnow_levels',
                'explicitsnow_iceprofile', 'explicitsnow_icemelt','explicitsnow_icelevels','explicitsnow_age', 'explicitsnow_compactn',
                'explicitsnow_drift', 'explicitsnow_gone']
        subs = ['explicitsnow_grain']
        for subroutine in subs:
            self.isolate_child_subroutine(cls, subroutine)
        '''
    def run(self):
        self.create_target_directory()
        self.process_subroutines()

if __name__ == "__main__":
    rest_of_path = "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    target_module = "hydrol" #"explicitsnow"
    work = os.getenv("WORK")
    isolator = Isolator(rest_of_path, target_module, work)
    isolator.run()

