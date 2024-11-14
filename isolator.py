import logging
import os
import shutil
from collections import deque
from processor import Processor
from extractor import Extractor
from modifier import Modifier
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from fparser.two import Fortran2008 as F28

class Isolator:
    """
    The Isolator class is responsible for isolating subroutines from Fortran modules. It 
    manages the environment setup, parsing, and transformation of the subroutines while handling 
    dependencies like global variable declarations. The class operates on Fortran code, separating parent 
    and child subroutines, and preparing the code for further processing or compilation.

    Key responsibilities include:
        - Setting up the directory structure and environment for isolating subroutines.
        - Extracting subroutines from Fortran modules and handling their dependencies.
        - Processing subroutines to identify and modify relevant loops and variables.
        - Merging global variable declarations across the processed subroutines.
        - Updating and managing main program files, ensuring the integration of isolated subroutines.

    Attributes:
        rest_of_path (str): The path to the module directory within the working environment.
        target_module (str): The name of the Fortran module being processed.
        scratch_dir (str): The working directory where intermediate files and outputs are stored.
        module_dir_sp (str): The specific path to the module within the scratch directory.
        module_file_sp (str): The full path to the target Fortran module file.
        processor_sp (Processor): A `Processor` instance to handle parsing and Fortran code operations.
        module_tree_sp (Node): The parsed tree representation of the Fortran module.
        target_module_dir (str): Directory for the isolated target module's output.
        global_vars_decl_sp (dict): Dictionary to hold global variable declarations from subroutines.
        child_subroutine_call_sp (dict): Stores subroutine calls within the processed child subroutines.
        child_error_flag (dict): Tracks errors or flags encountered during child subroutine isolation.

    Methods:
        setup_environment():
            Sets up the directory structure and paths for the target module and its processing.

        create_target_directory():
            Creates or cleans the directory where the isolated subroutines will be stored.

        separate_child_subroutine(cls, subroutine_key):
            Isolates a specific child subroutine from the Fortran module, parsing and processing it.

        merge_global_vars_decl(in_dict):
            Merges global variable declarations from child subroutines into a master list.

        process_subroutines():
            Processes all subroutines from the target module, handling both parent and child subroutines.

        run():
            Executes the full pipeline of environment setup, subroutine isolation, and processing.
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
        self.global_vars_decl_sp = {}
        self.child_subroutine_call_sp = {}
        self.child_error_flag = {}
    
    '''def setup_environment(self):
        self.module_dir_sp = os.path.join(self.scratch_dir, self.rest_of_path)
        self.module_file_sp = os.path.join(self.module_dir_sp, self.target_module)
    '''
    def create_target_directory(self):
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        os.makedirs(self.target_module_dir)

    def separate_child_subroutine(self, cls, child_subroutine_key, local_var_parent=None):
        '''assert os.path.exists(self.processor_sp.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor_sp.benchmark_dir, child_subroutine_key)
        os.makedirs(sub_dir, exist_ok=True)
        logging.info(f"{child_subroutine_key} directory created inside benchmark: {sub_dir}")
        '''
        dummy_as_local = set() 
        if local_var_parent is not None:
            for actual_arg_list in cls.actual_arg_spec_list[child_subroutine_key]:
                for iarg, arg in enumerate(actual_arg_list):
                    if arg in local_var_parent:
                        dummy_arg = cls.dummy_arg_list[child_subroutine_key][iarg]
                        dummy_as_local.add(dummy_arg)
                        print(f'\033[38;5;196mWarning: passing local variable "{arg}" as argument "{dummy_arg}" into procedure "{child_subroutine_key}"!\033[0m')

        print("Trying to separate a child subroutine .... ", child_subroutine_key)

        subroutine_tree = cls.subroutines[child_subroutine_key]
        cls.extract_intent(child_subroutine_key, subroutine_tree)
        cls.clean_subroutine(child_subroutine_key, subroutine_tree)
        code_string = subroutine_tree.tofortran()
        parse_tree = Processor().parse_fortran_string(code_string)
        #assert parse_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'
        cls.find_variables(subroutine_tree, child_subroutine_key)
        local_var_child = cls.extract_names(cls.var_local)

        cls.dec_global = {}
        cls.dec_child = {}
        cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, cls.var_global)

        cls.scalar_variables = set()
        cls.shapes_variables = set()
        cls.process_declaration_variables(cls.var_dummy)
        for key in cls.dec_global.keys():
            cls.process_declaration_variables(cls.dec_global[key])

        shape_to_search = cls.shapes_variables - cls.scalar_variables - cls.var_global
        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, shape_to_search)
            cls.var_global.update(shape_to_search)

        self.all_array_info = cls.extract_array_info(cls.dec_global, cls.var_dummy)
        if dummy_as_local:
            dummy_as_local = dummy_as_local.intersection(self.all_array_info.keys())
        if dummy_as_local:
            local_var_child.update(dummy_as_local)

        modifier = Modifier(
                self.all_array_info, 
                cls.loop_dict, 
                cls.var_declared, 
                cls.imp_shape,
                cls.allowed_external_subroutines, 
                local_var_child)
        parse_tree = modifier.replace_gpu_unsupported(parse_tree)
        modified_block = modifier.merge_vector_loop(parse_tree)
        
        assert modifier.do_index == 0 and modifier.enddo_index == 0, (
                f"Error: do_index and enddo_index are not reset properly. "
                f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
                )
        
        modified_subroutine_tree = modifier.add_vector_loop(modified_block)

        if not dummy_as_local:

            assert os.path.exists(self.processor_sp.benchmark_dir), "benchmark directory does not exist!"
            sub_dir = os.path.join(self.processor_sp.benchmark_dir, child_subroutine_key)
            os.makedirs(sub_dir, exist_ok=True)
            logging.info(f"{child_subroutine_key} directory created inside benchmark: {sub_dir}")

            subroutine_dir = os.path.join(self.target_module_dir, child_subroutine_key)
            os.makedirs(subroutine_dir)

            call_stmt_org = modifier.subroutine_call_act_org
            call_stmt_vec = modifier.subroutine_call_act_vec
        
            self.processor_sp.add_declarations(cls.dec_global, cls.var_modif_info)
            file_path = os.path.join(subroutine_dir, self.module_global_file)
            self.processor_sp.update_global_module(cls.dec_global, file_path, child_subroutine_key, self.module_tree_cp)
            file_path = os.path.join(subroutine_dir, self.main_program_file)
            sub_trees = [subroutine_tree, modified_subroutine_tree]
            call_stmts = [call_stmt_org, call_stmt_vec]

            self.processor_sp.update_main_program(cls.var_dummy, sub_trees, call_stmts, \
                    modifier.dummy_add_decl, \
                    modifier.error_flag, \
                    modifier.acc_enter_data_copyin, \
                    cls.var_modif_info, file_path, child_subroutine_key, \
                    cls.dummy_arg_list[child_subroutine_key], self.module_tree_cp)
            error_status = self.processor_sp.compile_and_run(os.getcwd(), self.target_module_dir)
            assert error_status == 0, "Error: Compilation failed or main_program not generated."
            self.processor_sp.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target)
        return modified_subroutine_tree, modifier.error_flag
        

    def separate_parent_subroutine(self, cls, subroutine_key, subroutines_parent=None):
        assert os.path.exists(self.processor_sp.benchmark_dir), "benchmark directory does not exist!"
        sub_dir = os.path.join(self.processor_sp.benchmark_dir, subroutine_key)
        os.makedirs(sub_dir, exist_ok=True)
        logging.info(f"{subroutine_key} directory created inside benchmark: {sub_dir}")

        subroutine_tree = cls.subroutines[subroutine_key]
        cls.find_variables(subroutine_tree, subroutine_key)
        local_var_parent = cls.extract_names(cls.var_local)

        queue = deque(cls.call_within_sub[subroutine_key])
        if subroutine_key not in self.child_subroutine_call_sp:
            self.child_subroutine_call_sp[subroutine_key] = []
            self.child_error_flag[subroutine_key] = {}

        while queue:
            child_subroutine_key = queue.popleft()
            child_subroutine_tree = cls.subroutines[child_subroutine_key]
            self.child_subroutine_call_sp[subroutine_key].append(child_subroutine_tree)
            #if subroutines_parent is not None:
            #    for subroutine_key_parent in subroutines_parent:
            #        self.child_subroutine_call_sp[subroutine_key_parent].append(subroutine_tree)
            if child_subroutine_key not in cls.call_within_sub:
                modified_child_subroutine_tree, error_flag = self.separate_child_subroutine(cls, child_subroutine_key, local_var_parent)
                self.child_subroutine_call_sp[subroutine_key].append(modified_child_subroutine_tree)
                self.child_error_flag[subroutine_key][child_subroutine_key] = error_flag
                self.merge_global_vars_decl(cls.dec_global)
            #else:
            #    self.parent_subroutine_call.add(subroutine_key)
            #    self.separate_parent_subroutine(cls, child_subroutine_key, self.parent_subroutine_call)


        print("Now, trying to separate a parent subroutine .... ", subroutine_key)
        subroutine_dir = os.path.join(self.target_module_dir, subroutine_key)
        os.makedirs(subroutine_dir)

        subroutine_tree = cls.subroutines[subroutine_key]
        cls.extract_intent(subroutine_key, subroutine_tree, cls.call_within_sub[subroutine_key])
        cls.clean_subroutine(subroutine_key, subroutine_tree)
        code_string = subroutine_tree.tofortran()
        parse_tree = Processor().parse_fortran_string(code_string)
        #assert parse_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'

        cls.find_variables(subroutine_tree, subroutine_key)
        local_var_parent = cls.extract_names(cls.var_local)
        cls.var_global = cls.var_global - cls.call_within_sub[subroutine_key] - set(self.global_vars_decl_sp.keys())
        cls.dec_global = {}
        cls.dec_child = {}
        cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, cls.var_global)

        cls.scalar_variables = set()
        cls.shapes_variables = set()
        cls.process_declaration_variables(cls.var_dummy)
        for key in cls.dec_global.keys():
            cls.process_declaration_variables(cls.dec_global[key])
        shape_to_search = cls.shapes_variables - cls.scalar_variables - cls.var_global

        if shape_to_search:
            cls.find_global_variables(self.module_dir_sp, self.module_tree_sp, shape_to_search)
            cls.var_global.update(shape_to_search)

        self.merge_global_vars_decl(cls.dec_global)

        self.all_array_info = cls.extract_array_info(self.global_vars_decl_sp, cls.var_dummy)
        modifier = Modifier(
                self.all_array_info, 
                cls.loop_dict, 
                cls.var_declared, 
                cls.imp_shape,
                cls.allowed_external_subroutines, 
                local_var_parent,
                cls.call_within_sub[subroutine_key],
                self.child_error_flag[subroutine_key]
                )
        parse_tree = modifier.replace_gpu_unsupported(parse_tree)
        modified_block = modifier.merge_vector_loop(parse_tree)

        assert modifier.do_index == 0 and modifier.enddo_index == 0, (
                f"Error: do_index and enddo_index are not reset properly. "
                f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
                )

        modified_subroutine_tree = modifier.add_vector_loop(modified_block)
        call_stmt_org = modifier.subroutine_call_act_org
        call_stmt_vec = modifier.subroutine_call_act_vec

        self.processor_sp.add_declarations(self.global_vars_decl_sp, cls.var_modif_info)
        file_path = os.path.join(subroutine_dir, self.module_global_file)
        self.processor_sp.update_global_module(self.global_vars_decl_sp, file_path, subroutine_key, self.module_tree_cp)
        file_path = os.path.join(subroutine_dir, self.main_program_file)
        sub_trees = [subroutine_tree, modified_subroutine_tree]
        call_stmts = [call_stmt_org, call_stmt_vec]
        '''cls.var_dummy.append(modifier.dummy_add_decl)

        if modifier.error_flag.keys():
            for key in modifier.error_flag.keys():
                cls.var_dummy.append(modifier.error_flag[key]['error_flag_decl'])
        '''
        self.processor_sp.update_main_program(cls.var_dummy, sub_trees, call_stmts, \
                modifier.dummy_add_decl,\
                modifier.error_flag,\
                modifier.acc_enter_data_copyin, \
                cls.var_modif_info, file_path, subroutine_key, \
                cls.dummy_arg_list[subroutine_key], self.module_tree_cp, \
                self.child_subroutine_call_sp[subroutine_key])

        error_status = self.processor_sp.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."
        self.processor_sp.write_fortran_code_to_file(self.module_tree_cp, self.path_to_target) 

    def merge_global_vars_decl(self, in_dict):
        for child_key, child_value in in_dict.items():
                if child_key not in self.global_vars_decl_sp:
                    self.global_vars_decl_sp[child_key] = child_value

    def process_subroutines(self):
        cls = Extractor(self.module_dir_sp, self.module_tree_sp)
        cls.find_subroutines()
        cls.extract_loop_indices()

        for subroutine in ['hydrol_soil']:#cls.subroutine_keys_ncl:
            self.parent_subroutine_call = set()
            self.separate_parent_subroutine(cls, subroutine)
        '''
        subs = ['hydrol_diag_soil','hydrol_diag_soil_flux','hydrol_nudge_mc','hydrol_root_profile',
                'hydrol_soil_coef','hydrol_soil_froz','hydrol_soil_infilt','hydrol_soil_setup',
                'hydrol_soil_smooth_over_mcs2','hydrol_soil_smooth_under_mcr','hydrol_soil_tridiag',
                'hydrol_split_soil']
        
        subs = ['hydrol_diag_soil']
        for subroutine in subs:
            self.separate_child_subroutine(cls, subroutine)
        '''
    def run(self):
        #self.setup_environment()
        self.create_target_directory()
        self.process_subroutines()

if __name__ == "__main__":
    rest_of_path = "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    target_module = "hydrol"
    work = os.getenv("work")
    isolator = Isolator(rest_of_path, target_module, work)
    isolator.run()

