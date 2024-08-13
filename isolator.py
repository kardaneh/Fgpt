import os
import shutil
from collections import deque
from processor import Processor
from extractor import Extractor
from modifier import Modifier

class Isolator:
    def __init__(self, rest_of_path, target_module, work):
        self.module_global_file = "module_global.f90"
        self.main_program_file = "main.f90"
        self.rest_of_path = rest_of_path
        self.target_module = target_module
        self.scratch_dir = work
        self.module_dir_sp = os.path.join(self.scratch_dir, self.rest_of_path)
        self.module_file_sp = os.path.join(self.module_dir_sp, self.target_module)
        self.processor_sp = Processor()
        self.module_tree_sp = self.processor_sp.parse_fortran_file(self.module_file_sp)
        self.target_module_dir = os.path.join(os.getcwd(), self.target_module.split('.')[0])
        self.global_vars_decl_sp = {}
        self.child_subroutine_call_sp = {}
        self.child_error_flag = {}
    
    def setup_environment(self):
        self.module_dir_sp = os.path.join(self.scratch_dir, self.rest_of_path)
        self.module_file_sp = os.path.join(self.module_dir_sp, self.target_module)
    
    def create_target_directory(self):
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        os.makedirs(self.target_module_dir)

    def separate_child_subroutine(self, cls, subroutine_key):
        print("Trying to separate a child subroutine .... ", subroutine_key)
        subroutine_dir = os.path.join(self.target_module_dir, subroutine_key)
        os.makedirs(subroutine_dir)

        subroutine_tree = cls.subroutines[subroutine_key]
        code_string = subroutine_tree.tofortran()
        parse_tree = Processor().parse_fortran_string(code_string)
        #assert parse_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'

        cls.find_variables(subroutine_tree, subroutine_key)
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
        modifier = Modifier(
                self.all_array_info, 
                cls.loop_dict, 
                cls.var_declared, 
                cls.imp_shape,
                cls.allowed_external_subroutines, 
                cls.var_global)
        parse_tree = modifier.replace_gpu_unsupported(parse_tree)
        modified_block = modifier.merge_vector_loop(parse_tree)
        
        assert modifier.do_index == 0 and modifier.enddo_index == 0, (
                f"Error: do_index and enddo_index are not reset properly. "
                f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
                )
        
        modified_subroutine_tree = modifier.add_vector_loop(modified_block)
        call_stmt_org = modifier.subroutine_call_act_org
        call_stmt_vec = modifier.subroutine_call_act_vec
        
        self.processor_sp.add_declarations(cls.dec_global, cls.var_modif_info)
        file_path = os.path.join(subroutine_dir, self.module_global_file)
        self.processor_sp.update_global_module(cls.dec_global, file_path)
        file_path = os.path.join(subroutine_dir, self.main_program_file)
        sub_trees = [subroutine_tree, modified_subroutine_tree]
        call_stmts = [call_stmt_org, call_stmt_vec]
        cls.var_dummy.append(modifier.dummy_add_decl)

        if modifier.error_flag.keys():
            for key in modifier.error_flag.keys():
                cls.var_dummy.append(modifier.error_flag[key]['error_flag_decl'])
        
        self.processor_sp.update_main_program(cls.var_dummy, sub_trees, call_stmts, \
                modifier.error_flag, 
                modifier.acc_enter_data_copyin, \
                cls.var_modif_info, file_path)

        error_status = self.processor_sp.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."
        return modified_subroutine_tree, modifier.error_flag

    def separate_parent_subroutine(self, cls, subroutine_key, subroutines_parent=None):
        queue = deque(cls.call_within_sub[subroutine_key])
        if subroutine_key not in self.child_subroutine_call_sp:
            self.child_subroutine_call_sp[subroutine_key] = []
            self.child_error_flag[subroutine_key] = {}

        while queue:
            child_subroutine_key = queue.popleft()
            subroutine_tree = cls.subroutines[child_subroutine_key]
            self.child_subroutine_call_sp[subroutine_key].append(subroutine_tree)
            #if subroutines_parent is not None:
            #    for subroutine_key_parent in subroutines_parent:
            #        self.child_subroutine_call_sp[subroutine_key_parent].append(subroutine_tree)
            if child_subroutine_key not in cls.call_within_sub:
                modified_subroutine_tree, error_flag = self.separate_child_subroutine(cls, child_subroutine_key)
                self.child_subroutine_call_sp[subroutine_key].append(modified_subroutine_tree)
                self.child_error_flag[subroutine_key][child_subroutine_key] = error_flag
                self.merge_global_vars_decl(cls.dec_global)
            #else:
            #    self.parent_subroutine_call.add(subroutine_key)
            #    self.separate_parent_subroutine(cls, child_subroutine_key, self.parent_subroutine_call)


        print("Trying to separate a parent subroutine .... ", subroutine_key)
        subroutine_dir = os.path.join(self.target_module_dir, subroutine_key)
        os.makedirs(subroutine_dir)

        subroutine_tree = cls.subroutines[subroutine_key]
        code_string = subroutine_tree.tofortran()
        parse_tree = Processor().parse_fortran_string(code_string)
        #assert parse_tree == subroutine_tree, 'Error: Parsed code differs from original subroutine tree.'

        cls.find_variables(subroutine_tree, subroutine_key)
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
                cls.var_global,
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
        self.processor_sp.update_global_module(self.global_vars_decl_sp, file_path)
        file_path = os.path.join(subroutine_dir, self.main_program_file)
        sub_trees = [subroutine_tree, modified_subroutine_tree]
        call_stmts = [call_stmt_org, call_stmt_vec]
        cls.var_dummy.append(modifier.dummy_add_decl)

        if modifier.error_flag.keys():
            for key in modifier.error_flag.keys():
                cls.var_dummy.append(modifier.error_flag[key]['error_flag_decl'])

        self.processor_sp.update_main_program(cls.var_dummy, sub_trees, call_stmts, \
                modifier.error_flag,\
                modifier.acc_enter_data_copyin, \
                cls.var_modif_info, file_path, self.child_subroutine_call_sp[subroutine_key])

        error_status = self.processor_sp.compile_and_run(os.getcwd(), self.target_module_dir)
        assert error_status == 0, "Error: Compilation failed or main_program not generated."
        

    def merge_global_vars_decl(self, in_dict):
        for child_key, child_value in in_dict.items():
                if child_key not in self.global_vars_decl_sp:
                    self.global_vars_decl_sp[child_key] = child_value

    def process_subroutines(self):
        cls = Extractor(self.module_dir_sp, self.module_tree_sp)
        cls.find_subroutines()
        cls.extract_loop_indices()

        for subroutine in ['hydrol_vegupd']:#cls.subroutine_keys_ncl:
            self.parent_subroutine_call = set()
            self.separate_parent_subroutine(cls, subroutine)
        
        '''['hydrol_diag_soil','hydrol_diag_soil_flux','hydrol_nudge_mc','hydrol_root_profile',
                'hydrol_soil_coef','hydrol_soil_froz','hydrol_soil_infilt','hydrol_soil_setup',
                'hydrol_soil_smooth_over_mcs2','hydrol_soil_smooth_under_mcr','hydrol_soil_tridiag',
                'hydrol_split_soil']:
        '''
        '''for subroutine in ['hydrol_tmc_update']:
            self.separate_child_subroutine(cls, subroutine)
        '''
    def run(self):
        self.setup_environment()
        self.create_target_directory()
        self.process_subroutines()

if __name__ == "__main__":
    rest_of_path = "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/"
    target_module = "hydrol.f90"
    work = os.getenv("WORK")
    processor = Isolator(rest_of_path, target_module, work)
    processor.run()

