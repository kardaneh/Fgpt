import os, sys
from processor import Processor
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from navigator import Navigator
from shaper import Shaper

class Extractor:
    """
    A class to analyze and extract information from Fortran subroutines within a module.

    Attributes:
        module_dir (str): Directory of the Fortran module.
        module_tree: Parsed tree of the Fortran module.
        subroutine_keys_all (set): Set of all subroutine names.
        subroutine_keys_ncl (set): Set of subroutine names with no-call lists (NCL).
        subroutines (dict): Dictionary of subroutine trees keyed by subroutine names.
        dummy_arg_list (dict): Dictionary of dummy arguments for each subroutine.
        actual_arg_spec_list (dict): Dictionary of actual argument specifications for each call.
        external_subroutines (set): Set of external subroutines not declared within the module.
        call_subroutines (dict): Dictionary of call statements within subroutines.
        call_within_sub (dict): Dictionary of calls made within a specific subroutine.
        loop_dict (dict): Dictionary of loop indices found within the module.
        exclude (set): Set of variable names to exclude from analysis.
        cases_to_exclude (list): List of subroutine names to exclude from analysis.
        allowed_external_subroutines (set): Set of external subroutines allowed in calls.
        dec_global (dict): Dictionary of global declarations found in subroutines.
        dec_child (dict): Dictionary of child declarations found in subroutines.
        imp_shape (dict): Dictionary of implicit shape declarations.
        scalar_variables (set): Set of scalar variable names.
        shapes_variables (set): Set of shaped variable names.
    """

    def __init__(self, module_dir, module_tree):
        """
        Initializes the Extractor with the given module directory and tree.

        Args:
            module_dir (str): Directory of the Fortran module.
            module_tree: Parsed tree of the Fortran module.
        """
        try:
            self.module_dir = module_dir
            self.module_tree = module_tree
            self.subroutine_keys_all = set()
            self.subroutine_keys_ncl = set()
            self.subroutines = {}
            self.dummy_arg_list = {}
            self.actual_arg_spec_list = {}
            self.external_subroutines = {}
            self.call_subroutines = {}
            self.call_within_sub = {}
            self.loop_dict = {}
            self.exclude = {'kjpindex', 'nslm', 'nstm', 'nvm', 'nsnow', 'DIM', 'dim', 'MASK', 'next_calc_loop'}
            self.cases_to_exclude = ['clear', 'finalize', 'init', 'initialize', 'read']
            self.allowed_external_subroutines = {'ipslerr_p', 'xios_orchidee_send_field'}
            self.dec_global = {}
            self.dec_child = {}
            self.imp_shape = {}
            self.scalar_variables = set()
            self.shapes_variables = set()
        except Exception as e:
            raise RuntimeError(f"Error in __init__: {str(e)}")

    def extract_loop_indices(self):
        """
        Extracts loop indices and their start and end values from the module tree.

        Updates the loop_dict attribute with loop indices keyed by their end values.
        """
        try:
            for loop in walk(self.module_tree, F23.Nonlabel_Do_Stmt):
                if walk(loop, F23.Or_Operand) or walk(loop, F23.Part_Ref):
                    continue
                line_parts = loop.tostr().split('=')
                loop_index = line_parts[0].split()[-1]
                start_end_stride_values = line_parts[1].split(',')
                loop_start = start_end_stride_values[0].strip()
                loop_end = start_end_stride_values[1].strip()
                loop_end = ''.join(filter(str.isalpha, loop_end))
                if loop_end:
                    if loop_end not in self.loop_dict:
                        self.loop_dict[loop_end] = set()
                    self.loop_dict[loop_end].add(loop_index)
        except Exception as e:
            raise RuntimeError(f"Error in extract_loop_indices: {str(e)}")


    def find_subroutines(self):
        for sub in walk(self.module_tree, F23.Subroutine_Subprogram):
            subroutine_name = walk(sub, F23.Name)[0].string
            check = all(case not in subroutine_name for case in self.cases_to_exclude)
            if check:
                self.subroutine_keys_all.add(subroutine_name)
                self.subroutines[subroutine_name] = sub
                if subroutine_name not in self.dummy_arg_list:
                    self.dummy_arg_list[subroutine_name] = []
                for node in walk(sub, F23.Subroutine_Stmt):
                    arg_list = walk(node, F23.Dummy_Arg_List)
                    if arg_list:
                        arg_string = [string.strip() for string in arg_list[0].tostr().split(',')]
                        self.dummy_arg_list[subroutine_name] = arg_string
                call_stmt = walk(sub, F23.Call_Stmt)
                if call_stmt:
                    for item in call_stmt:
                        call_name = walk(item, F23.Name)[0].string
                        if call_name not in self.allowed_external_subroutines:
                            if subroutine_name not in self.call_within_sub:
                                self.call_within_sub[subroutine_name] = set()
                            self.call_within_sub[subroutine_name].add(call_name)
                        else:
                            self.subroutine_keys_ncl.add(subroutine_name)
                        if call_name not in self.actual_arg_spec_list:
                            self.actual_arg_spec_list[call_name] = []
                            self.call_subroutines[call_name] = []
                        arg_list = walk(item, F23.Actual_Arg_Spec_List)
                        if arg_list:
                            arg_string = [string.strip() for string in arg_list[0].tostr().split(',')]
                            self.actual_arg_spec_list[call_name].append(arg_string)
                        self.call_subroutines[call_name].append(item)
                else:
                    self.subroutine_keys_ncl.add(subroutine_name)
        self.external_subroutines = {item for item in self.actual_arg_spec_list.keys() \
                if item not in self.dummy_arg_list.keys()}

    def extract_names(self, var_local):
        """
        Extracts variable names from the given list of local variables 
        and stores them in the var_local_names set, resetting the set each time.
        
        Args:
            var_local (list): A list of local variables, where each item is 
                              a Fortran code structure.
        
        Raises:
            Exception: Captures and prints any exceptions encountered during
                       the name extraction process.
        """
        var_local_names = set() 
        for item in var_local:
            for entity in walk(item, F23.Entity_Decl):
                for child in entity.children:
                    if isinstance(child, F23.Name):
                        var_local_names.add(child.tostr())
        return var_local_names

    def find_variables(self, subroutine_tree, subroutine_name):
        self.var_global = set()
        self.var_in_local = set()
        self.var_dummy = []
        self.var_local = []
        self.var_modif = set()

        declared, used = walk(subroutine_tree, F23.Specification_Part), walk(subroutine_tree, F23.Execution_Part)
        self.var_declared = {name.tostr() for name in  walk(declared, F23.Entity_Decl)}
        names_declared, names_used = walk(declared, F23.Name), walk(used, F23.Name)

        var_declared = {name.string for name in names_declared }
        var_used = {name.string for name in names_used}
        self.var_global = var_used - var_declared

        shape = walk(walk(subroutine_tree, F23.Explicit_Shape_Spec), F23.Name)
        shapes = {name.string for name in shape}

        for declaration_stmt in walk(declared, F23.Type_Declaration_Stmt):
            #implicit_shape = walk(declaration_stmt, F23.Assumed_Shape_Spec)
            #if implicit_shape and len(walk(declaration_stmt, F23.Entity_Decl)) > 1:
            if len(walk(declaration_stmt, F23.Entity_Decl)) > 1:
                node_list = Processor().separate_entity_declarations(declaration_stmt)
            else:
                node_list = [declaration_stmt]
            for node in node_list:
                implicit_shape = walk(node, F23.Assumed_Shape_Spec)
                if implicit_shape:
                    shape_finder = Shaper(self.module_dir, self.dummy_arg_list, self.actual_arg_spec_list, self.call_subroutines)
                    nodes = shape_finder.find_implicit_shape(node, subroutine_name)
                    print('\033[32m' + 'found :' + '\033[0m', nodes, '\033[32m' + 'for :' + '\033[0m', node)
                    node = Processor().map_declaration(node, nodes)
                    entity_decl = walk(node, F23.Entity_Decl)[0].tostr()
                    if entity_decl not in self.imp_shape:
                        self.imp_shape[entity_decl] = node
                intent = walk(node, F23.Intent_Spec)
                if intent:
                    if walk(walk(node,F23.Entity_Decl),F23.Name)[0].string not in self.exclude:
                        self.var_dummy.append(node)
                        if F23.Intent_Attr_Spec('INTENT(IN)') in walk(node, F23.Intent_Attr_Spec):
                            self.var_in_local = {name.tostr() for name in  walk(node, F23.Entity_Decl)}
                else:
                    for name in walk(node, F23.Entity_Decl):
                        if name.string in self.dummy_arg_list[subroutine_name]:
                            raise ValueError(f"Variable '{name.string}' in subroutine '{subroutine_name}' "
                                    f"at statement '{node.tostr()}' is a dummy argument without intent.")
                        else:
                            self.var_in_local.add(name.string)
                    self.var_local.append(node)

        shape ={name.string for name in  walk(walk(self.var_dummy, F23.Explicit_Shape_Spec), F23.Name)}
        shapes.update(shape)

        self.var_global -= self.exclude
        shapes -= self.exclude
        self.var_global.update(shapes)

        #self.extract_names(self.var_local)

        for stmt in walk(subroutine_tree, F23.Execution_Part):
            for assign_stmt in walk(stmt, F23.Assignment_Stmt):
                lhs = assign_stmt.items[0]
                if isinstance (lhs, F23.Name):
                    if lhs.tostr() not in self.var_in_local:
                        self.var_modif.add(lhs.tostr())
                elif isinstance(lhs, F23.Part_Ref):
                    if lhs.children[0].tostr() not in self.var_in_local:
                        self.var_modif.add(lhs.children[0].tostr())
                else:
                    raise ValueError(f"Unexpected assignment left-hand side type: {type(lhs)} in statement: {assign_stmt.tostr()}")
    
    def find_global_variables(self, module_dir, module_tree, var_global):
        for declaration in var_global:
            self.finder = Navigator(module_dir, module_tree)
            if declaration not in self.external_subroutines:
                sys.stdout.write('\r' + '\033[32m' + 'Searching for variable: ' + declaration + ' ... ⏳' + '\033[0m\n')
                sys.stdout.flush()
                self.finder.variable_finder(declaration)
                sys.stdout.write('\r' + '')
                sys.stdout.flush()
                if self.finder.var_declaration:
                    print('\033[32m' + '✅ Variable found!' + '\033[0m\n')
                    self.dec_global[declaration] = [item for item in self.finder.var_declaration]
                else:
                    raise ValueError(f"Variable '{declaration}' is not found in any child modules.")
                if self.finder.var_initial:
                    self.dec_child[declaration] = self.finder.var_initial
                    print('\033[91m' + 'Attention: there are additional to search:', self.finder.var_initial)
                    print('\033[91m' + 'in the directory:', self.finder.module_dir_sc)
                    ffile = walk(self.finder.module_tree_sc, F23.Name)[0].string
                    print('\033[91m' + 'in the module', ffile)
                    self.find_global_variables(self.finder.module_dir_sc, self.finder.module_tree_sc, self.finder.var_initial)
            elif declaration in self.external_subroutines:
                sys.stdout.write('\r' + '\033[32m' + 'Searching for procedure: ' + declaration + ' ... ⏳' + '\033[0m\n')
                sys.stdout.flush()
                self.finder.external_subroutine_finder(declaration)
                sys.stdout.write('\r' + '')
                sys.stdout.flush()
                if self.finder.var_declaration:
                    print('\033[32m' + '✅ Procedure found!' + '\033[0m\n')
                    self.dec_global[declaration] = [item for item in self.finder.var_declaration]
                else:
                    raise ValueError(f"Procedure '{declaration}' is not found in any child modules.")

    def extract_array_info(self, dec_global, var_dummy_list):
        all_array_info = {}
        normalized_items = []
        self.var_modif_info = {}
        for key in dec_global:
            for item in dec_global[key]:
                is_var_modified = False
                if isinstance(item, F23.Type_Declaration_Stmt):
                    var_type = item.children[0].children[0]
                    entity_decls = walk(item, F23.Entity_Decl)
                    assert len(entity_decls) == 1,\
                            "In extract_array_info: walk(item, F23.Entity_Decl)=1, but got a different number."
                    entity_decl = entity_decls[0].tostr()
                    if entity_decl in self.var_modif:
                        is_var_modified = True
                        self.var_modif_info[entity_decl] = []
                        self.var_modif_info[entity_decl].append(var_type)
                    attr_spec = walk(item, F23.Attr_Spec)
                    if walk(item, F23.Explicit_Shape_Spec):
                        normalized_items.append(item)
                        if is_var_modified:
                            self.var_modif_info[entity_decl].append('DIMENSION')
                    if F23.Attr_Spec('ALLOCATABLE') in attr_spec:
                        declaration_stmt = Processor().combine_allocate_declaration(dec_global[key])
                        assert isinstance(declaration_stmt, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
                        assert walk(declaration_stmt, F23.Explicit_Shape_Spec), "In extract_array_info: failed to combine_allocate_declaration!"
                        normalized_items.append(declaration_stmt)
                        if is_var_modified:
                            self.var_modif_info[entity_decl].append('DIMENSION')

        for item in var_dummy_list:
            is_var_modified = False
            assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
            var_type = item.children[0].children[0]
            entity_decls = walk(item, F23.Entity_Decl)
            assert len(entity_decls) == 1,\
                    "In extract_array_info: walk(item, F23.Entity_Decl)=1, but got a different number."
            entity_decl = entity_decls[0].tostr()
            if entity_decl in self.var_modif:
                is_var_modified = True
                self.var_modif_info[entity_decl] = []
                self.var_modif_info[entity_decl].append(var_type)
            if walk(item, F23.Explicit_Shape_Spec):
                normalized_items.append(item)
                if is_var_modified:
                    self.var_modif_info[entity_decl].append('DIMENSION')
        for item in self.var_local:
            assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
            if walk(item, F23.Explicit_Shape_Spec):
                normalized_items.append(item)

        for item in normalized_items:
            current_var_info = []
            array_name = walk(item, F23.Entity_Decl)[0].tostr()
            for dim in walk(item, F23.Explicit_Shape_Spec):
                start_end = [part.strip() for part in dim.tostr().split(':')]
                lse = len(start_end)
                if lse == 1:
                    current_var_info.append({'dim_str': '1', 'dim_end': start_end[0]})
                elif lse == 2:
                    current_var_info.append({'dim_str': start_end[0], 'dim_end': start_end[1]})
                else:
                    raise ValueError("dimension control error!")
            all_array_info[array_name] = [part for part in current_var_info]

        return all_array_info

    def process_declaration_variables(self, items):
        for item in items:
            dec_stmt = isinstance(item, F23.Type_Declaration_Stmt)
            alo_stmt = isinstance(item, F23.Allocate_Stmt)
            use_stmt = isinstance(item, F23.Use_Stmt)
            if use_stmt:
                continue
            if not dec_stmt and not alo_stmt:
                raise ValueError("Item is neither Type_Declaration_Stmt nor Allocate_Stmt!")
            else:
                shape = walk(walk(item, F23.Allocate_Shape_Spec), F23.Name) if alo_stmt \
                        else walk(walk(item, F23.Explicit_Shape_Spec), F23.Name)
                if shape:
                    self.shapes_variables.update(name.string for name in shape if name.string not in self.exclude)
                else:
                    assert dec_stmt, 'The scalar must be a Type_Declaration_Stmt!'
                    array = walk(item, F23.Dimension_Attr_Spec)
                    if not array:
                        name = walk(walk(item, F23.Entity_Decl), F23.Name)
                        if name[0].string not in self.exclude:
                            self.scalar_variables.add(name[0].string)
