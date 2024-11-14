import os, sys
from processor import Processor
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from navigator import Navigator
from shaper import Shaper
import re

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
            self.general_usage_dict = {}
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

    def extract_intent(self, subroutine_key, subroutine_tree, within_calls=None):
        """
        """
        dummy_arg_list = self.dummy_arg_list[subroutine_key]
        usage = {arg: {'intent': None, 'first_use_parent': None} for arg in dummy_arg_list}
        def traverse_block(block):
            if hasattr(block, "content"):
                for child in block.content:
                    if isinstance(child, F23.Assignment_Stmt):
                        lhs_expr = child.items[0].tostr()
                        rhs_expr = child.items[-1].tostr()
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            pattern = r'\b' + re.escape(var_name) + r'\b'
                            if var_name in dummy_arg_list:
                                if isinstance(name.parent, F23.Section_Subscript_List):
                                    if usage[var_name]['intent'] is None:
                                        usage[var_name]['intent'] = 'IN'
                                    if usage[var_name]['first_use_parent'] is None:
                                        usage[var_name]['first_use_parent'] = child
                                else:
                                    if re.search(pattern, lhs_expr):
                                        if usage[var_name]['intent'] is None:
                                            usage[var_name]['intent'] = 'OUT'
                                        elif usage[var_name]['intent'] == 'IN':
                                            usage[var_name]['intent'] = 'INOUT'
                                        if usage[var_name]['first_use_parent'] is None:
                                            usage[var_name]['first_use_parent'] = child

                                    if re.search(pattern, rhs_expr):
                                        if usage[var_name]['intent'] is None:
                                            usage[var_name]['intent'] = 'IN'
                                        elif usage[var_name]['intent'] == 'OUT' and usage[var_name]['first_use_parent'] == child:
                                            usage[var_name]['intent'] = 'INOUT'
                                        if usage[var_name]['first_use_parent'] is None:
                                            usage[var_name]['first_use_parent'] = child

                    elif isinstance(child, F23.Nonlabel_Do_Stmt):
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            if var_name in dummy_arg_list:
                                if usage[var_name]['intent'] is None:
                                    usage[var_name]['intent'] = 'IN'
                                if usage[var_name]['first_use_parent'] is None:
                                    usage[var_name]['first_use_parent'] = child

                    elif isinstance(child, (F23.If_Then_Stmt, F23.Else_If_Stmt)):
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            if var_name in dummy_arg_list:
                                if usage[var_name]['intent'] is None:
                                    usage[var_name]['intent'] = 'IN'
                                if usage[var_name]['first_use_parent'] is None:
                                    usage[var_name]['first_use_parent'] = child

                    elif isinstance(child, (F23.Where_Construct_Stmt, F23.Masked_Elsewhere_Stmt)):
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            if var_name in dummy_arg_list:
                                if usage[var_name]['intent'] is None:
                                    usage[var_name]['intent'] = 'IN'
                                if usage[var_name]['first_use_parent'] is None:
                                    usage[var_name]['first_use_parent'] = child

                    elif isinstance(child, F23.Call_Stmt):
                        call_name = None
                        if child.children[0].tostr() not in self.allowed_external_subroutines:
                            for grandchild in child.children:
                                if grandchild is None:
                                    continue
                                if isinstance(grandchild, F23.Name):
                                    call_name = grandchild.tostr()
                                    assert call_name in within_calls, f"Error: {call_name} not found in within_calls"
                                    assert call_name in self.general_usage_dict, f"Error: {call_name} not found in self.general_usage_dict"
                                elif isinstance(grandchild, F23.Actual_Arg_Spec_List):
                                    assert call_name is not None, 'call_name is not defined yet'
                                    for name in grandchild.children:
                                        if name.tostr() in dummy_arg_list:
                                            var_name = name.tostr()
                                            current_intent = usage[var_name]['intent']
                                            call_intent = self.general_usage_dict[call_name][var_name]
                                            if current_intent is None:
                                                usage[var_name]['intent'] = call_intent
                                            elif current_intent == 'IN' and call_intent in {'OUT', 'INOUT'}:
                                                usage[var_name]['intent'] ='INOUT'
                                            elif current_intent == 'OUT' and call_intent == 'INOUT':
                                                usage[var_name]['intent'] == 'INOUT'

                                            if usage[var_name]['first_use_parent'] is None:
                                                usage[var_name]['first_use_parent'] = child
                    else:
                        traverse_block(child)
        execution_part = walk(subroutine_tree, F23.Execution_Part)[0]
        traverse_block(execution_part)
        self.general_usage_dict[subroutine_key] = {var: props['intent'] for var, props in usage.items()}

    @staticmethod
    def add_intent(block, intent):
        """
        Adds intent to a Fortran block by extracting the intrinsic type, explicit shape, and entity declaration list.

        Parameters:
            block: The Fortran block to traverse.
            intent (str): The intent specifier (e.g., "IN", "OUT", "INOUT").

        Returns:
            F23.Type_Declaration_Stmt: A Type Declaration Statement with the specified intent.
        """
        assert isinstance(block, F23.Type_Declaration_Stmt), (
            f"Expected block to be of type 'F23.Type_Declaration_Stmt', "
            f"but got '{type(block).__name__}' instead."
        )
        intrinsic_type_spec = None
        explicit_shape_spec_list = None
        entity_decl_list = None

        def traverse_block(block):
            nonlocal intrinsic_type_spec, explicit_shape_spec_list, entity_decl_list
            if hasattr(block, "children"):
                for child in block.children:
                    if isinstance(child, F23.Intrinsic_Type_Spec):
                        intrinsic_type_spec = child.tostr()
                    elif isinstance(child, F23.Explicit_Shape_Spec_List):
                        explicit_shape_spec_list = child.tostr()
                    elif isinstance(child, F23.Entity_Decl_List):
                        entity_decl_list = child.tostr()
                    else:
                        traverse_block(child)

        traverse_block(block)
        return F23.Type_Declaration_Stmt(
            f'{intrinsic_type_spec},dimension({explicit_shape_spec_list}),intent({intent})::{entity_decl_list}'
        )

    def clean_subroutine(self, subroutine_key, subroutine_tree):
        def traverse_subroutine(block):
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    child = block.content[idc]
                    if isinstance(child, F23.Type_Declaration_Stmt):
                        intent = walk(child, F23.Intent_Spec)
                        if intent:
                            intent_spec = intent[0].tostr()
                        if len(walk(child, F23.Entity_Decl)) > 1:
                            for stmt in Processor().separate_entity_declarations(child):
                                entity_decls = walk(stmt, F23.Entity_Decl)
                                assert len(entity_decls) == 1,\
                                        "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                                name = entity_decls[0].tostr()
                                if intent:
                                    intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                    if intent_spec_exp is None:
                                        print('\033[38;5;214m' + "Warning: Name %s is not used %s."%(name, stmt.tostr()) + '\033[0m')
                                    else:
                                        if intent_spec_exp != intent_spec:
                                            print('\033[38;5;214m' + "Warning: The intent is incorrect. Correction block" + '\033[0m')
                                            print('\033[38;5;214m' + "Name:%s, Expected:%s, Found: %s"%(name, intent_spec_exp, intent_spec) + '\033[0m')
                                            obj_org = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec)
                                            obj_mod = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec_exp)
                                            print('\033[38;5;214m' + "Original Declaration Statement: %s"%(stmt.tostr()) + '\033[0m')
                                            child_string = stmt.tostr().replace(obj_org.tostr(), obj_mod.tostr())
                                            stmt = F23.Type_Declaration_Stmt(child_string)
                                            print('\033[32m' + 'Modified Declaration Statement: %s'%child_string + '\033[0m')
                                else:
                                    if name in self.dummy_arg_list[subroutine_key]:
                                        print('\033[38;5;214m' + "Warning: Name %s is a dummy arguments without intent."%(name)+'\033[0m')
                                        print('\033[38;5;214m' + "Original Declaration Statement: %s"%(stmt.tostr()) + '\033[0m')
                                        intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                        if intent_spec_exp is not None:
                                            print('\033[38;5;214m' + "The expected intent is :  %s"%intent_spec_exp + '\033[0m')
                                            stmt = self.add_intent(stmt, intent_spec_exp)
                                            print('\033[32m' + 'Modified Declaration Statement: %s'%(stmt.tostr()) + '\033[0m')
                                        else:
                                            print('\033[38;5;214m' + "Warning: Name %s is not used %s."%(name, stmt.tostr()) + '\033[0m')
                                block.content.insert(idc + 1, stmt)
                            del block.content[idc]
                        else:
                            entity_decls = walk(child, F23.Entity_Decl)
                            assert len(entity_decls) == 1,\
                                    "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                            name = entity_decls[0].tostr()
                            if intent:
                                intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                if intent_spec_exp is None:
                                    print('\033[38;5;214m' + "Warning: Name %s is not used %s."%(name, child.tostr()) + '\033[0m')
                                else:
                                    if intent_spec_exp != intent_spec:
                                        print('\033[38;5;214m' + "Warning: incorrect intent for %s. Expected : %s, Found : %s. Correct it!" \
                                                %(name, intent_spec_exp, intent_spec) + '\033[0m')
                                        obj_org = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec)
                                        obj_mod = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec_exp)
                                        print('\033[38;5;214m' + "Original Declaration Statement: %s"%(child.tostr()) + '\033[0m')
                                        child_string = child.tostr().replace(obj_org.tostr(), obj_mod.tostr())
                                        block.content[idc] = F23.Type_Declaration_Stmt(child_string)
                                        print('\033[32m' + 'Modified Declaration Statement: %s'%child_string + '\033[0m')
                            else:
                                if name in self.dummy_arg_list[subroutine_key]:
                                    print('\033[38;5;214m' + "Warning: Name %s is a dummy arguments without intent."%(name)+'\033[0m')
                                    print('\033[38;5;214m' + "Original Declaration Statement: %s"%(child.tostr()) + '\033[0m')
                                    intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                    if intent_spec_exp is not None:
                                        print('\033[38;5;214m' + "Its expected intent is :  %s"%intent_spec_exp + '\033[0m')
                                        block.content[idc] = self.add_intent(child, intent_spec_exp)
                                        print('\033[32m' + 'Modified Declaration Statement: %s'%(block.content[idc].tostr()) + '\033[0m')
                                    else:
                                        print('\033[38;5;214m' + "Warning: Name %s is not used %s."%(name, child.tostr()) + '\033[0m')
                    else:
                        traverse_subroutine(child)
                    idc += 1
        traverse_subroutine(subroutine_tree)

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
                entity_decls = walk(node, F23.Entity_Decl)
                assert len(entity_decls) == 1,\
                        "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                name = entity_decls[0].tostr()
                if intent:
                    intent_spec = intent[0].tostr()
                    if walk(walk(node,F23.Entity_Decl),F23.Name)[0].string not in self.exclude:
                        self.var_dummy.append(node)
                        if F23.Intent_Attr_Spec('INTENT(IN)') in walk(node, F23.Intent_Attr_Spec):
                            #self.var_in_local = {name.tostr() for name in  walk(node, F23.Entity_Decl)}
                            for name in  walk(node, F23.Entity_Decl):
                                self.var_in_local.add(name.string)
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
