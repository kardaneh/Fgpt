import os, sys
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from collections import deque
from processor import Processor
from navigator import Navigator

class Shaper:
    """
    """

    def __init__(self, module_dir, parsed_modules, dummy_arg_list, actual_arg_spec_list=None, call_subroutines=None):
        """
        """
        self.module_dir_imp = module_dir
        self.dummy_arg_list = dummy_arg_list
        self.actual_arg_spec_list = actual_arg_spec_list
        self.call_subroutines = call_subroutines
        self.current_module_imp = ''
        self.module_tree_imp = None
        self.parsed_modules = parsed_modules

    def find_fortran_files_subroutine(self, subroutine_key):
        """
        """
        try:
            fortran_file_queue_imp = deque()
            for file in os.listdir(self.module_dir_imp):
                if file.endswith(('.f90', '.F90')):
                    file_base, _ = os.path.splitext(file)
                    if file_base != self.current_module_imp:
                        module_file_path = os.path.join(self.module_dir_imp, file)
                        fortran_file_queue_imp.append(module_file_path)
            
            while fortran_file_queue_imp:
                module_file_path = fortran_file_queue_imp.popleft()
                module_file_name = os.path.basename(module_file_path)
                module_name, _ = os.path.splitext(module_file_name)
                if module_name in self.parsed_modules:
                    self.module_tree_imp = self.parsed_modules[module_name]
                else:
                    self.module_tree_imp = Processor().parse_fortran_file(module_file_path)
                    self.parsed_modules[module_name] = self.module_tree_imp
                
                for sub in walk(self.module_tree_imp, F23.Subroutine_Subprogram):
                    call_stmt = walk(sub, F23.Call_Stmt)
                    if call_stmt:
                        for item in call_stmt:
                            call_name = walk(item, F23.Name)[0].string
                            if call_name == subroutine_key:
                                print(f'\033[32mSubroutine "{call_name}" is called inside module "{file_base}"!\033[0m')
                                arg_list = walk(item, F23.Actual_Arg_Spec_List)
                                if arg_list:
                                    arg_string = [string.strip() for string in arg_list[0].tostr().split(',')]
                                    self.actual_arg_spec_list[call_name].append(arg_string)
                                
                                self.call_subroutines[call_name].append(item)
                                return
            
            raise FileNotFoundError(f"Subroutine '{subroutine_key}' not found in any Fortran files within {self.module_dir_imp}")
        
        except Exception as e:
            raise RuntimeError(f"An error occurred while searching for Fortran files in method 'find_fortran_files_subroutine': {e}")

    def shaper_subroutine(self, node, subroutine_key):
        """
        """
        try:
            assert self.actual_arg_spec_list is not None, "Error: actual_arg_spec_list is None!"
            assert self.call_subroutines is not None, "Error: call_subroutines is None!"
            name = walk(walk(node, F23.Entity_Decl), F23.Name)[0].string
            if name not in self.dummy_arg_list[subroutine_key]:
                raise ValueError(f"An implicit array '{name}' with an unknown shape is declared locally!")
            
            if subroutine_key not in self.actual_arg_spec_list or subroutine_key not in self.call_subroutines:
                self.current_module_imp = walk(node.get_root(), F23.Name)[0].string
                print(f"The subroutine '{subroutine_key}' is called outside of the module '{self.current_module_imp}'. Searching the module...")
                self.find_fortran_files_subroutine(subroutine_key)
            
            for arg, call in zip(self.actual_arg_spec_list[subroutine_key], self.call_subroutines[subroutine_key]):
                act_arg = arg[self.dummy_arg_list[subroutine_key].index(name)]
                enclosing_subroutine = self.find_enclosing_subroutine(call)
                subroutine_key = walk(walk(enclosing_subroutine, F23.Subroutine_Stmt), F23.Name)[0].string
                declaration_part = walk(enclosing_subroutine, F23.Specification_Part)
                
                print(f'\033[32mCorresponding element of "{name}" is "{act_arg}" in call statement in subroutine "{subroutine_key}"!\033[0m')
                
                if declaration_part:
                    for decl in walk(declaration_part, F23.Type_Declaration_Stmt):
                        declarations = [name.string for name in walk(decl, F23.Entity_Decl)]
                        if act_arg in declarations:
                            explicit_shape = walk(decl, F23.Explicit_Shape_Spec)
                            if explicit_shape:
                                print(f'\033[32mFound explicit shape in subroutine "{subroutine_key}"!\033[0m')
                                return decl
                            else:
                                return self.shaper_subroutine(decl, subroutine_key)
            
            self.module_tree_imp = call.get_root()
            module_name = walk(self.module_tree_imp, F23.Name)[0].string
            print(f'"{act_arg}" is not a dummy argument in subroutine "{subroutine_key}". Searching in module "{module_name}"...')
            self.finder = Navigator(self.module_dir_imp, self.module_tree_imp, self.parsed_modules)
            self.finder.variable_finder(act_arg)
            return Processor().combine_allocate_declaration(self.finder.var_declaration)
        
        except Exception as e:
            raise RuntimeError(f"An error occurred in method 'shaper_subroutine': {e}")

    def shaper_function(self, node, function_tree, function_key, all_array_info):
        """
        """
        try:
            self.module_tree_imp = node.get_root()
            self.current_module_imp = walk(self.module_tree_imp, F23.Name)[0].string
            last_processed_module_dir = None
            function_assignment_stmt = None
            fortran_file_queue_imp = deque()

            name =  walk(node, F23.Entity_Decl)[0].tostr()
            if name not in self.dummy_arg_list[function_key]:
                raise ValueError(f"An implicit array '{name}' with an unknown shape is declared locally!")

            while function_assignment_stmt is None:
                for assignment_stmt in walk(self.module_tree_imp, F23.Assignment_Stmt):
                    for part_ref in walk(assignment_stmt, F23.Part_Ref):
                        for child in part_ref.children:
                            if isinstance(child, F23.Name):
                                if child.tostr() == function_key:
                                    function_assignment_stmt = True
                            if isinstance(child, F23.Section_Subscript_List):
                                if function_assignment_stmt:
                                    act_arg_list = child
                        if function_assignment_stmt:
                            act_arg = act_arg_list.children[self.dummy_arg_list[function_key].index(name)]
                            if ':' in act_arg.tostr() and isinstance(act_arg, F23.Part_Ref):
                                array_name = act_arg.children[0].tostr()
                                dims = act_arg.children[1].children
                                shape = []
                                assert array_name in all_array_info, (
                                        f"Error in modify_colon_array_vec: Array '{array_name}' not present in all_array_info."
                                        )
                                array_info = all_array_info[array_name]
                                for idim, dim in enumerate(dims):
                                    if dim.tostr() == ':':
                                        lb = array_info[idim]['dim_str']
                                        ub = array_info[idim]['dim_end']
                                        shape.append(f'{lb}:{ub}')
                            elif isinstance(act_arg, F23.Name):
                                array_name = act_arg.tostr()
                                assert array_name in all_array_info, (
                                        f"Error in modify_colon_array_vec: Array '{array_name}' not present in all_array_info."
                                        )
                                array_info = all_array_info[array_name]
                                shape = []
                                for idim in range(len(array_info)):
                                    lb = array_info[idim]['dim_str']
                                    ub = array_info[idim]['dim_end']
                                    shape.append(f'{lb}:{ub}')
                            else:
                                raise TypeError(
                                        f"Unexpected type for act_arg: {type(act_arg).__name__}. "
                                        "Expected 'F23.Part_Ref' or 'F23.Name'.")
                            dimensions = ', '.join([name for name in shape])
                            new_dec = Processor().map_declaration(node, explicit_dec=None, dimensions=dimensions)
                            break
                    if function_assignment_stmt:
                        break
                if function_assignment_stmt is None:
                    if self.module_dir_imp != last_processed_module_dir:
                        last_processed_module_dir = self.module_dir_imp
                        for file in os.listdir(self.module_dir_imp):
                            if file.endswith(('.f90', '.F90')):
                                file_base, _ = os.path.splitext(file)
                                if file_base != self.current_module_imp:
                                    module_file_path = os.path.join(self.module_dir_imp, file)
                                    fortran_file_queue_imp.append(module_file_path)
                    if fortran_file_queue_imp:
                        module_file_path = fortran_file_queue_imp.popleft()
                        module_file_name = os.path.basename(module_file_path)
                        module_name, _ = os.path.splitext(module_file_name)
                        if module_name in self.parsed_modules:
                            self.module_tree_imp = self.parsed_modules[module_name]
                        else:
                            self.module_tree_imp = Processor().parse_fortran_file(module_file_path)
                            self.parsed_modules[module_name] = self.module_tree_imp
                    else:
                        raise ValueError(
                                f"Function key '{function_key}' not found in any module, need to go to higher directory.")
            
            for child in function_tree.children:
                if isinstance(child, F23.Specification_Part):
                    for igc,gchild in enumerate(child.children):
                        if isinstance(gchild, F23.Type_Declaration_Stmt):
                            entity_decl = walk(gchild, F23.Entity_Decl)[0].tostr()
                            if entity_decl == name:
                                child.children[child.children.index(gchild)] = new_dec
            #return new_dec
        except Exception as e:
            raise RuntimeError(f"An error occurred in method 'shaper_subroutine': {e}")

    def shaper_intrinsic_size(self, node):
        """
        """
        try:
            shape = []
            name =  walk(node, F23.Entity_Decl)[0].tostr()
            for explicit_shape in walk(node, F23.Explicit_Shape_Spec):
                intrinsics = walk(explicit_shape, F23.Intrinsic_Name)
                if not intrinsics:
                    shape.append(explicit_shape.tostr())
                else:
                    intrinsic = intrinsics[0]
                    if intrinsic.tostr() == 'SIZE':
                        dim_value = None
                        intrinsic_parent = intrinsic.parent
                        assert len(intrinsic_parent.children) == 2, "Intrinsic parent must have exactly two children."
                        assert intrinsic_parent.children[0] == intrinsic, "First child must be the intrinsic function itself."
                        intrinsic_args = intrinsic.parent.children[1]
                        assert isinstance(intrinsic_args, F23.Actual_Arg_Spec_List), "Second child must be an Actual_Arg_Spec_List."
                        args = intrinsic_args.children
                        if len(args) == 1:
                            print(f"SIZE for {args[0].tostr()} without an explicit 'DIM'. "
                                    "This implies sum of all dimensions.")
                            dim_value = 'ALL'
                        if len(args) == 2:
                            if isinstance(args[1], F23.Actual_Arg_Spec):
                                dim_key, dim_value_node = args[1].items
                                assert isinstance(dim_key, F23.Name), "First item in Actual_Arg_Spec must be a Name."
                                if dim_key.tostr().lower() == 'dim':
                                    assert isinstance(dim_value_node, F23.Int_Literal_Constant), "Second item in must be an Int_Literal_Constant."
                                    dim_value = dim_value_node.tostr()
                            elif isinstance(args[1], F23.Int_Literal_Constant):
                                raise ValueError(
                                        f"Unexpected argument structure in intrinsic. Expected 'dim={args[1].tostr()}' but found: {args[1].tostr()}")
                            else:
                                raise ValueError("Unexpected structure for intrinsic arguments.")
                        
                        assert isinstance(node.parent, F23.Specification_Part), \
                                f"Expected node's parent to be of type F23.Specification_Part, but got {type(node.parent).__name__} instead."
                        for declaration_stmt in walk(node.parent, F23.Type_Declaration_Stmt):
                            entity_decls = walk(declaration_stmt, F23.Entity_Decl)
                            assert len(entity_decls) == 1,"walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                            if entity_decls[0].tostr() == args[0].tostr():
                                print('the shape is find')
                                size = '1'
                                for idim, explicit_shape in enumerate(walk(declaration_stmt, F23.Explicit_Shape_Spec), start=1):
                                    parts = [part.strip() for part in explicit_shape.tostr().split(':')]
                                    if len(parts) == 2:
                                        dim_size = '(' + parts[1] + '-' + parts[0] + '+1' + ')'
                                    elif len(parts) == 1:
                                        dim_size = parts[0]
                                    else:
                                        raise ValueError(f"Unexpected number of parts in explicit shape: {len(parts)}. Parts: {parts}")
                                    if dim_value != 'ALL':
                                        if idim == int(dim_value):
                                            size = dim_size
                                    else:
                                        size = dim_size + '*' + size
                        shape.append(size)
                    else:
                        raise ValueError(f"Unexpected intrinsic function: {intrinsic.tostr()}")
            dimensions = ', '.join([name for name in shape])
            new_dec = Processor().map_declaration(node, explicit_dec=None, dimensions=dimensions)
            for igc,gchild in enumerate(node.parent.children):
                if isinstance(gchild, F23.Type_Declaration_Stmt):
                    entity_decl = walk(gchild, F23.Entity_Decl)[0].tostr()
                    if entity_decl == name:
                        node.parent.children[node.parent.children.index(gchild)] = new_dec
        except Exception as e:
            raise RuntimeError(f"An error occurred in method 'find_enclosing_subroutine': {e}")


    def find_enclosing_subroutine(self, node):
        """
        Find the enclosing subroutine for a given AST node.

        Args:
            node (object): The AST node to search from.

        Returns:
            object: The enclosing subroutine node, or None if not found.
        """
        try:
            while node is not None:
                if isinstance(node, F23.Subroutine_Subprogram):
                    return node
                node = getattr(node, 'parent', None)
            return None
        
        except Exception as e:
            raise RuntimeError(f"An error occurred in method 'find_enclosing_subroutine': {e}")
