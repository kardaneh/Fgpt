import os, sys
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from collections import deque
from processor import Processor
from navigator import Navigator

class Shaper:
    """
    A class to find and analyze implicit array shapes within Fortran modules and subroutines.

    Attributes:
        module_dir_imp (str): Directory containing Fortran modules.
        dummy_arg_list (dict): Dictionary mapping subroutine names to dummy argument lists.
        actual_arg_spec_list (dict): Dictionary storing actual argument specifications for subroutines.
        call_subroutines (dict): Dictionary mapping subroutines to their corresponding call statements.
        fortran_file_queue_imp (deque): Queue for processing Fortran files.
        current_module_imp (str): The currently processed module.
        module_tree_imp (object): The abstract syntax tree of the currently processed module.
    """

    def __init__(self, module_dir, dummy_arg_list, actual_arg_spec_list, call_subroutines, parsed_modules):
        """
        Initialize the Shaper with the given directory and argument lists.

        Args:
            module_dir (str): Directory containing Fortran modules.
            dummy_arg_list (dict): Dictionary of subroutine dummy arguments.
            actual_arg_spec_list (dict): Dictionary of actual argument specifications.
            call_subroutines (dict): Dictionary of subroutine call statements.
        """
        self.module_dir_imp = module_dir
        self.dummy_arg_list = dummy_arg_list
        self.actual_arg_spec_list = actual_arg_spec_list
        self.call_subroutines = call_subroutines
        self.fortran_file_queue_imp = deque()
        self.current_module_imp = ''
        self.module_tree_imp = None
        self.parsed_modules = parsed_modules

    def find_fortran_files(self, subroutine_name):
        """
        Find and process Fortran files in the module directory to locate a specific subroutine.

        Args:
            subroutine_name (str): The name of the subroutine to search for.

        Raises:
            FileNotFoundError: If no matching subroutine is found in the provided files.
        """
        try:
            for file in os.listdir(self.module_dir_imp):
                if file.endswith(('.f90', '.F90')):
                    file_base, _ = os.path.splitext(file)
                    if file_base != self.current_module_imp:
                        module_file_path = os.path.join(self.module_dir_imp, file)
                        self.fortran_file_queue_imp.append(module_file_path)
            
            while self.fortran_file_queue_imp:
                module_file_path = self.fortran_file_queue_imp.popleft()
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
                            if call_name == subroutine_name:
                                print(f'\033[32mSubroutine "{call_name}" is called inside module "{file_base}"!\033[0m')
                                
                                if call_name not in self.actual_arg_spec_list:
                                    self.actual_arg_spec_list[call_name] = []
                                    self.call_subroutines[call_name] = []
                                
                                arg_list = walk(item, F23.Actual_Arg_Spec_List)
                                if arg_list:
                                    arg_string = [string.strip() for string in arg_list[0].tostr().split(',')]
                                    self.actual_arg_spec_list[call_name].append(arg_string)
                                
                                self.call_subroutines[call_name].append(item)
                                return
            
            raise FileNotFoundError(f"Subroutine '{subroutine_name}' not found in any Fortran files within {self.module_dir_imp}")
        
        except Exception as e:
            raise RuntimeError(f"An error occurred while searching for Fortran files in method 'find_fortran_files': {e}")

    def find_implicit_shape(self, node, subroutine_name):
        """
        Locate the implicit shape of an array declared within a subroutine.

        Args:
            node (object): AST node representing the subroutine or declaration.
            subroutine_name (str): The name of the subroutine to search in.

        Returns:
            object: The declaration statement with the explicit or implicit array shape.

        Raises:
            ValueError: If an implicit array with an unknown shape is declared locally.
        """
        try:
            name = walk(walk(node, F23.Entity_Decl), F23.Name)[0].string
            
            if name not in self.dummy_arg_list[subroutine_name]:
                raise ValueError(f"An implicit array '{name}' with an unknown shape is declared locally!")
            
            if subroutine_name not in self.actual_arg_spec_list or subroutine_name not in self.call_subroutines:
                self.current_module_imp = walk(node.get_root(), F23.Name)[0].string
                print(f"The subroutine '{subroutine_name}' is called outside of the module '{self.current_module_imp}'. Searching the module...")
                self.find_fortran_files(subroutine_name)
            
            for arg, call in zip(self.actual_arg_spec_list[subroutine_name], self.call_subroutines[subroutine_name]):
                corresponding_element = arg[self.dummy_arg_list[subroutine_name].index(name)]
                enclosing_subroutine = self.find_enclosing_subroutine(call)
                subroutine_key = walk(walk(enclosing_subroutine, F23.Subroutine_Stmt), F23.Name)[0].string
                declaration_part = walk(enclosing_subroutine, F23.Specification_Part)
                
                print(f'\033[32mCorresponding element of "{name}" is "{corresponding_element}" in call statement in subroutine "{subroutine_key}"!\033[0m')
                
                if declaration_part:
                    for decl in walk(declaration_part, F23.Type_Declaration_Stmt):
                        declarations = [name.string for name in walk(decl, F23.Entity_Decl)]
                        if corresponding_element in declarations:
                            explicit_shape = walk(decl, F23.Explicit_Shape_Spec)
                            if explicit_shape:
                                print(f'\033[32mFound explicit shape in subroutine "{subroutine_key}"!\033[0m')
                                return decl
                            else:
                                return self.find_implicit_shape(decl, subroutine_key)
            
            self.module_tree_imp = call.get_root()
            module_name = walk(self.module_tree_imp, F23.Name)[0].string
            print(f'"{corresponding_element}" is not a dummy argument in subroutine "{subroutine_key}". Searching in module "{module_name}"...')
            self.finder = Navigator(self.module_dir_imp, self.module_tree_imp, self.parsed_modules)
            self.finder.variable_finder(corresponding_element)
            return Processor().combine_allocate_declaration(self.finder.var_declaration)
        
        except Exception as e:
            raise RuntimeError(f"An error occurred in method 'find_implicit_shape': {e}")

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
