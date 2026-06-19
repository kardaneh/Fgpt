import os, sys
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from collections import deque
from processor import Processor
from navigator import Navigator
import unittest
import tempfile
import shutil
from collections import defaultdict

class Shaper:
    """
    A shape resolver for Fortran arrays that determines explicit shapes from implicit declarations.
    
    This class analyzes Fortran code to resolve implicit array shapes (: notation) by tracing
    through subroutine calls and function assignments to find the corresponding explicit
    declarations. It handles complex call hierarchies and intrinsic functions like SIZE.
    
    Parameters
    ----------
    module_dir : str
        Directory containing the Fortran module files
    parsed_modules : dict
        Dictionary of pre-parsed modules for efficient lookup
    dummy_arg_list : defaultdict
        Dictionary mapping subroutine names to their dummy argument lists
    actual_arg_spec_list : defaultdict, optional
        Dictionary mapping subroutine names to actual argument specifications
    call_subroutines : defaultdict, optional
        Dictionary mapping subroutine names to their call statement nodes
    
    Attributes
    ----------
    module_dir_imp : str
        Directory containing Fortran module files
    dummy_arg_list : defaultdict
        Dictionary of subroutine dummy arguments
    actual_arg_spec_list : defaultdict
        Dictionary of actual argument specifications
    call_subroutines : defaultdict
        Dictionary of subroutine call statements
    current_module_imp : str
        Current module being processed
    module_tree_imp : fparser.two.Fortran2003.Module
        Current parsed module tree
    parsed_modules : dict
        Dictionary of pre-parsed modules
    processor : Processor
        Processor instance for parsing and utility functions
    cases_to_exclude : list
        List of subroutine names to exclude from processing
    
    Methods
    -------
    find_fortran_files_subroutine(subroutine_key)
        Search Fortran files for a specific subroutine and its calls.
    shaper_subroutine(node, subroutine_key)
        Resolve implicit array shapes by tracing through subroutine calls.
    shaper_function(node, function_tree, function_key, all_array_info)
        Resolve implicit array shapes in function assignments.
    shaper_intrinsic_size(node)
        Handle SIZE intrinsic functions to determine array dimensions.
    find_enclosing_subroutine(node)
        Find the enclosing subroutine for a given AST node.
    
    Notes
    -----
    The class handles complex call hierarchies by performing breadth-first search
    through module dependencies. It can resolve shapes across multiple levels of
    subroutine calls and function assignments.
    
    Raises
    ------
    ValueError
        When unexpected AST node types are encountered
    AssertionError
        When expected conditions are not met during shape resolution
    RuntimeError
        When subroutines cannot be found or shapes cannot be resolved
    
    See Also
    --------
    Navigator : For module traversal and dependency analysis
    Processor : For Fortran parsing utilities
    """

    def __init__(self, module_dir, parsed_modules, module_path,dummy_arg_list, actual_arg_spec_list=None, call_subroutines=None, logger=None):
        """
        Initialize the Shaper with module directory, parsed modules, and argument mappings.
        
        Parameters
        ----------
        module_dir : str
            Directory containing the Fortran module files
        parsed_modules : dict
            Dictionary of pre-parsed modules for efficient lookup
        dummy_arg_list : defaultdict
            Dictionary mapping subroutine names to their dummy argument lists
        actual_arg_spec_list : defaultdict, optional
            Dictionary mapping subroutine names to actual argument specifications
        call_subroutines : defaultdict, optional
            Dictionary mapping subroutine names to their call statement nodes
        """
        self.module_dir_imp = module_dir
        self.dummy_arg_list = dummy_arg_list
        self.actual_arg_spec_list = actual_arg_spec_list
        self.call_subroutines = call_subroutines
        self.logger = logger
        self.current_module_imp = ''
        self.module_tree_imp = None
        self.parsed_modules = parsed_modules
        self.module_path = module_path
        self.processor = Processor(logger=self.logger)
        self.cases_to_exclude = ['clear', 'finalize', 'init', 'initialize', 'read']

    def find_fortran_files_subroutine(self, subroutine_key):
        """
        Search Fortran files for a specific subroutine and its call sites.
        
        This method searches through all Fortran files in the module directory to
        find where a subroutine is called. It populates the call information
        dictionaries with the found call statements and argument specifications.
        
        Parameters
        ----------
        subroutine_key : str
            The name of the subroutine to search for
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If the subroutine cannot be found in any Fortran files
        
        Notes
        -----
        - Searches all .f90 and .F90 files in the module directory
        - Excludes files matching the current module
        - Populates actual_arg_spec_list and call_subroutines dictionaries
        - Uses breadth-first search through the file queue
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
                    self.module_tree_imp = self.processor.parse_fortran_file(module_file_path)
                    self.parsed_modules[module_name] = self.module_tree_imp
                    self.module_path[module_name] = module_file_path
                
                for sub in walk(self.module_tree_imp, F23.Subroutine_Subprogram):
                    subroutine_name, arg_list = None, None
                    subroutine_stmt = walk(sub, F23.Subroutine_Stmt)[0]
                    call_stmt = walk(sub, F23.Call_Stmt)
                    for child in subroutine_stmt.children:
                        if child is None:
                            continue
                        if isinstance(child, F23.Name):
                            subroutine_name = child.tostr()
                        elif isinstance(child, F23.Dummy_Arg_List):
                            arg_list = child
                        else:
                            raise ValueError(f"Unexpected type '{type(child)}' encountered in children.")
                    assert subroutine_name is not None, f"Unexpected type {subroutine_name} encountered in children."
                    check = all(case not in subroutine_name for case in self.cases_to_exclude)
                    if not check:
                        continue
                    #self.subroutine_names_all.add(subroutine_name)
                    #self.subroutines[subroutine_name] = sub
                    if arg_list is not None :
                        if subroutine_name not in self.dummy_arg_list:
                            for child in arg_list.children:
                                self.dummy_arg_list[subroutine_name].append(child.tostr())

                    call_stmt = walk(sub, F23.Call_Stmt)
                    if call_stmt:
                        for item in call_stmt:
                            call_name = walk(item, F23.Name)[0].string
                            if call_name == subroutine_key:
                                self.processor.logger.info(f"Subroutine '{call_name}' is called inside module '{file_base}'!")
                                arg_list = walk(item, F23.Actual_Arg_Spec_List)
                                if arg_list:
                                    arg_string = [string.strip() for string in arg_list[0].tostr().split(',')]
                                    self.actual_arg_spec_list[call_name].append(arg_string)
                                
                                self.call_subroutines[call_name].append(item)
                                return
            
            self.processor.logger.error(f"Subroutine '{subroutine_key}' not found in any Fortran files within {self.module_dir_imp}")
            raise
        
        except Exception as e:
            self.processor.logger.exception(f"An error occurred while searching for Fortran files in method 'find_fortran_files_subroutine': ", e)
            raise

    def shaper_subroutine(self, node, subroutine_key):
        """
        Resolve implicit array shapes by tracing through subroutine call hierarchies.
        
        This method takes an implicit array declaration and traces through subroutine
        calls to find the corresponding explicit shape declaration. It handles
        multiple levels of call nesting and module dependencies.
        
        Parameters
        ----------
        node : fparser.two.Fortran2003.Type_Declaration_Stmt
            The implicit array declaration node to resolve
        subroutine_key : str
            The name of the subroutine where the array is used
        
        Returns
        -------
        fparser.two.Fortran2003.Type_Declaration_Stmt
            The resolved explicit array declaration
        
        Raises
        ------
        ValueError
            If the array is not found in dummy argument lists
        Exception
            If shape resolution fails at any level
        
        Notes
        -----
        - Traces through actual argument specifications to find explicit shapes
        - Uses Navigator for module-level variable resolution when needed
        - Handles both direct and nested subroutine calls
        - Logs the resolution path for debugging
        """
        try:
            assert self.actual_arg_spec_list is not None, "Error: actual_arg_spec_list is None!"
            assert self.call_subroutines is not None, "Error: call_subroutines is None!"
            name = walk(walk(node, F23.Entity_Decl), F23.Name)[0].string
            if name not in self.dummy_arg_list[subroutine_key]:
                raise ValueError(f"An implicit array '{name}' with an unknown shape is declared locally!")
            
            if subroutine_key not in self.actual_arg_spec_list or subroutine_key not in self.call_subroutines:
                self.current_module_imp = walk(node.get_root(), F23.Name)[0].string
                self.processor.logger.info(
                    f"The subroutine '{subroutine_key}' is called outside of the module '{self.current_module_imp}'. Searching the module...")

                self.find_fortran_files_subroutine(subroutine_key)
            
            for arg, call in zip(self.actual_arg_spec_list[subroutine_key], self.call_subroutines[subroutine_key]):
                act_arg = arg[self.dummy_arg_list[subroutine_key].index(name)]
                enclosing_subroutine = self.find_enclosing_subroutine(call)

                subroutine_key = walk(walk(enclosing_subroutine, F23.Subroutine_Stmt), F23.Name)[0].string
                declaration_part = walk(enclosing_subroutine, F23.Specification_Part)
                
                self.processor.logger.info(
                    f"Corresponding element of '{name}' is '{act_arg}' in call statement in subroutine '{subroutine_key}'!")

                if declaration_part:
                    for decl in walk(declaration_part, F23.Type_Declaration_Stmt):
                        declarations = [name.children[0].string for name in walk(decl, F23.Entity_Decl)]
                        if act_arg in declarations:
                            explicit_shape = walk(decl, F23.Explicit_Shape_Spec)
                            if explicit_shape:
                                self.processor.logger.info(f"Found explicit shape in subroutine '{subroutine_key}'!")
                                return decl
                            else:
                                return self.shaper_subroutine(decl, subroutine_key)
            
            self.module_tree_imp = call.get_root()
            module_name = walk(self.module_tree_imp, F23.Name)[0].string
            self.processor.logger.info(
                    f"'{act_arg}' is not a dummy argument in subroutine '{subroutine_key}'. Searching in module '{module_name}'...")

            self.finder = Navigator(self.module_dir_imp, self.module_tree_imp, self.parsed_modules, self.module_path, logger=self.logger)
            self.finder.variable_finder(act_arg)
            return self.processor.combine_allocate_declaration(self.finder.var_declaration)
        
        except Exception as e:
            self.processor.logger.exception(f"An error occurred in method 'shaper_subroutine': ", e)
            raise

    def shaper_function(self, node, function_tree, function_key, all_array_info):
        """
        Resolve implicit array shapes in function assignment contexts.
        
        This method handles shape resolution for arrays used in function
        assignments. It searches for function calls and extracts shape
        information from array references in the assignment statements.
        
        Parameters
        ----------
        node : fparser.two.Fortran2003.Type_Declaration_Stmt
            The implicit array declaration node to resolve
        function_tree : fparser.two.Fortran2003.Function_Subprogram
            The function tree containing the assignment
        function_key : str
            The name of the function being analyzed
        all_array_info : dict
            Dictionary containing shape information for all known arrays
        
        Returns
        -------
        fparser.two.Fortran2003.Type_Declaration_Stmt
            The resolved explicit array declaration
        
        Raises
        ------
        AssertionError
            If array information is missing or invalid
        Exception
            If function assignment analysis fails
        
        Notes
        -----
        - Handles both slice notation (:) and explicit array references
        - Supports multi-dimensional array shape resolution
        - Searches through module files when function is not found locally
        """
        try:
            self.module_tree_imp = node.get_root()
            self.current_module_imp = walk(self.module_tree_imp, F23.Name)[0].string
            last_processed_module_dir = None
            function_assignment_stmt = None
            act_arg_list = None
            fortran_file_queue_imp = deque()

            name =  walk(node, F23.Entity_Decl)[0].tostr()
            if name not in self.dummy_arg_list[function_key]:
                self.processor.logger.error(f"An implicit array '{name}' with an unknown shape is declared locally!")
                raise

            while function_assignment_stmt is None:
                for assignment_stmt in walk(self.module_tree_imp, F23.Assignment_Stmt):
                    for part_ref in walk(assignment_stmt, F23.Part_Ref):
                        if isinstance(part_ref.children[0], F23.Name) and part_ref.children[0].tostr() == function_key:
                            function_assignment_stmt = True
                            assert isinstance(part_ref.children[1], F23.Section_Subscript_List), f"Expected Section_Subscript_List as second child, got {type(part_ref.children[1])}"
                            act_arg_list = part_ref.children[1]
                        #for child in part_ref.children:
                        #    if isinstance(child, F23.Name):
                        #        if child.tostr() == function_key:
                        #            function_assignment_stmt = True
                        #    if isinstance(child, F23.Section_Subscript_List):
                        #        if function_assignment_stmt:
                        #            act_arg_list = child
                        if function_assignment_stmt:
                            act_arg = act_arg_list.children[self.dummy_arg_list[function_key].index(name)]
                            #print(act_arg, "llllllllllllll======")
                            if ':' in act_arg.tostr() and isinstance(act_arg, F23.Part_Ref):
                                array_name = act_arg.children[0].tostr()
                                dims = act_arg.children[1].children
                                #print('dims:',dims)
                                shape = []
                                assert array_name in all_array_info, (
                                        f"Error in shaper_function: Array '{array_name}' not present in all_array_info."
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
                                        f"Error in shaper_function: Array '{array_name}' not present in all_array_info."
                                        )
                                array_info = all_array_info[array_name]
                                shape = []
                                for idim in range(len(array_info)):
                                    lb = array_info[idim]['dim_str']
                                    ub = array_info[idim]['dim_end']
                                    shape.append(f'{lb}:{ub}')
                            else:
                                self.processor.logger.error(
                                        f"Unexpected type for act_arg: {type(act_arg).__name__}. "
                                        "Expected 'F23.Part_Ref' or 'F23.Name'."
                                        )
                                raise
                            dimensions = ', '.join([name for name in shape])
                            new_dec = self.processor.map_declaration(node, explicit_dec=None, dimensions=dimensions)
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
                            self.module_tree_imp = self.processor.parse_fortran_file(module_file_path)
                            self.parsed_modules[module_name] = self.module_tree_imp
                            self.module_path[module_name] = module_file_path
                    else:
                        self.processor.logger.error(
                                f"Function key '{function_key}' not found in any module, need to go to higher directory."
                                )
                        raise
            
            for child in function_tree.children:
                if isinstance(child, F23.Specification_Part):
                    for igc,gchild in enumerate(child.children):
                        if isinstance(gchild, F23.Type_Declaration_Stmt):
                            entity_decl = walk(gchild, F23.Entity_Decl)[0].tostr()
                            if entity_decl == name:
                                child.children[child.children.index(gchild)] = new_dec
            return new_dec
        except Exception as e:
            self.processor.logger.exception(f"An error occurred in method 'shaper_subroutine': ", e)
            raise

    def shaper_intrinsic_size(self, node):
        """
        Handle SIZE intrinsic functions to determine array dimensions.
        
        This method processes SIZE intrinsic function calls to compute
        explicit array dimensions. It handles both explicit DIM arguments
        and implicit dimension calculations.
        
        Parameters
        ----------
        node : fparser.two.Fortran2003.Type_Declaration_Stmt
            The declaration node containing SIZE intrinsic calls
        
        Returns
        -------
        fparser.two.Fortran2003.Type_Declaration_Stmt
            The declaration with SIZE intrinsics resolved to explicit dimensions
        
        Raises
        ------
        AssertionError
            If intrinsic function structure is invalid
        Exception
            If SIZE argument processing fails
        
        Notes
        -----
        - Handles SIZE with and without DIM arguments
        - Computes dimension sizes from explicit shape specifications
        - Supports multi-dimensional size calculations
        - Replaces intrinsic calls with computed dimension expressions
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
                            self.processor.logger.info(
                                    f"SIZE for {args[0].tostr()} without an explicit 'DIM'. This implies size of all dimensions.")
                            dim_value = 'ALL'
                        if len(args) == 2:
                            if isinstance(args[1], F23.Actual_Arg_Spec):
                                dim_key, dim_value_node = args[1].items
                                assert isinstance(dim_key, F23.Name), "First item in Actual_Arg_Spec must be a Name."
                                if dim_key.tostr().lower() == 'dim':
                                    assert isinstance(dim_value_node, F23.Int_Literal_Constant), "Second item in must be an Int_Literal_Constant."
                                    dim_value = dim_value_node.tostr()
                            elif isinstance(args[1], F23.Int_Literal_Constant):
                                dim_value = args[1].tostr()
                            else:
                                self.processor.logger.error("Unexpected structure for intrinsic arguments.")
                                raise
                        
                        assert isinstance(node.parent, F23.Specification_Part), \
                                f"Expected node's parent to be of type F23.Specification_Part, but got {type(node.parent).__name__} instead."
                        for declaration_stmt in walk(node.parent, F23.Type_Declaration_Stmt):
                            entity_decls = walk(declaration_stmt, F23.Entity_Decl)
                            # bug to fix, if there is multiple variables
                            assert len(entity_decls) == 1,"walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                            if entity_decls[0].tostr() == args[0].tostr():
                                size = '1'
                                for idim, explicit_shape in enumerate(walk(declaration_stmt, F23.Explicit_Shape_Spec), start=1):
                                    parts = [part.strip() for part in explicit_shape.tostr().split(':')]
                                    if len(parts) == 2:
                                        dim_size = '(' + parts[1] + '-' + parts[0] + '+1' + ')'
                                    elif len(parts) == 1:
                                        dim_size = parts[0]
                                    else:
                                        self.processor.logger.error(f"Unexpected number of parts in explicit shape: {len(parts)}. Parts: {parts}")
                                        raise
                                    if dim_value != 'ALL':
                                        if idim == int(dim_value):
                                            size = dim_size
                                    else:
                                        size = dim_size + '*' + size
                        shape.append(size)
                    else:
                        self.processor.logger.error(f"Unexpected intrinsic function: {intrinsic.tostr()}")
                        raise
            dimensions = ', '.join([name for name in shape])
            new_dec = self.processor.map_declaration(node, explicit_dec=None, dimensions=dimensions)
            for igc,gchild in enumerate(node.parent.children):
                if isinstance(gchild, F23.Type_Declaration_Stmt):
                    entity_decl = walk(gchild, F23.Entity_Decl)[0].tostr()
                    if entity_decl == name:
                        node.parent.children[node.parent.children.index(gchild)] = new_dec
            return new_dec
        except Exception as e:
            self.processor.logger.exception(f"An error occurred in method 'find_enclosing_subroutine': ", e)
            raise


    def find_enclosing_subroutine(self, node):
        """
        Find the enclosing subroutine for a given AST node.
        
        Traverses up the AST parent hierarchy to find the subroutine
        that contains the given node.
        
        Parameters
        ----------
        node : fparser.two.Fortran2003.Base
            The AST node to find the enclosing subroutine for
        
        Returns
        -------
        fparser.two.Fortran2003.Subroutine_Subprogram or None
            The enclosing subroutine node, or None if not found
        
        Raises
        ------
        Exception
            If AST traversal fails
        
        Notes
        -----
        - Uses parent pointer traversal up the AST
        - Returns the first enclosing subroutine found
        - Returns None if no subroutine is found in the hierarchy
        """
        try:
            while node is not None:
                if isinstance(node, F23.Subroutine_Subprogram):
                    return node
                node = getattr(node, 'parent', None)
            return None
        
        except Exception as e:
            self.processor.logger.exception(f"An error occurred in method 'find_enclosing_subroutine': ", e)
            raise
