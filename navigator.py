import os
from collections import deque
from processor import Processor
from logger import Logger
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
import unittest
import tempfile
import shutil

class Navigator:
    """
    A navigator for analyzing Fortran code to locate variables and subroutines within modules.
    
    This class performs a breadth-first search to traverse module dependencies and
    checks for variable declarations or subroutine definitions. It can handle complex
    module hierarchies and external subroutine interfaces.
    
    Parameters
    ----------
    subroutine_dir : str
        Directory containing the Fortran subroutine files
    module_tree : fparser.two.Fortran2003.Module
        Parsed Fortran module tree from fparser
    parsed_modules : dict
        Dictionary of pre-parsed modules for efficient lookup
    
    Attributes
    ----------
    module_dir_sc : str
        Directory containing the Fortran subroutine files
    module_tree_sc : fparser.two.Fortran2003.Module
        Parsed Fortran module tree
    variable_name_sc : str
        Name of the variable or subroutine to find
    var_declaration : list
        List of variable declaration statements found
    var_initial : list
        List of variable initialization names found
    return_key_sc : bool
        Flag indicating if the target was found
    visited_modules_sc : set
        Set of visited modules to avoid redundant checks
    child_modules_sc : set
        Set of child modules encountered during search
    module_set_sc : set
        Set of all encountered modules
    queue_sc : deque
        Queue to manage module traversal
    main_fortran_file : str
        The main Fortran driver file name (default: "orchideedriver.f90")
    main_dir : str
        Directory of the main Fortran driver file (default: 'src_driver')
    full_scout : bool
        Flag indicating if search has been extended to main program
    parsed_modules : dict
        Dictionary of pre-parsed modules
    processor : Processor
        Processor instance for parsing and utility functions
    
    Methods
    -------
    find_variable_in_module()
        Search for variable in current module and update attributes.
    find_external_subroutine_in_module()
        Search for external subroutine in current module.
    external_subroutine_finder(variable_name)
        Initialize search for external subroutine.
    variable_finder(variable_name)
        Initialize search for variable.
    add_modules_to_queue()
        Add modules from USE statements to traversal queue.
    find_var_in_child_modules(key='variable')
        Traverse child modules to find variable or subroutine.
    
    Notes
    -----
    The class uses breadth-first search to ensure the shortest path to the target
    is found first. It handles both variable declarations and subroutine definitions,
    including those in interface blocks.
    
    Raises
    ------
    RuntimeError
        If errors occur during module parsing or traversal
    AssertionError
        If entity declaration separation fails validation
    
    See Also
    --------
    Processor : Supporting class for Fortran parsing utilities
    """

    def __init__(self, subroutine_dir, module_tree, parsed_modules, module_path, logger=None):
        """
        Initialize the Navigator with directory, module tree, and parsed modules.
        
        Parameters
        ----------
        subroutine_dir : str
            Directory containing Fortran subroutine files
        module_tree : fparser.two.Fortran2003.Module
            Parsed Fortran module tree
        parsed_modules : dict
            Dictionary of pre-parsed modules for efficient lookup
        """
        self.module_dir_sc = subroutine_dir
        self.module_tree_sc = module_tree
        self.variable_name_sc = ''
        self.var_declaration = []
        self.var_initial = []
        self.return_key_sc = False
        self.visited_modules_sc = set()
        self.child_modules_sc = set()
        self.module_set_sc = set()
        self.queue_sc = deque()
        self.main_fortran_file = "orchideedriver.f90"
        self.main_dir = 'src_driver'
        self.full_scout = False
        self.parsed_modules = parsed_modules
        self.module_path = module_path
        self.logger = logger
        self.processor = Processor(logger=self.logger)
        #self.logger = Logger(Module_name="Navigator")
        #self.logger.show_header()

    def find_variable_in_module(self):
        """
        Search for the variable in the current module and update attributes.
        
        This method searches for the target variable in the current module tree.
        If found, it extracts declaration information and checks allocation status.
        The method handles both simple declarations and multi-variable declarations.
        
        Returns
        -------
        None
        
        Raises
        ------
        RuntimeError
            If errors occur during the search process
        
        Notes
        -----
        - For multi-variable declarations, the method separates individual entities
        - Checks for allocatable attributes and allocation statements
        - Logs findings using the processor's logger
        - Updates return_key_sc to True if variable is found with valid allocation
        """
        try:
            module_name = walk(self.module_tree_sc, F23.Name)[0].string
            names = walk(self.module_tree_sc, F23.Name)
            name_strings = [name.string for name in names]
            if self.variable_name_sc in name_strings:
                for child in names:
                    if child.string == self.variable_name_sc:
                        stmts = child.parent.parent.parent
                        any_allocate = walk(self.var_declaration, F23.Allocate_Stmt)
                        any_declarat = walk(self.var_declaration, F23.Type_Declaration_Stmt)
                        if (isinstance(stmts, F23.Type_Declaration_Stmt) and not any_declarat) or (isinstance(stmts, F23.Allocate_Stmt) and not any_allocate):
                            current = stmts
                            while current is not None and not isinstance(current, (F23.Subroutine_Subprogram, F23.Function_Subprogram, F23.Module)):
                                current = getattr(current, "parent", None)
                            morr = walk(current, F23.Name)[0].string
                            if isinstance(stmts, F23.Type_Declaration_Stmt) and len(walk(stmts, F23.Entity_Decl)) > 1:
                                for decleration in self.processor.separate_entity_declarations(stmts):
                                    entity_decls = walk(decleration, F23.Entity_Decl)
                                    assert len(entity_decls) == 1, \
                                        "Assertion failed in 'find_variable_in_module': Expected 1 entity declaration, found a different number."
                                    entity_decl = entity_decls[0].tostr()
                                    if self.variable_name_sc == entity_decl:
                                        stmt = decleration
                                        break
                            elif isinstance(stmts, F23.Allocate_Stmt) and len(walk(stmts, F23.Allocation)) > 1:
                                for allocation in self.processor.separate_entity_allocation(stmts):
                                    allocations = walk(allocation, F23.Allocation)
                                    assert len(allocations) == 1, \
                                        "Assertion failed in 'find_variable_in_module': Expected 1 allocation, found a different number."
                                    allocation = allocations[0].children[0].tostr()
                                    if self.variable_name_sc == allocation:
                                        stmt = allocation
                                        break
                            else:
                                stmt = stmts
                            self.var_initial = walk(walk(stmt, F23.Initialization), F23.Name)
                            self.var_declaration.append(stmt)
                            self.processor.logger.info(
                                    f"'{self.variable_name_sc}' is found in '{morr}' of the module '{module_name}'"
                                    )
                            self.processor.logger.info(str(stmt))
                        if isinstance(child.parent, F23.Function_Stmt):
                            self.processor.logger.warning(f"Warning: '{self.variable_name_sc}' is a function")
                            self.processor.logger.warning(f"The containing directory is: '{self.module_dir_sc}'")
                            function_name = child
                            function_subprogram = child.parent.parent
                            self.var_declaration.extend([function_name,function_subprogram, module_name])

            if self.var_declaration:
                self.return_key_sc = True
                allocate_stmt, attr_spec = walk(self.var_declaration, F23.Allocate_Stmt), walk(self.var_declaration, F23.Attr_Spec)
                if F23.Attr_Spec('ALLOCATABLE') not in attr_spec and allocate_stmt != [] or \
                        F23.Attr_Spec('ALLOCATABLE') in attr_spec and allocate_stmt == []:
                    self.return_key_sc = False
        except Exception as e:
            self.processor.logger.exception(f"Error in 'find_variable_in_module': ", e)

    def find_external_subroutine_in_module(self):
        """
        Search for an external subroutine within the current module.
        
        Searches for subroutine definitions in both interface blocks and
        regular subroutine subprograms. If found, creates a USE statement
        for the subroutine.
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If errors occur during interface or subroutine parsing
        
        Notes
        -----
        - Searches both interface blocks and subroutine subprograms
        - Creates a USE statement for found subroutines
        - Sets return_key_sc to True if subroutine is found
        """
        try:
            interfaces = walk(self.module_tree_sc, F23.Interface_Block)
            if interfaces:
                for interface in interfaces:
                    for node in walk(interface, F23.Interface_Stmt):
                        interface_name = walk(node, F23.Name)[0].string
                        if interface_name == self.variable_name_sc:
                            module_name = walk(self.module_tree_sc, F23.Name)[0].string
                            use_stmt = f'use {module_name}, ONLY: {self.variable_name_sc}'
                            self.var_declaration.append(F23.Use_Stmt(use_stmt))
                            self.return_key_sc = True
                            self.processor.logger.info(f"'{self.variable_name_sc}' procedure is found in the module '{module_name}' ")
                            return
            for sub in walk(self.module_tree_sc, F23.Subroutine_Subprogram):
                for node in walk(sub, F23.Subroutine_Stmt):
                    subroutine_name = walk(node, F23.Name)[0].string
                    if subroutine_name == self.variable_name_sc:
                        module_name = walk(self.module_tree_sc, F23.Name)[0].string
                        use_stmt = f'use {module_name}, ONLY: {self.variable_name_sc}'
                        self.var_declaration.append(F23.Use_Stmt(use_stmt))
                        self.return_key_sc = True
                        self.processor.logger.info(f"'{self.variable_name_sc}' procedure is found in the module '{module_name}' ")
                        return
        except Exception as e:
            self.processor.logger.exception(f"Error in 'find_external_subroutine_in_module': ", e)
            raise

    def external_subroutine_finder(self, variable_name):
        """
        Initialize the search for an external subroutine.
        
        Parameters
        ----------
        variable_name : str
            The name of the external subroutine to search for
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If errors occur during the search initialization
        
        Notes
        -----
        - Sets up initial module tracking and queue
        - Initiates module traversal with subroutine key
        - Logs the containing directory upon completion
        """
        try:
            self.variable_name_sc = variable_name
            module_name = walk(self.module_tree_sc, F23.Name)[0].string
            self.module_set_sc.add(module_name)
            self.child_modules_sc.add(module_name)
            self.visited_modules_sc.add(module_name)
            self.add_modules_to_queue()
            self.find_var_in_child_modules(key='subroutine')
            self.processor.logger.info(f"The containing directory is: {self.module_dir_sc}")
        except Exception as e:
            self.processor.logger.exception(f"Error in 'external_subroutine_finder':", e)
            raise

    def variable_finder(self, variable_name):
        """
        Initialize the search for a variable.
        
        Parameters
        ----------
        variable_name : str
            The name of the variable to search for
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If errors occur during the search initialization
        
        Notes
        -----
        - First searches current module, then traverses dependencies
        - Uses breadth-first search for module dependencies
        - Extends search to main program if needed
        - Logs the containing directory upon completion
        """
        try:
            self.variable_name_sc = variable_name
            self.find_variable_in_module()
            if self.return_key_sc:
                self.processor.logger.info(f"The containing directory is: {self.module_dir_sc}")
            else:
                module_name = walk(self.module_tree_sc, F23.Name)[0].string
                self.module_set_sc.add(module_name)
                self.child_modules_sc.add(module_name)
                self.visited_modules_sc.add(module_name)
                self.add_modules_to_queue()
                self.find_var_in_child_modules(key='variable')
                self.processor.logger.info(f"The containing directory is: {self.module_dir_sc}")
        except Exception as e:
            self.processor.logger.exception(f"Error in 'variable_finder': ", e)
            raise

    def add_modules_to_queue(self):
        """
        Add modules referenced by USE statements to the traversal queue.
        
        Extracts module names from USE statements in the current module
        and adds them to the queue for further analysis if they haven't
        been processed yet.
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If module name extraction fails or other errors occur
        
        Notes
        -----
        - Only adds modules not already in module_set_sc
        - Logs each module added to the queue
        - Skips modules that cannot be parsed or found
        """
        try:
            for module in walk(self.module_tree_sc, F23.Use_Stmt):
                module_name = None
                for entity in module.children:
                    if isinstance(entity, F23.Name):
                        module_name = entity.string
                        break
                if module_name is None:
                    self.processor.logger.error("Module name not found in the Fortran code.")
                    raise
                if module_name not in self.module_set_sc:
                    self.processor.logger.info(f"Module '{module_name}' is added into the queue.")
                    self.queue_sc.append(module)
                    self.module_set_sc.add(module_name)
        except Exception as e:
            self.processor.logger.exception(f"Error in 'add_modules_to_queue': ", e)
            raise
    
    def find_var_in_child_modules(self, key='variable'):
        """
        Traverse child modules in the queue to find a variable or subroutine.
        
        Parameters
        ----------
        key : str, optional
            Specifies search type: 'variable' or 'subroutine' (default: 'variable')
        
        Returns
        -------
        None
        
        Raises
        ------
        Exception
            If module traversal or parsing fails
        
        Notes
        -----
        - Processes modules in FIFO order (breadth-first)
        - Handles module file discovery in multiple directories
        - Extends search to main program if queue is exhausted
        - Updates module directory and tree during traversal
        """
        try:
            while self.queue_sc:
                module_name = None
                node = self.queue_sc.popleft()
                for entity in node.children:
                    if isinstance(entity, F23.Name):
                        module_name = entity.string
                        self.processor.logger.info(f"Checking the child module ...'{module_name}'")
                        break
                if module_name is None:
                    self.processor.logger.error("Module name not found in the Fortran code.")
                    raise
                if module_name in self.visited_modules_sc:
                    self.processor.logger.info(f"Pass! Module '{module_name}' has already been visited.")
                    continue
                module_found = False
                self.child_modules_sc.add(module_name)
                if module_name in self.parsed_modules:
                    module_found = True
                    child_module_tree = self.parsed_modules[module_name]
                    self.module_dir_sc = os.path.dirname(self.module_path[module_name])
                else:
                    module_filename_lower = os.path.join(self.module_dir_sc, module_name + ".f90")
                    module_filename_upper = os.path.join(self.module_dir_sc, module_name + ".F90")
                    # Check for module files and parse them
                    if os.path.exists(module_filename_lower) or os.path.exists(module_filename_upper):
                        selected_filename = module_filename_lower if os.path.exists(module_filename_lower) else module_filename_upper
                        child_module_tree = self.processor.parse_fortran_file(selected_filename)
                        module_found = True
                        self.parsed_modules[module_name] = child_module_tree
                        self.module_path[module_name] = selected_filename
                    else:
                        parent_directory = os.path.abspath(os.path.join(self.module_dir_sc, '../'))
                        for directory in os.listdir(parent_directory):
                            if directory.startswith(".") or directory == "__pycache__":
                                continue
                            if directory.startswith('src_'):
                                full_directory = os.path.join(parent_directory, directory)
                                if os.path.isdir(full_directory):
                                    module_filename_lower = os.path.join(full_directory, module_name + ".f90")
                                    module_filename_upper = os.path.join(full_directory, module_name + ".F90")
                                    if os.path.exists(module_filename_lower) or os.path.exists(module_filename_upper):
                                        selected_filename = module_filename_lower if os.path.exists(module_filename_lower) else module_filename_upper
                                        child_module_tree = self.processor.parse_fortran_file(selected_filename)
                                        self.module_dir_sc = full_directory
                                        module_found = True
                                        self.parsed_modules[module_name] = child_module_tree
                                        self.module_path[module_name] = selected_filename
                                        break 
                if module_found:
                    self.visited_modules_sc.add(module_name)
                    self.module_tree_sc = child_module_tree
                    self.add_modules_to_queue()
                    if key == 'variable':
                        self.find_variable_in_module()
                    elif key == 'subroutine':
                        self.find_external_subroutine_in_module()
                    if self.return_key_sc:
                        return

                # Handle empty queue and reset if needed
                if not self.queue_sc and not self.return_key_sc and not self.full_scout:
                    self.full_scout = True
                    self.processor.logger.warning("Warning: Queue is empty and return_key is still False! Extending queue with the main program!")
                    if not os.path.exists(os.path.join(self.module_dir_sc, self.main_fortran_file)):
                        parent_directory = os.path.abspath(os.path.join(self.module_dir_sc, '../'))
                        for directory in os.listdir(parent_directory):
                            if directory.startswith(self.main_dir):
                                full_directory = os.path.join(parent_directory, directory)
                                if os.path.exists(os.path.join(full_directory, self.main_fortran_file)):
                                    main_programm = os.path.join(full_directory, self.main_fortran_file)
                                    self.module_tree_sc = self.processor.parse_fortran_file(main_programm)
                                    self.module_dir_sc = full_directory
                                    self.add_modules_to_queue()
                if not self.queue_sc and not self.return_key_sc and self.full_scout:
                    self.processor.logger.error("Queue is empty, return key is False, and full scout is True. Unable to proceed.")
                    raise
        except Exception as e:
            self.processor.logger.exception(f"Error in 'find_var_in_child_modules': ", e)
            raise


class FortranSearcher:
    def __init__(self, module_path, parsed_modules, org_files_loaded, logger=None):
        """
        Args:
            processor: object providing `logger` and `parse_fortran_file` methods.
            module_path (dict): maps module names -> source file paths.
            parsed_modules (dict): maps module names -> parsed fparser trees.
        """
        self.module_path = module_path
        self.parsed_modules = parsed_modules
        self.org_files_loaded = org_files_loaded
        self.logger = logger
        self.processor = Processor(logger=self.logger)
        self.module_dir_sc = None  # Current search directory context

    # =====================================================
    # Main search routine
    # =====================================================
    def search_subroutine_in_dependencies(self, subroutine_name, current_module_name, current_dir):
        """
        Search for a subroutine by traversing module dependencies (USE statements).

        Returns:
            (found: bool, module_file_path: str | None, module_tree: object | None)
        """
        try:
            self.module_dir_sc = os.path.normpath(current_dir)
            visited_modules = set()
            module_queue = deque([current_module_name])

            while module_queue:
                module_name = module_queue.popleft()
                if module_name.lower() in ['mpi', 'xios', 'ioipsl', 'ieee_arithmetic', 'netcdf']:
                    continue
                if module_name in visited_modules:
                    continue
                visited_modules.add(module_name)

                # --- Parse or reuse module tree ---
                self.processor.logger.info(
                                f"🔎 Searching for module name '{module_name}'"
                                )
                module_tree, module_file_path = self._get_or_parse_module(module_name)
                if module_tree is None:
                    self.processor.logger.warning(f"⚠️  Module '{module_name}' not found, skipping.")
                    continue

                # --- Search for the subroutine inside this module ---
                for sub in walk(module_tree, F23.Subroutine_Subprogram):
                    subroutine_stmt = walk(sub, F23.Subroutine_Stmt)[0]
                    for child in subroutine_stmt.children:
                        if isinstance(child, F23.Name) and child.tostr() == subroutine_name:
                            self.processor.logger.info(
                                f"✅ Found subroutine '{subroutine_name}' in module '{module_name}', '{module_file_path}' ")

                            path_to_original = module_file_path.replace('.f90', '_org.fgpt').replace('.F90', '_org.Fgpt')
                            if not os.path.exists(path_to_original):
                                self.processor.logger.info(f"💾 Created backup: {path_to_original}")
                                shutil.copy(module_file_path, path_to_original)
                            return True, module_file_path, module_tree

                # --- Collect USE dependencies ---
                used_modules = self._collect_used_modules(module_tree, subroutine_name)
                for used_module in used_modules:
                    if used_module not in visited_modules and used_module not in module_queue:
                        module_queue.append(used_module)

            self.processor.logger.warning(
                    f"⚠️  Subroutine '{subroutine_name}' not found in '{current_module_name}' or its dependencies."
                    )
            return False, None, None

        except Exception as e:
            self.processor.logger.exception(f"Error while searching for subroutine '{subroutine_name}'", e)
            return False, None, None

    # =====================================================
    # Parse or reuse existing module (with org file handling)
    # =====================================================
    def _get_or_parse_module(self, module_name):
        """
        Search for a Fortran module definition by name using a breadth-first directory search.

        Returns:
            (module_tree, module_file_path) if found, else (None, None)
        """
        # Reuse cached result if available
        if module_name in self.parsed_modules:
            self.processor.logger.info(
                    f"✅ Module '{module_name}' is already cached (path: '{self.module_path[module_name]}')"
                    )
            return self.parsed_modules[module_name], self.module_path.get(module_name)

        visited_dirs = set()
        dir_queue = deque([self.module_dir_sc])

        while dir_queue:
            dir_path = dir_queue.popleft()
            dir_path = os.path.abspath(dir_path)
            if dir_path in visited_dirs or not os.path.isdir(dir_path):
                continue
            visited_dirs.add(dir_path)

            # --- Search all .f90 / .F90 files in this directory ---
            files_in_dir = [
                    f for f in os.listdir(dir_path)
                    if f.endswith(".f90") or f.endswith(".F90")
                    ]

            def file_similarity_score(filename):
                name = os.path.splitext(filename.lower())[0]  # remove extension
                mod = module_name.lower()
                if name == mod:
                    return -float('inf')  # exact match first
                if mod in name or name in mod:
                    return -1000  # partial match next
                common_prefix_len = len(os.path.commonprefix([name, mod]))
                return -common_prefix_len  # longer prefix = higher priority

            files_in_dir.sort(key=file_similarity_score)

            for filename in files_in_dir:
                file_path = os.path.join(dir_path, filename)

                # --- Prefer original backup if available ---
                path_to_original = (
                    file_path.replace('.f90', '_org.fgpt')
                             .replace('.F90', '_org.Fgpt')
                )
                if os.path.exists(path_to_original) and filename not in self.org_files_loaded:
                    file_to_parse = path_to_original
                    self.org_files_loaded.add(filename)
                    self.processor.logger.info(
                            f"💾 Using original backup for module '{filename}' (from '{path_to_original}')"
                            )
                else:
                    file_to_parse = file_path

                if file_path in self.module_path.values():
                    for cached_mod, cached_path in self.module_path.items():
                        if cached_path == file_path:
                            module_tree = self.parsed_modules[cached_mod]
                            self.processor.logger.info(
                                    f"♻️  Reusing cached parse tree for file '{cached_mod}' module/program: '{file_path}'")
                            if cached_mod.lower() == module_name.lower():
                                self.processor.logger.info(
                                        f"✅ Module '{cached_mod}' is already cached (path: '{file_path}')")
                                return module_tree, file_path
                            break
                else:
                    module_tree = self.processor.parse_fortran_file(file_to_parse)


                # --- Extract and cache all modules in this file ---
                found_target = False
                for mod_stmt in walk(module_tree):
                    if not isinstance(mod_stmt, (F23.Module_Stmt, F23.Program_Stmt)):
                        continue
                    mod_name_node = walk(mod_stmt, F23.Name)
                    if not mod_name_node:
                        continue

                    mod_name = mod_name_node[0].tostr()
                    mod_name_lower = mod_name.lower()

                    # Cache module for reuse
                    if mod_name not in self.parsed_modules:
                        self.parsed_modules[mod_name] = module_tree
                        self.module_path[mod_name] = file_path

                    # Check if this is the requested module
                    if mod_name_lower == module_name.lower():
                        self.processor.logger.info(f"✅ Found module '{module_name}' in file: '{file_to_parse}'")
                        self.module_dir_sc = dir_path
                        return module_tree, file_path

            # --- Add sibling src_* directories dynamically to the queue ---
            parent_directory = os.path.abspath(os.path.join(dir_path, "../"))
            for directory in os.listdir(parent_directory):
                full_dir = os.path.join(parent_directory, directory)
                if (
                        directory.startswith("src_")
                        and os.path.isdir(full_dir)
                        and full_dir not in visited_dirs
                        ):
                    dir_queue.append(full_dir)

        # --- Not found after scanning all directories ---
        self.processor.logger.warning(
                f"⚠️  Module '{module_name}' not found starting from '{self.module_dir_sc}'"
                )
        return None, None

    # =====================================================
    # Extract USE dependencies
    # =====================================================
    def _collect_used_modules(self, module_tree, target_name=None):
        """
        Return a set of module names referenced by USE statements.
        """
        used_modules = set()
        for use_stmt in walk(module_tree, F23.Use_Stmt):
            for child in use_stmt.children:
                if isinstance(child, F23.Name):
                    used_modules.add(child.tostr().lower())  # normalize

        if target_name:
            def similarity_score(mod_name):
                target = target_name.lower()
                if mod_name == target:
                    return -float('inf')  # exact match first
                if target in mod_name or mod_name in target:
                    return -1000  # partial match next
                common_prefix_len = len(os.path.commonprefix([mod_name, target]))
                return -common_prefix_len
            return sorted(used_modules, key=similarity_score)

        return used_modules

'''
class TestNavigator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()

        # Create test Fortran files
        cls.simple_module = os.path.join(cls.test_dir, "simple_mod.f90")
        with open(cls.simple_module, "w") as f:
            f.write("""
            module simple_mod
            implicit none
            integer, parameter :: global_param = 42
            real, dimension(:), allocatable :: global_array
            contains

            subroutine test_sub(a, b, n)
                integer, intent(in) :: a
                integer, intent(in) :: n
                real, dimension(n), intent(out) :: b
                integer :: i
                real :: local_scalar

                local_scalar = real(a) * 2.0

                if (.not. allocated(global_array)) then
                    allocate(global_array(5))
                    global_array = [1.0, 2.0, 3.0, 4.0, 5.0]
                end if

                do i = 1, n
                    b(i) = local_scalar + real(i) + global_param + global_array(mod(i-1, 5) + 1)
                end do

            end subroutine test_sub
            end module simple_mod
            """)

        cls.dependent_module = os.path.join(cls.test_dir, "dependent_mod.f90")
        with open(cls.dependent_module, "w") as f:
            f.write("""
            module dependent_mod
            use simple_mod
            implicit none
            integer :: dependent_var

            contains

            subroutine dependent_sub(x, m)
            integer, intent(in) :: m
            real, dimension(m),intent(inout) :: x
            dependent_var = 5
            call test_sub(dependent_var, x, m)
            global_array =  [10.0, 20.0, 30.0, 40.0, 50.0]
            end subroutine dependent_sub
            end module dependent_mod
            """)

        # Parse the module trees
        processor = Processor()
        cls.simple_tree = processor.parse_fortran_file(cls.simple_module)
        cls.dependent_tree = processor.parse_fortran_file(cls.dependent_module)

        # Create a parsed_modules dictionary
        cls.parsed_modules = {
            "simple_mod": cls.simple_tree,
            "dependent_mod": cls.dependent_tree
        }
        cls.module_path = {
                "simple_mod": cls.simple_module,
                "dependent_mod":cls.dependent_module
                }

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Create fresh Navigator instances for each test
        self.simple_navigator = Navigator(self.test_dir, self.simple_tree, self.parsed_modules, self.module_path)
        self.dependent_navigator = Navigator(self.test_dir, self.dependent_tree, self.parsed_modules, self.module_path)

    def test_initialization(self):
        # Test that initialization sets up all attributes correctly
        self.assertEqual(self.simple_navigator.module_dir_sc, self.test_dir)
        self.assertIsInstance(self.simple_navigator.module_tree_sc.children[1], F23.Module)
        self.assertEqual(self.simple_navigator.var_declaration, [])
        self.assertEqual(self.simple_navigator.var_initial, [])
        self.assertFalse(self.simple_navigator.return_key_sc)
        self.assertEqual(self.simple_navigator.visited_modules_sc, set())
        self.assertEqual(self.simple_navigator.child_modules_sc, set())
        self.assertEqual(self.simple_navigator.module_set_sc, set())
        self.assertEqual(len(self.simple_navigator.queue_sc), 0)
        self.assertFalse(self.simple_navigator.full_scout)

    def test_find_variable_in_module(self):
        # Test finding a variable in the current module
        self.simple_navigator.variable_name_sc = "global_param"
        self.simple_navigator.find_variable_in_module()
        self.assertTrue(self.simple_navigator.return_key_sc)
        self.assertEqual(len(self.simple_navigator.var_declaration), 1)
        self.assertIsInstance(self.simple_navigator.var_declaration[0], F23.Type_Declaration_Stmt)

        # Test finding an array
        self.simple_navigator.variable_name_sc = "global_array"
        self.simple_navigator.var_declaration = []
        self.simple_navigator.return_key_sc = False
        self.simple_navigator.find_variable_in_module()
        self.assertTrue(self.simple_navigator.return_key_sc)
        self.assertEqual(len(self.simple_navigator.var_declaration), 2)
        type_decl_found = any(isinstance(stmt, F23.Type_Declaration_Stmt) for stmt in self.simple_navigator.var_declaration)
        allocate_found = any(isinstance(stmt, F23.Allocate_Stmt) for stmt in self.simple_navigator.var_declaration)
        self.assertTrue(allocate_found, "Allocate_Stmt not found in the parse tree")
        self.assertTrue(type_decl_found, "Type_Declaration_Stmt not found in the parse tree")
        
    
    def test_find_external_subroutine_in_module(self):
        # Test finding a subroutine in the current module
        self.simple_navigator.variable_name_sc = "test_sub"
        self.simple_navigator.find_external_subroutine_in_module()
        self.assertTrue(self.simple_navigator.return_key_sc)
        self.assertEqual(len(self.simple_navigator.var_declaration), 1)
        self.assertIsInstance(self.simple_navigator.var_declaration[0], F23.Use_Stmt)
    
    
    def test_add_modules_to_queue(self):
        # Test adding modules to the queue from USE statements
        self.dependent_navigator.add_modules_to_queue()

        # Verify the queue was populated correctly
        self.assertEqual(len(self.dependent_navigator.queue_sc), 1)
        self.assertEqual(len(self.dependent_navigator.module_set_sc), 1)
        self.assertIn("simple_mod", self.dependent_navigator.module_set_sc)
    
    def test_variable_finder(self):
        # Test finding a variable that requires module traversal
        self.dependent_navigator.variable_finder("global_param")

        # Verify the variable was found through module dependencies
        self.assertTrue(self.dependent_navigator.return_key_sc)
        self.assertEqual(len(self.dependent_navigator.var_declaration), 1)
        self.assertEqual(len(self.dependent_navigator.visited_modules_sc), 2)  # dependent_mod and simple_mod

        # Test finding an array
        self.dependent_navigator.var_declaration = []
        self.dependent_navigator.return_key_sc = False
        self.dependent_navigator.variable_finder("global_array")
        self.assertTrue(self.dependent_navigator.return_key_sc)
        self.assertEqual(len(self.dependent_navigator.var_declaration), 2)
        type_decl_found = any(isinstance(stmt, F23.Type_Declaration_Stmt) for stmt in self.dependent_navigator.var_declaration)
        allocate_found = any(isinstance(stmt, F23.Allocate_Stmt) for stmt in self.dependent_navigator.var_declaration)
        self.assertTrue(allocate_found, "Allocate_Stmt not found in the parse tree")
        self.assertTrue(type_decl_found, "Type_Declaration_Stmt not found in the parse tree")

    def test_external_subroutine_finder(self):
        # Test finding an external subroutine that requires module traversal
        self.dependent_navigator.external_subroutine_finder("test_sub")

        # Verify the subroutine was found through module dependencies
        self.assertTrue(self.dependent_navigator.return_key_sc)
        self.assertEqual(len(self.dependent_navigator.var_declaration), 1)
        self.assertEqual(len(self.dependent_navigator.visited_modules_sc), 2)  # dependent_mod and simple_mod

    def test_find_var_in_child_modules(self):
        # Test the module traversal logic for variables
        self.dependent_navigator.variable_name_sc = "global_param"
        self.dependent_navigator.module_set_sc.add("dependent_mod")
        self.dependent_navigator.child_modules_sc.add("dependent_mod")
        self.dependent_navigator.visited_modules_sc.add("dependent_mod")
        self.dependent_navigator.add_modules_to_queue()

        # Verify the variable is found in child modules
        self.dependent_navigator.find_var_in_child_modules(key='variable')
        self.assertTrue(self.dependent_navigator.return_key_sc)
        self.assertEqual(len(self.dependent_navigator.var_declaration), 1)

    def test_error_handling(self):
        self.simple_navigator.variable_finder("nonexistent_var")
        self.assertFalse(self.simple_navigator.return_key_sc)
        self.simple_navigator.external_subroutine_finder("nonexistent_sub")
        self.assertFalse(self.simple_navigator.return_key_sc)

if __name__ == "__main__":
    unittest.main()
'''
