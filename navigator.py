import os
from collections import deque
from processor import Processor
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23

class Navigator:
    """
    Navigator class used to analyze Fortran code to find specific variables or subroutines within modules.
    It performs a breadth-first search to traverse module dependencies and checks for variable or subroutine declarations.

    Attributes:
        module_dir_sc (str): Directory containing the Fortran subroutine files.
        module_tree_sc (F23 object): Parsed Fortran module tree.
        variable_name_sc (str): Name of the variable or subroutine to find.
        var_declaration (list): List of variable declaration statements found.
        var_initial (list): List of variable initialization names found.
        return_key_sc (bool): Flag to indicate if the target was found.
        visited_modules_sc (set): Set of visited modules to avoid redundant checks.
        child_modules_sc (set): Set of child modules encountered during the search.
        module_set_sc (set): Set of all encountered modules.
        queue_sc (deque): Queue to manage module traversal.
        main_fortran_file (str): The main Fortran driver file name.
        main_dir (str): Directory of the main Fortran driver file.
        full_scout (bool): Flag indicating if the search has been extended to the main program.
    """
    def __init__(self, subroutine_dir, module_tree):
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

    def find_variable_in_module(self):
        """
        Search for the variable in the current module and update the relevant attributes.
        If the variable is found, it checks the type and allocates status, then stores the information.
        """
        try:
            module_name = walk(self.module_tree_sc, F23.Name)[0].string
            names = walk(self.module_tree_sc, F23.Name)
            name_strings = [name.string for name in names]
            if self.variable_name_sc in name_strings:
                for child in names:
                    if child.string == self.variable_name_sc:
                        stmts = child.parent.parent.parent
                        name = stmts.parent.parent
                        morr = walk(name, F23.Name)[0].string
                        any_allocate = walk(self.var_declaration, F23.Allocate_Stmt)
                        any_declarat = walk(self.var_declaration, F23.Type_Declaration_Stmt)
                        if (isinstance(stmts, F23.Type_Declaration_Stmt) and not any_declarat) or \
                                (isinstance(stmts, F23.Allocate_Stmt) and not any_allocate):
                            if isinstance(stmts, F23.Type_Declaration_Stmt) and len(walk(stmts, F23.Entity_Decl)) > 1:
                                for decleration in Processor().separate_entity_declarations(stmts):
                                    entity_decls = walk(decleration, F23.Entity_Decl)
                                    assert len(entity_decls) == 1, \
                                        "Assertion failed in 'find_variable_in_module': Expected 1 entity declaration, found a different number."
                                    entity_decl = entity_decls[0].tostr()
                                    if self.variable_name_sc == entity_decl:
                                        stmt = decleration
                                        break
                            elif isinstance(stmts, F23.Allocate_Stmt) and len(walk(stmts, F23.Allocation)) > 1:
                                for allocation in Processor().separate_entity_allocation(stmts):
                                    allocations = walk(allocation, F23.Allocation)
                                    assert len(allocations) == 1, \
                                        "Assertion failed in 'find_variable_in_module': Expected 1 allocation, found a different number."
                                    allocation = allocations[0].children[0].tostr()
                                    if self.variable_name_sc == allocation:
                                        stmt = allocation
                                        break
                            else:
                                stmt = stmts
                            any_additional = walk(walk(stmt, F23.Initialization), F23.Name)
                            self.var_initial = [nadi.string for nadi in any_additional]
                            self.var_declaration.append(stmt)
                            print('\033[32m' + f"<{self.variable_name_sc}> is found in <<{morr}>> of the module <<< {module_name} >>>" + '\033[0m')
                            print('\033[32m' + str(stmt) + '\033[0m')
            if self.var_declaration:
                self.return_key_sc = True
                allocate_stmt, attr_spec = walk(self.var_declaration, F23.Allocate_Stmt), walk(self.var_declaration, F23.Attr_Spec)
                if F23.Attr_Spec('ALLOCATABLE') not in attr_spec and allocate_stmt != [] or \
                        F23.Attr_Spec('ALLOCATABLE') in attr_spec and allocate_stmt == []:
                    self.return_key_sc = False
        except Exception as e:
            raise RuntimeError(f"Error in 'find_variable_in_module': {str(e)}")

    def find_external_subroutine_in_module(self):
        """
        Search for an external subroutine within the current module.
        If the subroutine is found, it adds a use statement to the variable declarations.
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
                            print('\033[32m' + f"<{self.variable_name_sc} procedure> is found in the module <<< {module_name} >>>" + '\033[0m')
                            return
            for sub in walk(self.module_tree_sc, F23.Subroutine_Subprogram):
                for node in walk(sub, F23.Subroutine_Stmt):
                    subroutine_name = walk(node, F23.Name)[0].string
                    if subroutine_name == self.variable_name_sc:
                        module_name = walk(self.module_tree_sc, F23.Name)[0].string
                        use_stmt = f'use {module_name}, ONLY: {self.variable_name_sc}'
                        self.var_declaration.append(F23.Use_Stmt(use_stmt))
                        self.return_key_sc = True
                        print('\033[32m' + f"<{self.variable_name_sc} procedure> is found in the module <<< {module_name} >>>" + '\033[0m')
                        return
        except Exception as e:
            raise RuntimeError(f"Error in 'find_external_subroutine_in_module': {str(e)}")

    def external_subroutine_finder(self, variable_name):
        """
        Initialize the search for an external subroutine by setting the target variable name,
        adding the module to the visited set, and starting the search process.

        Args:
            variable_name (str): The name of the external subroutine to search for.
        """
        try:
            self.variable_name_sc = variable_name
            module_name = walk(self.module_tree_sc, F23.Name)[0].string
            self.module_set_sc.add(module_name)
            self.child_modules_sc.add(module_name)
            self.visited_modules_sc.add(module_name)
            self.add_modules_to_queue()
            self.find_var_in_child_modules(key='subroutine')
            print('\033[32m' + f"The containing directory is: {self.module_dir_sc}" + '\033[0m')
        except Exception as e:
            raise RuntimeError(f"Error in 'external_subroutine_finder': {str(e)}")

    def variable_finder(self, variable_name):
        """
        Initialize the search for a variable by setting the target variable name and starting the search process.

        Args:
            variable_name (str): The name of the variable to search for.
        """
        try:
            self.variable_name_sc = variable_name
            self.find_variable_in_module()
            if self.return_key_sc:
                print('\033[32m' + f"The containing directory is: {self.module_dir_sc}" + '\033[0m')
            else:
                module_name = walk(self.module_tree_sc, F23.Name)[0].string
                self.module_set_sc.add(module_name)
                self.child_modules_sc.add(module_name)
                self.visited_modules_sc.add(module_name)
                self.add_modules_to_queue()
                self.find_var_in_child_modules(key='variable')
                print('\033[32m' + f"The containing directory is: {self.module_dir_sc}" + '\033[0m')
        except Exception as e:
            raise RuntimeError(f"Error in 'variable_finder': {str(e)}")

    def add_modules_to_queue(self):
        """
        Add the modules referenced by USE statements in the current module to the queue for further analysis.
        """
        try:
            for module in walk(self.module_tree_sc, F23.Use_Stmt):
                module_name = None
                for entity in module.children:
                    if isinstance(entity, F23.Name):
                        module_name = entity.string
                        break
                if module_name is None:
                    raise ValueError("Module name not found in the Fortran code.")
                if module_name not in self.module_set_sc:
                    print('\033[38;5;27m' + f"Add! Module {module_name} is added into the queue." + '\033[0m')
                    self.queue_sc.append(module)
                    self.module_set_sc.add(module_name)
        except Exception as e:
            raise RuntimeError(f"Error in 'add_modules_to_queue': {str(e)}")
    
    def find_var_in_child_modules(self, key='variable'):
        """
        Traverse child modules in the queue to find a variable or subroutine. Updates the module directory and tree as needed.
        
        Args:
            key (str): Specifies whether to search for a 'variable' or 'subroutine'.
        """
        try:
            while self.queue_sc:
                module_name = None
                node = self.queue_sc.popleft()
                for entity in node.children:
                    if isinstance(entity, F23.Name):
                        module_name = entity.string
                        print("Checking the child module ....", module_name)
                        break
                if module_name is None:
                    raise ValueError("Module name not found in the Fortran code.")
                if module_name in self.visited_modules_sc:
                    print('\033[38;5;208m' + f"Pass! Module {module_name} has already been visited." + '\033[0m')
                    continue
                self.child_modules_sc.add(module_name)
                module_filename_lower = os.path.join(self.module_dir_sc, module_name + ".f90")
                module_filename_upper = os.path.join(self.module_dir_sc, module_name + ".F90")

                # Check for module files and parse them
                if os.path.exists(module_filename_lower) or os.path.exists(module_filename_upper):
                    child_module_tree = Processor().parse_fortran_file(
                        module_filename_lower if os.path.exists(module_filename_lower) else module_filename_upper)
                    self.visited_modules_sc.add(module_name)
                    self.module_tree_sc = child_module_tree
                    self.add_modules_to_queue()
                    if key == 'variable':
                        self.find_variable_in_module()
                    elif key == 'subroutine':
                        self.find_external_subroutine_in_module()
                    if self.return_key_sc:
                        return
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
                                    child_module_tree = Processor().parse_fortran_file(
                                        module_filename_lower if os.path.exists(module_filename_lower) else module_filename_upper)
                                    self.visited_modules_sc.add(module_name)
                                    self.module_dir_sc = full_directory
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
                    print('\033[38;5;196m' + "Warning: Queue is empty and return_key is still False! Extending queue with the main program!" + '\033[0m')
                    if not os.path.exists(os.path.join(self.module_dir_sc, self.main_fortran_file)):
                        parent_directory = os.path.abspath(os.path.join(self.module_dir_sc, '../'))
                        for directory in os.listdir(parent_directory):
                            if directory.startswith(self.main_dir):
                                full_directory = os.path.join(parent_directory, directory)
                                if os.path.exists(os.path.join(full_directory, self.main_fortran_file)):
                                    main_programm = os.path.join(full_directory, self.main_fortran_file)
                                    self.module_tree_sc = Processor().parse_fortran_file(main_programm)
                                    self.module_dir_sc = full_directory
                                    self.add_modules_to_queue()
                if not self.queue_sc and not self.return_key_sc and self.full_scout:
                    raise RuntimeError("Queue is empty, return key is False, and full scout is True. Unable to proceed.")
        except Exception as e:
            raise RuntimeError(f"Error in 'find_var_in_child_modules': {str(e)}")
