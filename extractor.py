import os, sys
from processor import Processor
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from navigator import Navigator, FortranSearcher
from shaper import Shaper
import re
from fparser.common.readfortran import FortranStringReader
from collections import defaultdict, deque
import shutil 

class Extractor:
    """
    """
    def __init__(self, module_dir, module_tree, logger=None):
        """
        """
        try:
            self.module_dir = module_dir
            self.module_tree = module_tree
            self.logger = logger
            self.subroutine_keys_all = set()
            self.subroutine_keys_ncl = set()
            self.subroutines = defaultdict()
            self.func_result = defaultdict()
            self.dummy_arg_list = defaultdict(list)
            self.actual_arg_spec_list = defaultdict(list)
            self.external_subroutines = set()
            self.call_subroutines = defaultdict(list)
            self.call_within_sub = defaultdict(lambda: defaultdict(list))
            self.loop_dict = defaultdict(lambda: defaultdict(set)) #defaultdict(set)
            self.loop_vect = defaultdict(lambda: None)
            self.exclude = {'kjpindex', 'nslm', 'nstm', 'nvm', 'nsnow', 'nice', 'ncirc', 'DIM', 'dim', 'MASK', 'next_calc_loop'}
            self.cases_to_exclude = [
                    'clear', 
                    'finalize', 
                    'init', 
                    'initialize', 
                    'read', 
                    'write',
                    'albedo_surface_soilalb'
                    ]
            self.allowed_external_subroutines = {
                    'ipslerr_p', 
                    'xios_orchidee_send_field', 
                    'xios_orchidee_recv_field', 
                    'flinget', 
                    'flininfo', 
                    'scatter',
                    'getin', 
                    'bcast'}
            self.dec_global = defaultdict(lambda: defaultdict(list))
            self.all_array_info = defaultdict(lambda: defaultdict(list))
            self.imp_shape = defaultdict(dict)
            self.scalar_variables = defaultdict(list)
            self.shapes_variables = defaultdict(list)
            self.var_modif_info = defaultdict(lambda: defaultdict(list))
            self.general_usage_dict = defaultdict()
            self.parsed_modules = defaultdict()
            self.var_global = defaultdict(list)
            self.var_dummy = defaultdict(list)
            self.var_local = defaultdict(list)
            self.var_modif = defaultdict(set)
            self.var_in_local = defaultdict(set)
            self.var_local_names = defaultdict(set)
            self.var_declared = defaultdict(set)
            self.module_global_stock = {}
            self.module_path = {}
            self.org_files_loaded = set()
            self.processor = Processor(logger=self.logger)
            self.procedure_search = FortranSearcher(self.module_path, self.parsed_modules, self.org_files_loaded, logger=self.logger)
        except Exception as e:
            self.processor.logger.exception(f"Error in __init__: ", e)
            raise

    def extract_loop_vect(self, subroutine_key, subroutine_tree):
        """
        Identifies and extracts the vector loop structure from a given subroutine.

        This method searches for non-labeled `DO` loops in the subroutine body
        that are structured like:
        DO index = start, end, stride
        Specifically, it looks for loops whose end value is `'kjpindex'`, 
        which is used here as an indicator of a vectorized loop.

        For each such loop:
            - If `kjpindex` is used as the upper bound, the loop is stored.
            - If a vector loop is already stored and a new one is found,
                the method checks for consistency. If they differ, it raises an error.

        Skips any loops that involve logical operations (e.g., `Or_Operand`) or 
        part references (e.g., array accesses) in their structure.

        Parameters:
            subroutine_key (str): Name of the current subroutine.
            subroutine_tree (Fparser node): Parsed tree of the subroutine.

        Raises:
            RuntimeError: If an error occurs during parsing or inconsistency is detected.
            ValueError: If multiple inconsistent vector loops are found in the same subroutine.
        """
        try:
            for loop in walk(subroutine_tree, F23.Nonlabel_Do_Stmt):
                if walk(loop, F23.Or_Operand) or walk(loop, F23.Part_Ref):
                    continue
                line_parts = loop.tostr().split('=')
                loop_index = line_parts[0].split()[-1]
                start_end_stride_values = line_parts[1].split(',')
                loop_start = start_end_stride_values[0].strip()
                loop_end = start_end_stride_values[1].strip()
                if loop_end and loop_end == 'kjpindex':
                    loop_string = loop.tostr()
                    if self.loop_vect[subroutine_key] is not None:
                        if self.loop_vect[subroutine_key] != loop_string:
                            raise ValueError(
                                    f"Inconsistent vector loops for {subroutine_key}: "
                                    f"existing '{self.loop_vect[subroutine_key]}' and new '{loop_string}' do not match."
                                    )
                    else:
                        self.loop_vect[subroutine_key] = loop_string
        except Exception as e:
            self.processor.logger.exception(f"Error in extract_loop_indices: ", e)
            raise

    def find_subroutines(self):
        """
        """
        # Iterate through all subroutine subprograms in the module AST
        current_module_name = walk(self.module_tree, F23.Module_Stmt)[0].children[1].tostr()
        module_subroutines_queue = deque(walk(self.module_tree, F23.Subroutine_Subprogram))
        module_subroutines_avail = set(stmt.children[1].tostr() for stmt in walk(self.module_tree, F23.Subroutine_Stmt))

        #for sub in walk(self.module_tree, F23.Subroutine_Subprogram):
        while module_subroutines_queue:
            sub = module_subroutines_queue.popleft()
            subroutine_key, dummy_arg_list = None, None

            # Extract the main subroutine statement
            subroutine_stmt = walk(sub, F23.Subroutine_Stmt)[0]
            call_stmt = walk(sub, F23.Call_Stmt)

            # Parse subroutine statement children to extract name and dummy arguments
            for child in subroutine_stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Name):
                    subroutine_key = child.tostr()
                elif isinstance(child, F23.Dummy_Arg_List):
                    dummy_arg_list = child
                else:
                    raise ValueError(f"Unexpected type '{type(child)}' encountered in children.")

            # Validate subroutine key extraction
            assert subroutine_key is not None, f"Unexpected type {subroutine_key} encountered in children."

            for loop in walk(sub, F23.Nonlabel_Do_Stmt):
                if len(loop.children) < 2 or loop.children[1] is None:
                    continue

                if walk(loop, F23.Or_Operand) or walk(loop, F23.Part_Ref):
                    continue

                loop_control = loop.children[1]

                # The structure is: (None, (loop_var, [start, end, stride]), None)
                if (len(loop_control.children) >= 2 and isinstance(loop_control.children[1], tuple) and len(loop_control.children[1]) >= 2):
                    loop_var_tuple = loop_control.children[1]
                    loop_index = loop_var_tuple[0].tostr()  # The loop variable (e.g., 'j')
                    bounds_list = loop_var_tuple[1]         # The bounds [start, end, stride]
                    # Check if we have at least start and end bounds
                    if len(bounds_list) >= 2 and bounds_list[1] is not None:
                        loop_end = bounds_list[1].tostr()   # The end bound (e.g., 'm')
                        if loop_end:
                            self.loop_dict[subroutine_key][loop_end].add(loop_index)

            # Filter out subroutines with excluded naming patterns
            check = all(case not in subroutine_key for case in self.cases_to_exclude)
            if not check:
                continue

            # Register the subroutine in internal data structures
            self.subroutine_keys_all.add(subroutine_key)
            self.subroutines[subroutine_key] = sub

            # Extract dummy arguments if present
            if dummy_arg_list is not None:
                assert isinstance(dummy_arg_list, F23.Dummy_Arg_List), f"Expected dummy_arg_list, got {type(dummy_arg_list).__name__.lower()}"
                for child in dummy_arg_list.children:
                    self.dummy_arg_list[subroutine_key].append(child.tostr())

            # Process call statements within the subroutine
            if call_stmt:
                for item in call_stmt:
                    call_name = item.children[0].tostr()

                    # Skip calls to excluded subroutines
                    check = all(case not in call_name for case in self.cases_to_exclude)
                    if not check:
                        continue

                    actual_arg_spec_list = item.children[1]
                    assert call_name is not None, f"Unexpected type {subroutine_key} encountered in children."

                    # Classify as internal or external call (ioipsl, xios)
                    if call_name not in self.allowed_external_subroutines:
                        self.call_within_sub[subroutine_key][call_name].append(item) #.add(call_name)
                        if call_name not in module_subroutines_avail:
                            self.processor.logger.warning(
                                f"Subroutine '{subroutine_key}' calls '{call_name}' which is not defined in current module")
                            self.external_subroutines.add(call_name)
                            found, module_file_path, module_tree = self.procedure_search.search_subroutine_in_dependencies(call_name, current_module_name, self.module_dir)
                            if found:
                                self.processor.logger.info(
                                        f"Found external subroutine '{call_name}' in file: {call_name}, adding to processing queue")
                                # Add the found subroutine to the right end of the queue for processing
                                all_subroutines_in_module = walk(module_tree, F23.Subroutine_Subprogram)
                                for subroutine_subprogram in all_subroutines_in_module:
                                    module_subroutines_queue.append(subroutine_subprogram)
                                    subroutine_stmt = walk(subroutine_subprogram, F23.Subroutine_Stmt)[0]
                                    for child in subroutine_stmt.children:
                                        if isinstance(child, F23.Name):
                                            sub_name = child.tostr()
                                            module_subroutines_avail.add(sub_name)
                        
                    else:
                        continue
                        # call to ioipsl, xios are allowed, so, instead of call_name, subroutine_key is added!
                        #self.subroutine_keys_ncl.add(subroutine_key) 

                    # Extract and store actual arguments from call
                    if actual_arg_spec_list is not None:
                        assert isinstance(actual_arg_spec_list, F23.Actual_Arg_Spec_List), f"Expected actual_arg_spec_list, got {type(actual_arg_spec_list).__name__.lower()}"
                        arg_string = []
                        for child in actual_arg_spec_list.children:
                            arg_string.append(child.tostr())
                        #arg_string = [child.tostr() for child in arg_list.children]
                        self.actual_arg_spec_list[call_name].append(arg_string)

                    # Register the call statement
                    self.call_subroutines[call_name].append(item)
            else:
                # Subroutine has no call statements or call to ioipsl, xios
                self.subroutine_keys_ncl.add(subroutine_key)

        # Identify external subroutines (called but not defined in module)
        self.external_subroutines.update(self.allowed_external_subroutines)

    def extract_function_dummy_args(self, function_tree):
        """
        Extract dummy arguments from a function subprogram.
    
        Parameters
        ----------
        function_subprogram : fparser.two.Fortran2003.Function_Subprogram
            The parsed function subprogram
        
        Returns
        -------
        list
            List of dummy argument names
        """

        function_stmt = walk(function_tree, F23.Function_Stmt)[0]
        for child in function_stmt.children:
            if child is None:
                continue
            if isinstance(child, F23.Name):
                function_key = child.tostr()
                self.subroutines[function_key] = function_tree
            elif isinstance(child, F23.Dummy_Arg_List):
                arg_list = child
            elif isinstance(child, F23.Suffix):
                self.func_result[function_key] = child.children[0].tostr()
        assert function_key is not None, f"Unexpected type {function_key} encountered in children."
        if arg_list is not None:
            for child in arg_list.children:
                self.dummy_arg_list[function_key].append(child.tostr())
        
        self.subroutines[function_key] = function_tree


    def _process_intent_assignment_statement(self, child, child_parent, dummy_arg_list, usage):
        """
        """
        lhs_expr = child.items[0]
        rhs_expr = child.items[-1]

        # Find all variable names in this assignment statement
        for name in walk(child, F23.Name):
            var_name = name.tostr()

            # Compute read/write once
            is_write = any(var_name == node.tostr() for node in walk(lhs_expr, F23.Name))
            is_read = any(var_name == node.tostr() for node in walk(rhs_expr, F23.Name))
            # Only process if this is a dummy argument
            if var_name in dummy_arg_list:

                # CONDITIONAL CONTEXT TRACKING
                # Check if we're inside an IF construct AND we already recorded a first use
                # AND that first use was in the same IF construct
                if isinstance(child_parent, F23.If_Construct):

                    # Get the index of current child within the IF construct
                    current_index = child_parent.children.index(child)

                    # Find all branch boundary indices (IF, ELSEIF, ELSE)
                    branch_boundaries = []
                    for i, stmt in enumerate(child_parent.children):
                        if isinstance(stmt, (F23.If_Then_Stmt, F23.Else_If_Stmt, F23.Else_Stmt)):
                            branch_boundaries.append(i)
                    
                    # Determine which branch the current statement is in
                    current_branch = 0
                    for boundary_index in sorted(branch_boundaries):
                        if current_index > boundary_index:
                            current_branch += 1
                        else:
                            break

                    # RESET first_use_update for this variable at the start
                    usage[var_name]['first_use_update'] = False

                    # Check if we should allow update
                    if usage[var_name]['first_use_assign'] is not None and usage[var_name]['first_use_assign'].parent == child_parent:
                        first_use_index = child_parent.children.index(usage[var_name]['first_use_assign'])

                        # Determine which branch the first use was in
                        first_use_branch = 0
                        for boundary_index in sorted(branch_boundaries):
                            if first_use_index > boundary_index:
                                first_use_branch += 1
                            else:
                                break

                        # Only update if we're in a DIFFERENT branch than the first use
                        if current_branch != first_use_branch:
                            usage[var_name]['first_use_update'] = True


                # UPDATE FIRST USE ASSIGNMENT
                # If this is the first time we see this variable OR we're in update mode
                #if is_write:
                if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                        # Record this statement as the (current) first use
                        usage[var_name]['first_use_assign'] = child

                # CHECK IF VARIABLE IS IN ARRAY SUBSCRIPT (read-only context)
                if isinstance(name.parent, (F23.Section_Subscript_List, F23.Subscript_Triplet)):
                    # Array subscripts are always read-only usage
                    if usage[var_name]['intent'] is None:
                        usage[var_name]['intent'] = 'IN'
                else:
                    # CHECK LEFT-HAND SIDE USAGE (WRITE)
                    if is_write:
                        if usage[var_name]['intent'] is None:
                            # First usage is write → INTENT(OUT)
                            usage[var_name]['intent'] = 'OUT'
                        elif usage[var_name]['intent'] == 'IN':
                            # Was read-only, now being written → INTENT(INOUT)
                            usage[var_name]['intent'] = 'INOUT'

                    # CHECK RIGHT-HAND SIDE USAGE (READ)
                    if is_read:
                        if usage[var_name]['intent'] is None:
                            # First usage is read → INTENT(IN)
                            usage[var_name]['intent'] = 'IN'
                        # safer for inout than checking out
                        elif usage[var_name]['intent'] == 'OUT': #and usage[var_name]['first_use_assign'] == child:
                            # Special case: reading and writing in SAME first use statement → INTENT(INOUT)
                            usage[var_name]['intent'] = 'INOUT'


    def _process_intent_read_only_statement(self, child, child_parent, dummy_arg_list, usage):
        """
        """
        for name in walk(child, F23.Name):
            var_name = name.tostr()
            if var_name in dummy_arg_list:
                # Conditional context tracking
                if isinstance(child_parent, F23.If_Construct) and \
                        usage[var_name]['first_use_assign'] is not None and \
                        usage[var_name]['first_use_assign'].parent == child_parent:
                    usage[var_name]['first_use_update'] = True

                # Update first_use_assign
                if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                    usage[var_name]['first_use_assign'] = child

                # All variables in DO statements are read-only
                if usage[var_name]['intent'] is None:
                    usage[var_name]['intent'] = 'IN'

    def _process_intent_call_statement(self, child, child_parent, dummy_arg_list, usage, within_calls):
        """
        """
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
                    actual_args = [arg.tostr() for arg in grandchild.children]

                    # Process by POSITION to handle duplicate arguments correctly
                    for i, name in enumerate(grandchild.children):
                        var_name = name.tostr()
                        if var_name in dummy_arg_list:
                            # Conditional context tracking
                            if isinstance(child_parent, F23.If_Construct) and \
                                    usage[var_name]['first_use_assign'] is not None and \
                                    usage[var_name]['first_use_assign'].parent == child_parent:
                                usage[var_name]['first_use_update'] = True

                            # Update first_use_assign
                            if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                                usage[var_name]['first_use_assign'] = child

                            # Map by POSITION: actual argument i → dummy argument i
                            corresponding_element = self.dummy_arg_list[call_name][i]
                            call_intent = self.general_usage_dict[call_name][corresponding_element]
                            current_intent = usage[var_name]['intent']

                            if current_intent is None:
                                usage[var_name]['intent'] = call_intent
                            elif current_intent == 'IN' and call_intent in {'OUT', 'INOUT'}:
                                usage[var_name]['intent'] = 'INOUT'
                            elif current_intent == 'OUT' and call_intent == 'INOUT':
                                usage[var_name]['intent'] = 'INOUT'
                            # current_intent == 'INOUT' remains 'INOUT' (implicit)

    def extract_intent(self, subroutine_key, subroutine_tree, within_calls=None):
        """
        """
        dummy_arg_list = self.dummy_arg_list[subroutine_key]
        usage = {arg: {'intent': None, 'first_use_assign': None, 'first_use_update': False} for arg in dummy_arg_list}
        def traverse_block(block):
            if hasattr(block, "content"):
                for child in block.content:
                    child_parent = child.parent
                    if isinstance(child, F23.Assignment_Stmt):
                        self._process_intent_assignment_statement(child, child_parent, dummy_arg_list, usage)
                    elif isinstance(child, (
                        F23.Nonlabel_Do_Stmt,
                        F23.If_Then_Stmt, 
                        F23.Else_If_Stmt,
                        F23.Where_Construct_Stmt, 
                        F23.Masked_Elsewhere_Stmt,
                        F23.Select_Case_Stmt, 
                        F23.Case_Stmt
                        )
                        ):
                        self._process_intent_read_only_statement(child, child_parent, dummy_arg_list, usage)

                    elif isinstance(child, F23.Call_Stmt):
                        self._process_intent_call_statement(child, child_parent, dummy_arg_list, usage, within_calls)
                    # ToDo, pointer assosiation, etc  
                    else:
                        traverse_block(child)
        execution_part = walk(subroutine_tree, F23.Execution_Part)[0]
        traverse_block(execution_part)
        self.general_usage_dict[subroutine_key] = {var: props['intent'] for var, props in usage.items()}
        #self.processor.logger.info(f"Induced INTENT for subroutine '{subroutine_key}':")
        #for var, props in usage.items():
        #    intent = props['intent'] if props['intent'] is not None else 'UNKNOWN'
        #    self.processor.logger.info(f"  '{var}': '{intent}'")

    @staticmethod
    def add_intent(block, intent):
        """
        Adds an `INTENT` attribute to a Fortran type declaration block.

        This function inspects a given `F23.Type_Declaration_Stmt` AST node, extracts its type,
        shape, and declared variables, and reconstructs it with the specified `INTENT` attribute 
        (e.g., `INTENT(IN)`, `INTENT(OUT)`, or `INTENT(INOUT)`).

        Parameters:
            block (F23.Type_Declaration_Stmt): 
                The Fortran type declaration statement node to modify. Must contain
                intrinsic type, shape specification, and entity declarations.
            intent (str): 
                The desired INTENT attribute to inject (e.g., "in", "out", "inout").

        Returns:
            F23.Type_Declaration_Stmt: 
                A new type declaration node with the added INTENT attribute.

        Raises:
            AssertionError: If `block` is not of type `F23.Type_Declaration_Stmt`.
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
        if not intrinsic_type_spec:
            raise ValueError("Could not find intrinsic type specification in the block")
        if not entity_decl_list:
            raise ValueError("Could not find entity declaration list in the block")

        attributes = []
        if explicit_shape_spec_list:
            attributes.append(f"dimension({explicit_shape_spec_list})")
        attributes.append(f"intent({intent})")
        attributes_str = ",".join(attributes)
        return   F23.Type_Declaration_Stmt(
                f'{intrinsic_type_spec},{attributes_str}::{entity_decl_list}'
                )

    def clean_subroutine(self, subroutine_key, subroutine_tree):
        """
        Validates and corrects INTENT specifications for dummy arguments in a given subroutine.

        This method traverses all type declaration statements in the subroutine and ensures that:
        - Each dummy argument (from the subroutine's signature) has an `INTENT` attribute.
        - The `INTENT` (IN, OUT, or INOUT) matches the intent previously inferred using `extract_intent`.
        - If multiple variables are declared in one statement, they are separated into individual declarations 
        for more accurate handling.
        - If the existing `INTENT` is missing or incorrect, it modifies the statement accordingly.
        - Warnings are printed for any inconsistencies found (e.g., undeclared usage or missing intents).

        The method modifies the subroutine tree **in-place** by:

        - Replacing or inserting corrected declaration statements.
        - Using `self.general_usage_dict[subroutine_key]` to lookup expected intents.
        - Calling `self.add_intent()` to apply missing intent attributes.

        Parameters:
        -----------
        subroutine_key : str
            The name of the subroutine to clean.

        subroutine_tree : Fortran2003.Subroutine_Subprogram
            The parsed tree (AST node) of the subroutine being cleaned.

        Notes:
        ------
        - This method assumes that `extract_intent()` has already been run, and intent info is available in 
            self.general_usage_dict`.
        - The processor internally uses `walk()` and `Processor().separate_entity_declarations()` to navigate and 
            restructure declaration blocks where needed.
    
        Returns:
        --------
            None (modifies subroutine_tree in-place)
        """
        def traverse_subroutine(block):
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    child = block.content[idc]
                    if isinstance(child, F23.Type_Declaration_Stmt):
                        intent = walk(child, F23.Intent_Spec)
                        entity_decls = walk(child, F23.Entity_Decl)
                        if intent:
                            intent_spec = intent[0].tostr()
                        if len(entity_decls) > 1:
                            self.processor.logger.warning(
                                    f"Expected exactly one Entity_Decl but found {len(entity_decls)}. "
                                    f"Found: {[decl.tostr() for decl in entity_decls]}. Breaking ...  "
                                    )
                            for stmt in self.processor.separate_entity_declarations(child):
                                stmt.parent = block
                                entity_decls = walk(stmt, F23.Entity_Decl)
                                assert len(entity_decls) == 1, \
                                        f"walk(declaration_stmt, F23.Entity_Decl) should return exactly one, but got {len(entity_decls)}"
                                name = entity_decls[0].tostr()
                                if intent:
                                    intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                    if intent_spec_exp is None:
                                        self.processor.logger.warning(f"Name '{name}' is not used. Declaration: {stmt.tostr()}")
                                    else:
                                        if intent_spec_exp != intent_spec:
                                            self.processor.logger.warning("The intent is incorrect. Correction block")
                                            self.processor.logger.warning(f"Name: '{name}', Expected: '{intent_spec_exp}', Found: '{intent_spec}'")
                                            obj_org = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec)
                                            obj_mod = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec_exp)
                                            self.processor.logger.info(f"Original Declaration Statement: {stmt.tostr()}")
                                            child_string = stmt.tostr().replace(obj_org.tostr(), obj_mod.tostr())
                                            stmt = F23.Type_Declaration_Stmt(child_string)
                                            self.processor.logger.info(f'Modified Declaration Statement: {child_string}')
                                else:
                                    if name in self.dummy_arg_list[subroutine_key]:
                                        self.processor.logger.warning(f"Name '{name}' is a dummy argument without intent.")
                                        self.processor.logger.warning(f"Original Declaration Statement: {stmt.tostr()}")
                                        intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                        if intent_spec_exp is not None:
                                            self.processor.logger.warning(f"The expected intent is: {intent_spec_exp}")
                                            stmt = self.add_intent(stmt, intent_spec_exp)
                                            self.processor.logger.info(f"Modified Declaration Statement: {stmt.tostr()}")
                                        else:
                                            self.processor.logger.warning(f"Name {name} is not used. Declaration: {stmt.tostr()}")
                                block.content.insert(idc + 1, stmt)
                            del block.content[idc]
                        else:
                            name = entity_decls[0].children[0].tostr()
                            if intent:
                                intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                if intent_spec_exp is None:
                                    self.processor.logger.warning(f"Name '{name}' is not used. Declaration: {child.tostr()}")
                                else:
                                    if intent_spec_exp != intent_spec:
                                        self.processor.logger.warning("The intent is incorrect. Correction block")
                                        self.processor.logger.warning(f"Name '{name}', Expected: '{intent_spec_exp}', Found: '{intent_spec}'")
                                        obj_org = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec)
                                        obj_mod = F23.Intent_Attr_Spec('INTENT(%s)'%intent_spec_exp)
                                        self.processor.logger.warning(f"Original Declaration Statement: {child.tostr()}")
                                        child_string = child.tostr().replace(obj_org.tostr(), obj_mod.tostr())
                                        block.content[idc] = F23.Type_Declaration_Stmt(child_string)
                                        self.processor.logger.warning(f"Modified Declaration Statement: {child_string}")
                            else:
                                if name in self.dummy_arg_list[subroutine_key]:
                                    self.processor.logger.warning(f"Name '{name}' is a dummy argument without intent.")
                                    self.processor.logger.warning(f"Original Declaration Statement: {child.tostr()}")
                                    intent_spec_exp = self.general_usage_dict[subroutine_key][name]
                                    if intent_spec_exp is not None:
                                        self.processor.logger.warning("Its expected intent is: '{intent_spec_exp}'")
                                        block.content[idc] = self.add_intent(child, intent_spec_exp)
                                        self.processor.logger.warning("Modified Declaration Statement: {block.content[idc].tostr()}")
                                    else:
                                        self.processor.logger.warning(f"Name '{name}' is not used in declaration: {child.tostr()}")
                    else:
                        traverse_subroutine(child)
                    idc += 1
        traverse_subroutine(subroutine_tree)

    def find_variables(self, subroutine_tree, subroutine_key, parent_subroutine_key=None):
        """
        """

        shapes = {}
        self.var_dummy[subroutine_key].clear()
        self.var_local[subroutine_key].clear()

        specification_part = None
        execution_part = None

        # Walk through the function_tree content to find parts
        for idx, item in enumerate(subroutine_tree.content):
            if isinstance(item, F23.Specification_Part):
                specification_part = item
            elif isinstance(item, F23.Execution_Part):
                execution_part = item
        # Check that both parts were found and have content
        if specification_part is None:
            raise ValueError("Specification_Part not found in function_tree")
        if execution_part is None:
            raise ValueError("Execution_Part not found in function_tree")
        if not hasattr(specification_part, 'content'):
            raise AttributeError("Specification_Part has no 'content' attribute")
        if not hasattr(execution_part, 'content'):
            raise AttributeError("Execution_Part has no 'content' attribute")

        stmt_list = specification_part.content
        idx = 0
        while idx < len(stmt_list):
            stmt = stmt_list[idx]
            if isinstance(stmt, F23.Type_Declaration_Stmt):
                entity_decls = walk(stmt, F23.Entity_Decl)
                if len(entity_decls) > 1:
                    new_stmts = self.processor.separate_entity_declarations(stmt)
                    for new_stmt in new_stmts:
                        new_stmt.parent = specification_part
                    stmt_list[idx:idx+1] = new_stmts
                    idx += len(new_stmts)-1
                    continue
            idx += 1

        self.var_declared[subroutine_key] = {name.tostr() for name in  walk(specification_part, F23.Entity_Decl)}
        names_declared, names_used = walk(specification_part, F23.Name), walk(execution_part, F23.Name)

        declared_names_str = {name.string for name in names_declared }
        
        seen = {}
        for name in names_used:
            if (
                    name.string not in declared_names_str and 
                    name.string not in self.exclude
                    ):
                seen[name.string] = name

        self.var_global[subroutine_key] = list(seen.values()) #var_used - var_declared

        for idx, node in enumerate(specification_part.children):
            if isinstance(node, F23.Type_Declaration_Stmt):
                assert len(walk(node, F23.Entity_Decl)) == 1,\
                        "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                implicit_shape = walk(node, F23.Assumed_Shape_Spec)
                intrinsic_name = walk(node, F23.Intrinsic_Name)
                if implicit_shape:
                    self.processor.logger.warning(f"Implicit shape detected in the declaration {node}")
                    if isinstance(subroutine_tree, F23.Subroutine_Subprogram):
                        shape_finder = Shaper(self.module_dir, self.parsed_modules, self.module_path,\
                                self.dummy_arg_list, self.actual_arg_spec_list, \
                                self.call_subroutines, logger=self.logger)
                        explicit_node = shape_finder.shaper_subroutine(node, subroutine_key)
                        self.processor.logger.info(f"An explicit similar declaration is found: {explicit_node}")
                        node = self.processor.map_declaration(node, explicit_dec=explicit_node, dimensions=None)
                        entity_decl = walk(node, F23.Entity_Decl)[0].tostr()
                        if entity_decl not in self.imp_shape[subroutine_key]:
                            self.imp_shape[subroutine_key][entity_decl] = node
                        specification_part.children[idx] = node

                    elif isinstance(subroutine_tree, F23.Function_Subprogram):
                        assert parent_subroutine_key is not None, "Error: 'parent_subroutine_key' must not be None."
                        shape_finder = Shaper(self.module_dir, self.parsed_modules, self.module_path, self.dummy_arg_list, logger=self.logger)
                        node = shape_finder.shaper_function(node, subroutine_tree, subroutine_key, self.all_array_info[parent_subroutine_key])
                        self.processor.logger.info(f"An explicit similar declaration is found: {node}")
                        entity_decl = walk(node, F23.Entity_Decl)[0].tostr()
                        if entity_decl not in self.imp_shape[subroutine_key]:
                            self.imp_shape[subroutine_key][entity_decl] = node
                if intrinsic_name:
                    self.processor.logger.warning(f"Intrinsic name detected in the declaration {node}")
                    if isinstance(subroutine_tree, F23.Function_Subprogram):
                        shape_finder = Shaper(self.module_dir, self.parsed_modules, self.module_path, self.dummy_arg_list, logger=self.logger)
                        node = shape_finder.shaper_intrinsic_size(node)
                        self.processor.logger.info(f"An explicit similar declaration is found: {node}")
                        #node = nodes
                    else:
                        raise ValueError(f"intrinsic_name found in {type(subroutine_tree).__name__} declarations! "
                                f"This case is not implemented yet!")
                
                entity_decls = walk(node, F23.Entity_Decl)
                assert len(entity_decls) == 1,\
                        "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                name = entity_decls[0].tostr()
                for shape_spec in walk(node, F23.Explicit_Shape_Spec):
                    for dim in walk(shape_spec, F23.Name):
                        dim_str = dim.tostr()
                        if (
                                dim_str not in self.dummy_arg_list[subroutine_key] and 
                                dim_str not in self.exclude
                                ):
                            shapes[dim_str] = dim
                if name in self.dummy_arg_list[subroutine_key]:
                    if name not in self.exclude:
                        self.var_dummy[subroutine_key].append(node)
                else:
                    if (
                            name == subroutine_key 
                            or (
                                subroutine_key in self.func_result 
                                and name == self.func_result[subroutine_key]
                                )
                            ):
                        new_decl = self.add_intent(node, "out")
                        self.var_dummy[subroutine_key].append(new_decl)
                    else:
                        self.var_local[subroutine_key].append(node)

        self.var_dummy[subroutine_key].sort(key=lambda node: node.children[-1].tostr().lower())
        existing_names = {v.tostr() for v in self.var_global[subroutine_key]}
        self.var_global[subroutine_key].extend(
                dim_node for dim_str, dim_node in seen.items()
                if dim_str not in existing_names
                )

        for item in self.var_local[subroutine_key]:
            for entity in walk(item, F23.Entity_Decl):
                for child in entity.children:
                    if isinstance(child, F23.Name):
                        self.var_local_names[subroutine_key].add(child.tostr())
    
    def extract_local_in_variables(self, subroutine_key, subroutine_tree):
        """
        """
        try:
            # Initialize the set for this subroutine if it doesn't exist
            if subroutine_key not in self.var_in_local:
                self.var_in_local[subroutine_key] = set()
            
            # Walk through all declaration statements in the subroutine
            for node in walk(subroutine_tree, F23.Type_Declaration_Stmt):
                try:
                    # Get the variable name from Entity_Decl - should be exactly one
                    entity_decls = walk(node, F23.Entity_Decl)
                    assert len(entity_decls) == 1, \
                        f"In extract_local_variables: walk(node, F23.Entity_Decl)=1, but got {len(entity_decls)}."
                        
                    name = entity_decls[0].children[0].tostr()
                    
                    # Check intent specifications
                    intent_specs = walk(node, F23.Intent_Attr_Spec)
                    
                    if intent_specs:
                        # Variable has intent specification
                        if F23.Intent_Attr_Spec('INTENT(IN)') in intent_specs:
                            if name not in self.exclude:
                                self.var_in_local[subroutine_key].add(name)
                        # For INTENT(OUT) or INTENT(INOUT), don't add to var_in_local
                        # as they can be modified
                    else:
                        # check the function result
                        if (
                                name == subroutine_key
                                or (
                                    subroutine_key in self.func_result
                                    and name == self.func_result[subroutine_key]
                                    )
                                ):
                            continue
                        else:
                            # No intent specification - this is a local variable
                            self.var_in_local[subroutine_key].add(name)
                            
                except Exception as e:
                    self.processor.logger.exception(f"Error processing declaration statement: {node.tostr()}", e)
                    raise
                    
        except Exception as e:
            self.processor.logger.exception(f"Error extracting local variables for '{subroutine_key}': ", e)
            raise

    def extract_modified_variables(self, subroutine_key, subroutine_tree):
        """
        """
        try:
            # Initialize the set for this subroutine if it doesn't exist
            if subroutine_key not in self.var_modif:
                self.var_modif[subroutine_key] = set()

            var_in_local = self.var_in_local[subroutine_key]
            dec_global = self.dec_global[subroutine_key]
            var_dummy_list = self.var_dummy[subroutine_key]

            # Walk through all assignment statements in the subroutine
            for assign_stmt in walk(subroutine_tree, F23.Assignment_Stmt):
                try:
                    lhs = assign_stmt.items[0]

                    if isinstance(lhs, F23.Name):
                        # Simple variable assignment: var = value
                        var_name = lhs.tostr()
                        if var_name not in var_in_local:
                            self.var_modif[subroutine_key].add(var_name)

                    elif isinstance(lhs, F23.Part_Ref):
                        # Array or function reference: arr(i) = value or func() = value
                        base_var_name = lhs.children[0].tostr()
                        if base_var_name not in var_in_local:
                            self.var_modif[subroutine_key].add(base_var_name)

                    else:
                        raise ValueError(
                            f"Unexpected assignment left-hand side type: {type(lhs)} "
                            f"in statement: {assign_stmt.tostr()}"
                        )
                except Exception as e:
                    self.processor.logger.exception(f"Error processing assignment statement: {assign_stmt.tostr()}", e)
                    raise

            # Process global declarations for modified variables info
            for key in dec_global:
                for item in dec_global[key]:
                    try:
                        if isinstance(item, F23.Type_Declaration_Stmt):
                            var_type = item.children[0].children[0]
                            entity_decls = walk(item, F23.Entity_Decl)
                            assert len(entity_decls) == 1, \
                                f"In extract_modified_variables: walk(item, F23.Entity_Decl)=1, but got {len(entity_decls)}."
                            entity_decl = entity_decls[0].children[0].tostr()

                            is_var_modified = entity_decl in self.var_modif[subroutine_key]

                            if is_var_modified:
                                self.var_modif_info[subroutine_key][entity_decl].append(var_type)

                            # Handle explicit shape specifications
                            if walk(item, F23.Explicit_Shape_Spec):
                                if is_var_modified:
                                    self.var_modif_info[subroutine_key][entity_decl].append('DIMENSION')

                            # Handle ALLOCATABLE arrays
                            attr_spec = walk(item, F23.Attr_Spec)
                            if F23.Attr_Spec('ALLOCATABLE') in attr_spec:
                                try:
                                    declaration_stmt = self.processor.combine_allocate_declaration(dec_global[key])
                                    assert isinstance(declaration_stmt, F23.Type_Declaration_Stmt), \
                                        f"Item is not of type F23.Type_Declaration_Stmt!"
                                    assert walk(declaration_stmt, F23.Explicit_Shape_Spec), \
                                        "In extract_modified_variables: failed to combine_allocate_declaration!"

                                    if is_var_modified:
                                        self.var_modif_info[subroutine_key][entity_decl].append('DIMENSION')
                                except Exception as e:
                                    self.processor.logger.exception(f"Error processing ALLOCATABLE declaration for {entity_decl}", e)
                                    raise

                    except Exception as e:
                        self.processor.logger.exception(f"Error processing global declaration item: {item}", e)
                        raise

            # Process dummy arguments for var_modif_info
            for item in var_dummy_list:
                try:
                    assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
                    var_type = item.children[0].children[0]
                    entity_decls = walk(item, F23.Entity_Decl)
                    assert len(entity_decls) == 1, \
                        f"In extract_modified_variables: walk(item, F23.Entity_Decl)=1, but got {len(entity_decls)}."
                    entity_decl = entity_decls[0].tostr()

                    is_var_modified = entity_decl in self.var_modif[subroutine_key]

                    if is_var_modified:
                        self.var_modif_info[subroutine_key][entity_decl].append(var_type)
                        if walk(item, F23.Explicit_Shape_Spec):
                            self.var_modif_info[subroutine_key][entity_decl].append('DIMENSION')
                except Exception as e:
                    self.processor.logger.exception(f"Error processing dummy argument item: {item}", e)
                    raise

            # Sort the var_modif_info for consistency
            if subroutine_key in self.var_modif_info:
                try:
                    sorted_inner = sorted(self.var_modif_info[subroutine_key].items())
                    self.var_modif_info[subroutine_key] = defaultdict(list, sorted_inner)
                except Exception as e:
                    self.processor.logger.exception(f"Error sorting var_modif_info for '{subroutine_key}': ", e)
                    raise

        except Exception as e:
            self.processor.logger.exception(f"Error extracting modified variables for '{subroutine_key}':", e)
            raise

    def find_global_variables(self, module_dir, module_tree, var_global, subroutine_key):
        """
        Recursively searches for the declarations of global variables and external procedures
        within the module hierarchy starting from a given module directory and parse tree.

        Parameters
        ----------
        module_dir : str
            The directory path of the current module where the search starts.

        module_tree : F23.Module
            The parse tree of the current module, typically obtained via a Fortran parser.

        var_global : set or list
            A collection of variable or procedure names (strings) that are considered global
            and need to be resolved within the module or its children.

        subroutine_key : str
            The name of the subroutine whose global variables are being resolved; used as a key
            in `self.dec_global` to store found declarations.

        Behavior
        --------
        - For each variable/procedure name in `var_global`, determines whether it is an external
            subroutine or a variable.
        - If it is a variable (not external), uses `Navigator.variable_finder` to search for its
            declaration within the given module tree and directory.
        - If found, stores the declarations in `self.dec_global[subroutine_key]` keyed by the variable name.
        - If additional unresolved variables are discovered during the search (`var_initial`),
            recursively searches child modules for these variables.
        - For external subroutines, uses `Navigator.external_subroutine_finder` to locate the
            procedure declarations and similarly stores them.
        - Raises an error if a variable or external procedure cannot be found in the accessible modules.

        Outputs
        -------
        - Prints progress messages with color highlighting to indicate the status of each search:
        - Searching (with spinner)
        - Found (success)
        - Attention messages for additional recursive searches.

        Raises
        ------
        ValueError
            When a global variable or external procedure is not found in the module hierarchy.
        """
        for var in var_global:
            assert isinstance(var, F23.Name), "Expected var_global to contain F23.Name nodes"
            declaration = var.tostr()
            if declaration in self.module_global_stock:
                cached_data = self.module_global_stock[declaration]
                decl_type = "procedure" if declaration in self.external_subroutines else "variable"
                self.processor.logger.info(f"ℹ️  Found '{decl_type}' '{declaration}' in global stock ➡️  reusing:")
                for i, item in enumerate(cached_data, 1):
                    self.processor.logger.info(f"   {i}. {item}")
                if walk(cached_data, F23.Function_Subprogram):
                    parent = self.processor.find_enclosing_parent(var, F23.Assignment_Stmt)
                    self.processor.logger.info(f"The global {declaration} used in {parent} is a Function_Subprogram.")
                    self.call_subroutines[declaration].append(parent)
                    self.call_within_sub[subroutine_key][declaration].append(parent)
                self.dec_global[subroutine_key][declaration] = cached_data
                #any_initialization
                var_initial = walk(walk(cached_data, F23.Initialization), F23.Name)
                if var_initial:
                    self.processor.logger.warning(f"Attention: there are additional variables to search: {var_initial}")
                    self.processor.logger.warning(f"In the directory: {module_dir}")
                    ffile = walk(module_tree, F23.Name)[0].string
                    self.processor.logger.warning(f"In the module: {ffile}")
                    self.find_global_variables(module_dir, module_tree, var_initial, subroutine_key)
                continue
            self.finder = Navigator(module_dir, module_tree, self.parsed_modules, self.module_path, logger=self.logger)
            if declaration not in self.external_subroutines:
                self.processor.logger.info(f"⏳... Searching for variable '{declaration}'")
                self.finder.variable_finder(declaration)
                if self.finder.var_declaration:
                    self.processor.logger.info("✅ Variable found!")
                    declaration_data = list(self.finder.var_declaration)
                    if walk(declaration_data, F23.Function_Subprogram):
                        function_name = declaration_data[0] 
                        function_subprogram = declaration_data[1]
                        module_name = declaration_data[2]
                        assert module_name in self.module_path, \
                                f"Module '{module_name}' not found in module_path. Available modules: {list(self.module_path.keys())}"
                        current_module_path = self.module_path[module_name]
                        path_to_original = current_module_path.replace('.f90', '_org.fgpt').replace('.F90', '_org.Fgpt')
                        if os.path.exists(path_to_original) and module_name not in self.org_files_loaded:
                            self.processor.logger.info(f"Loading original file for function '{declaration}': {path_to_original}")
                            original_module_tree = self.processor.parse_fortran_file(path_to_original)
                            self.parsed_modules[module_name] = original_module_tree
                            self.module_path[module_name] = current_module_path
                            self.org_files_loaded.add(module_name)
                            # Search for the function in the original file
                            for sub in walk(original_module_tree, F23.Function_Subprogram):
                                function_stmt = walk(sub, F23.Function_Stmt)[0]
                                for func_child in function_stmt.children:
                                    if isinstance(func_child, F23.Name) and func_child.tostr() == declaration:
                                        # Use the function from the original file
                                        function_name = func_child
                                        function_subprogram = sub
                                        break
                        elif not os.path.exists(path_to_original):
                            shutil.copy(current_module_path, path_to_original)
                            self.processor.logger.info(
                                    f"Created backup of original file: {path_to_original}")
                        parent = self.processor.find_enclosing_parent(var, F23.Assignment_Stmt)
                        self.processor.logger.info(f"The global {declaration} used in {parent} is a Function_Subprogram.")
                        self.call_subroutines[declaration].append(parent)
                        self.call_within_sub[subroutine_key][declaration].append(parent)
                        
                        self.processor.logger.info(
                                f"Calling Function_Subprogram {declaration} in Subroutine_Subprogram {subroutine_key}."
                                )
                        
                        assert isinstance(function_subprogram, F23.Function_Subprogram), (
                                f"Expected type 'F23.Function_Subprogram', but got '{type(function_subprogram).__name__}' instead.")

                        assert isinstance(function_name, F23.Name), (
                                f"Expected type 'F23.Name', but got '{type(function_name).__name__}' instead."
                                )
                        self.subroutines[function_name.tostr()] = function_subprogram
                        self.extract_function_dummy_args(function_subprogram)

                    self.dec_global[subroutine_key][declaration] = declaration_data
                    self.module_global_stock[declaration] = declaration_data
                else:
                    self.processor.logger.error(f"Variable '{declaration}' is not found in any child modules.")
                    raise
                if self.finder.var_initial:
                    self.processor.logger.warning(f"Attention: there are additional variables to search: '{self.finder.var_initial}'")
                    self.processor.logger.warning(f"In the directory: '{self.finder.module_dir_sc}'")
                    ffile = walk(self.finder.module_tree_sc, F23.Name)[0].string
                    self.processor.logger.warning(f"In the module: '{ffile}'")
                    self.find_global_variables(self.finder.module_dir_sc, self.finder.module_tree_sc, self.finder.var_initial, subroutine_key)
            elif declaration in self.external_subroutines:
                self.processor.logger.info("⏳... Searching for procedure '{declaration}'")
                self.finder.external_subroutine_finder(declaration)
                if self.finder.var_declaration:
                    self.processor.logger.info("✅ Procedure found!")
                    declaration_data = list(self.finder.var_declaration)
                    self.dec_global[subroutine_key][declaration] = declaration_data
                    self.module_global_stock[declaration] = declaration_data
                else:
                    self.processor.logger.error(f"Procedure '{declaration}' is not found in any child modules.")
                    raise

    def extract_all_array_info(self, dec_global, var_dummy_list, subroutine_key):
        """
        """
        try:
            normalized_arrays = []
            normalized_scalars = []

            # Process global declarations
            for key in dec_global:
                for item in dec_global[key]:
                    try:
                        if isinstance(item, F23.Type_Declaration_Stmt):
                            entity_decls = walk(item, F23.Entity_Decl)
                            assert len(entity_decls) == 1, \
                                f"In extract_all_array_info: walk(item, F23.Entity_Decl)=1, but got {len(entity_decls)}."

                            if walk(item, F23.Explicit_Shape_Spec):
                                normalized_arrays.append(item)
                            else:
                                normalized_scalars.append(item)

                            # Handle ALLOCATABLE arrays
                            attr_spec = walk(item, F23.Attr_Spec)
                            if F23.Attr_Spec('ALLOCATABLE') in attr_spec:
                                declaration_stmt = self.processor.combine_allocate_declaration(dec_global[key])
                                assert isinstance(declaration_stmt, F23.Type_Declaration_Stmt), \
                                    f"Item is not of type F23.Type_Declaration_Stmt!"
                                assert walk(declaration_stmt, F23.Explicit_Shape_Spec), \
                                    "In extract_all_array_info: failed to combine_allocate_declaration!"
                                normalized_arrays.append(declaration_stmt)
                    except Exception as e:
                        self.processor.logger.exception(f"Error processing global declaration in extract_all_array_info: {item}", e)
                        raise

            # Process dummy arguments
            for item in var_dummy_list:
                try:
                    assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
                    entity_decls = walk(item, F23.Entity_Decl)
                    assert len(entity_decls) == 1, \
                        f"In extract_all_array_info: walk(item, F23.Entity_Decl)=1, but got {len(entity_decls)}."

                    if walk(item, F23.Explicit_Shape_Spec):
                        normalized_arrays.append(item)
                    else:
                        normalized_scalars.append(item)
                except Exception as e:
                    self.processor.logger.exception(f"Error processing dummy argument in extract_all_array_info: {item}", e)
                    raise

            # Process local variables
            for item in self.var_local[subroutine_key]:
                try:
                    assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
                    if walk(item, F23.Explicit_Shape_Spec):
                        normalized_arrays.append(item)
                    else:
                        normalized_scalars.append(item)
                except Exception as e:
                    self.processor.logger.exception(f"Error processing local variable in extract_all_array_info: {item}", e)
                    raise

            # Extract dimension information
            for item in normalized_arrays:
                try:
                    current_var_info = []
                    array_name = walk(item, F23.Entity_Decl)[0].children[0].tostr()
                    for dim in walk(item, F23.Explicit_Shape_Spec):
                        start_end = [part.strip() for part in dim.tostr().split(':')]
                        lse = len(start_end)
                        if lse == 1:
                            current_var_info.append({'dim_str': '1', 'dim_end': start_end[0]})
                        elif lse == 2:
                            current_var_info.append({'dim_str': start_end[0], 'dim_end': start_end[1]})
                        else:
                            raise ValueError(f"Invalid dimension format: {dim.tostr()}")
                    self.all_array_info[subroutine_key][array_name] = [part for part in current_var_info]
                except Exception as e:
                    self.processor.logger.exception(f"Error processing array dimension info for item: {item}", e)
                    raise

        except Exception as e:
            self.processor.logger.exception(f"Error in extract_all_array_info for '{subroutine_key}': ", e)
            raise

    def process_declaration_variables(self, items, subroutine_key):
        """
        Analyze a list of declaration-related statements within a subroutine and 
        categorize variables as either scalar or array (with explicit shape).

        Parameters
        ----------
        items : list
            A list of parsed Fortran statements (typically from the Specification_Part)
            to analyze. These may include type declarations and allocation statements.
    
        subroutine_key : str
            The name of the subroutine being processed, used as a key to store 
            results in class-level dictionaries.

        Behavior
        --------
        - Skips irrelevant statements (e.g., USE statements).
        - For `Allocate_Stmt` or `Type_Declaration_Stmt` with explicit shape info,
            it extracts the array shape variables and stores them in `shapes_variables[subroutine_key]`.
        - For scalar variable declarations (i.e., without DIMENSION attributes), 
            it identifies and stores them in `scalar_variables[subroutine_key]`.
        - Excludes specific names defined in `self.exclude` from both scalar and shape tracking.

        Notes
        -----
        - This method helps differentiate between scalar and shaped variables in
            preparation for isolation, dependency resolution, and rewriting.
        - If a declaration is neither a type declaration nor an allocation, it's ignored.
        - Dimension attributes (e.g., `DIMENSION(:,:)`) are used to distinguish arrays 
            when shape specs are not explicitly present.

        Raises
        ------
        AssertionError
            If an unshaped variable is not declared with a `Type_Declaration_Stmt`.
        """
        for item in items:
            dec_stmt = isinstance(item, F23.Type_Declaration_Stmt)
            alo_stmt = isinstance(item, F23.Allocate_Stmt)
            use_stmt = isinstance(item, F23.Use_Stmt)
            if use_stmt or (not dec_stmt and not alo_stmt):
                continue
            else:
                shape = walk(walk(item, F23.Allocate_Shape_Spec), F23.Name) if alo_stmt \
                        else walk(walk(item, F23.Explicit_Shape_Spec), F23.Name)
                if shape:
                    seen = {n.string for n in self.shapes_variables[subroutine_key]}
                    for name in shape:
                        if (name is not None and
                                name.tostr() is not None and
                                name.tostr() != 'None' and
                                name.tostr().strip() != '' and
                                name.tostr() not in self.exclude and 
                                name.tostr() not in seen
                                ):
                            self.shapes_variables[subroutine_key].append(name)
                            seen.add(name.string)
                else:
                    assert dec_stmt, 'The scalar must be a Type_Declaration_Stmt!'
                    array = walk(item, F23.Dimension_Attr_Spec)
                    if not array:
                        seen = {n.string for n in self.scalar_variables[subroutine_key]}
                        names = walk(walk(item, F23.Entity_Decl), F23.Name)
                        for name in names:
                            if (name is not None and
                                    name.tostr() is not None and
                                    name.tostr() != 'None' and
                                    name.tostr().strip() != '' and
                                    name.tostr() not in self.exclude and
                                    name.tostr() not in seen
                                    ):
                                self.scalar_variables[subroutine_key].append(name)
                                seen.add(name.string)
    
    def organize_code_components(self, subroutine_key, input_dict, openacc=False):
        try:
            var_modif = self.var_modif_info[subroutine_key]
            # Initialize result dictionary to store all processed components
            result = {
                    'add_to_module': [],           # Declarations to add to module section
                    'add_to_routin': [],           # Allocation statements for routine body
                    'add_to_usestm': [],           # Collected USE statements
                    'acc_declare_create': [],      # Variables for OpenACC CREATE directive
                    'acc_declare_copyin': [],      # Variables for OpenACC COPYIN directive
                    'reads_non_allocatables': [],  # Read statements for declaration routine
                    'reads_allocatables': [],      # Read statements for read routine
                    'write_stmt': []               # Write statements for file output
                    }
            # Process each variable in the input dictionary
            for key in sorted(input_dict):
                var_in_modif = key in var_modif  # Check if variable is modified
                # Process each declaration item for this variable
                for item in input_dict[key]:
                    is_dec_stmt = isinstance(item, F23.Type_Declaration_Stmt)
                    is_alo_stmt = isinstance(item, F23.Allocate_Stmt)
                    is_use_stmt = isinstance(item, F23.Use_Stmt)
                    
                    # Handle type declaration statements
                    if is_dec_stmt:
                        # Add to module declarations, with OpenACC processing if needed
                        result['add_to_module'].append(
                                self.processor.add_entity_to_declaration(item, var_modif)
                                if var_in_modif and openacc
                                else item
                                )
                        # Extract declaration components
                        all_entity_names = walk(item, F23.Entity_Decl)
                        initialized = walk(item, F23.Initialization)
                        attr_spec = walk(item, F23.Attr_Spec)
                        # Process uninitialized variables
                        if not initialized:
                            # Add to OpenACC create list if OpenACC is enabled
                            for entity_name in all_entity_names:
                                if openacc:
                                    result['acc_declare_create'].append(entity_name.tostr())
                            # Categorize based on whether variable is allocatable
                            if F23.Attr_Spec('ALLOCATABLE') not in attr_spec:
                                result['reads_non_allocatables'].append(item)
                            else:
                                # Combine allocate and declaration for allocatable arrays
                                combined = self.processor.combine_allocate_declaration(input_dict[key])
                                result['reads_allocatables'].append(combined)
                        else:
                            # Add initialized variables to OpenACC copyin list
                            if openacc:
                                for entity_name in all_entity_names:
                                    for child in entity_name.children:
                                        if isinstance(child, F23.Name):
                                            result['acc_declare_copyin'].append(child.tostr())

                    # Handle allocation statements
                    if is_alo_stmt:
                        for allocation_stmt in self.processor.add_entity_to_allocation(item, var_modif, openacc):
                            result['add_to_routin'].append(allocation_stmt.children[0])

                    # Handle USE statements
                    if is_use_stmt:
                        result['add_to_usestm'].append(item)

            # Post-process the collected items
            result['add_to_module'] = self.processor.remove_intent_and_save(result['add_to_module'])
            result['add_to_module'] = self.processor.process_queue(result['add_to_module'])

            result['reads_non_allocatables'] = self.processor.remove_intent_and_save(result['reads_non_allocatables'])
            read_list, write_stmt = self.generated_wr_statement(result['reads_non_allocatables'])
            result['reads_non_allocatables'] = read_list
            result['write_stmt'].extend(write_stmt)

            read_list, write_stmt = self.generated_wr_statement(result['reads_allocatables'])
            result['reads_allocatables'] = read_list
            result['write_stmt'].extend(write_stmt)

            # Create OpenACC directive commands if OpenACC is enabled
            if result['acc_declare_copyin']:
                acc_declare_copyin_str = ', '.join(result['acc_declare_copyin'])
                result['acc_declare_copyin_cmd'] = self.processor.parse_fortran_comment(f"!$ACC DECLARE COPYIN({acc_declare_copyin_str})")

            if result['acc_declare_create']:
                acc_declare_create_str = ', '.join(result['acc_declare_create'])
                result['acc_declare_create_cmd'] = self.processor.parse_fortran_comment(f"!$ACC DECLARE CREATE({acc_declare_create_str})")
                result['acc_update_device_cmd'] = self.processor.parse_fortran_comment(f"!$ACC UPDATE DEVICE({acc_declare_create_str})")

            self.processor.logger.info("Declarations and allocations processed successfully")
            return result

        except Exception as e:
            self.processor.logger.exception(f"Failed to process declarations and allocations, Error: ", e)
            raise

    def generated_wr_statement(self, items):
        """
        """
        try:
            read_list = []    # Generated read statements for variable initialization
            write_stmt= []
            items_sep = []
            # Separate multi-variable declarations into individual declarations
            for item in items:
                if len(walk(item, F23.Entity_Decl)) > 1:
                    node_list = self.processor.separate_entity_declarations(item)
                    items_sep.extend(node_list)
                else:
                    items_sep.append(item)

            # Generate I/O statements for each declaration
            for item in items_sep:
                init = True
                intent = walk(item, F23.Intent_Spec)

                # Skip initialization for OUT intent variables
                if F23.Intent_Spec('OUT') in intent:
                    init = False

                # Extract variable name from declaration
                var_name = None
                for child in item.children:
                    if isinstance(child, F23.Entity_Decl_List):
                        var_name = child.tostr()
                        break  # Found the variable name, no need to continue

                # Assert that var_name was successfully extracted
                assert var_name is not None, f"Failed to extract variable name from declaration: {item}"

                # Generate read/write statements for variables that need initialization
                if init:
                    # Create read statement with error handling
                    code_template = f"""
                    read(1363, iostat = ier){var_name}
                    if (ier /= 0) then
                    write(*,*) 'Error reading from file for {var_name}. ',' IOSTAT : ', ier
                    endif
                    """
                    read_list.append(self.processor.parse_fortran_statement(code_template))

                    # Create corresponding write statement
                    write_stmt.append(F23.Write_Stmt(f"write(1363){var_name}"))

            self.processor.logger.info("Processing initialization completed!")
            return read_list, write_stmt

        except Exception as e:
            self.processor.logger.exception(f"Error processing initialization: ", e)
            raise


'''if __name__ == "__main__":
    import unittest
    import tempfile
    import shutil

    class TestExtractor(unittest.TestCase):
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
                integer, parameter :: i_std = 4
            
                contains
            
                subroutine test_sub(a, b)
                integer, intent(in) :: a
                real, intent(out) :: b(:)
                integer :: i
                do i = 1, size(b)
                    b(i) = a * 2.0
                end do
                end subroutine test_sub
            
                end module simple_mod
                """)
        
            # Create a more complex module with dependencies
            cls.complex_module = os.path.join(cls.test_dir, "complex_mod.f90")
            with open(cls.complex_module, "w") as f:
                f.write("""
                module complex_mod
                use simple_mod
                implicit none
                integer, parameter :: n = 10
            
                contains
            
                subroutine complex_sub(x, y)
                real, intent(inout) :: x(n)
                real, intent(out) :: y
                integer :: j
                call test_sub(5, x)
                y = sum(x)
                end subroutine complex_sub
            
                subroutine helper_fn(z, res)
                real, intent(in) :: z(n)
                real, intent(out):: res
                res = sqrt(sum(z**2))
                end subroutine  helper_fn
            
                end module complex_mod
                """)
        
            # Parse the module trees
            cls.processor = Processor()
            cls.simple_tree = cls.processor.parse_fortran_file(cls.simple_module)
            cls.complex_tree = cls.processor.parse_fortran_file(cls.complex_module)

        @classmethod
        def tearDownClass(cls):
            # Remove the temporary directory
            shutil.rmtree(cls.test_dir)

        def setUp(self):
            # Create a fresh Extractor instance for each test
            self.simple_extractor = Extractor(self.test_dir, self.simple_tree)
            self.complex_extractor = Extractor(self.test_dir, self.complex_tree)

        def test_initialization(self):
            # Test that initialization sets up all attributes correctly
            self.assertEqual(self.simple_extractor.module_dir, self.test_dir)
            self.assertIsInstance(self.simple_extractor.module_tree.children[1], F23.Module)
            self.assertIsInstance(self.simple_extractor.subroutines, defaultdict)
            self.assertIsInstance(self.simple_extractor.dummy_arg_list, defaultdict)
            self.assertEqual(self.simple_extractor.exclude, {'kjpindex', 'nslm', 'nstm', 'nvm', 'nsnow', 'DIM', 'dim', 'MASK', 'next_calc_loop'})

        def test_extract_loop_indices(self):
            # Test loop index extraction
            self.simple_extractor.extract_loop_indices()
            self.assertEqual(len(self.simple_extractor.loop_dict), 1)
            self.assertIn('SIZE(b)', self.simple_extractor.loop_dict)
            self.assertEqual(self.simple_extractor.loop_dict['SIZE(b)'], {'i'})

        def test_extract_loop_vect(self):
            # Test vector loop extraction
            self.simple_extractor.find_subroutines()
            sub_key = "test_sub"
            sub_tree = self.simple_extractor.subroutines[sub_key]
        
            # Test with a non-vector loop
            self.simple_extractor.extract_loop_vect(sub_key, sub_tree)
            self.assertIsNone(self.simple_extractor.loop_vect[sub_key])
        
            # Test with a vector loop (kjpindex)
            code = """
            subroutine vect_sub(a)
            integer, intent(in) :: a
            real :: b(10)
            integer :: i
            do i = 1, kjpindex
                b(i) = a * 2.0
            end do
            end subroutine vect_sub
            """
            sub_tree = self.processor.parse_fortran_string(code)
            self.simple_extractor.extract_loop_vect("vect_sub", sub_tree)
            self.assertIsNotNone(self.simple_extractor.loop_vect["vect_sub"])

        def test_find_subroutines(self):
            # Test subroutine finding
            self.simple_extractor.find_subroutines()
            self.assertEqual(self.simple_extractor.subroutine_keys_all, {"test_sub"})
            self.assertEqual(self.simple_extractor.subroutine_keys_ncl, {"test_sub"})
        
            # Test with complex module
            self.complex_extractor.find_subroutines()
            self.assertEqual(self.complex_extractor.subroutine_keys_all, {'test_sub', 'complex_sub', 'helper_fn'})
            self.assertEqual(set(self.complex_extractor.call_within_sub["complex_sub"].keys()), {"test_sub"})

        def test_extract_names(self):
            # Test name extraction
            self.simple_extractor.find_subroutines()
            sub_key = "test_sub"
            sub_tree = self.simple_extractor.subroutines[sub_key]
            self.simple_extractor.find_variables(sub_tree, sub_key)
        
            # Verify local names are extracted
            self.simple_extractor.extract_names(sub_key)
            self.assertEqual(self.simple_extractor.var_local_names[sub_key], {"i"})

        def test_extract_intent_clean_subroutine(self):
            # Test intent extraction
            self.complex_extractor.find_subroutines()
            sub_key = "complex_sub"
            sub_tree = self.complex_extractor.subroutines[sub_key]
        
            # First need to process the called subroutine
            self.simple_extractor.find_subroutines()
            called_key = "test_sub"
            called_tree = self.simple_extractor.subroutines[called_key]
            self.simple_extractor.find_variables(called_tree, called_key)
            self.simple_extractor.extract_intent(called_key, called_tree)

            # Verify intents
            self.assertEqual(self.simple_extractor.general_usage_dict[called_key]["a"], "IN")
            self.simple_extractor.clean_subroutine(called_key, called_tree)
            self.assertEqual(self.simple_extractor.general_usage_dict[called_key]["b"], "INOUT")

    
        def test_add_intent(self):
            # Test adding intent to declarations
            decl = "real :: a(10)"
            parsed_decl = Processor().parse_fortran_statement(decl)
        
            # Add IN intent
            new_decl = Extractor.add_intent(parsed_decl.children[0], "in")
            self.assertIn("INTENT(IN)", new_decl.tostr())
        
            # Add OUT intent
            new_decl = Extractor.add_intent(parsed_decl.children[0], "out")
            self.assertIn("INTENT(OUT)", new_decl.tostr())

        def test_find_variables(self):
            # Test variable finding and categorization
            self.simple_extractor.find_subroutines()
            sub_key = "test_sub"
            sub_tree = self.simple_extractor.subroutines[sub_key] 
            self.simple_extractor.find_variables(sub_tree, sub_key)
        
            # Verify variable categorization

            self.assertEqual(len(self.simple_extractor.var_dummy[sub_key]), 2)  # a and b
            self.assertEqual(len(self.simple_extractor.var_local[sub_key]), 1)  # i
            self.assertEqual(self.simple_extractor.var_modif[sub_key], {"b"})

        def test_find_global_variables(self):
            # Test global variable finding
            self.complex_extractor.find_subroutines()
            sub_key = "complex_sub"
            sub_tree = self.complex_extractor.subroutines[sub_key]
            # First find variables to get globals
            self.complex_extractor.find_variables(sub_tree, sub_key)
            # Now find globals (test_sub in this case)
            
            self.complex_extractor.find_global_variables(
                self.test_dir, 
                self.complex_tree, 
                self.complex_extractor.var_global[sub_key], 
                sub_key
            )
        
            # Verify global was found
            self.assertIn("test_sub", self.complex_extractor.dec_global[sub_key])

        def test_extract_array_info(self):
            # Test array information extraction
            self.complex_extractor.find_subroutines()
            sub_key = "complex_sub"
            sub_tree = self.complex_extractor.subroutines[sub_key]
        
            # First find variables and globals
            self.complex_extractor.find_variables(sub_tree, sub_key)
            self.complex_extractor.find_global_variables(
                self.test_dir, 
                self.complex_tree, 
                self.complex_extractor.var_global[sub_key], 
                sub_key
            )
        
            # Now extract array info
            self.complex_extractor.extract_array_info(
                self.complex_extractor.dec_global[sub_key],
                self.complex_extractor.var_dummy[sub_key],
                sub_key
            )
        
            # Verify array info
            self.assertIn("x", self.complex_extractor.all_array_info[sub_key])
            self.assertEqual(len(self.complex_extractor.all_array_info[sub_key]["x"]), 1)

        def test_process_declaration_variables(self):
            # Test processing of declaration variables
            self.simple_extractor.find_subroutines()
            sub_key = "test_sub"
            sub_tree = self.simple_extractor.subroutines[sub_key]
        
            # First find variables
            self.simple_extractor.find_variables(sub_tree, sub_key)
        
        
            # Process declarations
            self.simple_extractor.process_declaration_variables(self.simple_extractor.var_dummy[sub_key], sub_key)
        
            # Verify scalar and shaped variables
            self.assertEqual(self.simple_extractor.scalar_variables[sub_key], [F23.Name("a")])
            self.assertEqual(self.simple_extractor.shapes_variables[sub_key], [F23.Name("n")])
        
    unittest.main()
'''
