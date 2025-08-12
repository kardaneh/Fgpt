import os, sys
from processor import Processor
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from navigator import Navigator
from shaper import Shaper
import re
from fparser.common.readfortran import FortranStringReader
from collections import defaultdict

class Extractor:
    """
    """

    def __init__(self, module_dir, module_tree):
        """
        The Extractor class is responsible for analyzing a parsed Fortran module and extracting
        all relevant structural and dependency information needed to isolate subroutines or functions.

        It builds an internal representation of:
            - Subroutines and their relationships (calls, dependencies, dummy args)
            - Variables (local, global, dummy, modified)
            - Loops and vectorization patterns
            - Shape and type declarations
            - Cross-module dependencies

        This information is critical for use in:
            - Fortran subroutine isolation (used by the `Isolator` class)
            - Source-to-source transformation (e.g., Fortran to Python)
            - Refactoring or static analysis

        Parameters
        ----------
        module_dir : str
            Path to the directory containing the target module source.
        module_tree : fparser.two.Fortran2008.Program
            The parsed AST of the target Fortran module.

        Attributes
        ----------
        subroutine_keys_all : set
            All subroutine/function names found in the module.
        subroutine_keys_ncl : set
            Subroutines without any `CALL` statements inside.
        subroutines : defaultdict
            Mapping from subroutine names to their AST node representations.
        func_result : defaultdict
            Return variable names of functions (if applicable).
        dummy_arg_list : defaultdict[list]
            List of dummy arguments for each subroutine.
        actual_arg_spec_list : defaultdict[list]
            Actual arguments used in calls within each subroutine.
        external_subroutines : set
            Subroutines that are declared external (outside the current module).
        call_subroutines : defaultdict[list]
            All `CALL` statements found within each subroutine.
        call_within_sub : defaultdict[set]
            Subroutines invoked from within each subroutine.
        loop_dict : defaultdict[set]
            Loops associated with each subroutine (parsed structurally).
        loop_vect : defaultdict
            Optional vectorization-related loops (used in optimizations).
        exclude : set
            Reserved variable names that are ignored during extraction.
        cases_to_exclude : list
            Naming patterns used to skip uninteresting subroutines.
        allowed_external_subroutines : set
            External calls allowed during isolation or transformation.
        dec_global : defaultdict[dict[list]]
            Global declarations (e.g., types, dimensions) extracted from modules.
        all_array_info : defaultdict[dict[list]]
            Metadata about arrays (e.g., shape, intent).
        imp_shape : defaultdict[dict]
            Mapping of implicitly shaped variables and their dimensions.
        scalar_variables : defaultdict[set]
            Set of scalar variables for each subroutine.
        shapes_variables : defaultdict[set]
            Set of shaped (array) variables per subroutine.
        var_modif_info : defaultdict[dict[list]]
            Variables modified within each subroutine and how.
        general_usage_dict : defaultdict
            Generalized usage metadata (e.g., access patterns).
        parsed_modules : defaultdict
            Cache of other parsed modules for dependency resolution.
        var_global : defaultdict[set]
            Global variables accessed within each subroutine.
        var_dummy : defaultdict[list]
            Dummy arguments declared for each subroutine.
        var_local : defaultdict[list]
            Locally declared variables in each subroutine.
        var_modif : defaultdict[set]
            Variables modified (assigned) in each subroutine.
        var_local_names : defaultdict[set]
            Set of all local variable names in each subroutine.
        var_declared : defaultdict[set]
            All explicitly declared variables per subroutine.

        Notes
        -----
        The Extractor class serves as the central analysis engine for understanding the full
        semantic structure of a Fortran module. It supports and powers transformations such as:
        - Isolating subroutines
        - Rewriting declarations
        - Tracing variable usage and modifications
        - Validating or rewriting loop patterns
        This makes it ideal for high-level program understanding, static analysis, or test-case generation.
        """
        try:
            self.module_dir = module_dir
            self.module_tree = module_tree
            self.subroutine_keys_all = set()
            self.subroutine_keys_ncl = set()
            self.subroutines = defaultdict()
            self.func_result = defaultdict()
            self.dummy_arg_list = defaultdict(list)
            self.actual_arg_spec_list = defaultdict(list)
            self.external_subroutines = set()
            self.call_subroutines = defaultdict(list)
            self.call_within_sub = defaultdict(set)
            self.loop_dict = defaultdict(set)
            self.loop_vect = defaultdict(lambda: None)
            self.exclude = {'kjpindex', 'nslm', 'nstm', 'nvm', 'nsnow', 'DIM', 'dim', 'MASK', 'next_calc_loop'}
            self.cases_to_exclude = ['clear', 'finalize', 'init', 'initialize', 'read']
            self.allowed_external_subroutines = {'ipslerr_p', 'xios_orchidee_send_field'}
            self.dec_global = defaultdict(lambda: defaultdict(list))
            self.all_array_info = defaultdict(lambda: defaultdict(list))
            self.imp_shape = defaultdict(dict)
            self.scalar_variables = defaultdict(set)
            self.shapes_variables = defaultdict(set)
            self.var_modif_info = defaultdict(lambda: defaultdict(list))
            self.general_usage_dict = defaultdict()
            self.parsed_modules = defaultdict()
            self.var_global = defaultdict(set)
            self.var_dummy = defaultdict(list)
            self.var_local = defaultdict(list)
            self.var_modif = defaultdict(set)
            self.var_local_names = defaultdict(set)
            self.var_declared = defaultdict(set)
        except Exception as e:
            raise RuntimeError(f"Error in __init__: {str(e)}")

    def extract_loop_indices(self):
        """
        Extract loop indices and their associated loop bounds from all non-labeled DO loops in the module.

        This method traverses the parsed Fortran module tree to find all DO loops without labels. For each loop, 
        it parses the loop control statement to identify the loop index variable and the loop's end bound. It skips 
        loops that involve complex expressions (such as logical OR operands or part references).

        The extracted loop indices are stored in a dictionary (`self.loop_dict`), where each key is a loop end bound 
        and the corresponding value is a set of loop index variables that iterate up to that bound.

        This information is useful for analyzing loop structures, dependencies, and vectorization opportunities 
        within the module's subroutines.
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
                if loop_end:
                    self.loop_dict[loop_end].add(loop_index)

        except Exception as e:
            raise RuntimeError(f"Error in extract_loop_indices: {str(e)}")


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
            raise RuntimeError(f"Error in extract_loop_indices: {str(e)}")

    def find_subroutines(self):
        """
        Extracts all subroutines from the parsed Fortran module and gathers metadata for each one.

        This method performs the following:
        - Walks the module tree to identify all `Subroutine_Subprogram` nodes.
        - Extracts the subroutine name (`subroutine_key`) and its dummy argument list.
        - Filters out subroutines whose names match patterns in `self.cases_to_exclude` (e.g., 'init', 'clear', 'read').
        - Saves the full subroutine structure in `self.subroutines`, and tracks the name in `self.subroutine_keys_all`.
        - Detects all internal and external subroutine calls:
            - Internal calls are stored in `self.call_within_sub[subroutine_key]`.
            - External calls (e.g., library or unrelated modules) are detected by checking against `self.allowed_external_subroutines`.
        - Records all actual arguments passed during calls in `self.actual_arg_spec_list`.
        - Stores all call statements in `self.call_subroutines`.
        - Tracks subroutines that are considered "having no call" in `self.subroutine_keys_ncl`.
        - Determines external subroutines as those that are called but not defined within the current module.

        This function is essential for analyzing dependencies between procedures and for enabling subroutine isolation.
        """
        for sub in walk(self.module_tree, F23.Subroutine_Subprogram):
            subroutine_key, arg_list = None, None
            subroutine_stmt = walk(sub, F23.Subroutine_Stmt)[0]
            call_stmt = walk(sub, F23.Call_Stmt)
            for child in subroutine_stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Name):
                    subroutine_key = child.tostr()
                elif isinstance(child, F23.Dummy_Arg_List):
                    arg_list = child
                else:
                    raise ValueError(f"Unexpected type '{type(child)}' encountered in children.")
            assert subroutine_key is not None, f"Unexpected type {subroutine_key} encountered in children."
            check = all(case not in subroutine_key for case in self.cases_to_exclude)
            if not check:
                continue
            self.subroutine_keys_all.add(subroutine_key)
            self.subroutines[subroutine_key] = sub
            if arg_list is not None:
                for child in arg_list.children:
                    self.dummy_arg_list[subroutine_key].append(child.tostr())
            if call_stmt:
                for item in call_stmt:
                    call_name = item.children[0].tostr()
                    check = all(case not in call_name for case in self.cases_to_exclude)
                    if not check:
                        continue
                    arg_list = item.children[1]
                    assert call_name is not None, f"Unexpected type {subroutine_key} encountered in children."
                    if call_name not in self.allowed_external_subroutines:
                        self.call_within_sub[subroutine_key].add(call_name)
                    else:
                        self.subroutine_keys_ncl.add(subroutine_key)
                    if arg_list is not None:
                        arg_string = []
                        for child in arg_list.children:
                            arg_string.append(child.tostr())
                        self.actual_arg_spec_list[call_name].append(arg_string)
                    self.call_subroutines[call_name].append(item)
            else:
                self.subroutine_keys_ncl.add(subroutine_key)
        self.external_subroutines = {item for item in self.actual_arg_spec_list.keys() \
                if item not in self.dummy_arg_list.keys()}


    def extract_names(self, subroutine_key):
        """
        Extracts the names of the local variables using the var_local attribute based on the given subroutine key argument. 
        The var_local attribute contains a list of Type_Declaration_Stmt from which extraction of the variables names is done. 

        params
        ------
        - subroutine_key (str):  
        
        return
        ------
        - None 
        """
        for item in self.var_local[subroutine_key]:
            for entity in walk(item, F23.Entity_Decl):
                for child in entity.children:
                    if isinstance(child, F23.Name):
                        self.var_local_names[subroutine_key].add(child.tostr())
        #return self.var_local_names[subroutine_key]

    def extract_intent(self, subroutine_key, subroutine_tree, within_calls=None):
        """
        Analyze the usage of dummy arguments within a Fortran subroutine and 
        infer their INTENT attribute (IN, OUT, or INOUT).

        This function traverses the execution part of the given subroutine and 
        tracks how each dummy argument is used:
    
        - If a variable is used only on the right-hand side of expressions (e.g., in conditions or RHS of assignments), 
            it is classified as INTENT(IN).
        - If a variable appears on the left-hand side (LHS) of an assignment, it is classified as INTENT(OUT).
        - If a variable is both read and written (RHS and LHS), or passed to a child subroutine with a known 
            INTENT(INOUT), it is classified as INTENT(INOUT).
    
        The function handles various Fortran constructs:
            - Assignment statements
            - Loop headers (e.g., DO loops)
            - Conditional branches (IF-THEN, ELSEIF)
            - WHERE and ELSEWHERE constructs
            - Nested subroutine calls
    
        In the case of subroutine calls, if a dummy argument is passed to a child subroutine, the method also 
        references the intent already extracted for the child (from `self.general_usage_dict`) to propagate 
        intent information upward through the call hierarchy.

        The results are stored in `self.general_usage_dict` under the corresponding `subroutine_key`.

        Parameters:
        -----------
        subroutine_key : str
            The name of the subroutine being analyzed.
    
        subroutine_tree : Fparser node (Subroutine_Subprogram)
            The parsed tree representation of the subroutine.

        within_calls : set[str], optional
            Set of subroutine names called within the current subroutine. Used for validating and propagating
            intent information when dummy arguments are passed to internal calls.

        Raises:
        -------
        AssertionError:
            If expected subroutine calls or argument lists are not found.
    
        RuntimeError:
            If traversal encounters unexpected structures.

        Returns:
        --------
        None (results stored in self.general_usage_dict)
        """
        dummy_arg_list = self.dummy_arg_list[subroutine_key]
        usage = {arg: {'intent': None, 'first_use_assign': None, 'first_use_update': False} for arg in dummy_arg_list}
        def traverse_block(block):
            if hasattr(block, "content"):
                for child in block.content:
                    child_parent = child.parent
                    if isinstance(child, F23.Assignment_Stmt):
                        lhs_expr = child.items[0].tostr()
                        rhs_expr = child.items[-1].tostr()
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            pattern = r'\b' + re.escape(var_name) + r'\b'
                            if var_name in dummy_arg_list:

                                if isinstance(child_parent, F23.If_Construct) and \
                                        usage[var_name]['first_use_assign'] is not None and \
                                        usage[var_name]['first_use_assign'].parent == child_parent:
                                    usage[var_name]['first_use_update'] = True

                                if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                                    usage[var_name]['first_use_assign'] = child

                                if isinstance(name.parent, F23.Section_Subscript_List):
                                    if usage[var_name]['intent'] is None:
                                        usage[var_name]['intent'] = 'IN'
                                else:
                                    if re.search(pattern, lhs_expr):
                                        if usage[var_name]['intent'] is None:
                                            usage[var_name]['intent'] = 'OUT'
                                        elif usage[var_name]['intent'] == 'IN':
                                            usage[var_name]['intent'] = 'INOUT'

                                    if re.search(pattern, rhs_expr):
                                        if usage[var_name]['intent'] is None:
                                            usage[var_name]['intent'] = 'IN'
                                        elif usage[var_name]['intent'] == 'OUT' and usage[var_name]['first_use_assign'] == child:
                                            usage[var_name]['intent'] = 'INOUT'

                    elif isinstance(child, F23.Nonlabel_Do_Stmt):
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            if var_name in dummy_arg_list:
                                if isinstance(child_parent, F23.If_Construct) and \
                                        usage[var_name]['first_use_assign'] is not None and \
                                        usage[var_name]['first_use_assign'].parent == child_parent:
                                    usage[var_name]['first_use_update'] = True
                                if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                                    usage[var_name]['first_use_assign'] = child
                                if usage[var_name]['intent'] is None:
                                    usage[var_name]['intent'] = 'IN'

                    elif isinstance(child, (F23.If_Then_Stmt, F23.Else_If_Stmt)):
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            if var_name in dummy_arg_list:
                                if isinstance(child_parent, F23.If_Construct) and \
                                        usage[var_name]['first_use_assign'] is not None and \
                                        usage[var_name]['first_use_assign'].parent == child_parent:
                                    usage[var_name]['first_use_update'] = True
                                if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                                    usage[var_name]['first_use_assign'] = child
                                if usage[var_name]['intent'] is None:
                                    usage[var_name]['intent'] = 'IN'

                    elif isinstance(child, (F23.Where_Construct_Stmt, F23.Masked_Elsewhere_Stmt)):
                        for name in walk(child, F23.Name):
                            var_name = name.tostr()
                            if var_name in dummy_arg_list:
                                if isinstance(child_parent, F23.If_Construct) and \
                                        usage[var_name]['first_use_assign'] is not None and \
                                        usage[var_name]['first_use_assign'].parent == child_parent:
                                    usage[var_name]['first_use_update'] = True
                                if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                                    usage[var_name]['first_use_assign'] = child
                                if usage[var_name]['intent'] is None:
                                    usage[var_name]['intent'] = 'IN'

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
                                            if isinstance(child_parent, F23.If_Construct) and \
                                                    usage[var_name]['first_use_assign'] is not None and \
                                                    usage[var_name]['first_use_assign'].parent == child_parent:
                                                usage[var_name]['first_use_update'] = True
                                            if usage[var_name]['first_use_assign'] is None or usage[var_name]['first_use_update']:
                                                usage[var_name]['first_use_assign'] = child
                                            current_intent = usage[var_name]['intent']
                                            call_intent = self.general_usage_dict[call_name][var_name]
                                            if current_intent is None:
                                                usage[var_name]['intent'] = call_intent
                                            elif current_intent == 'IN' and call_intent in {'OUT', 'INOUT'}:
                                                usage[var_name]['intent'] ='INOUT'
                                            elif current_intent == 'OUT' and call_intent == 'INOUT':
                                                usage[var_name]['intent'] == 'INOUT'
                    else:
                        traverse_block(child)
        execution_part = walk(subroutine_tree, F23.Execution_Part)[0]
        traverse_block(execution_part)
        self.general_usage_dict[subroutine_key] = {var: props['intent'] for var, props in usage.items()}

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
        return F23.Type_Declaration_Stmt(
            f'{intrinsic_type_spec},dimension({explicit_shape_spec_list}),intent({intent})::{entity_decl_list}'
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
      `     self.general_usage_dict`.
        - Color-coded terminal warnings (`\033[38;5;214m`) and confirmations (`\033[32m`) are printed for user visibility.
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

    def find_variables(self, subroutine_tree, subroutine_key, parent_subroutine_key=None):
        """
        Analyzes a subroutine or function's declarations and execution to extract and categorize variables.

        This method performs a full scan of a subroutine's abstract syntax tree to identify:
            - **Declared** vs **Used** variables.
            - **Dummy arguments** (with or without `INTENT`).
            - **Local variables** (non-dummy and declared within the subroutine).
            - **Global variables** (used but not declared in the subroutine).
            - **Modified variables** (i.e., appearing on the LHS of an assignment).
            - **Array shape dependencies** (variables that define array bounds).
            - Handles implicit and intrinsic shape declarations, and replaces them with explicit forms when possible.

        The analysis is stored internally in various class attributes:
            - `self.var_dummy[subroutine_key]` — List of dummy arguments.
            - `self.var_local[subroutine_key]` — List of local variables.
            - `self.var_global[subroutine_key]` — Set of global variables (used but undeclared).
            - `self.var_declared[subroutine_key]` — Set of explicitly declared variable names.
            - `self.var_modif[subroutine_key]` — Set of modified variable names.
            - `self.imp_shape[subroutine_key]` — Dictionary of array variables with implicitly shaped declarations.
    
        If any dummy argument is found without an `INTENT`, an error is raised (for strict isolation handling).

        Parameters
        ----------
        subroutine_tree : Fortran2003.Subroutine_Subprogram or Function_Subprogram
            Parsed Fortran AST of the subroutine or function to be analyzed.

        subroutine_key : str
            Name of the subroutine (or function) being analyzed.

        parent_subroutine_key : str, optional
            Required when analyzing a function, used to retrieve array shape info from its caller.

        Raises
        ------
        ValueError
            If a dummy argument is declared without an `INTENT` or intrinsic shape handling is not implemented
            for some specific cases.

        Notes
        -----
        - This method relies on `Shaper` to resolve implicit or intrinsic shapes into explicit ones.
        - Multiple variable declarations in one line are separated for clarity and safety.
        - Warnings are printed for intrinsic names or implicit shapes found during traversal.
        - `self.exclude` is used to filter out known indices or ignored variables from global list.

        Returns
        -------
        None (modifies internal state of the Extractor instance)
        """

        var_in_local = set()
        shapes = set()
        self.var_dummy[subroutine_key].clear()
        self.var_local[subroutine_key].clear()

        declared, used = walk(subroutine_tree, F23.Specification_Part), walk(subroutine_tree, F23.Execution_Part)
        self.var_declared[subroutine_key] = {name.tostr() for name in  walk(declared, F23.Entity_Decl)}
        names_declared, names_used = walk(declared, F23.Name), walk(used, F23.Name)

        var_declared = {name.string for name in names_declared }
        var_used = {name.string for name in names_used}
        self.var_global[subroutine_key] = var_used - var_declared

        #shape = walk(walk(subroutine_tree, F23.Explicit_Shape_Spec), F23.Name)
        #shapes = {name.string for name in shape}

        for declaration_stmt in walk(declared, F23.Type_Declaration_Stmt):
            if len(walk(declaration_stmt, F23.Entity_Decl)) > 1:
                node_list = Processor().separate_entity_declarations(declaration_stmt)
            else:
                node_list = [declaration_stmt]
            for node in node_list:
                implicit_shape = walk(node, F23.Assumed_Shape_Spec)
                intrinsic_name = walk(node, F23.Intrinsic_Name)
                if implicit_shape:
                    print('\033[38;5;214m' + "Warning: Implicit shape detected in the decleration!" + '\033[0m')
                    print(f'\033[38;5;214mNode: {node}\033[0m')
                    if isinstance(subroutine_tree, F23.Subroutine_Subprogram):
                        shape_finder = Shaper(self.module_dir, self.parsed_modules, \
                                self.dummy_arg_list, self.actual_arg_spec_list, \
                                self.call_subroutines)
                        nodes = shape_finder.shaper_subroutine(node, subroutine_key)
                        print(f'\033[32mfound: {nodes}\033[0m')
                        node = Processor().map_declaration(node, explicit_dec=nodes, dimensions=None)
                        entity_decl = walk(node, F23.Entity_Decl)[0].tostr()
                        if entity_decl not in self.imp_shape[subroutine_key]:
                            self.imp_shape[subroutine_key][entity_decl] = node
                    elif isinstance(subroutine_tree, F23.Function_Subprogram):
                        assert parent_subroutine_key is not None, "Error: 'parent_subroutine_key' must not be None."
                        shape_finder = Shaper(self.module_dir, self.parsed_modules, self.dummy_arg_list)
                        nodes = shape_finder.shaper_function(node, subroutine_tree, subroutine_key, self.all_array_info[parent_subroutine_key])
                        print(f'\033[32mfound: {nodes}\033[0m')
                        node = nodes
                        entity_decl = walk(node, F23.Entity_Decl)[0].tostr()
                        if entity_decl not in self.imp_shape[subroutine_key]:
                            self.imp_shape[subroutine_key][entity_decl] = node
                if intrinsic_name:
                    print(f'\033[38;5;214mWarning: Intrinsic name detected in the decleration!\033[0m')
                    print(f'\033[38;5;214mNode: {node}\033[0m')
                    if isinstance(subroutine_tree, F23.Function_Subprogram):
                        shape_finder = Shaper(self.module_dir, self.parsed_modules, self.dummy_arg_list)
                        nodes = shape_finder.shaper_intrinsic_size(node)
                        print(f'\033[32mfound: {nodes}\033[0m')
                        node = nodes
                    else:
                        raise ValueError(f"intrinsic_name found in {type(subroutine_tree).__name__} declarations! "
                                f"This case is not implemented yet!")
                intent = walk(node, F23.Intent_Spec)
                entity_decls = walk(node, F23.Entity_Decl)
                assert len(entity_decls) == 1,\
                        "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                name = entity_decls[0].tostr()
                for explicit_shape_spec in walk(node, F23.Explicit_Shape_Spec):
                    for dim in explicit_shape_spec.children:
                        if isinstance(dim, F23.Name):
                            shapes.add(dim.tostr())
                if intent:
                    intent_spec = intent[0].tostr()
                    #if walk(walk(node,F23.Entity_Decl),F23.Name)[0].string not in self.exclude:
                    if name not in self.exclude:
                        self.var_dummy[subroutine_key].append(node)
                        if F23.Intent_Attr_Spec('INTENT(IN)') in walk(node, F23.Intent_Attr_Spec):
                            #for name in  walk(node, F23.Entity_Decl):
                            var_in_local.add(name)
                else:
                    #for name in walk(node, F23.Entity_Decl):
                    if name in self.dummy_arg_list[subroutine_key]:
                        raise ValueError(f"Variable '{name.string}' in subroutine '{subroutine_key}' "
                                f"at statement '{node.tostr()}' is a dummy argument without intent.")
                    if (
                            name == subroutine_key 
                            or (
                                subroutine_key in self.func_result 
                                and name == self.func_result[subroutine_key]
                                )
                            ):
                        self.var_dummy[subroutine_key].append(node)
                    else:
                        var_in_local.add(name)
                        self.var_local[subroutine_key].append(node)

        self.var_dummy[subroutine_key].sort(key=lambda node: node.children[-1].tostr().lower())
        #shape ={name.string for name in  walk(walk(self.var_dummy[subroutine_key], F23.Explicit_Shape_Spec), F23.Name)}
        #shapes.update(shape)

        self.var_global[subroutine_key] -= self.exclude
        shapes -= self.exclude
        self.var_global[subroutine_key].update(shapes)

        #self.extract_names(self.var_local[subroutine_key])

        for stmt in walk(subroutine_tree, F23.Execution_Part):
            for assign_stmt in walk(stmt, F23.Assignment_Stmt):
                lhs = assign_stmt.items[0]
                if isinstance (lhs, F23.Name):
                    if lhs.tostr() not in var_in_local:
                        self.var_modif[subroutine_key].add(lhs.tostr())
                elif isinstance(lhs, F23.Part_Ref):
                    if lhs.children[0].tostr() not in var_in_local:
                        self.var_modif[subroutine_key].add(lhs.children[0].tostr())
                else:
                    raise ValueError(f"Unexpected assignment left-hand side type: {type(lhs)} in statement: {assign_stmt.tostr()}")
    
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
        for declaration in var_global:
            self.finder = Navigator(module_dir, module_tree, self.parsed_modules)
            if declaration not in self.external_subroutines:
                sys.stdout.write('\r' + '\033[32m' + 'Searching for variable: ' + declaration + ' ... ⏳' + '\033[0m\n')
                sys.stdout.flush()
                self.finder.variable_finder(declaration)
                sys.stdout.write('\r' + '')
                sys.stdout.flush()
                if self.finder.var_declaration:
                    print('\033[32m' + '✅ Variable found!' + '\033[0m\n')
                    self.dec_global[subroutine_key][declaration] = [item for item in self.finder.var_declaration]
                else:
                    raise ValueError(f"Variable '{declaration}' is not found in any child modules.")
                if self.finder.var_initial:
                    print('\033[91m' + 'Attention: there are additional to search:', self.finder.var_initial)
                    print('\033[91m' + 'in the directory:', self.finder.module_dir_sc)
                    ffile = walk(self.finder.module_tree_sc, F23.Name)[0].string
                    print('\033[91m' + 'in the module', ffile)
                    self.find_global_variables(self.finder.module_dir_sc, self.finder.module_tree_sc, self.finder.var_initial, subroutine_key)
            elif declaration in self.external_subroutines:
                sys.stdout.write('\r' + '\033[32m' + 'Searching for procedure: ' + declaration + ' ... ⏳' + '\033[0m\n')
                sys.stdout.flush()
                self.finder.external_subroutine_finder(declaration)
                sys.stdout.write('\r' + '')
                sys.stdout.flush()
                if self.finder.var_declaration:
                    print('\033[32m' + '✅ Procedure found!' + '\033[0m\n')
                    self.dec_global[subroutine_key][declaration] = [item for item in self.finder.var_declaration]
                else:
                    raise ValueError(f"Procedure '{declaration}' is not found in any child modules.")

    def extract_array_info(self, dec_global, var_dummy_list, subroutine_key):
        """
        Extracts detailed dimensional information for arrays used within a given subroutine, 
        and stores this information in `self.all_array_info`.

        The method processes:
            - Global variable declarations (from other modules)
            - Dummy argument declarations (with potential shape info)
            - Local variable declarations
        and normalizes array-related declarations to extract shape dimensions such as:
            - Lower and upper bounds
            - Dimensionality (rank)
    
        It also tracks whether a variable has been modified and annotates it with additional 
        properties like `DIMENSION` or type information (e.g., REAL, INTEGER, etc.) 
        in `self.var_modif_info`.

        Parameters
        ----------
        dec_global : dict
            Dictionary of external/global declarations used in the subroutine, typically obtained 
            from imported modules. Keys are module names; values are lists of `Type_Declaration_Stmt`.
    
        var_dummy_list : list
            List of dummy argument declaration statements (`Type_Declaration_Stmt`) for the current subroutine.
    
        subroutine_key : str
            Identifier (name) for the current subroutine being analyzed.

        Populates
        ---------
        - self.all_array_info[subroutine_key]: dict
            Stores detailed dimension info for each array variable in the subroutine.
            Example:
            {
                'arr': [
                    {'dim_str': '1', 'dim_end': 'N'},
                    {'dim_str': '1', 'dim_end': 'M'}
                ],
            }

        - self.var_modif_info[subroutine_key]: defaultdict
            Annotates modified variables with associated types and whether they're arrays.

        Notes
        -----
        - The method combines `ALLOCATE` statements and declaration statements for `ALLOCATABLE` arrays.
        - It skips scalars and focuses only on variables with `Explicit_Shape_Spec`.
        - Dimensions are normalized into a list of dicts containing start and end bounds.
        - If dimensions are improperly formatted or too many colon-separated parts are found, it raises an error.
        """
        normalized_items = []
        for key in dec_global:
            for item in dec_global[key]:
                is_var_modified = False
                if isinstance(item, F23.Type_Declaration_Stmt):
                    var_type = item.children[0].children[0]
                    entity_decls = walk(item, F23.Entity_Decl)
                    assert len(entity_decls) == 1,\
                            "In extract_array_info: walk(item, F23.Entity_Decl)=1, but got a different number."
                    entity_decl = entity_decls[0].children[0].tostr()
                    if entity_decl in self.var_modif[subroutine_key]:
                        is_var_modified = True
                        self.var_modif_info[subroutine_key][entity_decl].append(var_type)
                    attr_spec = walk(item, F23.Attr_Spec)
                    if walk(item, F23.Explicit_Shape_Spec):
                        normalized_items.append(item)
                        if is_var_modified:
                            self.var_modif_info[subroutine_key][entity_decl].append('DIMENSION')
                    if F23.Attr_Spec('ALLOCATABLE') in attr_spec:
                        declaration_stmt = Processor().combine_allocate_declaration(dec_global[key])
                        assert isinstance(declaration_stmt, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
                        assert walk(declaration_stmt, F23.Explicit_Shape_Spec), "In extract_array_info: failed to combine_allocate_declaration!"
                        normalized_items.append(declaration_stmt)
                        if is_var_modified:
                            self.var_modif_info[subroutine_key][entity_decl].append('DIMENSION')

        for item in var_dummy_list:
            is_var_modified = False
            assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
            var_type = item.children[0].children[0]
            entity_decls = walk(item, F23.Entity_Decl)
            assert len(entity_decls) == 1,\
                    "In extract_array_info: walk(item, F23.Entity_Decl)=1, but got a different number."
            entity_decl = entity_decls[0].tostr()
            if entity_decl in self.var_modif[subroutine_key]:
                is_var_modified = True
                self.var_modif_info[subroutine_key][entity_decl].append(var_type)
            if walk(item, F23.Explicit_Shape_Spec):
                normalized_items.append(item)
                if is_var_modified:
                    self.var_modif_info[subroutine_key][entity_decl].append('DIMENSION')
        for item in self.var_local[subroutine_key]:
            assert isinstance(item, F23.Type_Declaration_Stmt), f"Item is not of type F23.Type_Declaration_Stmt!"
            if walk(item, F23.Explicit_Shape_Spec):
                normalized_items.append(item)

        for item in normalized_items:
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
                    raise ValueError("dimension control error!")
            self.all_array_info[subroutine_key][array_name] = [part for part in current_var_info]
        
        for key in self.var_modif_info:
            sorted_inner = sorted(self.var_modif_info[key].items())
            self.var_modif_info[key] = defaultdict(list,sorted_inner)

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
            #if not dec_stmt and not alo_stmt:
            #    raise ValueError("Item is neither Type_Declaration_Stmt nor Allocate_Stmt!")
            else:
                shape = walk(walk(item, F23.Allocate_Shape_Spec), F23.Name) if alo_stmt \
                        else walk(walk(item, F23.Explicit_Shape_Spec), F23.Name)
                if shape:
                    self.shapes_variables[subroutine_key].update(name.string for name in shape if name.string not in self.exclude)
                else:
                    assert dec_stmt, 'The scalar must be a Type_Declaration_Stmt!'
                    array = walk(item, F23.Dimension_Attr_Spec)
                    if not array:
                        name = walk(walk(item, F23.Entity_Decl), F23.Name)
                        if name[0].string not in self.exclude:
                            self.scalar_variables[subroutine_key].add(name[0].string)
