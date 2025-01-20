import re
import logging
from processor import Processor
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from fparser.two import Fortran2008 as F28
from collections import deque, defaultdict

class Modifier:
    """
    The Modifier class is responsible for transforming Fortran source code to handle various
    computational and array-related transformations, particularly when porting or optimizing the code for
    modern architectures such as GPUs.

    The class includes methods for:
        - Replacing unsupported Fortran GPU features with compatible code.
        - Modifying and optimizing array operations.
        - Handling vectorization and manually adjusting loops for performance improvements.
        - Parsing and modifying `If-Then` and `Else-If` statements in the Fortran code.
        - Creating call statements for subroutines, processing dummy arguments and vector dimensions.
        - Merging and adding vector loops within the Fortran code blocks for better performance.

    Attributes:
        all_array_info (dict): Stores information about all arrays found in the code.
        loop_dict (dict): Maps loops and their relevant metadata.
        var_declared (set): A set of declared variables in the code.
        imp_shape (list): A list containing the shapes of implicit arrays.
        allowed_external_subroutines (list): A list of allowed external subroutines.
        var_local (dict): Stores global variables for the Fortran code being processed.
        seen (set): A set to track visited nodes during the traversal of expressions.
        arrays_queue (collections.deque): A queue to store arrays during expression traversal.
        error_flag (dict): Flags for error handling in child nodes.

    Methods:
        replace_gpu_unsupported(parse_tree):
            Replaces GPU-unsupported features in the Fortran code.
        
        merge_vector_loop(parse_tree):
            Merges vector loops to optimize performance.
        
        add_vector_loop(modified_block):
            Adds vector loops to the transformed Fortran code.
        
        replace_vec_colon_with_index(output_stmt):
            Replaces vectorized colons in array accesses with index-based accesses.
        
        edit_if_else_stmt(stmt):
            Edits `If-Then` and `Else-If` statements, processing vector dimensions.

        create_act_call_stmt(subroutine_stmt):
            Creates and modifies a call statement for a subroutine with dummy arguments and vector dimensions.
    """
    def __init__(self, loop_vect_value, all_array_info, loop_dict, var_declared, imp_shape, io_subroutines, var_local,\
            within_calls=None, child_error_flag=None):
        self.all_array_info = all_array_info
        self.loop_dict = loop_dict
        self.var_declared = var_declared
        self.var_local = var_local
        self.imp_shape = imp_shape
        self.within_calls = within_calls
        self.child_error_flag = child_error_flag
        self.io_subroutines = io_subroutines
        self.do_index = 0
        self.do_vector_loop = None
        self.enddo_index = 0
        self.remove_return = False
        self.acc_enter_data_copyin = None
        self.vector_loop = F23.Nonlabel_Do_Stmt(loop_vect_value) if loop_vect_value else None
        self.dummy_add = (
                self.vector_loop.tostr().split('=')[0].split()[-1] if loop_vect_value else None
                )
        self.unsupported_functions = ['MINLOC', 'MAXLOC']
        self.fortran_math_functions_no_dim = ["ABS", "SQRT", "EXP", "LOG", "SIN", "COS",\
                "TAN", "ASIN", "ACOS", "ATAN", "MOD", "SIGN", "MAX", "MIN", "FLOOR",\
                "CEILING", "NINT", "RAND"]
        self.gpu_unsupported_decl = []
    def contains_unsupported_function(self, assignment_stmt):
        """
        Check if an assignment statement contains any unsupported functions.

        Parameters:
        assignment_stmt (F23.Assignment_Stmt): The Fortran assignment statement to check.

        Returns:
        bool: True if the statement contains any unsupported function, False otherwise.
        """
        for func_ref in walk(assignment_stmt, F23.Intrinsic_Function_Reference):
            if func_ref.children[0].tostr() in self.unsupported_functions:
                return True
        return False

    def traverse_expression(self, node):
        """
        Traverse an expression node and process arrays and names within it.
        
        Args:
            node: The node to be traversed, which can be of various types
                  including Part_Ref, Name, or a node with children.
        """
        try:
            if isinstance(node, F23.Part_Ref):
                part_refs = walk(node.children, F23.Part_Ref)
                self.arrays_queue.append(node)
                #if hasattr(node, 'children'):
                if part_refs:
                    for part_ref in part_refs:
                        self.arrays_queue.append(part_ref)
                        #self.traverse_expression(child)
                '''if part_refs:
                    print(f"There is indirect addressing in {node.tostr()}")
                    section_subscript_lists = [gc for gc in node.children if isinstance(gc, F23.Section_Subscript_List)]
                    for ssl in section_subscript_lists:
                        part_refs_in_ssl = [ggc for ggc in ssl.children if isinstance(ggc, F23.Part_Ref)]
                        for part_ref in part_refs_in_ssl:
                            self.arrays_queue.append(part_ref)
                else:
                    self.arrays_queue.append(node)
                '''
            elif isinstance(node, F23.Name):
                name_str = node.tostr()
                if (name_str in self.all_array_info.keys() and
                        name_str not in self.seen):
                    self.arrays_queue.append(node)
                    self.seen.add(name_str)
            elif hasattr(node, 'children'):
                for child in node.children:
                    self.traverse_expression(child)
        except Exception as e:
            raise Exception(f"Error in traverse_expression: {e}")

    def add_dos(self, assignment_stmt):
        """
        Add DO statements for array assignments, processing the dimensions and
        generating the corresponding loops.
        
        Args:
            assignment_stmt: The assignment statement to process.
        """
        try:
            assert isinstance(assignment_stmt, F23.Assignment_Stmt), (
                'Error in add_dos: the stmt must be an assignment!'
            )

            array_names_all = [
                name for name in walk(assignment_stmt, F23.Name)
                if name.tostr() in self.all_array_info.keys()
            ]
            array_col = walk(assignment_stmt, F23.Part_Ref)
            if array_col:
                node = array_col[0]
            else:
                node = array_names_all[0]

            self.dims = deque()
            self.do_stmts = {}
            self.dos = {}
            self.code = ""

            if ':' in node.tostr() and isinstance(node, F23.Part_Ref):
                for child in node.children:
                    if isinstance(child, F23.Name):
                        array_name = child.tostr()
                        assert array_name in self.all_array_info, (
                            f"Error in add_dos: Array '{array_name}' not present in all_array_info."
                        )
                        array_info = self.all_array_info[array_name]
                    if isinstance(child, F23.Section_Subscript_List):
                        for idim, dim in enumerate(child.children):
                            if dim.tostr() == ':':
                                lb = array_info[idim]['dim_str']
                                ub = array_info[idim]['dim_end']
                                assert ub in self.loop_dict, (
                                    f"Assertion failed in add_dos: `ub` ({ub}) is not in `self.loop_dict` ({self.loop_dict})."
                                )
                                index = list(self.loop_dict[ub] & self.var_declared)
                                do_stmt = f'DO {index[0]} = {lb}, {ub}'
                                if index[0] not in self.do_stmts:
                                    self.dos[index[0]] = [lb, ub]
                                    self.do_stmts[index[0]] = do_stmt
                                    self.dims.appendleft(index[0])
            elif ':' not in node.tostr() and isinstance(node, F23.Name):
                array_name = node.tostr()
                assert array_name in self.all_array_info, (
                    f"Error in add_dos: Array '{array_name}' not present in all_array_info."
                )
                array_info = self.all_array_info[array_name]
                for idim in range(len(array_info)):
                    lb = array_info[idim]['dim_str']
                    ub = array_info[idim]['dim_end']
                    assert ub in self.loop_dict, (
                        f"Assertion failed in add_dos: `ub` ({ub}) is not in `self.loop_dict` ({self.loop_dict})."
                    )
                    index = list(self.loop_dict[ub] & self.var_declared)
                    do_stmt = f'DO {index[0]} = {lb}, {ub}'
                    if index[0] not in self.dos.keys():
                        self.dos[index[0]] = [lb, ub]
                        self.do_stmts[index[0]] = do_stmt
                        self.dims.appendleft(index[0])
            self.code = ""
            while self.dims:
                dim = self.dims.popleft()
                if F23.Nonlabel_Do_Stmt(self.do_stmts[dim]).tostr() != self.vector_loop.tostr():
                    self.code += self.do_stmts[dim] + '\n'
        except Exception as e:
            raise Exception(f"Error in add_dos: {e}")


    def extract_intrinsic_names(self, stmt):
        """
        Extract intrinsic names and their associated details from a statement.

        Args:
            stmt: The statement to process for intrinsic names.

        Returns:
            A dictionary where keys are parent names of intrinsics and values
            are dictionaries with details about the intrinsics.
        """
        try:
            intrinsic_name = {}
            for intrinsic in walk(stmt, F23.Intrinsic_Name):
                if intrinsic.tostr() in self.fortran_math_functions_no_dim:
                    continue
                '''intrinsic_parent = intrinsic.parent.tostr()
                if intrinsic_parent not in intrinsic_name:
                    dict_list = []
                dict_list.append({'intrinsic': intrinsic.tostr()})
                if any(dim in intrinsic_parent for dim in ['dim', 'DIM']):
                    old_arg_spec = walk(intrinsic.parent, F23.Actual_Arg_Spec)[0].tostr()
                    dict_list.append({'dim': old_arg_spec})
                '''
                dim_value = None
                intrinsic_parent = intrinsic.parent
                fparser_class = getattr(F23, type(intrinsic_parent).__name__)
                assert len(intrinsic_parent.children) == 2, "Intrinsic parent must have exactly two children."
                assert intrinsic_parent.children[0] == intrinsic, "First child must be the intrinsic function itself."
                intrinsic_args = intrinsic.parent.children[1]
                assert isinstance(intrinsic_args, F23.Actual_Arg_Spec_List), "Second child must be an Actual_Arg_Spec_List."
                args = intrinsic_args.children
                if len(args) == 1:
                    print(f"Reduction operation detected for {args[0].tostr()} without an explicit 'DIM'. "
                            "This implies a vector reduction across all dimensions.")
                if len(args) == 2:
                    if isinstance(args[1], F23.Actual_Arg_Spec):
                        dim_key, dim_value_node = args[1].items
                        assert isinstance(dim_key, F23.Name), "First item in Actual_Arg_Spec must be a Name."
                        if dim_key.tostr().lower() == 'dim':
                            assert isinstance(dim_value_node, F23.Int_Literal_Constant), "Second item in Actual_Arg_Spec must be an Int_Literal_Constant."
                            dim_value = dim_value_node.tostr()
                    elif isinstance(args[1], F23.Int_Literal_Constant):
                        raise ValueError(
                                f"Unexpected argument structure in intrinsic. Expected 'dim={args[1].tostr()}' but found: {args[1].tostr()}")
                    else:
                        raise ValueError("Unexpected structure for intrinsic arguments.")
                intrinsic_parent_str = intrinsic_parent.tostr()
                if intrinsic_parent_str not in intrinsic_name:
                    dict_list = []
                dict_list.append({'intrinsic': intrinsic.tostr()})
                if dim_value:
                    dict_list.append({'dim': F23.Actual_Arg_Spec(f'DIM={dim_value}').tostr()})
                merged_dict = {k: v for d in dict_list for k, v in d.items()}
                intrinsic_name[intrinsic_parent_str] = merged_dict
            return intrinsic_name
        except Exception as e:
            raise Exception(f"Error in extract_intrinsic_names: {e}")

    def process_section_subscript_list(self, node, reduction_dim):
        """
        Process a section subscript list to identify the dimension for reduction.

        Args:
            node: The node containing the section subscript list.
            reduction_dim: The dimension number for reduction.

        Returns:
            The index of the dimension as a string if found, otherwise None.
        """
        try:
            for child in node.children:
                if isinstance(child, F23.Section_Subscript_List):
                    colon_dim = 0
                    for idim, dim in enumerate(child.children):
                        if dim.tostr() == ':':
                            colon_dim += 1
                            if colon_dim == int(reduction_dim):
                                return f'{idim}'
        except Exception as e:
            raise Exception(f"Error in process_section_subscript_list: {e}")
        
    def modify_colon_array(self, node, in_intrinsic_parent=None, reduction_dim=None):
        """
        Modify an array's section subscript list based on given parameters.

        Args:
            node: The node containing the section subscript list.
            in_intrinsic_parent: Optional parent intrinsic for dimensional changes.
            reduction_dim: Dimension to consider for reduction or modification.

        Returns:
            A tuple containing the modified node and optional new argument specification.
        """
        try:
            new_arg_spec = None
            for child in node.children:
                if isinstance(child, F23.Name):
                    array_name = child.tostr()
                    assert array_name in self.all_array_info, (
                        f"Error in modify_colon_array: Array '{array_name}' not present in all_array_info."
                    )
                    array_info = self.all_array_info[array_name]
                
                if isinstance(child, F23.Section_Subscript_List):
                    shape = list(child.children)
                    colon_dim = 0
                    for idim, dim in enumerate(child.children):
                        if dim.tostr() == ':':
                            colon_dim += 1
                            if in_intrinsic_parent is not None:
                                if reduction_dim == 'ALL':
                                    shape[idim] = dim.tostr()
                                    continue
                                elif idim == int(reduction_dim):
                                    shape[idim] = dim.tostr()
                                    print(f'The DIM value in {in_intrinsic_parent} must be changed to {colon_dim}.')
                                    new_arg_spec = F23.Actual_Arg_Spec(f'DIM = {colon_dim}')
                                    continue

                            lb = array_info[idim]['dim_str']
                            ub = array_info[idim]['dim_end']
                            assert ub in self.loop_dict, (
                                f"Assertion failed in modify_colon_array: `ub` ({ub}) is not in `self.loop_dict` "
                                f"({self.loop_dict})."
                            )
                            index = list(self.loop_dict[ub] & self.var_declared)
                            assert index[0] in self.dos.keys(), (
                                f"Assertion failed in modify_colon_array: `index[0]` ({index[0]}) is not a key in "
                                f"`self.dos` ({self.dos.keys()})."
                            )
                            assert lb == self.dos[index[0]][0] and ub == self.dos[index[0]][1], (
                                f"Assertion failed in modify_colon_array: Expected `lb` ({lb}) to be "
                                f"{self.dos[index[0]][0]} and `ub` ({ub}) to be {self.dos[index[0]][1]} "
                                f"for key `index[0]` ({index[0]}) in `self.dos`."
                            )
                            shape[idim] = index[0]
                            colon_dim -= 1
                        else:
                            shape[idim] = dim.tostr()
                    
                    dimensions = ', '.join([name for name in shape])
                    mod_node = F23.Part_Ref(f"{array_name}({dimensions})")
                    return mod_node, new_arg_spec
        except Exception as e:
            raise Exception(f"Error in modify_colon_array: {e}")

    def modify_colon_array_vec(self, node, in_intrinsic_parent=None, reduction_dim=None):
        """
        Modify an array's section subscript list based on given parameters, specifically
        handling cases where upper bounds are 'kjpindex'.

        Args:
            node: The node containing the section subscript list.
            in_intrinsic_parent: Optional parent intrinsic for dimensional changes.
            reduction_dim: Dimension to consider for reduction or modification.

        Returns:
            A tuple containing the modified node and optional new argument specification.
        """
        try:
            new_arg_spec = None
            for child in node.children:
                if isinstance(child, F23.Name):
                    array_name = child.tostr()
                    assert array_name in self.all_array_info, (
                        f"Error in modify_colon_array_vec: Array '{array_name}' not present in all_array_info."
                    )
                    array_info = self.all_array_info[array_name]

                if isinstance(child, F23.Section_Subscript_List):
                    shape = list(child.children)
                    colon_dim = 0
                    for idim, dim in enumerate(child.children):
                        if dim.tostr() == ':':
                            colon_dim += 1
                            if in_intrinsic_parent is not None:
                                if reduction_dim == 'ALL':
                                    shape[idim] = dim.tostr()
                                    continue
                                elif idim == int(reduction_dim):
                                    shape[idim] = dim.tostr()
                                    print(f'The DIM value in {in_intrinsic_parent} must be changed to {colon_dim}.')
                                    new_arg_spec = F23.Actual_Arg_Spec(f'DIM = {colon_dim}')
                                    continue

                            lb = array_info[idim]['dim_str']
                            ub = array_info[idim]['dim_end']
                            if ub == 'kjpindex':
                                assert ub in self.loop_dict, (
                                    f"Assertion failed in modify_colon_array_vec: `ub` ({ub}) is not in "
                                    f"`self.loop_dict` ({self.loop_dict})."
                                )
                                index = list(self.loop_dict[ub] & self.var_declared)
                                shape[idim] = index[0]
                                colon_dim -= 1
                            else:
                                shape[idim] = dim.tostr()
                        else:
                            shape[idim] = dim.tostr()
                    
                    dimensions = ', '.join([name for name in shape])
                    mod_node = F23.Part_Ref(f"{array_name}({dimensions})")
                    return mod_node, new_arg_spec
        except Exception as e:
            raise Exception(f"Error in modify_colon_array_vec: {e}")

    def remove_vec_for_locals_in_assigns(self, node):
        """
        Removes vector dimension for local arrays in assignments.

        Args:
            node (F23.Node): The node representing an array.

        Returns:
            F23.Node: A modified node with the appropriate dimensions.

        Raises:
            Exception: If any error occurs during the process.
        """
        try:
            for child in node.children:
                if isinstance(child, F23.Name):
                    array_name = child.tostr()
                    if array_name in self.var_local:
                        print(f"Array '{array_name}' is a local array and the vector dim will be removed!")
                    else:
                        print(f"Array '{array_name}' is a global/dummy array and the vector dim will be kept!")
                        return node
                    assert array_name in self.all_array_info, (
                        f"Error in remove_vec_for_locals_in_assigns: Array '{array_name}' not present in all_array_info."
                    )
                    array_info = self.all_array_info[array_name]

                if isinstance(child, F23.Section_Subscript_List):
                    shape = []
                    for idim, dim in enumerate(child.children):
                        lb = array_info[idim]['dim_str']
                        ub = array_info[idim]['dim_end']
                        if ub == 'kjpindex' and array_name in self.var_declared:
                            assert ub in self.loop_dict, (
                                f"Assertion failed in remove_vec_for_locals_in_assigns: `ub` ({ub}) is not in `self.loop_dict` "
                                f"({self.loop_dict})."
                            )
                            index = list(self.loop_dict[ub] & self.var_declared)
                            assert dim.tostr() == ':' or index[0] == dim.tostr(), (
                                f"Assertion failed in remove_vec_for_locals_in_assigns: `index[0]` ({index[0]}) is not equal to {dim.tostr()}."
                            )
                            continue
                        else:
                            shape.append(dim.tostr())
                    dimensions = ', '.join(shape)
                    mod_node = F23.Part_Ref(f"{array_name}({dimensions})") if dimensions \
                            else F23.Name(f"{array_name}")
                    return mod_node
        except Exception as e:
            raise Exception(f"Error in remove_vec_for_locals_in_assigns: {e}")

    def traverse_declaration_stmt(self, node):
        """
        Traverses a node to extract and construct a type declaration statement.

        Args:
            node (F23.Node): The node representing a declaration statement.

        Returns:
            F23.Type_Declaration_Stmt: A new type declaration statement constructed from the node.

        Raises:
            Exception: If any error occurs during the traversal or construction.
        """
        entity_decl_list = None
        intrinsic_type_spec = None
        explicit_shape_spec_list = None
        intent_attr_spec = None

        def inner_traverse(node):
            nonlocal entity_decl_list, intrinsic_type_spec, explicit_shape_spec_list, intent_attr_spec
            if isinstance(node, F23.Entity_Decl_List):
                any_shape_list = any(isinstance(child, F23.Explicit_Shape_Spec_List) for child in node.children[0].children)
                if any_shape_list:
                    for child in node.children[0].children:
                        if isinstance(child, F23.Name):
                            entity_decl_list = child.tostr()
                        elif isinstance(child, F23.Explicit_Shape_Spec_List):
                            shape = []
                            for dim in child.children:
                                if dim.tostr() != 'kjpindex':
                                    shape.append(dim.tostr())
                            explicit_shape_spec_list = ', '.join(shape)
                else:
                    entity_decl_list = node.tostr()
            elif isinstance(node, F23.Intrinsic_Type_Spec):
                intrinsic_type_spec = node.tostr()
            elif isinstance(node, F23.Explicit_Shape_Spec_List):
                shape = []
                for dim in node.children:
                    if dim.tostr() != 'kjpindex':
                        shape.append(dim.tostr())
                explicit_shape_spec_list = ', '.join(shape)
            elif isinstance(node, F23.Intent_Attr_Spec):
                intent_attr_spec = node.tostr()
            elif hasattr(node, 'children'):
                for child in node.children:
                    inner_traverse(child)
        try:
            inner_traverse(node)
        except Exception as e:
            raise Exception(f"Error in traverse_declaration_stmt: {e}")
        
        dimension_part = f",DIMENSION({explicit_shape_spec_list})" if explicit_shape_spec_list else ""
        intent_part = f",{intent_attr_spec}" if intent_attr_spec else ""
        new_declaration_stmt = F23.Type_Declaration_Stmt(
            f"{intrinsic_type_spec}{dimension_part}{intent_part}::{entity_decl_list}"
        )
        return new_declaration_stmt

    def modify_array_without_notation(self, node):
        """
        Modify an array's representation without notation based on array information and bounds.

        Args:
            node: The node containing the array name.

        Returns:
            A modified node with the updated dimensions.
        """
        try:
            array_name = node.tostr()
            assert array_name in self.all_array_info, (
                f"Error in modify_array_without_notation: Array '{array_name}' not present in all_array_info."
            )
            array_info = self.all_array_info[array_name]
            shape = []

            for idim in range(len(array_info)):
                lb = array_info[idim]['dim_str']
                ub = array_info[idim]['dim_end']
                assert ub in self.loop_dict, (
                    f"Assertion failed in modify_array_without_notation: `ub` ({ub}) is not in "
                    f"`self.loop_dict` ({self.loop_dict})."
                )
                index = list(self.loop_dict[ub] & self.var_declared)
                assert index[0] in self.dos.keys(), (
                    f"Assertion failed in modify_array_without_notation: `index[0]` ({index[0]}) is not a key in "
                    f"`self.dos` ({self.dos.keys()})."
                )
                assert lb == self.dos[index[0]][0] and ub == self.dos[index[0]][1], (
                    f"Assertion failed in modify_array_without_notation: Expected `lb` ({lb}) to be "
                    f"{self.dos[index[0]][0]} and `ub` ({ub}) to be {self.dos[index[0]][1]} "
                    f"for key `index[0]` ({index[0]}) in `self.dos`."
                )
                shape.append(index[0])

            dimensions = ', '.join([name for name in shape])
            mod_node = F23.Part_Ref(f"{array_name}({dimensions})")
            return mod_node
        except Exception as e:
            raise Exception(f"Error in modify_array_without_notation: {e}")

    def modify_array_without_notation_vec(self, node):
        """
        Modify an array's representation without notation, handling cases where the upper bound is 'kjpindex'.

        Args:
            node: The node containing the array name.

        Returns:
            A modified node with the updated dimensions.
        """
        try:
            array_name = node.tostr()
            assert array_name in self.all_array_info, (
                f"Error in modify_array_without_notation_vec: Array '{array_name}' not present in all_array_info."
            )
            array_info = self.all_array_info[array_name]
            shape = []

            for idim in range(len(array_info)):
                lb = array_info[idim]['dim_str']
                ub = array_info[idim]['dim_end']
                if ub == 'kjpindex':
                    assert ub in self.loop_dict, (
                        f"Assertion failed in modify_array_without_notation_vec: `ub` ({ub}) is not in "
                        f"`self.loop_dict` ({self.loop_dict})."
                    )
                    index = list(self.loop_dict[ub] & self.var_declared)
                    shape.append(index[0])
                else:
                    shape.append(':')

            dimensions = ', '.join([name for name in shape])
            mod_node = F23.Part_Ref(f"{array_name}({dimensions})")
            return mod_node
        except Exception as e:
            raise Exception(f"Error in modify_array_without_notation_vec: {e}")

    def modify_statement_where(self, stmt):
        """
        Modify a statement by traversing expressions and updating array notations as needed.

        Args:
            stmt: The statement to be modified.

        Returns:
            A modified version of the input statement.
        """
        try:
            self.seen = set()
            self.arrays_queue = deque()
            self.traverse_expression(stmt)
            intrinsic_name = self.extract_intrinsic_names(stmt)
            input_stmt = stmt.tostr()

            while self.arrays_queue:
                node = self.arrays_queue.popleft()
                node_str = node.tostr()

                if ':' in node_str and isinstance(node, F23.Part_Ref):
                    number_of_colon = node_str.count(':')
                    node_in_intrinsic = any(node_str in key for key in intrinsic_name.keys())

                    if node_in_intrinsic:
                        for intrinsic_parent in intrinsic_name.keys():
                            reduction_dim = 'ALL'
                            print(f'node {node_str} found within intrinsic parent {intrinsic_parent}')
                            if 'dim' in intrinsic_name[intrinsic_parent]:
                                old_arg_spec = intrinsic_name[intrinsic_parent]['dim']
                                reduction_dim = old_arg_spec.split('=')[-1].strip()
                                print(f'{reduction_dim}th of :s in {node_str} must stay as < : >.')
                                reduction_dim = self.process_section_subscript_list(node, reduction_dim)
                                print(f'dimension {reduction_dim} of {node_str} must stay as < : >.')
                            
                            mod_node, new_arg_spec = self.modify_colon_array(node, intrinsic_parent, reduction_dim)
                            mod_intrinsic_parent = re.sub(re.escape(node_str), mod_node.tostr(), intrinsic_parent)
                            if new_arg_spec is not None:
                                mod_intrinsic_parent = re.sub(re.escape(old_arg_spec), new_arg_spec.tostr(), mod_intrinsic_parent,
                                        flags=re.IGNORECASE)
                            input_stmt = re.sub(re.escape(intrinsic_parent), mod_intrinsic_parent, input_stmt)
                            node = mod_node
                    else:
                        mod_node, new_arg_spec = self.modify_colon_array(node)
                        input_stmt = re.sub(re.escape(node_str), mod_node.tostr(), input_stmt)
                        node = mod_node
                elif ':' not in node_str and isinstance(node, F23.Name):
                    array_name = node_str
                    mod_node = self.modify_array_without_notation(node)
                    input_stmt = re.sub(
                        r'\b' + re.escape(array_name) + r'\b(?!\()', mod_node.tostr(), input_stmt
                    )
                    node = mod_node
                mod_node = self.remove_vec_for_locals_in_assigns(node)
                input_stmt = re.sub(
                        re.escape(node.tostr()), mod_node.tostr(), input_stmt
                        )

            return input_stmt
        except Exception as e:
            raise Exception(f"Error in modify_statement_where: {e}")

    def replace_where(self, block):
        """
        Replace WHERE constructs in the given block with corresponding IF-ELSEIF-ELSE constructs.

        Args:
            block: The block containing WHERE constructs to be replaced.
        """
        try:
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    child = block.content[idc]
                    
                    if isinstance(child, F23.Where_Construct_Stmt):
                        modified_stmt = self.modify_statement_where(child.children[0])
                        if_then_stmt = F23.If_Then_Stmt(f'IF ({modified_stmt}) THEN')
                        self.code += if_then_stmt.tostr() + '\n'
                    elif isinstance(child, F23.Masked_Elsewhere_Stmt):
                        modified_stmt = self.modify_statement_where(child.children[0])
                        else_if_stmt = F23.Else_If_Stmt(f'ELSEIF ({modified_stmt}) THEN')
                        self.code += else_if_stmt.tostr() + '\n'
                    elif isinstance(child, F23.Elsewhere_Stmt):
                        self.code += 'ELSE' + '\n'
                    elif isinstance(child, F23.End_Where_Stmt):
                        self.code += 'ENDIF' + '\n'
                        for key in self.dos.keys():
                            lb = self.dos[key][0]
                            ub = self.dos[key][1]
                            do_stmt = f'DO {key} = {lb}, {ub}'
                            if F23.Nonlabel_Do_Stmt(do_stmt).tostr() != self.vector_loop.tostr():
                                self.code += 'ENDDO' + '\n'
                    elif isinstance(child, F23.Assignment_Stmt):
                        modified_stmt = self.modify_statement_where(child)
                        self.code += modified_stmt + '\n'
                    else:
                        self.replace_where(child)
                    
                    idc += 1
        except Exception as e:
            raise Exception(f"Error in replace_where: {e}")

    def modify_specification_part(self, block):
        """
        Modify the specification part of the block by updating entity declarations with assumed shape specifications
        and handling dummy additions.

        Args:
            block: The block containing entity declarations to be modified.

        Returns:
            The modified block.
        """
        try:
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    declaration_stmt = block.content[idc]
                    if not isinstance(declaration_stmt, F23.Type_Declaration_Stmt):
                        idc += 1
                        continue
                    if isinstance(declaration_stmt, F23.Implicit_Part):
                        idc += 1
                        continue
                    entity_decl = walk(declaration_stmt, F23.Entity_Decl)[0].tostr()
                    if len(walk(declaration_stmt, F23.Entity_Decl)) > 1:
                        separate_declarations = Processor().separate_entity_declarations(declaration_stmt)
                        block.content.pop(idc)
                        for declaration in separate_declarations:
                            entity_decl = walk(declaration, F23.Entity_Decl)[0].tostr()
                            if entity_decl == self.dummy_add:
                                continue
                            implicit_shape = walk(declaration, F23.Assumed_Shape_Spec)
                            if implicit_shape:
                                assert entity_decl in self.imp_shape, (
                                    f"Error: {entity_decl} is not in imp_shape."
                                )
                                block.content.insert(idc, self.imp_shape[entity_decl])
                            else:
                                block.content.insert(idc, declaration)
                            idc += 1
                        continue
                    else:
                        if entity_decl == self.dummy_add:
                            del block.content[idc]
                            continue
                        implicit_shape = walk(declaration_stmt, F23.Assumed_Shape_Spec)
                        if implicit_shape:
                            assert entity_decl in self.imp_shape, (
                                f"Error: {entity_decl} is not in imp_shape."
                            )
                            block.content[idc] = self.imp_shape[entity_decl]
                    idc += 1

                first_idx = None
                last_idx = None
                for idx, node in enumerate(block.content):
                    if isinstance(node, F23.Type_Declaration_Stmt):
                        first_idx = idx
                        break 

                block.content.insert(first_idx + 1, self.dummy_add_decl)
                if self.error_flag.keys():
                    for key in self.error_flag.keys():
                        block.content.insert(first_idx + 1, self.error_flag[key]['error_flag_decl'])

                for idx, node in enumerate(block.content):
                    if isinstance(node, F23.Type_Declaration_Stmt):
                        last_idx = idx

                if self.gpu_unsupported_decl:
                    for decl in self.gpu_unsupported_decl:
                        block.content.insert(last_idx + 1, decl)
                
            return block
        except Exception as e:
            raise Exception(f"Error, an error occurred while modifying the specification part: {e}")

    def remove_vec_for_locals_in_specification(self, block):
        """
        Processes a block of statements, updating type declaration statements to remove unnecessary vector dimensions
        for local arrays.

        Args:
            block (F23.Block): The block of statements to be processed.

        Returns:
            F23.Block: The updated block with modified type declaration statements.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    declaration_stmt = block.content[idc]
                    if not isinstance(declaration_stmt, F23.Type_Declaration_Stmt):
                        idc += 1
                        continue
                    else:
                        entity_decls = walk(declaration_stmt, F23.Entity_Decl)
                        assert len(entity_decls) == 1,\
                                "walk(declaration_stmt, F23.Entity_Decl), but got a different number."
                        entity_decl = entity_decls[0].tostr()
                        if entity_decl not in self.var_local or not walk(declaration_stmt, F23.Explicit_Shape_Spec_List):
                            idc += 1
                            continue
                        block.content[idc] = self.traverse_declaration_stmt(declaration_stmt)
                    idc += 1
            return block
        except Exception as e:
            raise Exception(f"Error, an error occurred in remove vec for locals in_specification: {e}")

    def replace_unsupported_function_with_manual_loop(self, assignment_stmt):
        """
        Replace unsupported GPU functions (e.g., MINLOC, MAXLOC) in a Fortran assignment with a manual loop.

        Parameters:
        assignment_stmt (F23.Assignment_Stmt): The Fortran assignment statement to process.

        Returns:
        str: The modified Fortran code with unsupported functions replaced by a manual loop.
        """
        func_node = next(
                (child for child in walk(assignment_stmt) if isinstance(child, F23.Intrinsic_Function_Reference)),
                None
                )
        if not func_node:
            return assignment_stmt
        
        func_body_str = None
        func_dim = 1
        self.seen = set()
        self.arrays_queue = deque()
        self.traverse_expression(func_node)

        for child in func_node.children[1].children:
            if isinstance(child, F23.Actual_Arg_Spec) and child.children[0].tostr().lower() == 'dim':
                func_dim = int(child.children[1].tostr())
            else:
                func_body_str = child.tostr()
        while self.arrays_queue:
            node = self.arrays_queue.popleft()
            assert ':' in node.tostr() and isinstance(node, F23.Part_Ref)
            reduction_dim = self.process_section_subscript_list(node, func_dim)
            for child in node.children:
                if isinstance(child, F23.Name):
                    array_name = child.tostr()
                    assert array_name in self.all_array_info, (
                            f"Error in replace_unsupported_function_with_manual_loop: Array '{array_name}' not present in all_array_info."
                            )
                    array_info = self.all_array_info[array_name]
                if isinstance(child, F23.Section_Subscript_List):
                    shape = list(child.children)
                    for idim, dim in enumerate(child.children):
                         if dim.tostr() == ':' and idim == int(reduction_dim):
                            lb = array_info[idim]['dim_str']
                            ub = array_info[idim]['dim_end']
                            print('lb, ub', ub)
                            print(self.loop_dict[ub])
                            print(self.var_declared)
                            index = list(self.loop_dict[ub] & self.var_declared)
                            shape[idim] = index[0]
                         else:
                            shape[idim] = dim.tostr()
                    dimensions = ', '.join([name for name in shape])
                    mod_node = F23.Part_Ref(f"{array_name}({dimensions})")
                    func_body_str = re.sub(
                            re.escape(node.tostr()), mod_node.tostr(), func_body_str
                            )
        loop_code = []
        loop_code.append(f'{self.minmax_value_str} = HUGE(0.0)')
        loop_code.append(f'{self.minmax_index_str} = 1')
        loop_code.append(f'DO {index[0]} = {lb}, {ub}')
        loop_code.append(f'  IF ({func_body_str} .LT. {self.minmax_value_str}) THEN')
        loop_code.append(f'    {self.minmax_value_str} = {func_body_str}')
        loop_code.append(f'    {self.minmax_index_str} = {index[0]} + 1 - {lb}')
        loop_code.append('  END IF')
        loop_code.append('END DO')
        new_assignment = re.sub(
                            re.escape(func_node.tostr()), self.minmax_index_str, assignment_stmt.tostr()
                            )
        return Processor().parse_fortran_statement("\n".join(loop_code) + f"\n{new_assignment}")


    def replace_vec_colon_with_index(self, assignment_stmt):
        """
        Replaces colons in array subscripts with appropriate index variables
        in an assignment statement.

        Args:
            assignment_stmt (F23.Assignment_Stmt): The assignment statement that
                contains array subscripts with colons to be replaced.

        Returns:
            F23.Assignment_Stmt: A new assignment statement with colons replaced by
                index variables.
        """
        try:
            assert isinstance(assignment_stmt, F23.Assignment_Stmt), (
                'Error: the stmt must be an assignment!'
            )
            self.seen = set()
            self.arrays_queue = deque()
            self.traverse_expression(assignment_stmt)
            intrinsic_name = self.extract_intrinsic_names(assignment_stmt)
            input_stmt = assignment_stmt.tostr()
            
            while self.arrays_queue:
                node = self.arrays_queue.popleft()
                if ':' in node.tostr() and isinstance(node, F23.Part_Ref):
                    node_in_intrinsic = any(
                        node.tostr() in key for key in intrinsic_name.keys()
                    )
                    if node_in_intrinsic:
                        for intrinsic_parent in intrinsic_name.keys():
                            reduction_dim = 'ALL'
                            print(f'node {node.tostr()} found within intrinsic parent {intrinsic_parent}')
                            if 'dim' in intrinsic_name[intrinsic_parent]:
                                old_arg_spec = intrinsic_name[intrinsic_parent]['dim']
                                reduction_dim = old_arg_spec.split('=')[-1].strip()
                                print(f'{reduction_dim}th of :s in {node.tostr()} must stay as < : >.')
                                reduction_dim = self.process_section_subscript_list(node, reduction_dim)
                                print(f'dimension {reduction_dim} of {node.tostr()} must stay as < : >.')
                            mod_node, new_arg_spec = self.modify_colon_array_vec(node, intrinsic_parent, reduction_dim)
                            mod_intrinsic_parent = re.sub(re.escape(node.tostr()), mod_node.tostr(),intrinsic_parent)
                            if new_arg_spec is not None:
                                mod_intrinsic_parent = re.sub(re.escape(old_arg_spec), new_arg_spec.tostr(),mod_intrinsic_parent,
                                        flags=re.IGNORECASE)
                            input_stmt = re.sub(re.escape(intrinsic_parent), mod_intrinsic_parent,input_stmt)
                            node = mod_node
                    else:
                        mod_node, new_arg_spec = self.modify_colon_array_vec(node)
                        input_stmt = re.sub(
                            re.escape(node.tostr()), mod_node.tostr(), input_stmt
                        )
                        node = mod_node

                elif ':' not in node.tostr() and isinstance(node, F23.Name):
                    array_name = node.tostr()
                    mod_node = self.modify_array_without_notation_vec(node)
                    input_stmt = re.sub(
                        r'\b' + re.escape(array_name) + r'\b(?!\()', mod_node.tostr(),
                        input_stmt
                    )
                    node = mod_node
                mod_node = self.remove_vec_for_locals_in_assigns(node)
                input_stmt = re.sub(
                        re.escape(node.tostr()), mod_node.tostr(), input_stmt
                        )
            output_stmt = F23.Assignment_Stmt(input_stmt)
            if self.contains_unsupported_function(output_stmt):
                print(f"Unsupported function found in assignment statement: {output_stmt.tostr()}")
                output_stmt = self.replace_unsupported_function_with_manual_loop(output_stmt) 
            return output_stmt

        except Exception as e:
            raise Exception(f"Error in replace_vec_colon_with_index: {e}")

    def edit_if_else_stmt(self, stmt):
        """
        """
        try:
            assert isinstance(stmt, (F23.If_Then_Stmt, F23.Else_If_Stmt)), (
                    'Error: the stmt must be an If_Then_Stmt or an Else_If_Stmt!'
                    )
            self.seen = set()
            self.arrays_queue = deque()
            self.traverse_expression(stmt)
            input_stmt = stmt.tostr()

            while self.arrays_queue:
                node = self.arrays_queue.popleft()
                mod_node = self.remove_vec_for_locals_in_assigns(node)
                input_stmt = re.sub(
                        re.escape(node.tostr()), mod_node.tostr(), input_stmt
                        )
            return F23.If_Then_Stmt(input_stmt)

        except Exception as e:
            raise Exception(f"Error in replace_vec_colon_with_index: {e}")

    def create_act_call_stmt(self, subroutine_stmt):
        """
        Creates a call statement for a subroutine, processing dummy arguments and handling vector dimensions
        where necessary.

        Args:
            subroutine_stmt (F23.Subroutine_Stmt): The subroutine statement node to process.

        Returns:
            None: Updates the `subroutine_call_act_vec` attribute with the constructed statement.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            dummy_arg_list = []
            subroutine_name = None

            for child in subroutine_stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Name):
                    subroutine_name = child.tostr()
                elif isinstance(child, F23.Dummy_Arg_List):
                    for grandchild in child.children:
                        grandchild_name = grandchild.tostr()
                        dummy_arg_list.append(grandchild_name)
            arg_list = ', '.join(dummy_arg_list)
            code = (
                f"{self.vector_loop.tostr()}\n"
                f"CALL {subroutine_name}({arg_list})\n"
                "ENDDO"
            )
            self.subroutine_call_act_vec = Processor().parse_fortran_statement(code)
        except Exception as e:
            raise Exception(f"Error in create_act_call_stmt: {e}")

    def merge_vector_loop(self, block):
        """
        Merges vector loops by modifying specific children if they match the `vector_loop`.

        Args:
            block (F23.Block): The block of code containing various statements.

        Returns:
            F23.Block: The modified block of code with merged vector loops.

        """
        try:
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    child = block.content[idc]
                    if isinstance(child, F23.Subroutine_Stmt):
                        subroutine_name, arg_list = None, None
                        subroutine_stmt = "subroutine "
                        if loop_vect_value:
                            add_decl = f"INTEGER(KIND = i_std), INTENT(IN) :: {self.dummy_add}"
                            self.dummy_add_decl = F23.Type_Declaration_Stmt(add_decl)
                        for grandchild in child.children:
                            if grandchild is None:
                                continue 
                            if isinstance(grandchild, F23.Name):
                                subroutine_name = grandchild.tostr()
                                subroutine_stmt += f"{grandchild.tostr()}_acc"
                            elif isinstance(grandchild, F23.Dummy_Arg_List):
                                arg_list = grandchild.tostr()
                                if loop_vect_value:
                                    dummy_arg_list_new = f"{self.dummy_add}, {arg_list}"
                                else:
                                    dummy_arg_list_new = arg_list
                                self.acc_enter_data_copyin = []
                                for grandgrandchild in grandchild.children:
                                    grandgrandchild_str = grandgrandchild.tostr()
                                    if grandgrandchild_str in self.all_array_info:
                                        self.acc_enter_data_copyin.append(grandgrandchild_str)
                                if self.error_flag.keys():
                                    error_flag_keys = ', '.join([name for name in self.error_flag.keys()])
                                    subroutine_stmt += f"({error_flag_keys}, {dummy_arg_list_new})"
                                else:
                                    subroutine_stmt += f"({dummy_arg_list_new})"
                            else:
                                raise ValueError(f"Unexpected type '{type(grandchild)}' encountered in children.")
                        assert subroutine_name is not None, "Subroutine name cannot be None."
                        subroutine_stmt_mod = F23.Subroutine_Stmt(subroutine_stmt)
                        subroutine_call = f"CALL {subroutine_name}({arg_list})"
                        self.subroutine_call_act_org = F23.Call_Stmt(subroutine_call)
                        self.create_act_call_stmt(subroutine_stmt_mod)
                        block.content[idc] = subroutine_stmt_mod
                        acc_routine_seq_cmd = Processor().parse_fortran_comment(f"!$ACC ROUTINE SEQ")
                        block.content.insert(idc + 1, acc_routine_seq_cmd)
                    elif isinstance(child, F23.Call_Stmt):
                        if child.children[0].tostr() not in self.io_subroutines:
                            add_error_flag_into_call_stmt = False
                            call_stmt = "call "
                            for grandchild in child.children:
                                if grandchild is None:
                                    continue 
                                if isinstance(grandchild, F23.Name):
                                    assert grandchild.tostr() in self.within_calls, (
                                        f"Error: '{grandchild.tostr()}' not found in within_calls"
                                    )
                                    #if self.child_error_flag and grandchild.tostr() in self.child_error_flag.keys():
                                    if (self.child_error_flag and
                                            grandchild.tostr() in self.child_error_flag and
                                            self.child_error_flag[grandchild.tostr()]):
                                        add_error_flag_into_call_stmt = True
                                        add_error_flags = self.child_error_flag[grandchild.tostr()]
                                    call_stmt += f"{grandchild.tostr()}_acc"
                                elif isinstance(grandchild, F23.Actual_Arg_Spec_List):
                                    if add_error_flag_into_call_stmt:
                                        error_flag_keys = ', '.join([name for name in add_error_flags.keys()])
                                        call_stmt += f"({error_flag_keys}, {self.dummy_add}, {grandchild.tostr()})"
                                    else:
                                        call_stmt += f"({self.dummy_add}, {grandchild.tostr()})"
                                else:
                                    raise ValueError(f"Unexpected type '{type(grandchild)}' encountered in children.")
                            block.content[idc] = F23.Call_Stmt(call_stmt)
                    elif isinstance(child, F23.End_Subroutine_Stmt):
                        end_subroutine_stmt = "end subroutine "
                        for grandchild in child.children:
                            if grandchild is None:
                                continue
                            if isinstance(grandchild, F23.Name):
                                end_subroutine_stmt += f"{grandchild.tostr()}_acc"
                        block.content[idc] = F23.End_Subroutine_Stmt(end_subroutine_stmt)
                    elif isinstance(child, F23.Specification_Part):
                        new_specification_part = self.modify_specification_part(child)
                        block.content[idc] = self.remove_vec_for_locals_in_specification(new_specification_part)
                    elif isinstance(child, F23.Where_Construct):
                        print("Where construc: ", child.tostr())
                        stmt = walk(child, F23.Assignment_Stmt)
                        self.add_dos(stmt[0])
                        self.replace_where(child)
                        block.content[idc] = Processor().parse_fortran_statement(self.code)
                    elif isinstance(child, F23.Nonlabel_Do_Stmt):
                        self.do_index += 1
                        self.enddo_index += 1
                        if child.tostr() == self.vector_loop.tostr():
                            self.do_vector_loop = self.do_index
                            del block.content[idc]
                            continue
                    elif isinstance(child, F23.End_Do_Stmt):
                        self.enddo_index -= 1
                        self.do_index -= 1
                        if self.enddo_index + 1 == self.do_vector_loop:
                            del block.content[idc]
                            self.do_vector_loop = None
                            continue
                    elif isinstance(child, F23.If_Then_Stmt):
                        print("Original If Then Stmt ", child.tostr())
                        modified_stmt = self.edit_if_else_stmt(child)
                        print("Modified If Then Stmt:", modified_stmt.tostr())
                        block.content[idc] = modified_stmt
                    elif isinstance(child, F23.If_Stmt):
                        if walk(child, F23.Write_Stmt):
                            del block.content[idc]
                            continue
                        else:
                            code = f"IF({child.children[0]})THEN\n{child.children[1]}\nENDIF\n"
                            block.content[idc] = Processor().parse_fortran_statement(code)
                            idc -= 1
                    elif isinstance(child, (F23.Open_Stmt, F23.Write_Stmt, F23.Close_Stmt)):
                        if '1363' in child.tostr():
                            del block.content[idc]
                            continue
                    elif isinstance(child, F23.Return_Stmt):
                        if isinstance(block.content[idc+2], F23.End_If_Stmt):
                            block.content[idc+2] = F23.Else_Stmt("ELSE")
                        else:
                            raise RuntimeError("Error: The next statement is not an End_If_Stmt. "
                                               "Cannot insert ELSE statement here.")
                        del block.content[idc]
                        self.remove_return = True
                        continue
                    elif isinstance(child, F23.Assignment_Stmt):
                        print("Original Assignment Statement:", child.tostr())
                        modified_stmt = self.replace_vec_colon_with_index(child)
                        print("Modified Assignment Statement:", modified_stmt.tostr())
                        block.content[idc] = modified_stmt
                    else:
                        self.merge_vector_loop(child)
                    idc += 1
            return block
        except Exception as e:
            raise Exception(f"Error in merge_vector_loop: {e}")

    def add_vector_loop(self, block):
        """
        Adds the vector loop to the block content.

        Args:
            block (F23.Block): The block of code where the vector loop should be added.

        Returns:
            F23.Block: The modified block of code with the vector loop added.
        """
        try:
            for node in block.content:
                if not hasattr(node, "content"):
                    continue
                if isinstance(node, F23.Execution_Part):
                    if self.remove_return:
                        idx = len(node.content)
                        node.content.insert(idx, F23.End_If_Stmt('ENDIF'))
            return Processor().parse_fortran_string(block.tofortran())
        except Exception as e:
            raise Exception(f"Error in add_vector_loop: {e}")

    def replace_gpu_unsupported(self, subroutine_tree):
        """
        Replace WRITE statements with error flags in a Fortran subroutine.
        This method walks through the provided subroutine parse tree, identifies WRITE statements,
        and replaces them with error flags. It also handles CALL statements containing 'ipslerr'..

        Args:
        subroutine_tree (Program): The parse tree of the Fortran subroutine.

        Returns:
        tuple: Updated subroutine_tree and a dictionary containing error flag information.
        """
        try:
            if self.contains_unsupported_function(subroutine_tree):
                self.minmax_value_str,  self.minmax_index_str = 'minmax_value', 'minmax_index'
                self.gpu_unsupported_decl = [F23.Type_Declaration_Stmt(f'INTEGER(KIND = i_std) :: {self.minmax_index_str}'),
                        F23.Type_Declaration_Stmt(f'REAL(r_std) :: {self.minmax_value_str}')
                        ]
            write_stmts = walk(subroutine_tree, F23.Write_Stmt)
            parent_to_writes = defaultdict(list)
            parent_node_to_str = {}
            self.error_flag = {}
            error_flag_counter = 0
            if not write_stmts and self.child_error_flag is None:
                return subroutine_tree

            subroutine_name = next((child.children[0].get_name().string for child in subroutine_tree.children
                                    if isinstance(child, F23.Subroutine_Subprogram)), None)

            for write_stmt in write_stmts:
                if '1363' in write_stmt.tostr():
                    continue
                parent = write_stmt.parent
                parent_str = parent.tostr()
                parent_to_writes[parent_str].append(write_stmt)
                parent_node_to_str[parent_str] = parent

            for parent_str, writes in parent_to_writes.items():
                dict_info = {'error_flag_decl': '', 'error_flag_init':'', 'write_calls': ''}
                error_flag_counter += 1
                error_flag_name = f'error_flag_{subroutine_name}_{error_flag_counter}'

                parent = parent_node_to_str[parent_str]
                if isinstance(parent.children, tuple):
                    continue
                error_flag_decl_str = f'INTEGER(KIND = i_std), INTENT(INOUT) :: {error_flag_name}'
                error_flag_decl = F23.Type_Declaration_Stmt(error_flag_decl_str)
                dict_info['error_flag_decl'] = error_flag_decl

                error_flag_init_str = f'{error_flag_name} = 0'
                error_flag_init = F23.Assignment_Stmt(error_flag_init_str)
                dict_info['error_flag_init'] = error_flag_init
                
                error_flag_set_str = f'{error_flag_name} = {error_flag_name} + 1'
                error_flag_set = F23.Assignment_Stmt(error_flag_set_str)
                parent.children[parent.children.index(writes[0])] = error_flag_set

                code = f"""
                WRITE(numout, *) 'Warning: in the {subroutine_name}, {error_flag_name} is > 0 :', {error_flag_name}
                """

                index = 0
                while index < len(parent.children):
                    child = parent.children[index]
                    if isinstance(child, F23.Write_Stmt):
                        parent.children.pop(index)
                    elif isinstance(child, F23.Call_Stmt):
                        if 'ipslerr' in child.children[0].tostr():
                            code += child.tostr() + '\n'
                            parent.children.pop(index)
                        else:
                            index += 1
                    else:
                        index += 1

                check_point = f"""
                IF ({error_flag_name} .GT. 0) THEN
                    {code}
                ENDIF
                """

                dict_info['write_calls'] = Processor().parse_fortran_statement(check_point)
                self.error_flag[error_flag_name] = dict_info

            if self.child_error_flag:
                for key in self.child_error_flag.keys():
                    for grandkey in self.child_error_flag[key]:
                        self.error_flag[grandkey] = self.child_error_flag[key][grandkey]

            return subroutine_tree
        except Exception as e:
            raise Exception(f"Error in replace_gpu_unsupported: {e}")

def main():
    """
    """
    code_string = subroutine_tree.tofortran()
    parse_tree = Processor().parse_fortran_string(code_string)
    #assert parse_tree == subroutine_tree, "Parsed tree does not match the original subroutine_tree"
    file_path = './org.f90'
    Processor().write_fortran_code_to_file(parse_tree, file_path)
    modifier = Modifier(cls.all_array_info[subroutine_key], cls.loop_dict, cls.var_declared[subroutine_key], cls.imp_shape[subroutine_key],
                                  cls.allowed_external_subroutines, cls.var_local_names[subroutine_key])
    parse_tree = modifier.replace_gpu_unsupported(parse_tree)
    modified_block = modifier.merge_vector_loop(parse_tree)
    assert modifier.do_index == 0 and modifier.enddo_index == 0, (
        f"Error: do_index and enddo_index are not reset properly. "
        f"do_index={modifier.do_index}, enddo_index={modifier.enddo_index}"
    )
    final_block = modifier.add_vector_loop(modified_block)
    file_path = './mod.f90'
    Processor().write_fortran_code_to_file(final_block, file_path)

if __name__ == "__main__":
    # Initialize SubroutineFinder with paths to module directory and module tree.
    cls = Extractor(isolator.module_dir_sp, isolator.module_tree_sp)
    cls.find_subroutines()
    subroutine_key = 'hydrol_root_profile'
    subroutine_tree = cls.subroutines[subroutine_key]
    cls.find_variables(subroutine_tree, subroutine_key)
    cls.find_global_variables(isolator.module_dir_sp, isolator.module_tree_sp, cls.var_global[subroutine_key], subroutine_key)
    cls.extract_loop_indices()
    cls.extract_array_info(cls.dec_global[subroutine_key], cls.var_dummy[subroutine_key], subroutine_key)
    cls.extract_names(subroutine_key)
    # Call the main function to process the Fortran code.
    
    main()

