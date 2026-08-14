# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import re
import subprocess
from collections import defaultdict, deque
from typing import Any

from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

from fgpt.core.common.logger import Logger
from fgpt.core.frontend.processor import Processor


class TapenadePass:
    """
    Pass for processing and cleaning Tapenade-generated tangent/adjoint code.

    This class handles the post-processing of Tapenade differentiated code,
    including validation of array declarations, cleaning of external statements,
    and replacement of 'isize' dimensions with actual array shapes.

    Parameters
    ----------
    logger : Logger, optional
        Logger instance for output. If None, a default Logger is created.
    allowed_external_subroutines : list of str, optional
        List of external subroutine names that are allowed to be imported.
    all_array_info : dict, optional
        Dictionary containing information about all arrays in the code.

    Attributes
    ----------
    logger : Logger
        Logger instance for output.
    processor : Processor
        Fortran code processor for parsing and manipulation.
    allowed_external_subroutines : list of str
        List of allowed external subroutine names.
    all_array_info : dict
        Dictionary containing array shape information.
    array_shape_not_defined : defaultdict(list)
        Dictionary mapping array names to shape info for undefined shapes.
    array_declr_not_defined : defaultdict(list)
        Dictionary mapping array names to declaration statements for undefined shapes.
    size_in_call_stmt : defaultdict(list)
        Dictionary mapping 'isize' arguments to call statements.
    """

    def __init__(
        self,
        logger: Logger,
        allowed_external_subroutines: list[str] | None = None,
        all_array_info: dict | None = None,
    ) -> None:
        """
        Initialize the TapenadePass instance.

        Parameters
        ----------
        logger : Logger
            Logger instance for output.
        allowed_external_subroutines : list of str, optional
            List of external subroutine names allowed to be imported.
        all_array_info : dict, optional
            Dictionary containing array shape information.
        """
        self.logger = logger or Logger()
        self.logger.show_header("TapenadePass Initialized")

        self.processor = Processor(logger=self.logger)
        self.allowed_external_subroutines = allowed_external_subroutines or []
        self.all_array_info = all_array_info or {}

        self.array_shape_not_defined = defaultdict(list)
        self.array_declr_not_defined = defaultdict(list)
        self.size_in_call_stmt = defaultdict(list)

    def check_tapenade_isize(self, parse_tree: Any) -> None:
        """
        Validate array declarations against all_array_info.

        This method walks through the AST and checks if all array declarations
        that have dimensions starting with "isize" are present in the
        all_array_info dictionary. If an array is not found, a warning is logged.

        Parameters
        ----------
        parse_tree : Any
            The Fortran AST to validate.

        Raises
        ------
        AssertionError
            If the AST structure is not as expected.
        ValueError
            If invalid dimension formats are encountered.
        """
        try:
            self.logger.info("Validating array declarations against all_array_info")
            self.array_shape_not_defined.clear()
            self.array_declr_not_defined.clear()
            self.size_in_call_stmt.clear()

            # Walk through all type declaration statements and check if there is any tapenade isize
            for decl in walk(parse_tree, F23.Type_Declaration_Stmt):
                explicit_shape_spec = walk(decl, F23.Explicit_Shape_Spec)
                if explicit_shape_spec:
                    # Get entity declarations from this type declaration
                    entity_decls = walk(decl, F23.Entity_Decl)
                    assert len(entity_decls) == 1, (
                        f"Expected exactly 1 Entity_Decl in Type_Declaration_Stmt, "
                        f"but got {len(entity_decls)} in {decl.tostr()}"
                    )

                    # Extract the entity declaration
                    array_name = entity_decls[0].tostr()

                    current_array_dims = []
                    for dim in explicit_shape_spec:
                        start_end = [part.strip() for part in dim.tostr().split(":")]
                        lse = len(start_end)
                        if lse == 1:
                            current_array_dims.append(
                                {"dim_str": "1", "dim_end": start_end[0]}
                            )
                        elif lse == 2:
                            current_array_dims.append(
                                {"dim_str": start_end[0], "dim_end": start_end[1]}
                            )
                        else:
                            raise ValueError(f"Invalid dimension format: {dim.tostr()}")

                    if any("isize" in dim.tostr() for dim in explicit_shape_spec):
                        self.logger.warning(
                            f"Array declaration '{decl.tostr()}' contains 'isize' "
                            "dimension. Shape needs to be identified!"
                        )
                        self.array_shape_not_defined[array_name] = current_array_dims
                        assert hasattr(decl.parent, "content"), (
                            f"Expected parent of declaration to have a 'content' "
                            f"attribute, got {type(decl.parent).__name__}"
                        )
                        self.array_declr_not_defined[array_name].append(decl)
                    else:
                        if array_name not in self.all_array_info:
                            self.logger.warning(
                                f"Array '{array_name}' not found in all_array_info."
                            )
                            self.logger.info(
                                f"Array '{array_name}' is added to all_array_info as "
                                "its shape is well defined!"
                            )
                            self.all_array_info[array_name] = current_array_dims

            if self.array_shape_not_defined:
                self.logger.info(
                    "The following arrays have shapes that need to be fully defined:"
                )
                for array_name, shape_info in self.array_shape_not_defined.items():
                    self.logger.info(
                        f" - Array '{array_name}' with shape_info '{shape_info}'"
                    )
            else:
                self.logger.info(
                    "No arrays with 'isize' dimensions found that are missing from "
                    "all_array_info."
                )

            # Process call statements with 'isize' arguments
            for call_stmt in walk(parse_tree, F23.Call_Stmt):
                actual_arg_list = call_stmt.children[1]
                actual_args_text = actual_arg_list.tostr()

                if "isize" not in actual_args_text:
                    continue

                args = actual_arg_list.children
                if len(args) != 2:
                    raise ValueError(
                        f"Expected exactly 2 actual arguments, got {len(args)}: "
                        f"{actual_arg_list}"
                    )

                size_arg = args[1]

                if not hasattr(call_stmt.parent, "content"):
                    parent = call_stmt.parent
                    self.logger.warning(
                        "Expected parent of Call_Stmt to have a 'content' attribute, "
                        f"got {type(parent).__name__}"
                    )

                    # Handle If_Stmt that needs normalization
                    if isinstance(parent, F23.If_Stmt):
                        condition = parent.children[0]
                        action = parent.children[1]

                        code = f"IF ({condition}) THEN\n{action}\nENDIF"

                        code_tree = self.processor.parse_fortran_statement(code)
                        new_if_node = code_tree.children[0]

                        if not isinstance(new_if_node, F23.If_Construct):
                            raise TypeError(
                                "Expected the first child of code_tree to be an "
                                f"If_Construct, got {type(new_if_node).__name__}: "
                                f"{new_if_node.tostr()}"
                            )

                        old_parent = getattr(parent, "parent", None)
                        if old_parent is None or not hasattr(old_parent, "content"):
                            raise TypeError(
                                f"Expected parent of '{type(parent).__name__}' "
                                "to have a 'content' attribute, "
                                f"got {type(old_parent).__name__}"
                            )

                        for index, node in enumerate(old_parent.content):
                            if node is parent:
                                old_parent.content[index] = new_if_node
                                new_if_node.parent = old_parent
                                break
                        else:
                            raise ValueError(
                                "Could not find the original If_Stmt in its "
                                "parent's content."
                            )

                        # Get the corresponding Call_Stmt from the new subtree
                        new_call_stmts = walk(new_if_node, F23.Call_Stmt)
                        if len(new_call_stmts) != 1:
                            raise ValueError(
                                "Expected exactly one Call_Stmt in the normalized "
                                f"If_Construct, got {len(new_call_stmts)}"
                            )
                        call_stmt = new_call_stmts[0]
                    else:
                        raise NotImplementedError("Not implemented yet!")

                if not hasattr(call_stmt.parent, "content"):
                    raise TypeError(
                        "Expected parent of Call_Stmt to have a 'content' attribute, "
                        f"got {type(call_stmt.parent).__name__}"
                    )

                self.size_in_call_stmt[size_arg.tostr()].append(call_stmt)

            if self.size_in_call_stmt:
                self.logger.info(
                    "The following 'isize' arguments are used in CALL statements "
                    "and need to be identified:"
                )
                for size_arg, statements in self.size_in_call_stmt.items():
                    self.logger.info(
                        f"Size argument '{size_arg}' "
                        f"with {len(statements)} occurrence(s):"
                    )
                    for call_stmt in statements:
                        self.logger.info(f" - CALL: '{call_stmt.tostr()}'")
            else:
                self.logger.info(
                    "No CALL statements containing 'isize' arguments were found."
                )

        except Exception:
            self.logger.exception("Failed to validate array declarations")
            raise

    def process_call_stmt(self, call_stmt: F23.Call_Stmt) -> F23.Call_Stmt:
        """
        Process CALL statements containing an array section followed by an
        isize* argument.

        This method replaces the 'isize*' argument with the actual dimension
        size derived from the all_array_info dictionary.

        Parameters
        ----------
        call_stmt : F23.Call_Stmt
            The CALL statement to process.

        Returns
        -------
        F23.Call_Stmt
            The processed CALL statement with replaced dimension size.

        Raises
        ------
        AssertionError
            If the CALL statement structure is not as expected.
        NotImplementedError
            If the array argument is not a Part_Ref.

        Example
        -------
        >>> call_stmt = F23.Call_Stmt("CALL pushreal8array(mc(ji, :, jst), isize2ofdrfmc)")
        >>> new_call_stmt = process_call_stmt(call_stmt)
        >>> print(new_call_stmt.tostr())
        CALL pushreal8array(mc(ji, :, jst), (nlon) - (1) + 1)
        """
        new_call_stmt = None
        assert isinstance(call_stmt, F23.Call_Stmt), (
            f"Expected F23.Call_Stmt, got {type(call_stmt).__name__}: {call_stmt}"
        )

        children = call_stmt.children
        assert len(children) == 2, (
            f"Expected Call_Stmt to have 2 children, got {len(children)}: {call_stmt}"
        )

        call_name = children[0]
        assert isinstance(call_name, F23.Name), (
            f"Expected first child of Call_Stmt to be F23.Name, "
            f"got {type(call_name).__name__}: {call_name}"
        )

        actual_arg_list = children[1]
        assert isinstance(actual_arg_list, F23.Actual_Arg_Spec_List), (
            f"Expected second child of Call_Stmt to be "
            f"F23.Actual_Arg_Spec_List, got {type(actual_arg_list).__name__}: "
            f"{actual_arg_list}"
        )

        args = actual_arg_list.children
        assert len(args) == 2, (
            f"Expected exactly 2 actual arguments, got {len(args)}: {actual_arg_list}"
        )

        array_arg = args[0]
        size_arg = args[1]

        assert isinstance(size_arg, F23.Name), (
            f"Expected second argument to be F23.Name, "
            f"got {type(size_arg).__name__}: {size_arg}"
        )

        assert "isize" in size_arg.tostr(), (
            f"Expected second argument to contain 'isize', "
            f"got '{size_arg.tostr()}' in CALL '{call_stmt.tostr()}'"
        )

        if isinstance(array_arg, F23.Part_Ref):
            array_name = next(
                (
                    child.tostr()
                    for child in array_arg.children
                    if isinstance(child, F23.Name)
                ),
                None,
            )
            assert array_name is not None, (
                f"Could not find F23.Name in array argument '{array_arg}'"
            )

            assert array_name in self.all_array_info, (
                f"Array '{array_name}' from CALL '{call_stmt.tostr()}' "
                f"is not present in all_array_info"
            )

            array_name_base = "".join(c for c in array_name if not c.isdigit())
            assert array_name_base in size_arg.tostr(), (
                f"Expected size argument '{size_arg.tostr()}' to contain "
                f"array name '{array_name_base}'"
            )

            section_subscript_list = next(
                (
                    child
                    for child in array_arg.children
                    if isinstance(child, F23.Section_Subscript_List)
                ),
                None,
            )

            assert section_subscript_list is not None, (
                f"Could not find F23.Section_Subscript_List in "
                f"array argument '{array_arg}'"
            )

            size_arg_name = size_arg.tostr()
            match = re.fullmatch(r"isize(\d+)of.+", size_arg_name, re.IGNORECASE)

            assert match is not None, (
                f"Expected size argument in the form 'isizeNofX', got '{size_arg_name}'"
            )

            idim = int(match.group(1)) - 1

            assert idim < len(section_subscript_list.children), (
                f"Dimension {idim + 1} from '{size_arg_name}' is outside "
                f"the array section '{array_arg.tostr()}'"
            )

            dim = section_subscript_list.children[idim]

            assert isinstance(dim, F23.Subscript_Triplet), (
                f"Expected dimension {idim + 1} of array '{array_name}' "
                f"to be a subscript triplet ':', got '{dim.tostr()}'"
            )

            assert all(child is None for child in dim.children), (
                f"Expected ':' on dimension {idim + 1} of array '{array_name}', "
                f"got '{dim.tostr()}'"
            )

            array_info = self.all_array_info[array_name][idim]
            lb = array_info["dim_str"]
            ub = array_info["dim_end"]

            if lb == "1":
                dimension_size = ub
            else:
                dimension_size = f"({ub}) - ({lb}) + 1"

            new_call_stmt = F23.Call_Stmt(
                f"CALL {call_name.tostr()}({array_arg.tostr()}, {dimension_size})"
            )
            return new_call_stmt

        elif isinstance(array_arg, F23.Name):
            # Handle case where array argument is just a Name
            array_name = array_arg.tostr()

            assert array_name in self.all_array_info, (
                f"Array '{array_name}' from CALL '{call_stmt.tostr()}' "
                f"is not present in all_array_info"
            )

            array_name_base = "".join(c for c in array_name if not c.isdigit())
            assert array_name_base in size_arg.tostr(), (
                f"Expected size argument '{size_arg.tostr()}' to contain "
                f"array name '{array_name_base}'"
            )

            size_arg_name = size_arg.tostr()
            match = re.fullmatch(r"isize(\d+)of.+", size_arg_name, re.IGNORECASE)

            assert match is not None, (
                f"Expected size argument in the form 'isizeNofX', got '{size_arg_name}'"
            )

            idim = int(match.group(1)) - 1
            array_info = self.all_array_info[array_name]

            assert idim < len(array_info), (
                f"Dimension {idim + 1} from '{size_arg_name}' is outside "
                f"the array '{array_name}' which has {len(array_info)} dimensions"
            )

            lb = array_info[idim]["dim_str"]
            ub = array_info[idim]["dim_end"]

            if lb == "1":
                dimension_size = ub
            else:
                dimension_size = f"({ub}) - ({lb}) + 1"

            new_call_stmt = F23.Call_Stmt(
                f"CALL {call_name.tostr()}({array_arg.tostr()}, {dimension_size})"
            )
            return new_call_stmt
        else:
            raise NotImplementedError("Not implemented yet!")

    def collect_all_arrays(
        self, node: Any, arrays_queue: deque, seen: set[str]
    ) -> None:
        """
        Traverse an expression node and collect arrays and names within it.

        This method recursively traverses the AST and collects all array
        references and names that are present in the all_array_info dictionary.

        Parameters
        ----------
        node : Any
            The AST node to traverse.
        arrays_queue : deque
            Queue to collect array nodes.
        seen : set of str
            Set to track already processed array names.

        Raises
        ------
        Exception
            If an error occurs during traversal.
        """
        try:
            if isinstance(node, F23.Part_Ref):
                part_refs = walk(node.children, F23.Part_Ref)
                arrays_queue.append(node)
                if part_refs:
                    for part_ref in part_refs:
                        arrays_queue.append(part_ref)
            elif isinstance(node, F23.Name):
                name_str = node.tostr()
                if name_str in self.all_array_info.keys() and name_str not in seen:
                    arrays_queue.append(node)
                    seen.add(name_str)
            elif hasattr(node, "children"):
                for child in node.children:
                    self.collect_all_arrays(child, arrays_queue, seen)
        except Exception as e:
            raise Exception(f"Error in collect_all_arrays: {e}")

    def process_array_shape(
        self, arrays_queue: deque, declared_array_shape_info: list[dict]
    ) -> list[str]:
        """
        Process array shape information from a queue of array nodes.

        This method extracts dimension information from arrays found in the
        queue and replaces any 'isize' dimensions with actual values from
        the all_array_info dictionary.

        Parameters
        ----------
        arrays_queue : deque
            Queue containing array nodes to process.
        declared_array_shape_info : list of dict
            List of declared shape information dictionaries.

        Returns
        -------
        list of str
            List of corrected dimension strings.

        Raises
        ------
        AssertionError
            If array information is missing from all_array_info.
        """
        found_dim_info = []
        array_shape_correctted = []

        while arrays_queue:
            array_node = arrays_queue.popleft()

            if ":" in array_node.tostr() and isinstance(array_node, F23.Part_Ref):
                children = getattr(array_node, "children", ())
                array_name = next(
                    (child for child in children if isinstance(child, F23.Name)), None
                )
                assert array_name.tostr() in self.all_array_info, (
                    f"Array '{array_name.tostr()}' not present in all_array_info."
                )
                array_info = self.all_array_info[array_name.tostr()]

                section_subscript_list = next(
                    (
                        child
                        for child in children
                        if isinstance(child, F23.Section_Subscript_List)
                    ),
                    None,
                )
                self.logger.info(
                    f"Processing array '{array_name.tostr()}' with section "
                    f"subscripts: '{section_subscript_list.tostr()}'"
                )

                for idim, dim in enumerate(section_subscript_list.children):
                    if ":" in dim.tostr():
                        lb = array_info[idim]["dim_str"]
                        ub = array_info[idim]["dim_end"]
                        start_end = [part.strip() for part in dim.tostr().split(":")]
                        start = start_end[0]
                        end = start_end[1]

                        if start == "" and end == "":
                            # :
                            found_dim_info.append({"dim_str": lb, "dim_end": ub})
                        elif start == "" and end != "":
                            # :n
                            found_dim_info.append({"dim_str": lb, "dim_end": end})
                        elif start != "" and end == "":
                            # n:
                            found_dim_info.append({"dim_str": start, "dim_end": ub})
                        else:
                            # n:m
                            found_dim_info.append({"dim_str": start, "dim_end": end})

            elif ":" not in array_node.tostr() and isinstance(array_node, F23.Name):
                array_name = array_node.tostr()
                assert array_name in self.all_array_info, (
                    f"Array '{array_name}' not present in all_array_info."
                )
                array_info = self.all_array_info[array_name]

                for idim in range(len(array_info)):
                    lb = array_info[idim]["dim_str"]
                    ub = array_info[idim]["dim_end"]
                    found_dim_info.append({"dim_str": lb, "dim_end": ub})

            if found_dim_info:
                self.logger.info(
                    f"Comparing {len(found_dim_info)} found dimensions vs "
                    f"{len(declared_array_shape_info)} declared dimensions."
                )

                if len(found_dim_info) == len(declared_array_shape_info):
                    for idim, found_dim in enumerate(found_dim_info):
                        declared_dim = declared_array_shape_info[idim]
                        self.logger.info(
                            f"For dimension {idim}: Found {found_dim} vs "
                            f"declared {declared_dim}"
                        )
                        for key in declared_dim:
                            if "isize" in declared_dim[key]:
                                old_value = declared_dim[key]
                                declared_dim[key] = found_dim[key]
                                self.logger.info(
                                    f"For dimension {idim}: the '{key}' value "
                                    f"'{old_value}' is replaced with "
                                    f"'{declared_dim[key]}'"
                                )
                        array_shape_correctted.append(
                            f"{declared_dim['dim_str']}:{declared_dim['dim_end']}"
                        )
                    return array_shape_correctted

        return array_shape_correctted

    def clean_tapenade_statements(self, parse_tree: Any) -> None:
        """
        Clean Tapenade-generated external statements and replace 'isize'
        dimensions.

        This method removes unnecessary external statements, processes
        assignments to undefined arrays, and replaces 'isize' arguments in
        CALL statements with actual dimension sizes.

        Parameters
        ----------
        parse_tree : Any
            The Fortran AST to clean.

        Raises
        ------
        AssertionError
            If the AST structure is not as expected.
        NotImplementedError
            If assignment to Part_Ref is encountered.
        """
        try:
            allowed_names = {name.lower() for name in self.allowed_external_subroutines}
            imported_external_names = set()

            # Collect imported external names
            for use_stmt in walk(parse_tree, F23.Use_Stmt):
                children = getattr(use_stmt, "children", ())
                only_list = next(
                    (child for child in children if isinstance(child, F23.Only_List)),
                    None,
                )

                if only_list is not None:
                    # USE x, ONLY: y, z, ...
                    for item in getattr(only_list, "children", ()):
                        if not isinstance(item, F23.Name):
                            raise AssertionError(f"Expected F23.Name, got {type(item)}")
                        name = item.tostr()
                        if name in allowed_names:
                            imported_external_names.add(name)
                else:
                    # USE x without ONLY, assume all names are imported
                    module_name = next(
                        (child for child in children if isinstance(child, F23.Name)),
                        None,
                    )
                    if module_name is not None:
                        name = module_name.tostr()
                        if name in allowed_names:
                            imported_external_names.add(name)

            # If no allowed external procedures were imported, no cleaning needed
            if not imported_external_names:
                self.logger.info(
                    "No allowed external procedures imported through USE ... ONLY found"
                )

            self.check_tapenade_isize(parse_tree)

            self.logger.info(
                "Allowed imported external procedures found: "
                f"{sorted(imported_external_names)}"
            )

            def clean_use_and_external_statements(block: Any) -> None:
                """
                First pass: Remove unnecessary USE and EXTERNAL statements.

                Parameters
                ----------
                block : Any
                    The AST block to clean.
                """
                if not hasattr(block, "content"):
                    return

                i = 0
                while i < len(block.content):
                    child = block.content[i]
                    should_remove = False

                    # Check if this is a USE statement we should remove
                    if isinstance(child, F23.Use_Stmt):
                        children = getattr(child, "children", ())
                        only_list = next(
                            (
                                node
                                for node in children
                                if isinstance(node, F23.Only_List)
                            ),
                            None,
                        )

                        # USE x
                        if only_list is None:
                            module_name = next(
                                (
                                    node
                                    for node in children
                                    if isinstance(node, F23.Name)
                                ),
                                None,
                            )
                            if (
                                module_name is not None
                                and module_name.tostr() not in imported_external_names
                            ):
                                should_remove = True

                    # Check if this is an EXTERNAL statement we should remove
                    if isinstance(child, F23.External_Stmt):
                        external_names = set()
                        for external_name in getattr(child, "children", ()):
                            if isinstance(external_name, F23.External_Name_List):
                                for name in getattr(external_name, "children", ()):
                                    if not isinstance(name, F23.Name):
                                        raise AssertionError(
                                            f"Expected F23.Name, got {type(name)}"
                                        )
                                    external_names.add(name.tostr())

                        # Remove if the external statement references imported
                        # procedures
                        if external_names & imported_external_names:
                            should_remove = True

                    # Remove the statement if flagged
                    if should_remove:
                        self.logger.info(
                            f"Removing Tapenade-generated statement: {child.tostr()}"
                        )
                        block.content.pop(i)
                        continue

                    # Recursively clean child blocks
                    clean_use_and_external_statements(child)
                    i += 1

            def process_assignment_statements(block: Any) -> None:
                """
                Second pass: Process assignment statements with undefined arrays.

                Parameters
                ----------
                block : Any
                    The AST block to process.
                """
                if not hasattr(block, "content"):
                    return

                i = 0
                while i < len(block.content):
                    child = block.content[i]

                    # Process assignment statements with undefined arrays
                    if isinstance(child, F23.Assignment_Stmt):
                        lhs_expr = child.items[0]
                        rhs_expr = child.items[-1]

                        if (
                            isinstance(lhs_expr, F23.Name)
                            and lhs_expr.tostr() in self.array_shape_not_defined
                        ):
                            array_name = lhs_expr.tostr()
                            self.logger.info(
                                "Found assignment to undefined array "
                                f"'{array_name}' in statement: '{child.tostr()}'"
                            )

                            # Traverse RHS to find arrays with known shapes
                            arrays_queue = deque()
                            seen = set()
                            self.collect_all_arrays(rhs_expr, arrays_queue, seen)
                            array_shape_correctted = self.process_array_shape(
                                arrays_queue, self.array_shape_not_defined[array_name]
                            )

                            if array_shape_correctted:
                                dimensions = ", ".join(
                                    [name for name in array_shape_correctted]
                                )
                                self.logger.info(
                                    f"For the array '{array_name}' the identified "
                                    f"dimension is {dimensions}."
                                )

                                for old_decl in self.array_declr_not_defined[
                                    array_name
                                ]:
                                    mapped = self.processor.map_declaration(
                                        old_decl,
                                        explicit_dec=None,
                                        dimensions=dimensions,
                                    )
                                    self.logger.info(
                                        f"The original declaration {old_decl} is "
                                        f"mapped to {mapped}"
                                    )
                                    parent = old_decl.parent
                                    for ich in range(len(parent.content)):
                                        node = parent.content[ich]
                                        if node == old_decl:
                                            parent.content[ich] = mapped

                                # Add the array information to all_array_info
                                self.all_array_info[array_name] = [
                                    {
                                        "dim_str": dim.split(":")[0].strip(),
                                        "dim_end": dim.split(":")[1].strip(),
                                    }
                                    for dim in array_shape_correctted
                                ]
                                self.logger.info(
                                    f"Added array '{array_name}' to all_array_info with "
                                    f"shape: {self.all_array_info[array_name]}"
                                )

                                del self.array_shape_not_defined[array_name]
                            else:
                                raise ValueError(
                                    f"Could not determine shape for array '{array_name}'. "
                                    f"No known arrays found in RHS expression: '{rhs_expr.tostr()}'"
                                )

                        if (
                            isinstance(rhs_expr, F23.Part_Ref)
                            and lhs_expr.tostr() in self.array_shape_not_defined
                        ):
                            raise NotImplementedError("Not implemented yet!")

                    # Recursively process child blocks
                    process_assignment_statements(child)
                    i += 1

            def process_call_statements(block: Any) -> None:
                """
                Third pass: Process CALL statements with 'isize' arguments.

                Parameters
                ----------
                block : Any
                    The AST block to process.
                """
                if not hasattr(block, "content"):
                    return

                i = 0
                while i < len(block.content):
                    child = block.content[i]

                    # Process CALL statements with 'isize' arguments
                    if isinstance(child, F23.Call_Stmt):
                        children = child.children
                        actual_arg_list = children[1]

                        if "isize" in actual_arg_list.tostr():
                            args = actual_arg_list.children
                            assert len(args) == 2, (
                                f"Expected exactly 2 actual arguments, "
                                f"got {len(args)}: {actual_arg_list}"
                            )
                            size_arg = args[1]
                            assert size_arg.tostr() in self.size_in_call_stmt, (
                                f"'{size_arg.tostr()}' not in size_in_call_stmt!"
                            )

                            for statement in self.size_in_call_stmt[size_arg.tostr()]:
                                parent = statement.parent
                                mapped = self.process_call_stmt(statement)
                                assert mapped is not None, (
                                    f"Error in mapping the call statement "
                                    f"{statement.tostr()}"
                                )
                                self.logger.info(
                                    f"The original call stmt {statement} is "
                                    f"mapped to {mapped}"
                                )
                                for ich in range(len(parent.content)):
                                    node = parent.content[ich]
                                    if node == statement:
                                        parent.content[ich] = mapped

                    # Recursively process child blocks
                    process_call_statements(child)
                    i += 1

            # Execute passes in the correct order
            clean_use_and_external_statements(parse_tree)
            process_assignment_statements(parse_tree)
            process_call_statements(parse_tree)

        except Exception:
            self.logger.exception("Failed to clean Tapenade external statements")
            raise

    def generate_adjoint_and_tangent(
        self, file_path: str, module_name: str, subroutine_name: str
    ) -> None:
        """
        Generate tangent-linear and adjoint Fortran code via Tapenade.

        Invokes the external Tapenade tool twice on `file_path`: once
        in tangent mode and once in adjoint/reverse mode, both rooted at
        `module_name` as the differentiation head. Both subprocess calls
        are executed with their working directory set to the directory
        containing `file_path`, so generated output files are written
        alongside the source.

        Parameters
        ----------
        file_path : str
            Path to the Fortran source file to differentiate. Its parent
            directory is used as the working directory for the Tapenade
            subprocess calls.
        module_name : str
            Name of the module/procedure to use as the differentiation
            head. The generated tangent module is named
            ``{module_name}_tgt`` and the generated adjoint module is
            named ``{module_name}_adj``.
        subroutine_name : str
            Name of the subroutine to differentiate.

        Raises
        ------
        subprocess.CalledProcessError
            If either the tangent or adjoint Tapenade invocation exits with
            a non-zero return code.
        FileNotFoundError
            If the Tapenade executable is not found on the system PATH.

        Notes
        -----
        Requires Tapenade to be installed and available on the system
        PATH. The tangent command is run before the adjoint command;
        if the tangent run fails, the adjoint run is not attempted.
        """
        # Resolve file path and working directory
        abs_file_path = os.path.abspath(file_path)
        dir_name = os.path.dirname(abs_file_path) or "."

        # Tapenade commands
        tangent_cmd = [
            "tapenade",
            "-d",
            abs_file_path,
            "-tangent",
            "-head",
            subroutine_name,
            "-tgtmodulename",
            f"{module_name}_tgt",
        ]

        adjoint_cmd = [
            "tapenade",
            "-b",
            abs_file_path,
            "-head",
            subroutine_name,
            "-adjmodulename",
            f"{module_name}_adj",
        ]

        # Run tangent generation
        self.logger.info(f"Generating tangent code for {subroutine_name}")
        subprocess.run(tangent_cmd, cwd=dir_name, check=True)

        # Clean generated tangent file
        tangent_file = os.path.join(dir_name, f"{module_name}_d.f90")
        if os.path.isfile(tangent_file):
            self.logger.info(f"Cleaning Tapenade tangent file: {tangent_file}")
            tangent_tree = self.processor.parse_fortran_file(tangent_file)
            self.clean_tapenade_statements(tangent_tree)
            self.processor.write_fortran_code_to_file(tangent_tree, tangent_file)
        else:
            self.logger.warning(f"Tangent file not found: {tangent_file}")

        # Run adjoint generation
        self.logger.info(f"Generating adjoint code for {subroutine_name}")
        subprocess.run(adjoint_cmd, cwd=dir_name, check=True)

        # Clean generated adjoint file
        adjoint_file = os.path.join(dir_name, f"{module_name}_b.f90")
        if os.path.isfile(adjoint_file):
            self.logger.info(f"Cleaning Tapenade adjoint file: {adjoint_file}")
            adjoint_tree = self.processor.parse_fortran_file(adjoint_file)
            self.clean_tapenade_statements(adjoint_tree)
            self.processor.write_fortran_code_to_file(adjoint_tree, adjoint_file)
        else:
            self.logger.warning(f"Adjoint file not found: {adjoint_file}")

        self.logger.info(
            f"Successfully generated and cleaned tangent and adjoint code for "
            f"{subroutine_name}"
        )
