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
        self.array_declr_not_defined = {}
        self.size_in_call_stmt = defaultdict(list)

        self.fortran_math_functions_no_dim = [
            "ABS",
            "SQRT",
            "EXP",
            "LOG",
            "SIN",
            "COS",
            "TAN",
            "ASIN",
            "ACOS",
            "ATAN",
            "MOD",
            "SIGN",
            "MAX",
            "MIN",
            "FLOOR",
            "CEILING",
            "NINT",
            "RAND",
        ]

        # Module-level array info for arrays identified at module level
        self.all_array_info_module_level = {}

        # Subroutine-level array info
        self.all_array_info_subroutine_level = defaultdict(dict)

        # Track current subroutine being processed
        self.current_subroutine = None

    def extract_module_level_arrays(self, parse_tree: Any) -> None:
        """
        Extract module-level array declarations from the specification part.

        This method walks through the module specification part and collects
        array declarations that are at the module level (not inside subroutines).

        Parameters
        ----------
        parse_tree : Any
            The Fortran AST to process.
        """
        try:
            self.logger.info("Extracting module-level array declarations")

            # Get the specification part of the module
            spec_parts = walk(parse_tree, F23.Specification_Part)
            if not spec_parts:
                self.logger.info("No specification part found in module")
                return

            spec_part = spec_parts[0]

            # Walk through all type declaration statements in the specification part
            for decl in walk(spec_part, F23.Type_Declaration_Stmt):
                explicit_shape_spec = walk(decl, F23.Explicit_Shape_Spec)
                entity_decls = walk(decl, F23.Entity_Decl)

                if not entity_decls:
                    continue

                assert len(entity_decls) == 1, (
                    f"Expected exactly 1 Entity_Decl in Type_Declaration_Stmt, "
                    f"but got {len(entity_decls)} in {decl.tostr()}"
                )

                array_name = entity_decls[0].tostr()

                # Check if this is an ALLOCATABLE array
                attr_spec = walk(decl, F23.Attr_Spec)
                is_allocatable = F23.Attr_Spec("ALLOCATABLE") in attr_spec

                # Handle ALLOCATABLE arrays with assumed shape (:, :)
                if is_allocatable and not explicit_shape_spec:
                    # Try to find the base array name
                    base_array_name = None
                    if array_name.endswith("d") or array_name.endswith("b"):
                        base_array_name = array_name[:-1]
                    elif array_name.endswith("sd") or array_name.endswith("sb"):
                        base_array_name = array_name[:-2]
                    elif array_name.endswith("_tgt") or array_name.endswith("_adj"):
                        base_array_name = array_name.replace("_tgt", "").replace(
                            "_adj", ""
                        )

                    # Check if base array exists in all_array_info
                    if (
                        base_array_name is not None
                        and base_array_name in self.all_array_info
                    ):
                        self.logger.info(
                            f"Module-level array '{array_name}' is derived from '{base_array_name}'. "
                            f"Inheriting shape from '{base_array_name}': {self.all_array_info[base_array_name]}"
                        )
                        self.all_array_info_module_level[array_name] = (
                            self.all_array_info[base_array_name].copy()
                        )

                # Handle explicit shape arrays
                if explicit_shape_spec:
                    current_array_dims = []
                    for dim in explicit_shape_spec:
                        start_end = [
                            child.tostr() for child in dim.children if child is not None
                        ]
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
                        raise AssertionError(
                            f"Module-level array '{array_name}' contains 'isize' dimension. "
                            "while it is assumed to be handled at subroutine level."
                        )
                    else:
                        if array_name not in self.all_array_info_module_level:
                            self.logger.info(
                                f"Module-level array '{array_name}' added to module-level info: {current_array_dims}"
                            )
                            self.all_array_info_module_level[array_name] = (
                                current_array_dims
                            )

            self.logger.info(
                f"Extracted {len(self.all_array_info_module_level)} module-level arrays"
            )

        except Exception:
            self.logger.exception("Failed to extract module-level arrays")
            raise

    def get_array_info(self, array_name: str) -> dict | None:
        """
        Search for array information across all levels:
        1. all_array_info (original)
        2. all_array_info_module_level (module level)
        3. all_array_info_subroutine_level (current subroutine)

        Parameters
        ----------
        array_name : str
            The name of the array to search for.

        Returns
        -------
        dict | None
            The array information if found, None otherwise.
        """
        # Check original all_array_info
        if array_name in self.all_array_info:
            return self.all_array_info[array_name]

        # Check module level
        if array_name in self.all_array_info_module_level:
            return self.all_array_info_module_level[array_name]

        # Check current subroutine level
        if (
            self.current_subroutine
            and array_name
            in self.all_array_info_subroutine_level.get(self.current_subroutine, {})
        ):
            return self.all_array_info_subroutine_level[self.current_subroutine][
                array_name
            ]

        return None

    def set_array_info(self, array_name: str, array_info: dict) -> None:
        """
        Set array information at the appropriate level.

        Parameters
        ----------
        array_name : str
            The name of the array.
        array_info : dict
            The array information to store.
        """
        if self.current_subroutine:
            self.all_array_info_subroutine_level[self.current_subroutine][
                array_name
            ] = array_info
        else:
            self.all_array_info_module_level[array_name] = array_info

    def extract_reduction_chain(self, stmt: Any) -> dict:
        """
        Extract intrinsic names and their associated details from a statement.

        This handles nested reductions by building a chain of reductions and
        collecting all DIM values from innermost to outermost, adjusting them
        based on the current rank after each reduction.

        Parameters
        ----------
        stmt : Any
            The statement to process for intrinsic names.

        Returns
        -------
        dict
            A dictionary where the key is the outermost intrinsic parent string and
            the value contains all reduction information including the chain of DIMs.
        """
        try:
            # Collect all intrinsics with their depth
            intrinsic_nodes = []
            for intrinsic in walk(stmt, F23.Intrinsic_Name):
                intrinsic_str = intrinsic.tostr()
                if intrinsic_str in self.fortran_math_functions_no_dim:
                    continue

                intrinsic_parent = intrinsic.parent
                intrinsic_nodes.append(
                    {
                        "intrinsic": intrinsic,
                        "parent": intrinsic_parent,
                        "str": intrinsic_str,
                        "depth": self._get_node_depth(intrinsic),
                    }
                )

            if not intrinsic_nodes:
                return {}

            # Sort by depth (innermost first)
            intrinsic_nodes.sort(key=lambda x: x["depth"], reverse=True)

            # Build the chain of reductions from innermost to outermost
            reduction_chain = []
            # current_node = None
            outermost_parent = None

            # Track current rank and mapping from current dimensions to original dimensions
            current_rank = None
            # dim_mapping: current dimension index -> original dimension index (1-based)
            dim_mapping = []

            for node_info in intrinsic_nodes:
                intrinsic = node_info["intrinsic"]
                intrinsic_parent = node_info["parent"]
                intrinsic_str = node_info["str"]

                # Get the array argument and DIM
                intrinsic_args = intrinsic_parent.children[1]
                args = intrinsic_args.children

                if len(args) == 0:
                    continue

                # Get the array argument
                array_arg = args[0]

                # Determine the rank of the array argument by finding the actual Part_Ref
                array_rank = None
                root_array = None

                # Find the actual Part_Ref in the expression (it might be wrapped in ABS or other operations)
                part_refs = walk(array_arg, F23.Part_Ref)
                if part_refs:
                    # Take the first Part_Ref (the root array)
                    part_ref = part_refs[0]
                    root_array = self._get_root_array_name(part_ref)
                    for child in part_ref.children:
                        if isinstance(child, F23.Section_Subscript_List):
                            array_rank = len(child.children)
                            break

                # If we didn't find the rank, try from all_array_info using get_array_info
                if array_rank is None and root_array is not None:
                    array_info = self.get_array_info(root_array)
                    if array_info is not None:
                        array_rank = len(array_info)
                        self.logger.info(
                            f"Found rank {array_rank} for array '{root_array}' from array info"
                        )

                # If still not found, try to get from the Name
                if array_rank is None:
                    names = walk(array_arg, F23.Name)
                    for name in names:
                        name_str = name.tostr()
                        array_info = self.get_array_info(name_str)
                        if array_info is not None:
                            root_array = name_str
                            array_rank = len(array_info)
                            self.logger.info(
                                f"Found rank {array_rank} for array '{root_array}' from array info via Name walk"
                            )
                            break

                # If current_rank is not set, use the array's rank
                if current_rank is None and array_rank is not None:
                    current_rank = array_rank
                    dim_mapping = list(range(1, current_rank + 1))
                    self.logger.info(
                        f"Initial rank of array '{root_array}' is {current_rank}"
                    )
                    self.logger.info(f"Initial dimension mapping: {dim_mapping}")

                # Extract DIM value and find the original dimension
                dim_value = None
                original_dim = None
                rank_before = current_rank
                dim_mapping_before = dim_mapping.copy() if dim_mapping else []

                # Check if there's a DIM argument (either with or without DIM= keyword)
                has_dim = False
                if len(args) >= 2:
                    # The last argument might be the DIM
                    last_arg = args[-1]

                    # Case 1: DIM= keyword (Actual_Arg_Spec with items)
                    if isinstance(last_arg, F23.Actual_Arg_Spec):
                        items = getattr(last_arg, "items", [])
                        if len(items) == 2:
                            dim_key, dim_value_node = items
                            if (
                                isinstance(dim_key, F23.Name)
                                and dim_key.tostr().lower() == "dim"
                            ):
                                if isinstance(dim_value_node, F23.Int_Literal_Constant):
                                    dim_value = dim_value_node.tostr()
                                    has_dim = True
                        elif len(items) == 1:
                            # This is the case: SUM(..., 2) without DIM= keyword
                            # The Actual_Arg_Spec has a single item which is the value
                            value_node = items[0]
                            if isinstance(value_node, F23.Int_Literal_Constant):
                                dim_value = value_node.tostr()
                                has_dim = True
                    # Case 2: Direct Int_Literal_Constant (SUM(..., 2))
                    elif isinstance(last_arg, F23.Int_Literal_Constant):
                        dim_value = last_arg.tostr()
                        has_dim = True
                    # Case 3: Name (SUM(..., dim_var)) - could be a variable containing the dimension
                    elif isinstance(last_arg, F23.Name):
                        dim_value = last_arg.tostr()
                        has_dim = True

                if has_dim and dim_value is not None:
                    # Get the original dimension index from the mapping
                    try:
                        dim_idx = (
                            int(dim_value) - 1
                        )  # 0-based index in current dimensions
                        if dim_mapping and 0 <= dim_idx < len(dim_mapping):
                            original_dim = dim_mapping[dim_idx]
                        else:
                            original_dim = int(dim_value)

                        self.logger.info(
                            f"Reduction '{intrinsic_str}' reduces current dimension {dim_value} "
                            f"(original dimension {original_dim}) from array with rank {current_rank}"
                        )

                        # Update current rank and mapping (remove the reduced dimension)
                        if current_rank is not None and current_rank > 0:
                            if dim_mapping and 0 <= dim_idx < len(dim_mapping):
                                # Remove the dimension from mapping
                                removed_dim = dim_mapping.pop(dim_idx)
                                self.logger.info(
                                    f"Removed original dimension {removed_dim} from mapping"
                                )
                            current_rank -= 1
                            self.logger.info(
                                f"After '{intrinsic_str}', current rank is {current_rank}"
                            )
                            self.logger.info(
                                f"Updated dimension mapping: {dim_mapping}"
                            )
                    except (ValueError, TypeError):
                        # If dim_value is not an integer (e.g., a variable name), keep it as is
                        original_dim = dim_value
                        self.logger.info(
                            f"Reduction '{intrinsic_str}' uses dimension variable '{dim_value}'"
                        )
                else:
                    # No DIM argument - reduction across all dimensions
                    dim_value = "ALL"
                    original_dim = "ALL"
                    if current_rank is not None:
                        current_rank = 0
                        dim_mapping = []  # No dimensions left
                        self.logger.info(
                            f"Reducing all dimensions, rank becomes {current_rank}"
                        )

                # Find the root array name if not found yet
                if root_array is None:
                    root_array = self._get_root_array_name(array_arg)

                # Add to reduction chain
                reduction_chain.append(
                    {
                        "intrinsic": intrinsic_str,
                        "dim": dim_value,
                        "original_dim": original_dim,
                        "root_array": root_array,
                        "rank_before": rank_before,
                        "rank_after": current_rank,
                        "dim_mapping_before": dim_mapping_before,
                        "dim_mapping_after": dim_mapping.copy() if dim_mapping else [],
                    }
                )

                # Update current node for next iteration
                # current_node = intrinsic_parent
                # Keep track of the outermost parent (last one in the chain)
                outermost_parent = intrinsic_parent

            # Now build the result - the outermost intrinsic contains all the chain
            if reduction_chain and outermost_parent is not None:
                # The outermost is the last one in the chain
                outermost = reduction_chain[-1]

                # Collect adjusted DIM values from the chain (innermost to outermost)
                all_dims = []
                original_dims = []
                for chain_item in reduction_chain:
                    if chain_item["dim"] is not None:
                        all_dims.append(chain_item["dim"])
                    if chain_item["original_dim"] is not None:
                        original_dims.append(str(chain_item["original_dim"]))
                    elif chain_item["dim"] == "ALL":
                        original_dims.append("ALL")

                # Store the complete information
                result = {
                    outermost_parent.tostr(): {
                        "intrinsic": outermost["intrinsic"],
                        "dim": all_dims,  # Adjusted DIMs from innermost to outermost
                        "original_dims": original_dims,  # Original dimension indices
                        "root_array": outermost["root_array"],
                        "chain": reduction_chain,  # Full chain for debugging
                        "num_reductions": len(reduction_chain),
                        "final_rank": current_rank,
                    }
                }

                self.logger.info(
                    f"Found {len(reduction_chain)} nested reductions on array "
                    f"'{outermost['root_array']}' with DIMs: {all_dims}, "
                    f"original DIMs: {original_dims}, final rank: {current_rank}"
                )

                return result

            return {}

        except Exception as e:
            raise Exception(f"Error in extract_reduction_chain: {e}")

    def _get_node_depth(self, node: Any) -> int:
        """
        Get the depth of a node in the AST.

        Parameters
        ----------
        node : Any
            The node to get depth for.

        Returns
        -------
        int
            The depth of the node.
        """
        depth = 0
        current = node
        while hasattr(current, "parent") and current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def _get_root_array_name(self, node: Any) -> str | None:
        """
        Get the root array name from a node.

        Parameters
        ----------
        node : Any
            The node to get the root array name from.

        Returns
        -------
        str | None
            The root array name if found, otherwise None.
        """
        if isinstance(node, F23.Name):
            return node.tostr()
        elif isinstance(node, F23.Part_Ref):
            for child in node.children:
                if isinstance(child, F23.Name):
                    return child.tostr()
        elif hasattr(node, "children"):
            for child in node.children:
                result = self._get_root_array_name(child)
                if result is not None:
                    return result
        return None

    def check_tapenade_isize(self, tree: Any) -> None:
        """
        Validate array declarations against all_array_info for a given tree
        (either a subroutine or the whole module).

        This method walks through the AST and checks if all array declarations
        that have dimensions starting with "isize" are present in the
        all_array_info dictionary. If an array is not found, a warning is logged.

        Parameters
        ----------
        tree : Any
            The Fortran AST to validate (subroutine or module).

        Raises
        ------
        AssertionError
            If ALLOCATABLE arrays are found in subroutines (they should be at module level).
        """
        try:
            context = (
                f"subroutine '{self.current_subroutine}'"
                if self.current_subroutine
                else "module level"
            )
            self.logger.info(f"Validating array declarations for {context}")

            # Walk through all type declaration statements
            for decl in walk(tree, F23.Type_Declaration_Stmt):
                explicit_shape_spec = walk(decl, F23.Explicit_Shape_Spec)

                # Get entity declarations
                entity_decls = walk(decl, F23.Entity_Decl)
                if not entity_decls:
                    continue

                assert len(entity_decls) == 1, (
                    f"Expected exactly 1 Entity_Decl in Type_Declaration_Stmt, "
                    f"but got {len(entity_decls)} in {decl.tostr()}"
                )

                array_name = entity_decls[0].tostr()

                # Check if this is an ALLOCATABLE array
                attr_spec = walk(decl, F23.Attr_Spec)
                is_allocatable = F23.Attr_Spec("ALLOCATABLE") in attr_spec

                # ALLOCATABLE arrays should only be at module level
                if is_allocatable and self.current_subroutine is not None:
                    raise AssertionError(
                        f"ALLOCATABLE array '{array_name}' found in subroutine '{self.current_subroutine}'. "
                        "ALLOCATABLE arrays should be declared at module level, not inside subroutines."
                    )

                # Skip if no explicit shape spec (scalar or assumed shape)
                if not explicit_shape_spec:
                    continue

                # Extract current array dimensions
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

                # Check if any dimension contains "isize"
                if any("isize" in dim.tostr() for dim in explicit_shape_spec):
                    self.logger.warning(
                        f"Array declaration '{decl.tostr()}' contains 'isize' "
                        f"dimension in {context}. Shape needs to be identified!"
                    )
                    self.array_shape_not_defined[array_name] = current_array_dims
                    assert hasattr(decl.parent, "content"), (
                        f"Expected parent of declaration to have a 'content' "
                        f"attribute, got {type(decl.parent).__name__}"
                    )
                    self.array_declr_not_defined[array_name] = decl
                else:
                    # Check if array already exists in any level
                    existing_info = self.get_array_info(array_name)
                    if existing_info is None:
                        self.logger.info(
                            f"Array '{array_name}' added to info in {context} with shape: {current_array_dims}"
                        )
                        # Store at the appropriate level
                        if self.current_subroutine:
                            self.all_array_info_subroutine_level[
                                self.current_subroutine
                            ][array_name] = current_array_dims
                        else:
                            self.all_array_info_module_level[array_name] = (
                                current_array_dims
                            )
                    else:
                        self.logger.info(
                            f"Array '{array_name}' already exists in array info. Skipping."
                        )

            # Report any arrays with undefined shapes
            if self.array_shape_not_defined:
                self.logger.info(
                    f"The following arrays in {context} have shapes that need to be fully defined:"
                )
                for array_name, shape_info in self.array_shape_not_defined.items():
                    self.logger.info(
                        f" - Array '{array_name}' with shape_info '{shape_info}'"
                    )
            else:
                self.logger.info(
                    f"No arrays with 'isize' dimensions found in {context}"
                )

            # Process call statements with 'isize' arguments
            for call_stmt in walk(tree, F23.Call_Stmt):
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
                    f"The following 'isize' arguments are used in CALL statements "
                    f"in {context} and need to be identified:"
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
                    f"No CALL statements containing 'isize' arguments were found in {context}."
                )

        except Exception:
            self.logger.exception(
                f"Failed to validate array declarations for {context}"
            )
            raise

    def process_call_stmt(self, call_stmt: F23.Call_Stmt) -> F23.Call_Stmt:
        """
        Process CALL statements containing an array section followed by an
        isize* argument or multiplication of isize arguments.

        This method replaces the 'isize*' argument(s) with the actual dimension
        size(s) derived from the all_array_info dictionary.

        Parameters
        ----------
        call_stmt : F23.Call_Stmt
            The CALL statement to process.

        Returns
        -------
        F23.Call_Stmt
            The processed CALL statement with replaced dimension size(s).

        Raises
        ------
        AssertionError
            If the CALL statement structure is not as expected.
        NotImplementedError
            If the array argument is not a Part_Ref or Name.
        """
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

        # Get array name from array argument
        if isinstance(array_arg, F23.Part_Ref):
            array_name = next(
                (
                    child.tostr()
                    for child in array_arg.children
                    if isinstance(child, F23.Name)
                ),
                None,
            )
            # Get section subscript list
            section_subscript_list = next(
                (
                    child
                    for child in array_arg.children
                    if isinstance(child, F23.Section_Subscript_List)
                ),
                None,
            )
            assert section_subscript_list is not None, (
                f"Could not find F23.Section_Subscript_List in array argument '{array_arg}'"
            )
        elif isinstance(array_arg, F23.Name):
            array_name = array_arg.tostr()
            section_subscript_list = None
        else:
            raise NotImplementedError(
                f"Array argument type {type(array_arg).__name__} not supported"
            )

        assert array_name is not None, (
            f"Could not find F23.Name in array argument '{array_arg}'"
        )

        # Check if array exists in any level of array info
        array_info = self.get_array_info(array_name)
        assert array_info is not None, (
            f"Array '{array_name}' from CALL '{call_stmt.tostr()}' "
            f"is not present in any level of array info (all_array_info, module level, or subroutine level)"
        )

        # Validate array name base is in size argument
        array_name_base = "".join(c for c in array_name if not c.isdigit())
        size_arg_str = size_arg.tostr()
        assert array_name_base.lower() in size_arg_str.lower(), (
            f"Expected size argument '{size_arg_str}' to contain "
            f"array name '{array_name_base}'"
        )

        # Process the size argument - could be Name or Add_Operand (multiplication)
        dimension_size = None

        if isinstance(size_arg, F23.Name):
            # Single isize argument
            size_arg_name = size_arg.tostr()
            match = re.fullmatch(r"isize(\d+)of.+", size_arg_name, re.IGNORECASE)

            assert match is not None, (
                f"Expected size argument in the form 'isizeNofX', got '{size_arg_name}'"
            )

            idim = int(match.group(1)) - 1

            if section_subscript_list is not None:
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

            array_info = self.get_array_info(array_name)
            assert array_info is not None, (
                f"Array '{array_name}' not found in any level of array info"
            )
            assert idim < len(array_info), (
                f"Dimension {idim} out of bounds for array '{array_name}' with {len(array_info)} dimensions"
            )
            lb = array_info[idim]["dim_str"]
            ub = array_info[idim]["dim_end"]

            if lb == "1":
                dimension_size = ub
            else:
                dimension_size = f"({ub}) - ({lb}) + 1"

        elif isinstance(size_arg, F23.Add_Operand):
            # Multiplication of isize arguments: ISIZE2OFDrfsoilmoist_s * ISIZE3OFDrfsoilmoist_s
            # or nested: ISIZE1OFDrf... * ISIZE2OFDrf... * ISIZE3OFDrf...
            self.logger.info(
                f"Processing multiplication of size arguments: {size_arg_str}"
            )

            # Collect all operands and operators using a stack
            operands = []
            operators = []
            stack = [size_arg]

            while stack:
                node = stack.pop()
                if isinstance(node, F23.Add_Operand):
                    # Add operator
                    operators.append(node.children[1])
                    # Push right and left operands
                    stack.append(node.children[2])  # right
                    stack.append(node.children[0])  # left
                else:
                    operands.append(node)

            # Reverse to maintain original order
            operands.reverse()
            operators.reverse()

            # Process each operand
            dimension_sizes = []
            for operand in operands:
                if isinstance(operand, F23.Name):
                    size_arg_name = operand.tostr()

                    # Validate array name base is in each operand
                    assert array_name_base.lower() in size_arg_name.lower(), (
                        f"Expected size argument '{size_arg_name}' to contain "
                        f"array name '{array_name_base}'"
                    )

                    match = re.fullmatch(
                        r"isize(\d+)of.+", size_arg_name, re.IGNORECASE
                    )
                    assert match is not None, (
                        f"Expected size argument in the form 'isizeNofX', got '{size_arg_name}'"
                    )

                    idim = int(match.group(1)) - 1

                    if section_subscript_list is not None:
                        assert idim < len(section_subscript_list.children), (
                            f"Dimension {idim + 1} from '{size_arg_name}' is outside "
                            f"the array section '{array_arg.tostr()}'"
                        )

                    array_info = self.get_array_info(array_name)
                    assert array_info is not None, (
                        f"Array '{array_name}' not found in any level of array info"
                    )
                    assert idim < len(array_info), (
                        f"Dimension {idim} out of bounds for array '{array_name}' with {len(array_info)} dimensions"
                    )
                    lb = array_info[idim]["dim_str"]
                    ub = array_info[idim]["dim_end"]

                    if lb == "1":
                        dimension_sizes.append(ub)
                    else:
                        dimension_sizes.append(f"({ub}) - ({lb}) + 1")
                else:
                    raise NotImplementedError(
                        f"Operand type {type(operand).__name__} not supported"
                    )

            # Combine all dimension sizes with operators
            if operators:
                operator = operators[0]
                # Verify all operators are the same
                for op in operators[1:]:
                    assert op == operator, f"Mixed operators not supported: {operators}"

                dimension_size = f" {operator} ".join(dimension_sizes)
            else:
                dimension_size = dimension_sizes[0] if dimension_sizes else ""

        else:
            raise NotImplementedError(
                f"Size argument type {type(size_arg).__name__} not supported"
            )

        # Create the new CALL statement
        return F23.Call_Stmt(
            f"CALL {call_name.tostr()}({array_arg.tostr()}, {dimension_size})"
        )

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
                # Check if this Part_Ref contains a colon (array section)
                has_colon = False
                for child in node.children:
                    if isinstance(child, F23.Section_Subscript_List):
                        for subscript in child.children:
                            if isinstance(subscript, F23.Subscript_Triplet):
                                has_colon = True
                                break
                        if has_colon:
                            break

                if has_colon:
                    arrays_queue.append(node)
            elif isinstance(node, F23.Name):
                name_str = node.tostr()
                if self.get_array_info(name_str) is not None and name_str not in seen:
                    arrays_queue.append(node)
                    seen.add(name_str)
            elif hasattr(node, "children"):
                for child in node.children:
                    self.collect_all_arrays(child, arrays_queue, seen)
        except Exception as e:
            raise Exception(f"Error in collect_all_arrays: {e}")

    def process_array_shape(
        self,
        arrays_queue: deque,
        declared_array_shape_info: list[dict],
        reduction_info: dict | None = None,
        lhs_array_name: str | None = None,
    ) -> list[str]:
        """
        Process array shape information from a queue of array nodes.

        This method extracts dimension information from arrays found in the
        queue and replaces any 'isize' dimensions with actual values from
        the all_array_info dictionary. Handles reduction operations by
        adjusting dimensions based on reduction_info.

        Parameters
        ----------
        arrays_queue : deque
            Queue containing array nodes to process.
        declared_array_shape_info : list of dict
            List of declared shape information dictionaries.
        reduction_info : dict, optional
            Dictionary containing reduction information from extract_reduction_chain.

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
        array_shape_corrected = []
        excluded_dims = set()
        final_rank = None

        # First, check if lhs_array_name has a base array variant already defined
        if lhs_array_name:
            base_array_name = None
            if lhs_array_name.endswith("d") or lhs_array_name.endswith("b"):
                base_array_name = lhs_array_name[:-1]
            elif lhs_array_name.endswith("sd") or lhs_array_name.endswith("sb"):
                base_array_name = lhs_array_name[:-2]
            elif lhs_array_name.endswith("_tgt") or lhs_array_name.endswith("_adj"):
                base_array_name = lhs_array_name.replace("_tgt", "").replace("_adj", "")

            # Check if base array exists in any level of array info
            if base_array_name is not None:
                base_array_info = self.get_array_info(base_array_name)
                if base_array_info is not None:
                    self.logger.info(
                        f"Array '{lhs_array_name}' is derived from '{base_array_name}'. "
                        f"Inheriting shape from '{base_array_name}': {base_array_info}"
                    )
                    # Return the shape from base array
                    array_shape_corrected = [
                        f"{dim['dim_str']}:{dim['dim_end']}" for dim in base_array_info
                    ]
                    return array_shape_corrected

        # Process the arrays_queue
        while arrays_queue:
            array_node = arrays_queue.popleft()

            # Get the array name from the node
            array_name_str = None
            if isinstance(array_node, F23.Part_Ref):
                children = getattr(array_node, "children", ())
                array_name = next(
                    (child for child in children if isinstance(child, F23.Name)), None
                )
                assert array_name is not None, (
                    f"Could not find F23.Name in Part_Ref: {array_node.tostr()}"
                )
                array_name_str = array_name.tostr()
            elif isinstance(array_node, F23.Name):
                array_name_str = array_node.tostr()
            else:
                continue

            # Skip the LHS array to avoid self-reference
            if lhs_array_name and array_name_str == lhs_array_name:
                self.logger.info(
                    f"Skipping LHS array '{array_name_str}' to avoid self-reference"
                )
                continue

            assert array_name_str is not None, (
                f"Could not extract array name from node: {array_node}"
            )
            assert self.get_array_info(array_name_str) is not None, (
                f"Array '{array_name_str}' not present in any level of array info."
            )

            # Check if this array node is in any key of reduction_info
            if reduction_info:
                node_str = array_node.tostr()
                matching_key = None
                for key in reduction_info.keys():
                    if node_str in key:
                        matching_key = key
                        break

                if matching_key:
                    self.logger.info(
                        f"Array node '{node_str}' found in reduction_info key: '{matching_key}'"
                    )

                    # Get the reduction info for this specific array
                    array_reduction_info = reduction_info[matching_key]

                    # Extract chain
                    reduction_chain = array_reduction_info.get("chain", [])
                    final_rank = array_reduction_info.get("final_rank", None)

                    # Get the original dimensions that were removed
                    for item in reduction_chain:
                        before = item.get("dim_mapping_before", [])
                        after = item.get("dim_mapping_after", [])

                        # Find which dimensions were removed in this reduction
                        removed = set(before) - set(after)
                        for dim in removed:
                            # Convert to 0-based index
                            excluded_dims.add(int(dim) - 1)
                            self.logger.info(
                                f"Dimension {dim} (index {int(dim) - 1}) will be excluded "
                                f"from final shape due to reduction '{item['intrinsic']}'"
                            )

                    self.logger.info(
                        f"Excluding dimensions: {excluded_dims}, final rank: {final_rank}"
                    )

            array_info = self.get_array_info(array_name_str)
            assert array_info is not None, (
                f"Array '{array_name_str}' not found in any level of array info "
                f"(all_array_info, module level, or subroutine level)."
            )

            # Process the Part_Ref to get dimensions
            if isinstance(array_node, F23.Part_Ref):
                section_subscript_list = next(
                    (
                        child
                        for child in array_node.children
                        if isinstance(child, F23.Section_Subscript_List)
                    ),
                    None,
                )
                assert section_subscript_list is not None, (
                    f"Could not find F23.Section_Subscript_List in Part_Ref: {array_node.tostr()}"
                )

                self.logger.info(
                    f"Processing array '{array_name_str}' with section "
                    f"subscripts: '{section_subscript_list.tostr()}'"
                )

                for idim, dim in enumerate(section_subscript_list.children):
                    # Check if this dimension has a colon (Subscript_Triplet)
                    if isinstance(dim, F23.Subscript_Triplet):
                        # Get start, end, step from triplet children
                        start, end, step = getattr(dim, "children", (None, None, None))

                        # If start or end is a Part_Ref (indirect access), treat as None
                        if isinstance(start, F23.Part_Ref):
                            start = None
                            self.logger.info(
                                f"Dimension {idim} has indirect start (Part_Ref), treating as full colon"
                            )
                        if isinstance(end, F23.Part_Ref):
                            end = None
                            self.logger.info(
                                f"Dimension {idim} has indirect end (Part_Ref), treating as full colon"
                            )

                        assert idim < len(array_info), (
                            f"Dimension index {idim} is out of bounds for array '{array_name_str}' "
                            f"with {len(array_info)} dimensions"
                        )

                        if start is None and end is None:
                            # Full colon ":" - use all_array_info
                            lb = array_info[idim]["dim_str"]
                            ub = array_info[idim]["dim_end"]
                            found_dim_info.append({"dim_str": lb, "dim_end": ub})
                        elif start is None and end is not None:
                            # ":n" - upper bound specified
                            lb = array_info[idim]["dim_str"]
                            ub = end.tostr()
                            found_dim_info.append({"dim_str": lb, "dim_end": ub})
                        elif start is not None and end is None:
                            # "n:" - lower bound specified
                            lb = start.tostr()
                            ub = array_info[idim]["dim_end"]
                            found_dim_info.append({"dim_str": lb, "dim_end": ub})
                        else:
                            # "n:m" - both bounds specified
                            lb = start.tostr()
                            ub = end.tostr()
                            found_dim_info.append({"dim_str": lb, "dim_end": ub})
                    elif isinstance(dim, F23.Name):
                        # Single name (scalar) - not a colon, skip
                        pass

            elif isinstance(array_node, F23.Name):
                # Simple array name, take all dimensions
                for idim in range(len(array_info)):
                    lb = array_info[idim]["dim_str"]
                    ub = array_info[idim]["dim_end"]
                    found_dim_info.append({"dim_str": lb, "dim_end": ub})

            # Now process the found dimensions with the excluded_dims
            if found_dim_info:
                # Filter out excluded dimensions
                filtered_dim_info = []
                for idx, dim_info in enumerate(found_dim_info):
                    if idx not in excluded_dims:
                        filtered_dim_info.append(dim_info)
                    else:
                        self.logger.info(f"Dimension {idx} is excluded from shape")

                found_dim_info = filtered_dim_info

                self.logger.info(
                    f"Comparing {len(found_dim_info)} found dimensions vs "
                    f"{len(declared_array_shape_info)} declared dimensions."
                )

                if len(found_dim_info) != len(declared_array_shape_info):
                    self.logger.warning(
                        f"Dimension mismatch: found {len(found_dim_info)} dimensions, "
                        f"but declared shape has {len(declared_array_shape_info)} dimensions. "
                        f"Mapping to scalar."
                    )

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
                    array_shape_corrected.append(
                        f"{declared_dim['dim_str']}:{declared_dim['dim_end']}"
                    )

                return array_shape_corrected

        return array_shape_corrected

    def clean_tapenade_statements(self, parse_tree: Any) -> None:
        """
        Clean Tapenade-generated external statements and replace 'isize'
        dimensions. Processes subroutines individually.

        Parameters
        ----------
        parse_tree : Any
            The Fortran AST to clean.

        Raises
        ------
        AssertionError
            If module-level array contains 'isize' dimension.
            If ALLOCATABLE arrays are found in subroutines.
        """
        try:
            # First, extract module-level arrays
            self.extract_module_level_arrays(parse_tree)

            # Get all subroutine subprograms
            subroutines = walk(parse_tree, F23.Subroutine_Subprogram)

            if subroutines:
                for sub in subroutines:
                    subroutine_stmt = walk(sub, F23.Subroutine_Stmt)[0]
                    subroutine_name = None
                    for child in subroutine_stmt.children:
                        if isinstance(child, F23.Name):
                            subroutine_name = child.tostr()
                            break

                    if subroutine_name is None:
                        continue

                    self.logger.info(f"Processing subroutine: {subroutine_name}")

                    # Set current subroutine context
                    self.current_subroutine = subroutine_name

                    # Clear subroutine-specific data
                    self.array_shape_not_defined.clear()
                    self.array_declr_not_defined.clear()
                    self.size_in_call_stmt.clear()

                    # Process this subroutine's declarations using check_tapenade_isize
                    self.check_tapenade_isize(sub)

                    # Clean statements for this subroutine
                    self._clean_subroutine_statements(sub)

                    # Reset context
                    self.current_subroutine = None
            else:
                # No subroutines - process at module level
                self.logger.info(
                    "No subroutines found in module. Processing at module level."
                )
                self.current_subroutine = None
                self.check_tapenade_isize(parse_tree)
                # Use the same cleaning logic on the module tree
                self._clean_subroutine_statements(parse_tree)

        except Exception:
            self.logger.exception("Failed to clean Tapenade external statements")
            raise

    def _clean_subroutine_statements(self, sub: Any) -> None:
        """
        Clean statements within a subroutine.

        Parameters
        ----------
        sub : Any
            The subroutine AST to clean.
        """
        try:
            allowed_names = {name.lower() for name in self.allowed_external_subroutines}
            imported_external_names = set()

            # Collect imported external names within this subroutine
            for use_stmt in walk(sub, F23.Use_Stmt):
                children = getattr(use_stmt, "children", ())
                only_list = next(
                    (child for child in children if isinstance(child, F23.Only_List)),
                    None,
                )
                if only_list is not None:
                    for item in getattr(only_list, "children", ()):
                        if not isinstance(item, F23.Name):
                            raise AssertionError(f"Expected F23.Name, got {type(item)}")
                        name = item.tostr()
                        if name in allowed_names:
                            imported_external_names.add(name)
                else:
                    module_name = next(
                        (child for child in children if isinstance(child, F23.Name)),
                        None,
                    )
                    if module_name is not None:
                        name = module_name.tostr()
                        if name in allowed_names:
                            imported_external_names.add(name)

            self.logger.info(
                f"Subroutine '{self.current_subroutine}': Allowed imported external procedures found: "
                f"{sorted(imported_external_names)}"
            )

            def clean_use_and_external_statements(block: Any) -> None:
                """
                First pass: Remove unnecessary USE and EXTERNAL statements.
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

                        # USE x (without ONLY)
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

                        # Remove if the external statement references imported procedures
                        if external_names & imported_external_names:
                            should_remove = True

                    # Remove the statement if flagged
                    if should_remove:
                        self.logger.info(
                            f"Removing Tapenade-generated statement: '{child.tostr()}' "
                        )
                        block.content.pop(i)
                        continue

                    # Recursively clean child blocks
                    clean_use_and_external_statements(child)
                    i += 1

            def process_assignment_statements(block: Any) -> None:
                """
                Second pass: Process assignment statements with undefined arrays.
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

                        if isinstance(lhs_expr, F23.Part_Ref):
                            array_name = next(
                                (
                                    child.tostr()
                                    for child in lhs_expr.children
                                    if isinstance(child, F23.Name)
                                ),
                                None,
                            )
                        elif isinstance(lhs_expr, F23.Name):
                            array_name = lhs_expr.tostr()
                        else:
                            raise NotImplementedError(
                                f"Unsupported LHS type in assignment: {type(lhs_expr).__name__} in '{child.tostr()}'"
                            )

                        assert array_name is not None, (
                            f"Could not find array name in '{lhs_expr}'"
                        )

                        if array_name in self.array_shape_not_defined:
                            self.logger.info(
                                f"Found assignment to undefined array '{array_name}' "
                                f"in statement: '{child.tostr()}'"
                            )

                            # Extract intrinsic/reduction information from the RHS
                            reduction_info = self.extract_reduction_chain(rhs_expr)

                            # Traverse RHS to find arrays with known shapes
                            arrays_queue = deque()
                            seen = set()
                            self.collect_all_arrays(rhs_expr, arrays_queue, seen)

                            # If no arrays found in RHS (e.g., max5b = 0.0_8), skip and continue
                            if not arrays_queue:
                                self.logger.info(
                                    f"No arrays found in RHS expression '{rhs_expr.tostr()}' "
                                    f"for array '{array_name}'. Skipping assignment."
                                )
                                i += 1
                                continue

                            array_shape_correctted = self.process_array_shape(
                                arrays_queue,
                                self.array_shape_not_defined[array_name],
                                reduction_info,
                                array_name,
                            )

                            if array_shape_correctted:
                                dimensions = ", ".join(array_shape_correctted)
                                self.logger.info(
                                    f"For the array '{array_name}' the identified "
                                    f"dimension is {dimensions}."
                                )

                                old_decl = self.array_declr_not_defined[array_name]
                                mapped = self.processor.map_declaration(
                                    old_decl,
                                    explicit_dec=None,
                                    dimensions=dimensions,
                                )
                                self.logger.info(
                                    f"The original declaration {old_decl} is mapped to {mapped}"
                                )
                                parent = old_decl.parent
                                for ich in range(len(parent.content)):
                                    node = parent.content[ich]
                                    if node == old_decl:
                                        parent.content[ich] = mapped
                                        break

                                # Add the array information using set_array_info
                                array_info = [
                                    {
                                        "dim_str": dim.split(":")[0].strip(),
                                        "dim_end": dim.split(":")[1].strip(),
                                    }
                                    for dim in array_shape_correctted
                                ]
                                self.set_array_info(array_name, array_info)
                                self.logger.info(
                                    f"Added array '{array_name}' to array info with "
                                    f"shape: {array_info}"
                                )

                                del self.array_shape_not_defined[array_name]
                                if array_name in self.array_declr_not_defined:
                                    del self.array_declr_not_defined[array_name]
                            else:
                                # array_shape_corrected is empty - map to scalar
                                self.logger.warning(
                                    f"Array '{array_name}' has no dimensions identified. "
                                    f"Mapping to scalar declaration."
                                )

                                old_decl = self.array_declr_not_defined[array_name]
                                mapped = self.processor.map_declaration(old_decl)
                                self.logger.info(
                                    f"The original declaration {old_decl} is mapped to scalar: {mapped}"
                                )
                                parent = old_decl.parent
                                for ich in range(len(parent.content)):
                                    node = parent.content[ich]
                                    if node == old_decl:
                                        parent.content[ich] = mapped
                                        break

                                del self.array_shape_not_defined[array_name]
                                if array_name in self.array_declr_not_defined:
                                    del self.array_declr_not_defined[array_name]

                        if (
                            isinstance(rhs_expr, F23.Part_Ref)
                            and isinstance(lhs_expr, F23.Name)
                            and lhs_expr.tostr() in self.array_shape_not_defined
                        ):
                            raise NotImplementedError("Not implemented yet!")

                    # Recursively process child blocks
                    process_assignment_statements(child)
                    i += 1

            def process_call_statements(block: Any) -> None:
                """
                Third pass: Process CALL statements with 'isize' arguments.
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
                                    f"Error in mapping the call statement {statement.tostr()}"
                                )
                                self.logger.info(
                                    f"The original call stmt {statement} is mapped to {mapped}"
                                )
                                for ich in range(len(parent.content)):
                                    node = parent.content[ich]
                                    if node == statement:
                                        parent.content[ich] = mapped
                                        break

                    # Recursively process child blocks
                    process_call_statements(child)
                    i += 1

            # Execute passes in the correct order
            clean_use_and_external_statements(sub)
            process_assignment_statements(sub)
            process_call_statements(sub)

        except Exception:
            self.logger.exception(
                f"Failed to clean statements for subroutine '{self.current_subroutine}'"
            )
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
