# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Core AST transformation utilities for FGPT.

This module provides low-level building blocks for manipulating and
rewriting Python ASTs generated from Fortran code.

It includes:
- AST rewriting helpers (ReplaceGlobals, AdjustIndices)
- safe evaluation utilities for symbolic expressions
- instance binding utilities for object-oriented rewriting
- dependency analysis tools for variable tracking

These utilities are used across the FGPT transpilation pipeline,
particularly in the Fortran to Python transformation stages.
"""

import ast
import logging
import operator
import os
from collections import defaultdict, deque
from collections.abc import Generator, Iterable
from typing import Any

import yaml


class ReplaceGlobals(ast.NodeTransformer):
    """
    Replace global variables, attributes, and method calls with their
    corresponding object-qualified references.

    This transformer traverses a Python abstract syntax tree (AST) and
    rewrites unqualified names into attribute accesses based on metadata
    stored in :attr:`cls_info`. It is primarily intended for source-to-source
    transformations where variables, arrays, and methods originally defined
    in a class context have been emitted without their owning instance.

    During traversal, the transformer:

    * Replaces global variable references with instance-qualified
      attributes (e.g. ``temp`` → ``self_obj.temp``).
    * Resolves attributes belonging to composed objects and rewrites them
      as nested attribute accesses.
    * Converts standalone method calls into instance method invocations.
    * Updates references appearing in assignments, expressions,
      comparisons, loops, f-strings, and function arguments.
    * Preserves names recorded in :attr:`_local_scope` to avoid replacing
      locally scoped variables.

    Parameters
    ----------
    cls_info : dict
        Mapping describing available class instances, attributes,
        methods, and composed objects used for name resolution.

    Attributes
    ----------
    cls_info : dict
        Metadata used to resolve variable and method ownership.
    _local_scope : set[str]
        Collection of locally scoped names that must not be rewritten.

    Notes
    -----
    The transformer relies on the metadata structure supplied through
    :attr:`cls_info`. Incorrect or incomplete metadata may result in
    unresolved names or invalid attribute substitutions.
    """

    def __init__(self, cls_info):
        self.cls_info = cls_info
        self._local_scope: set = set()

    def get_attr_node(self, name: str) -> ast.Attribute | None:
        """
        Resolve a variable name to its owning instance attribute.

        Searches the metadata stored in :attr:`cls_info` and constructs an
        :class:`ast.Attribute` node representing the fully qualified attribute
        access corresponding to the supplied name.

        Names recorded in :attr:`_local_scope` are ignored to prevent
        replacement of local variables.

        Parameters
        ----------
        name : str
            Variable name to resolve.

        Returns
        -------
        ast.Attribute | None
            Attribute node referencing the owning instance, or ``None`` if no
            matching attribute is found.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the get_attr_node.
        """
        try:
            if name in self._local_scope:
                return None

            for _, instances in self.cls_info.items():
                for inst_key, value in instances.items():
                    cls_attr = value.get("attributes", {})
                    other_object_instances = value.get("instances", {})

                    if name.isupper():
                        name = name.lower()

                    if name in cls_attr:
                        return ast.Attribute(
                            value=ast.Name(id=inst_key, ctx=ast.Load()),
                            attr=name,
                            ctx=ast.Load(),
                        )
                    elif other_object_instances:
                        for key in list(other_object_instances.keys()):
                            other_object_attributes = other_object_instances[key].get(
                                "attributes"
                            )
                            if name in other_object_attributes:
                                return ast.Attribute(
                                    value=other_object_instances[key]["class_name"],
                                    attr=name,
                                    ctx=ast.Load(),
                                )
            return None
        except Exception as e:
            raise RuntimeError(
                "RuntimeError in get_attr_node of \
                                ReplaceGlobals"
            ) from e

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """
        Visit a name node and replace global references.

        Attempts to resolve the identifier through :meth:`get_attr_node`. If a
        matching attribute is found, the name is replaced with an attribute
        access node.

        Parameters
        ----------
        node : ast.Name
            Name node being visited.

        Returns
        -------
        ast.AST
            Replacement attribute node if resolved; otherwise the original
            name node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the visit_Name.
        """
        try:
            replacement = self.get_attr_node(node.id)
            if replacement:
                return replacement
            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_Name of \
                                ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e

    def visit_Attribute(self, node):
        """
        Visit an attribute access and resolve composed-object references.

        Preserves attributes belonging to the current object while rewriting
        references that belong to composed instances described in
        :attr:`cls_info`.

        Parameters
        ----------
        node : ast.Attribute
            Attribute node being visited.

        Returns
        -------
        ast.AST
            Original attribute node or a rewritten attribute access.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the visit_Attribute.
        """
        try:
            node = self.generic_visit(node)
            # Only handle self.xxx form
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                name = node.attr

                # Check if it's a local attribute
                for _, instances in self.cls_info.items():
                    for _, value in instances.items():
                        local_attrs = value.get("attributes", [])
                        other_instances = value.get("instances", [])

                        if name.isupper():
                            name = name.lower()

                        if name in local_attrs:
                            return node  # keep as self.name

                        for key in other_instances.keys():
                            other_object_attributes = other_instances[key].get(
                                "attributes", {}
                            )
                            if name in other_object_attributes:
                                # Build self.instance.name
                                return ast.Attribute(
                                    value=other_instances[key]["class_name"],
                                    attr=name,
                                    ctx=ast.Load(),
                                )
            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_Attribute of \
                                ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e

    def visit_List(self, node):
        """
        Visit a list literal and transform its elements.

        Recursively visits each element of the list so that nested names,
        attributes, and expressions can be rewritten consistently.

        Parameters
        ----------
        node : ast.List
            List node being visited.

        Returns
        -------
        ast.List
            The transformed list node.
        """
        node.elts = [self.visit(elt) for elt in node.elts]
        return node

    def visit_Assign(self, node):
        """
        Visit an assignment statement and replace global references.

        Processes assignment targets and values, replacing variable references
        with instance-qualified attributes where applicable.

        Subscript targets are handled specially so that array references are
        rewritten while preserving their indices.

        Parameters
        ----------
        node : ast.Assign
            Assignment node being visited.

        Returns
        -------
        ast.Assign
            The transformed assignment node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the visit_Assign.
        """
        try:
            node = self.generic_visit(node)
            for i, target in enumerate(node.targets):
                if isinstance(target, ast.Name):
                    replacement = self.get_attr_node(target.id)
                    if replacement:
                        node.targets[i] = replacement
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name):
                        replacement = self.get_attr_node(target.value.id)
                        if replacement:
                            node.targets[i].value = replacement
            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_Assign of \
                                ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e

    def visit_Call(self, node):
        """
        Visit a function or method call and resolve ownership.

        Converts standalone method invocations into instance-qualified method
        calls based on metadata contained in :attr:`cls_info`.

        Calls belonging to composed objects are also resolved. Logging calls
        containing f-strings are traversed so that embedded expressions can be
        rewritten through :meth:`visit_FormattedValue`.

        Parameters
        ----------
        node : ast.Call
            Call node being visited.

        Returns
        -------
        ast.Call
            The transformed call node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the visit_Call.
        """
        try:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                method_name = node.func.id
                # First the method name and then check the args
                for _, instances in self.cls_info.items():
                    for inst_key, value in instances.items():
                        methods = value.get("methods", [])
                        other_object_instances = value.get("instances", [])
                        if method_name in methods:
                            node.func = ast.Attribute(
                                value=ast.Name(id=inst_key, ctx=ast.Load()),
                                attr=method_name,
                                ctx=ast.Load(),
                            )
                            new_args = []
                            for arg in node.args:
                                new_arg = self.visit(arg)
                                new_args.append(new_arg)
                            node.args = new_args

                            return node

                        elif other_object_instances:
                            # This is to handle the cases on which we have other classes
                            # intialized inside the class itself and require to be attributed
                            # to the intialized class
                            for key in list(other_object_instances.keys()):
                                other_object_attributes = other_object_instances[
                                    key
                                ].get("methods")
                                if method_name in other_object_attributes:
                                    node.func = ast.Attribute(
                                        value=other_object_instances[key]["class_name"],
                                        attr=method_name,
                                        ctx=ast.Load(),
                                    )
                                    new_args = []
                                    for arg in node.args:
                                        new_arg = self.visit(arg)
                                        new_args.append(new_arg)
                                    node.args = new_args
                                return node

            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logging"
            ):
                for arg in node.args:
                    if isinstance(arg, ast.JoinedStr):
                        new_values = []
                        for value in arg.values:
                            # This will now call visit_FormattedValue
                            new_value = self.visit(value)
                            new_values.append(new_value)
                        arg.values = new_values

                return node

            return self.generic_visit(node)

        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_Call of \
                                ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e

    def visit_Expr(self, node):
        """
        Visit an expression statement.

        Ensures the contained expression is recursively transformed before
        returning the updated node.

        Parameters
        ----------
        node : ast.Expr
            Expression statement node.

        Returns
        -------
        ast.Expr
            The transformed expression node.
        """
        node.value = self.visit(node.value)
        return node

    def visit_FormattedValue(self, node):
        """
        Visit an f-string formatted value.

        Recursively transforms the embedded expression so that references
        inside formatted strings remain consistent with the rewritten AST.

        Parameters
        ----------
        node : ast.FormattedValue
            Formatted value node.

        Returns
        -------
        ast.FormattedValue
            The transformed formatted value node.
        """
        node.value = self.visit(node.value)
        return node

    def visit_For(self, node):
        """
        Visit a ``for`` loop and rewrite iterator arguments.

        Examines iterator arguments supplied to calls such as ``range`` and
        replaces references to global attributes with their corresponding
        instance-qualified forms.

        Parameters
        ----------
        node : ast.For
            Loop node being visited.

        Returns
        -------
        ast.For
            The transformed loop node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the visit_For.
        """
        try:
            node = self.generic_visit(node)
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    # Retrieve the args and see if one
                    # them is dependant on the global attribute
                    for i, arg in enumerate(node.iter.args):
                        if isinstance(arg, ast.Name):
                            replacement = self.get_attr_node(arg.id)
                            if replacement:
                                node.iter.args[i] = replacement
            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_For of \
                                ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e

    def visit_If(self, node):
        """
        Visit a conditional statement and rewrite referenced globals.

        Handles simple boolean conditions, comparison expressions, and logical
        operations containing nested comparisons. Global references are
        resolved through :meth:`get_attr_node`, while comparison expressions
        are delegated to :meth:`_replace_compare`.

        Parameters
        ----------
        node : ast.If
            Conditional node being visited.

        Returns
        -------
        ast.If
            The transformed conditional node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the visit_If.
        """
        try:
            node = self.generic_visit(node)
            # These primarily reference the cases of logical
            if isinstance(node.test, ast.Name):
                replacement = self.get_attr_node(node.test.id)
                if replacement:
                    node.test = replacement

            # Handle comparisons: a < i
            elif isinstance(node.test, ast.Compare):
                self._replace_compare(node.test)

            # Handle boolean operations: a < i and i > b
            elif isinstance(node.test, ast.BoolOp):
                for i, value in enumerate(node.test.values):
                    if isinstance(value, ast.Compare):
                        self._replace_compare(value)
                    elif isinstance(value, ast.Name):
                        replacement = self.get_attr_node(value.id)
                        if replacement:
                            node.test.values[i] = replacement
                    elif isinstance(value, ast.Subscript):
                        if isinstance(value.value, ast.Name):
                            replacement = self.get_attr_node(value.value.id)
                            if replacement:
                                value.value = replacement

            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_If of \
                                ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e

    def _replace_compare(self, compare_node: ast.Compare) -> None:
        """
        Replace global references appearing in comparison expressions.

        Processes both the left-hand side and comparator expressions,
        rewriting names, subscripts, and nested binary operations to their
        instance-qualified equivalents.

        Parameters
        ----------
        compare_node : ast.Compare
            Comparison node to transform.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the _replace_compare.
        """
        try:
            if isinstance(compare_node.left, ast.Name):
                replacement = self.get_attr_node(compare_node.left.id)
                if replacement:
                    compare_node.left = replacement

            elif isinstance(compare_node.left, ast.Subscript):
                if isinstance(compare_node.left.value, ast.Name):
                    replacement = self.get_attr_node(compare_node.left.value.id)
                    if replacement:
                        compare_node.left.value = replacement

            # Comparators (right side)
            for i, comp in enumerate(compare_node.comparators):
                if isinstance(comp, ast.Name):
                    replacement = self.get_attr_node(comp.id)
                    if replacement:
                        compare_node.comparators[i] = replacement

                elif isinstance(comp, ast.Subscript):
                    if isinstance(comp.value, ast.Name):
                        replacement = self.get_attr_node(comp.value.id)
                        if replacement:
                            comp.value = replacement

                elif isinstance(comp, ast.BinOp):
                    compare_node.comparators[i] = self.visit_BinOp(comp)
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in _replace_compare of \
                                ReplaceGlobals: {ast.dump(compare_node, indent=4)}"
            ) from e

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        """
        Visit a binary operation and resolve global references.

        Traverses both operands of the binary expression and replaces
        unqualified variable references with their corresponding instance
        attributes.

        Nested binary expressions are processed recursively to ensure
        consistent rewriting throughout the expression tree.

        Parameters
        ----------
        node : ast.BinOp
            Binary operation node being visited.

        Returns
        -------
        ast.BinOp
            The transformed binary operation node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the BinOp.
        """
        try:
            self.generic_visit(node)

            # Left side
            if isinstance(node.left, ast.Subscript):
                if hasattr(node.left, "value") and isinstance(
                    node.left.value, ast.Name
                ):
                    replacement = self.get_attr_node(node.left.value.id)
                    if replacement:
                        node.left.value = replacement

            elif isinstance(node.left, ast.Name):
                replacement = self.get_attr_node(node.left.id)
                if replacement:
                    node.left = replacement

            # Right side
            if isinstance(node.right, ast.Subscript):
                if hasattr(node.right, "value") and isinstance(
                    node.right.value, ast.Name
                ):
                    replacement = self.get_attr_node(node.right.value.id)
                    if replacement:
                        node.right.value = replacement

            elif isinstance(node.right, ast.Name):
                replacement = self.get_attr_node(node.right.id)
                if replacement:
                    node.left = replacement

            elif isinstance(node.right, ast.BinOp):
                self.visit_BinOp(node.right)

            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_BinOp \
                               of ReplaceGlobals: {ast.dump(node, indent=4)}"
            ) from e


# SemanticIndexAdjuster
class AdjustIndices(ast.NodeTransformer):
    """
    Adjust array indices and loop-bound expressions to preserve Fortran
    indexing semantics after conversion to Python.

    This transformer traverses a Python abstract syntax tree (AST) and
    rewrites array accesses, loop bounds, comparisons, assignments, and
    index-related expressions so that code originating from Fortran
    retains its original behavior despite Python's zero-based indexing.

    The transformation relies on metadata stored in :attr:`array_info`
    to determine the declared lower bounds of arrays. Conventional
    Fortran loop variables recorded in :attr:`CONV_VARS` are treated
    specially because they are typically already adapted to Python's
    iteration model. Arrays with non-default lower bounds are adjusted
    using offsets computed from their declared dimensions.

    During traversal, the transformer:

    * Converts Fortran-style array indices into their Python equivalents.
    * Applies lower-bound offsets for arrays whose indexing does not start
      at ``1``.
    * Tracks variables requiring subsequent index correction through
      :attr:`adjusted_vars`.
    * Rewrites loop bounds and conditional comparisons involving converted
      indices.
    * Adjusts results of operations such as ``argmin`` and ``argmax`` so
      that returned indices remain consistent with the original Fortran
      indexing convention.
    * Preserves explicitly excluded variables recorded in
      :attr:`exclude_index`.

    Parameters
    ----------
    conv_vars : Collection[str]
        Conventional loop variables originating from Fortran constructs
        (e.g. ``ji``, ``jj``, ``jk``, ``jst``) that should not be treated
        as ordinary indices.
    array_info : dict
        Mapping containing array dimension metadata, including lower-bound
        information used to compute index offsets.
    cls_attributes : dict
        Metadata describing class attributes and attributes belonging to
        composed objects that may be required to resolve dimension
        variables.
    **kwargs
        Additional configuration options.

    Other Parameters
    ----------------
    adjusted_vars : set[str], optional
        Variables already known to represent adjusted indices and requiring
        special handling during subsequent transformations.
    exclude_index : Collection[str], optional
        Variable names that should never be automatically converted to
        Python indexing.

    Attributes
    ----------
    CONV_VARS : Collection[str]
        Conventional loop variables exempt from standard index correction.
    array_info : dict
        Case-insensitive mapping of array dimension metadata.
    cls_attributes : dict
        Attributes belonging to the current class.
    instances_global_attributes : dict
        Attributes belonging to composed object instances.
    adjusted_vars : set[str]
        Variables whose values have already been converted or derived from
        converted indices.
    exclude_index : Collection[str] | None
        Variables excluded from automatic index adjustment.

    Notes
    -----
    Fortran arrays may use arbitrary lower bounds (e.g. ``1:n``,
    ``0:n``, or ``-3:n``), whereas Python sequences always use
    zero-based indexing. This transformer compensates for those
    differences so that translated code accesses the same logical
    elements as the original Fortran source.

    The implementation is based on :class:`ast.NodeTransformer`,
    allowing nodes to be replaced in-place during traversal while
    preserving the overall structure of the AST.
    """

    def __init__(self, conv_vars, array_info: dict, cls_attributes: dict, **kwargs):
        # This corresponds to the conventional loop variables such as ji,jst,jl etc...
        self.CONV_VARS = conv_vars

        self.array_info = {k.casefold(): v for k, v in array_info.items()}
        # Attributes of the class (arrays, scalars)
        self.cls_attributes = cls_attributes.get("attributes", {})
        # Attributes of probably other object class if present inside the parent class
        self.instances_global_attributes = cls_attributes.get("instances", {})

        self.adjusted_vars = kwargs.get("adjusted_vars", set())
        self.exclude_index = kwargs.get("exclude_index")

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        """
        Visit and adjust array subscript expressions for Fortran-to-Python indexing.

        Resolves the target array's lower-bound information from
        :attr:`array_info` and rewrites index expressions so that Python's
        zero-based indexing preserves the semantics of the original Fortran
        source.

        For arrays with non-default lower bounds, applies an offset through
        :meth:`_apply_offset_if_convvar`. For standard Fortran arrays
        (lower bound of ``1``), adjusts indices using :meth:`_adjust_index`.

        If the array metadata cannot be resolved, a fallback transformation is
        applied to all indices under the assumption that they originate from
        Fortran-style indexing.

        Parameters
        ----------
        node : ast.Subscript
            The subscript expression being visited.

        Returns
        -------
        ast.Subscript
            The transformed subscript node.

        Raises
        ------
        KeyError
            If dimension metadata cannot be resolved for an indexed array.
        RuntimeError
            If an unexpected error occurs while processing the Subscript.
        """
        try:
            self.generic_visit(node)

            arr_name = None
            if isinstance(node.value, ast.Name):
                arr_name = node.value.id
            elif isinstance(node.value, ast.Attribute):
                arr_name = node.value.attr

            # NOTE: Adjust only if dim_str == 1, which reflects the default lower bound in FORTRAN.
            # Arrays that do not follow this convention should be verified before applying any transformation,
            # to avoid incorrect conversions. In general, loop variables from conventional FORTRAN loops
            # don't require changes, as they're already adapted for Python. However, if an array has a non-default
            # lower bound (not starting at 1), the corresponding loop variable must be corrected to
            # account for the offset between FORTRAN and Python indexing, thus recovering to the original Fortran index
            if arr_name and arr_name in self.array_info:
                dims_info = self.array_info[arr_name.casefold()]
                if isinstance(node.slice, ast.Tuple):
                    new_elts = []
                    for i, elt in enumerate(node.slice.elts):
                        dim_info_str = dims_info[i].get("dim_str")
                        # Case 1: dim_str is a digit
                        if dim_info_str.isdigit():
                            if dim_info_str == "1":
                                new_elts.append(self._adjust_index(elt))
                            else:
                                offset = 1 - int(dim_info_str)
                                new_elts.append(
                                    self._apply_offset_if_convvar(elt, offset)
                                )

                        # Case 2: dim_str is a variable (needs global_attributes lookup
                        # or if it's present inside one of the composition class)
                        else:
                            # resolve via global attributes
                            # or the perhaps either in one of the composition classes if such instance is present
                            resolved_str = ""
                            dimension = self.cls_attributes.get(dim_info_str)

                            if dimension is not None:
                                resolved_str = str(dimension[0])
                            elif self.instances_global_attributes:
                                for key in list(
                                    self.instances_global_attributes.keys()
                                ):
                                    instance_attributes = (
                                        self.instances_global_attributes[key].get(
                                            "attributes"
                                        )
                                    )
                                    if instance_attributes:
                                        dimension = instance_attributes.get(
                                            dim_info_str
                                        )
                                        if dimension:
                                            resolved_str = str(dimension[0])

                            if resolved_str == "1":
                                new_elts.append(self._adjust_index(elt))
                            elif resolved_str:
                                offset = 1 - int(resolved_str)
                                new_elts.append(
                                    self._apply_offset_if_convvar(elt, offset)
                                )
                            else:
                                raise KeyError(
                                    f"Could not resolve dimension info for: {dim_info_str}"
                                )

                    node.slice.elts = new_elts
                else:
                    # Single-dimension array
                    dim_info_str = dims_info[0].get("dim_str")
                    if dim_info_str.isdigit():
                        if dim_info_str == "1":
                            node.slice = self._adjust_index(node.slice)
                        else:
                            offset = 1 - int(dim_info_str)
                            node.slice = self._apply_offset_if_convvar(
                                node.slice, offset
                            )

                    # Case 2: dim_str is a variable (needs global_attributes lookup)
                    else:
                        resolved_str = ""
                        dimension = self.cls_attributes.get(dim_info_str)

                        if dimension is not None:
                            resolved_str = str(dimension[0])
                        elif self.instances_global_attributes:
                            for key in list(self.instances_global_attributes.keys()):
                                instance_attributes = self.instances_global_attributes[
                                    key
                                ].get("attributes")
                                if instance_attributes:
                                    dimension = instance_attributes.get(dim_info_str)
                                    if dimension:
                                        resolved_str = str(dimension[0])

                        if resolved_str == "1":
                            node.slice = self._adjust_index(node.slice)
                        elif resolved_str:
                            offset = 1 - int(resolved_str)
                            node.slice = self._apply_offset_if_convvar(
                                node.slice, offset
                            )
                        else:
                            raise KeyError(
                                f"Could not resolve dimension info for: {dim_info_str}"
                            )

            else:
                # fallback: adjust everything if we don’t know the array:
                # which mostly means that it's a functions
                if isinstance(node.slice, ast.Tuple):
                    node.slice.elts = [
                        self._adjust_index(elt) for elt in node.slice.elts
                    ]
                else:
                    node.slice = self._adjust_index(node.slice)

            return node

        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_Subscript of AdjustIndices: {ast.dump(node, indent=4)}"
            ) from e

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """
        Visit an assignment statement and adjust index-related expressions.

        Processes both assignment targets and assigned values, ensuring that
        variables derived from conventional loop indices are converted to
        their Python-equivalent indexing scheme.

        Tracks variables requiring future adjustment through
        :attr:`adjusted_vars` and applies transformations using
        :meth:`_adjust_assignment_rhs` and :meth:`_adjust_index`.

        Parameters
        ----------
        node : ast.Assign
            Assignment node to transform.

        Returns
        -------
        ast.Assign
            The transformed assignment node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the Assignements.
        """
        try:
            node.value = self.visit(node.value)
            new_targets = [self.visit(t) for t in node.targets]

            node.targets = new_targets
            # Handle index variables that are not loop variables themselves but are
            # used as array indices and receive their values through assignments
            # (e.g., `jsl = value`). This also includes variables that appear in
            # comparison expressions. In such cases, we must ensure that logical/mask
            # arrays are not incorrectly offset-adjusted.
            #
            # The same logic applies to variables that have been assigned from loop
            # variables, making them indirect references to the translated loop index.
            # These indirect references may appear as a Name, an Attribute, or an
            # array element, and should be treated consistently when determining
            # whether index adjustments are required.
            if isinstance(node.targets[0], ast.Name | ast.Attribute):
                name = (
                    node.targets[0].id
                    if isinstance(node.targets[0], ast.Name)
                    else node.targets[0].attr
                )
                if (
                    name in self.CONV_VARS
                ):  # THIS IS to modify in the case of CONV_vars ARE in the left hand side
                    node.value = self._adjust_assignment_rhs(node.value)
                # Check if the right hand assigement is that of conv vars
                elif name in self.adjusted_vars:
                    node.value = self._adjust_index(node.value)

                elif isinstance(node.value, ast.Compare) or name == "mask":
                    self.adjusted_vars.add(name)

            elif isinstance(node.targets[0], ast.Subscript):
                if isinstance(node.targets[0].value, ast.Name | ast.Attribute):
                    name = (
                        node.targets[0].value.id
                        if isinstance(node.targets[0].value, ast.Name)
                        else node.targets[0].value.attr
                    )
                    if name in self.adjusted_vars:
                        node.value = self._adjust_index(node.value)

            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_Assign \
                               of AdjustIndices: {ast.dump(node, indent=4)}"
            ) from e

    def visit_For(self, node: ast.For) -> ast.For:
        """
        Visit a ``for`` loop and normalize iterator bounds.

        Identifies unused loop variables and replaces them with ``_`` when
        appropriate. Iterator arguments are processed through
        :meth:`_process_arg` to account for previously adjusted index
        variables and converted Fortran loop bounds.

        Parameters
        ----------
        node : ast.For
            Loop node being visited.

        Returns
        -------
        ast.For
            The transformed loop node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the For loops.
        """
        try:
            self.generic_visit(node)

            loop_vars = self._extract_loop_vars(node.target)

            used_vars = set()
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    used_vars.add(child.id)

            unused = [v for v in loop_vars if v not in used_vars]
            if unused:
                print(f" ⚠️ Unused loop variable(s): {unused}")

                for var in unused:
                    self._rename_var_in_target(node.target, var, "_")

            if not isinstance(node.iter, ast.Call) or not hasattr(node.iter, "args"):
                return node

            new_args = []
            for arg in node.iter.args:
                new_arg = self._process_arg(arg, node)
                if new_arg:
                    new_args.append(new_arg)

            node.iter.args = new_args
            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_For \
                               of AdjustIndices: {ast.dump(node, indent=4)}"
            ) from e

    def _extract_loop_vars(self, target: ast.AST) -> list[str]:
        """
        Extract variable names from a loop target.

        Supports simple loop variables as well as tuple and list unpacking
        targets used in constructs such as ``enumerate``.

        Parameters
        ----------
        target : ast.AST
            Loop target expression.

        Returns
        -------
        list[str]
            Names of all variables appearing in the loop target.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if isinstance(target, ast.Name):
                return [target.id]
            elif isinstance(target, ast.Tuple | ast.List):
                # In the case we have enumerate instead of range
                vars_ = []
                for elt in target.elts:
                    vars_.extend(self._extract_loop_vars(elt))
                return vars_
            return []
        except Exception:
            raise

    def _rename_var_in_target(self, target: ast.AST, old: str, new: str) -> None:
        """
        Rename a variable within a loop target expression.

        Traverses tuple and list unpacking targets recursively and replaces
        occurrences of a given variable name.

        Parameters
        ----------
        target : ast.AST
            Loop target to modify.
        old : str
            Variable name to replace.
        new : str
            Replacement variable name.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if isinstance(target, ast.Name) and target.id == old:
                target.id = new
            elif isinstance(target, ast.Tuple | ast.List):
                for elt in target.elts:
                    self._rename_var_in_target(elt, old, new)
        except Exception:
            raise

    def _process_arg(self, arg: ast.AST, node: ast.AST) -> ast.AST:
        """
        Process an iterator argument in a ``for`` loop.

        Determines whether the argument references an adjusted index variable
        and dispatches specialized handling through
        :meth:`_handle_adjusted_left` when required.

        Parameters
        ----------
        arg : ast.AST
            Iterator argument.
        node : ast.For
            Parent loop node.

        Returns
        -------
        ast.AST
            The transformed argument node.

        Raises
        ------
        NotImplementedError
            If *node* contains an AST node type is
            present in the :attr:``adjusted_vars``.
        Exception
            Re-raises any unexpected error.
        """
        try:
            if isinstance(arg, ast.BinOp):
                left, right = arg.left, arg.right

                if isinstance(left, ast.Name) and left.id in self.adjusted_vars:
                    return self._handle_adjusted_left(arg, node)

                elif isinstance(left, ast.Subscript) and isinstance(
                    left.value, ast.Name | ast.Attribute
                ):
                    return self._handle_adjusted_left(arg, node)

                elif isinstance(left, ast.Name) and left.id not in self.adjusted_vars:
                    return arg

                # Case 4: Right is an adjusted variable
                elif isinstance(right, ast.Name) and right.id in self.adjusted_vars:
                    raise NotImplementedError(
                        "Not implemented yet for adjusted right-hand variable in visit_For."
                    )

                else:
                    return arg
            return arg
        except Exception:
            raise

    def _handle_adjusted_left(
        self, binop_node: ast.BinOp, parent_node: ast.AST
    ) -> ast.AST:
        """
        Rewrite loop-bound expressions involving adjusted variables.

        Handles iterator bounds whose left-hand side references an adjusted
        index variable or array element and restores the expected Python
        range semantics.

        Parameters
        ----------
        binop_node : ast.BinOp
            Binary operation describing the loop bound.
        parent_node : ast.AST
            Parent AST node.

        Returns
        -------
        ast.AST
            Rewritten bound expression.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            right = binop_node.right

            if isinstance(right, ast.Constant) and right.value == 1:
                return binop_node.left
            else:
                return ast.BinOp(
                    left=parent_node, op=ast.Add(), right=ast.Constant(value=1)
                )
        except Exception:
            raise

    def visit_If(self, node: ast.If) -> ast.If:
        """
        Visit an ``if`` statement and adjust comparison expressions.

        Processes conditional comparisons through :meth:`_handle_compare`
        and removes empty ``if`` blocks that contain no executable
        statements.

        Parameters
        ----------
        node : ast.If
            Conditional statement node.

        Returns
        -------
        ast.If | None
            The transformed conditional node, or ``None`` if the node should
            be removed.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the For loops.
        """
        try:
            self.generic_visit(node)
            if isinstance(node.test, ast.Compare):
                self._handle_compare(node.test)

            if (
                not node.body or all(isinstance(n, ast.Pass) for n in node.body)
            ) and not node.orelse:
                # Return None to delete the empty 'if' node entirely
                return None

            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeError in visit_For of AdjustIndices: {ast.dump(node, indent=4)}"
            ) from e

    def _handle_compare(self, node: ast.Compare) -> ast.Compare:
        """
        Adjust comparison operands involving converted indices.

        Ensures that comparisons involving conventional loop variables or
        entries tracked in :attr:`adjusted_vars` preserve their original
        Fortran semantics after conversion.

        Parameters
        ----------
        node : ast.Compare
            Comparison node to modify.

        Returns
        -------
        ast.Compare
            The transformed comparison node.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            # NOTE:
            # When processing comparison expressions, we check whether any operands
            # (particularly loop variables) require index adjustment. Loop variables
            # may have already been transformed to match Python's `range()` semantics
            # during Fortran-to-Python conversion. Therefore, when a loop variable is
            # compared directly against a constant, variable, or other expression,
            # we must avoid applying the transformation a second time. Such cases are
            # treated as indirect references, where the loop variable already represents
            # the translated Python index space.

            if isinstance(node.left, ast.Name) and (
                node.left.id in self.CONV_VARS or node.left.id in self.adjusted_vars
            ):
                for i in range(len(node.comparators)):
                    # Need to handle the case where the nodes compare themselves
                    # but perhaps with a certain index +, for example : loc == loc + 1
                    node.comparators[i] = self._adjust_index(node.comparators[i])
            elif isinstance(node.left, ast.Subscript):
                if isinstance(node.left.value, ast.Name) and (
                    node.left.value.id in self.CONV_VARS
                    or node.left.value.id in self.adjusted_vars
                ):
                    for i in range(len(node.comparators)):
                        if (
                            isinstance(node.comparators[i], ast.Subscript)
                            and node.comparators[i].value not in self.adjusted_vars
                        ):
                            continue
                        else:
                            node.comparators[i] = self._adjust_index(
                                node.comparators[i]
                            )
            return node
        except Exception:
            raise

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Visit a function call and adjust index-returning operations.

        Special handling is applied to ``argmin`` and ``argmax`` calls so
        that returned indices match the original Fortran lower-bound
        convention. Lower-bound information is retrieved from
        :attr:`array_info`.

        Parameters
        ----------
        node : ast.Call
            Function call node being visited.

        Returns
        -------
        ast.AST
            The transformed call node.

        Raises
        ------
        RuntimeError
            If an unexpected error occurs while processing the call.
        """
        try:
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr in [
                "argmin",
                "argmax",
            ]:
                # Check for the subscript node inside
                subscript_nodes = ast_walk(node, ast.Subscript)
                if subscript_nodes:
                    subscript_node = next(iter(subscript_nodes))
                    arr_name = (
                        subscript_node.value.id
                        if isinstance(subscript_node.value, ast.Name)
                        else subscript_node.value.attr
                    )

                    dim_info = self.array_info[arr_name.casefold()]
                    if dim_info and len(dim_info) == 1:
                        # If the lower bound is 0 this doesnt' require the creation of BinOp
                        if dim_info[0]["dim_str"] != "0":
                            # lb + min_idx for python and lb + min_idx - 1 for fortran
                            return ast.BinOp(
                                left=node,
                                op=ast.Add(),
                                right=ast.Constant(value=int(dim_info[0]["dim_str"])),
                            )
                    elif dim_info and len(dim_info) > 1:
                        # First we need to retrieve the dim info based on where the SLICE is
                        slices = getattr(subscript_node.slice, "elts", [])
                        slice_positions = [
                            i
                            for i, elem in enumerate(slices)
                            if isinstance(elem, ast.Slice)
                        ]
                        if (
                            len(slice_positions) == 1
                        ):  # Means only one slice in the multi dimensional array
                            lb = dim_info[slice_positions[0]]["dim_str"]
                            if lb != "0":
                                return ast.BinOp(
                                    left=node,
                                    op=ast.Add(),
                                    right=ast.Constant(value=int(lb)),
                                )
                        else:
                            # NOTE: When multiple slices are present, NumPy requires the `axis` argument
                            # to identify which dimension the operation applies to. The axis value
                            # must be examined to ensure that the correct slice (i.e., the intended
                            # array dimension from the original Fortran code) is selected.
                            axis_value = None
                            if isinstance(node.keywords[0].value, ast.Constant):
                                axis_value = node.keywords[0].value.value + 1

                            lb = dim_info[slice_positions[axis_value]]["dim_str"]
                            if lb != "0":
                                return ast.BinOp(
                                    left=node,
                                    op=ast.Add(),
                                    right=ast.Constant(value=int(lb)),
                                )
            return node
        except Exception as e:
            raise RuntimeError(
                f"RuntimeERROR in visit_Call of AdjustIndices: {ast.dump(node, indent=4)}"
            ) from e

    def _apply_offset_if_convvar(self, node: ast.AST, offset: int) -> ast.AST:
        """
        Apply a lower-bound offset to converted index expressions.

        Transforms conventional loop variables and adjusted variables so that
        references to arrays with non-zero lower bounds access the same
        logical elements as the original Fortran code.

        Parameters
        ----------
        node : ast.AST
            Expression potentially representing an index.
        offset : int
            Offset required to translate between Fortran and Python indexing.

        Returns
        -------
        ast.AST
            Adjusted expression node.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            # Fortran arrays may use arbitrary lower bounds (e.g., 0:n, 1:n, -3:n),
            # whereas Python/NumPy arrays are zero-based. During Fortran-to-Python
            # conversion, each array is stored with Python indexing, and references
            # to the original Fortran indices are translated by applying the array's
            # lower-bound offset. This preserves the original element mapping and
            # ensures that Python accesses the same logical array elements and
            # produces the same results as the Fortran code.
            if isinstance(node, ast.Name) and (
                node.id in self.CONV_VARS or node.id in self.adjusted_vars
            ):
                node = ast.BinOp(
                    left=node, op=ast.Add(), right=ast.Constant(value=offset)
                )

            elif isinstance(node, ast.BinOp):  # THis is to handle cases when the
                left = node.left
                right = node.right
                op = node.op

                is_valid_left = False

                # Case 1: left is a Name
                if isinstance(left, ast.Name):
                    if left.id in self.CONV_VARS or left.id in self.adjusted_vars:
                        is_valid_left = True

                # Case 2: left is a Subscript of a Name (A[i]) and A is in adjusted_vars
                elif isinstance(left, ast.Subscript):
                    if (
                        isinstance(left.value, ast.Name)
                        and left.value.id in self.adjusted_vars
                    ):
                        is_valid_left = True

                if is_valid_left and isinstance(right, ast.Constant):
                    original_value = right.value

                    if isinstance(op, ast.Sub):
                        new_value = original_value - offset
                        if new_value == 0:
                            node = left
                        else:
                            node = ast.BinOp(
                                left=left,
                                op=ast.Sub(),
                                right=ast.Constant(value=new_value),
                            )

                    elif isinstance(op, ast.Add):
                        new_value = original_value + offset
                        if new_value == 0:
                            node = left
                        else:
                            node = ast.BinOp(
                                left=left,
                                op=ast.Add(),
                                right=ast.Constant(value=new_value),
                            )

            elif isinstance(node, ast.Call):
                if not self._check_internal_call_element(node):
                    node = ast.BinOp(
                        left=node, op=ast.Add(), right=ast.Constant(value=offset)
                    )

            return node
        except Exception:
            raise

    def _adjust_index(self, index_node: ast.AST) -> ast.AST:
        """
        Convert a Fortran-style index expression to Python indexing.

        Recursively traverses names, binary operations, slices, subscripts,
        attributes, and selected function calls, applying the appropriate
        index correction rules.

        Variables listed in :attr:`CONV_VARS`,
        :attr:`adjusted_vars`, or :attr:`exclude_index` are excluded from
        direct modification when appropriate.

        Parameters
        ----------
        index_node : ast.AST
            Index expression to transform.

        Returns
        -------
        ast.AST
            The transformed index expression.

        Raises
        ------
        RuntimeError
            If index adjustment fails for the supplied node.
        """

        try:
            if isinstance(index_node, ast.Name):
                if (
                    index_node.id not in self.CONV_VARS
                    and index_node.id not in self.adjusted_vars
                ):
                    # TODO: Need a checker method that checks if one of the varaibles
                    # has either been affected is that of int type
                    # thus requires us to modify it directly
                    if (not self.exclude_index) or (
                        index_node.id not in self.exclude_index
                    ):
                        return self._subtract_one(index_node)

            elif isinstance(index_node, ast.BinOp):
                return self._handle_binop(index_node)

            elif isinstance(index_node, ast.Subscript):
                return self.visit_Subscript(index_node)

            elif isinstance(index_node, ast.Constant):
                if index_node.value == 0:
                    return index_node
                else:
                    return ast.Constant(value=index_node.value - 1)

            elif isinstance(index_node, ast.Attribute):
                if (
                    index_node.attr not in self.CONV_VARS
                    and index_node.attr not in self.adjusted_vars
                ):
                    if (not self.exclude_index) or (
                        index_node.attr not in self.exclude_index
                    ):
                        return self._subtract_one(index_node)

            elif isinstance(index_node, ast.Call):
                # Need to check if the usually int() internal elemnt is
                # not that of the adjusted vars or that of the excluded_index
                if self._check_internal_call_element(index_node):
                    return ast.BinOp(
                        left=index_node, op=ast.Sub(), right=ast.Constant(value=1)
                    )
                else:
                    return index_node

            elif isinstance(index_node, ast.Slice):
                if index_node.lower is None and index_node.upper is None:
                    return index_node

                # Recursively adjust lower and upper if they exist
                new_lower = (
                    self._adjust_index(index_node.lower) if index_node.lower else None
                )
                new_upper = index_node.upper
                new_step = (
                    self._adjust_index(index_node.step) if index_node.step else None
                )

                return ast.Slice(lower=new_lower, upper=new_upper, step=new_step)

            return index_node
        except Exception as e:
            raise RuntimeError(
                f"_adjust_index failed for node={ast.dump(index_node, indent=4)}"
            ) from e

    def _check_internal_call_element(self, node: ast.AST) -> ast.AST:
        """
        Adjust right-hand-side expressions assigned to converted indices.

        Rewrites assignments derived from conventional loop variables so that
        subsequent references operate in Python index space while preserving
        the original Fortran behavior.

        Parameters
        ----------
        rhs : ast.AST
            Right-hand-side expression.

        Returns
        -------
        ast.AST
            Adjusted expression.

        Raises
        ------
        RuntimeError
            If transformation fails.
        ValueError
            If the arg is not that of the subscript or
            the number of elements inside the `int` > 1
        """
        try:
            # Check if the node is a call, and function called is `int`
            int_call = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
            )
            if int_call:
                # Ensure there is exactly one argument
                if len(node.args) == 1:
                    arg = node.args[0]

                    if isinstance(arg, ast.Subscript) and isinstance(
                        arg.value, ast.Name
                    ):
                        var_name = arg.value.id
                        if var_name not in self.adjusted_vars and (
                            (not self.exclude_index)
                            or (var_name not in self.exclude_index)
                        ):
                            return True
                    else:
                        raise ValueError(
                            f"arg is not that of Subscript but of : {type(node)}"
                        )
                else:
                    raise ValueError("Number of args inside the int() > 1")
            return False
        except Exception as e:
            raise RuntimeError(
                f"_check_internal_call_element failed during the transformation: node={ast.dump(node, indent=4)}"
            ) from e

    # This is for right hand side assignement handling
    def _adjust_assignment_rhs(self, rhs: ast.AST) -> ast.AST:
        """
        Adjust right-hand-side expressions assigned to converted indices.

        Rewrites assignments derived from conventional loop variables so that
        subsequent references operate in Python index space while preserving
        the original Fortran behavior.

        Parameters
        ----------
        rhs : ast.AST
            Right-hand-side expression.

        Returns
        -------
        ast.AST
            Adjusted expression.

        Raises
        ------
        RuntimeError
            If transformation fails.
        """
        try:
            if isinstance(rhs, ast.Name) and rhs.id not in self.CONV_VARS:
                return self._subtract_one(rhs)

            elif isinstance(rhs, ast.Subscript):
                return self._subtract_one(rhs)

            elif isinstance(rhs, ast.BinOp):
                if isinstance(rhs.op, ast.Sub):
                    return ast.BinOp(
                        left=rhs.left,
                        op=ast.Sub(),
                        right=ast.Constant(value=rhs.right.value + 1),
                    )

                elif isinstance(rhs.op, ast.Add):
                    if rhs.right.value == 1:
                        return rhs.left
                    else:
                        return ast.BinOp(
                            left=rhs.left,
                            op=ast.Add(),
                            right=ast.Constant(value=rhs.right.value - 1),
                        )

            return rhs
        except Exception as e:
            raise RuntimeError(
                f"_adjust_assignment_rhs failed for node={ast.dump(rhs, indent=4)}"
            ) from e

    def _subtract_one(self, node: ast.AST) -> ast.BinOp:
        """
        Create an expression equivalent to ``node - 1``.

        Used as a primitive building block for translating Fortran
        one-based indices into Python zero-based indices.

        Parameters
        ----------
        node : ast.AST
            Expression to decrement.

        Returns
        -------
        ast.BinOp
            Binary subtraction node representing ``node - 1``.
        """
        return ast.BinOp(left=node, op=ast.Sub(), right=ast.Constant(value=1))

    def _handle_binop(self, node: ast.BinOp) -> ast.AST:
        """
        Adjust binary index expressions for Python indexing semantics.

        Rewrites expressions such as ``i - 1`` or ``i + n`` so that they
        continue to reference the same logical Fortran element after
        conversion.

        Parameters
        ----------
        node : ast.BinOp
            Binary operation to transform.

        Returns
        -------
        ast.AST
            The transformed expression.

        Raises
        ------
        RuntimeError
            If processing fails.
        """
        try:
            self.generic_visit(node)
            check = (
                isinstance(node.left, ast.Name)
                and node.left.id not in self.CONV_VARS
                and node.left.id not in self.adjusted_vars
                and isinstance(node.right, ast.Constant)
            )
            if check:
                if (not self.exclude_index) or (node.left.id not in self.exclude_index):
                    if isinstance(node.op, ast.Sub):
                        # i - 1 -> i - 2
                        return ast.BinOp(
                            left=node.left,
                            op=ast.Sub(),
                            right=ast.Constant(value=node.right.value + 1),
                        )
                    elif isinstance(node.op, ast.Add):
                        # i + 2 -> i + 1 or i + 1 -> i
                        if node.right.value == 1:
                            return node.left
                        else:
                            return ast.BinOp(
                                left=node.left,
                                op=ast.Add(),
                                right=ast.Constant(value=node.right.value - 1),
                            )
            return node
        except Exception as e:
            raise RuntimeError(
                f"_handle_binop failed for node={ast.dump(node, indent=4)}"
            ) from e


def get_instance_name(node_name: str) -> str:
    """
    Generate a compact instance name from a node or class name string.

    Constructs a shortened instance identifier based on underscore-separated
    components of *node_name*. Single-word names are lowercased directly,
    two-part names are reduced to initials, and longer names are reduced to
    a concatenation of initial letters.

    Parameters
    ----------
    node_name : str
        Input name to convert into an instance identifier.

    Returns
    -------
    str
        A compact instance name derived from *node_name*.

    Examples
    --------
    - ``processor`` → ``processor``
    - ``global_module`` → ``gm``
    - ``long_class_name`` → ``lcn``
    """
    clean_name = "_".join(filter(None, node_name.strip().split("_")))
    split_name = clean_name.split("_")

    if len(split_name) == 1:
        # Format: processor, isolator
        instance_name = split_name[0].lower()
    elif len(split_name) == 2:
        # Format: global_module
        instance_name = split_name[0].lower()[0] + split_name[1].lower()[0]
    else:
        instance_name = "".join(part.lower()[0] for part in split_name)

    return instance_name


def identify_replace_all(
    ast_list: list, cls_info: dict, local_names: set[str] | None = None
) -> None:
    """
    Apply global replacement transformations to a list of AST nodes.

    Applies a :class:`ReplaceGlobals` transformer to each node in
    *ast_list*, optionally restricting transformations to a given
    local scope.

    The transformation is performed in-place.

    Parameters
    ----------
    ast_list : list
        List of AST nodes to transform.
    cls_info : dict
        Class metadata used to initialize the transformer.
    local_names : set[str], optional
        Optional set of local variable names defining the transformation
        scope.

    Raises
    ------
    Exception
        Re-raises any exception encountered during transformation after
        logging the error.
    """

    try:
        transformer = ReplaceGlobals(cls_info)
        if local_names:
            transformer._local_scope = local_names

        for i, node in enumerate(ast_list):
            ast_list[i] = transformer.visit(node)
    except Exception as e:
        logging.error(f"Error in identify_replace_all: {e}")
        raise


def ast_walk(node, node_type: ast.AST) -> Generator | None:
    """
    Recursively traverse an AST and yield nodes matching a given type.

    Walks the AST rooted at *node* and yields all nodes that match
    *node_type*. If *node_type* is ``None``, all nodes are yielded.

    Parameters
    ----------
    node : ast.AST
        Root AST node to traverse.
    node_type : type or None
        AST node type to filter by. If ``None``, yields all nodes.

    Yields
    ------
    ast.AST or None
        AST nodes matching the filter. Yields ``None`` if an exception
        occurs during traversal.

    Raises
    ------
    Exception
        Re-raises any exception encountered during transformation after
        logging the error.

    Notes
    -----
    - Traversal is depth-first using :func:`ast.iter_child_nodes`.
    - If an exception occurs, it is logged and ``None`` is yielded.
    """
    try:
        if node_type is None or isinstance(node, node_type):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from ast_walk(child, node_type)
    except Exception:
        logging.exception("Exception in ast_walk")
        yield None


def find_folder(root_dir: str, target_folder: str) -> str | None:
    """
    Locate a folder within a directory tree.

    Recursively searches *root_dir* for a directory named *target_folder*
    using :func:`os.walk`. Returns the full path of the first match found.

    Parameters
    ----------
    root_dir : str
        Root directory to begin the search.
    target_folder : str
        Name of the folder to locate.

    Returns
    -------
    str or None
        Absolute path to the target folder if found, otherwise ``None``.

    Notes
    -----
    - Only the first occurrence is returned.
    - The search is breadth-first by directory traversal order of
      :func:`os.walk`.
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if target_folder in dirnames:
            return os.path.join(dirpath, target_folder)
    return None


def find_used_globals(node: ast.AST, common_attributes: Iterable[str]) -> set[str]:
    """
    Extract global variable names used within an AST node.

    Traverses *node* recursively and collects identifiers that match
    entries in *common_attributes*, treating them as global or shared
    variables.

    Parameters
    ----------
    node : ast.AST
        Root AST node to analyze.
    common_attributes : Iterable[str]
        set or collection of variable names considered global.

    Returns
    -------
    set[str]
        set of global variable names referenced in the AST.

    Notes
    -----
    - Traversal is recursive over all AST children.
    - Only :class:`ast.Name` nodes are considered for matches.
    - Attribute accesses are not included in the current implementation.
    """
    used_globals = set()

    def visit(n):
        if isinstance(n, ast.Name):
            if n.id in common_attributes:
                used_globals.add(n.id)
        # Recursively visit all child nodes
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return used_globals


def attach_instance(node: ast.AST, instance_name: str | None = "self"):
    """
    Attach an instance reference to variable names within an AST subtree.

    Recursively traverses *node* and replaces each :class:`ast.Name`
    reference with an :class:`ast.Attribute` referencing
    *instance_name*. For example, ``x`` becomes ``self.x`` when
    the default instance name is used.

    The transformation is applied in-place to nested AST nodes and
    lists of AST nodes. Non-AST objects are left unchanged.

    Parameters
    ----------
    node : ast.AST
        Root AST node to transform.
    instance_name : str, optional
        Name of the instance object to attach to variable references.
        Defaults to ``"self"``.

    Returns
    -------
    ast.AST
        The transformed AST node with variable references rewritten
        as instance attributes.

    Notes
    -----
    - The transformation is applied recursively to all child nodes.
    - Existing :class:`ast.Attribute` nodes are preserved and only
      :class:`ast.Name` nodes are rewritten.
    - The input AST is modified in-place.
    """

    if isinstance(node, ast.Name):
        return ast.Attribute(
            value=ast.Name(id=instance_name, ctx=ast.Load()),
            attr=node.id,
            ctx=ast.Load(),
        )

    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, ast.AST):
                    new_list.append(attach_instance(item, instance_name))
                else:
                    new_list.append(item)
            setattr(node, field, new_list)
        elif isinstance(value, ast.AST):
            setattr(node, field, attach_instance(value, instance_name))

    return node


def safe_eval_expr(node, attributes: dict[str, tuple] | None = None) -> Any:
    """
    Evaluate a restricted AST expression using supplied attribute values.

    Evaluates a subset of Python expressions represented by AST nodes
    without executing arbitrary code. Supported constructs include
    constants, variable references, attribute references, and common
    binary arithmetic operations.

    Variable and attribute values are resolved from *attributes*,
    where each entry is expected to contain the runtime value as the
    first element of a tuple.

    Parameters
    ----------
    node : ast.AST
        Expression node to evaluate.
    attributes : dict[str, tuple], optional
        Mapping of variable names to tuples containing runtime values.
        If ``None``, an empty mapping is used.

    Returns
    -------
    Any
        Result of evaluating the expression.

    Raises
    ------
    NameError
        If a referenced variable cannot be found in *attributes*.
    NotImplementedError
        If *node* contains an unsupported AST node type or operator.
    KeyError
        If an attribute reference exists but is not present in
        *attributes*.

    Notes
    -----
    Supported operators include:

    - :class:`ast.Add`
    - :class:`ast.Sub`
    - :class:`ast.Mult`
    - :class:`ast.Div`
    - :class:`ast.FloorDiv`
    - :class:`ast.Mod`
    - :class:`ast.Pow`

    This function is intended as a safe alternative to :func:`eval`
    for a limited subset of expressions.
    """
    if attributes is None:
        attributes = {}

    if isinstance(node, ast.Constant):
        return node.value

    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left, attributes)
        right = safe_eval_expr(node.right, attributes)

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }

        op_type = type(node.op)
        if op_type in ops:
            return ops[op_type](left, right)
        else:
            raise NotImplementedError(f"Operator {op_type} not supported.")

    elif isinstance(node, ast.Name):
        var_name = node.id
        if var_name in attributes:
            return attributes[var_name][0]  # Get the actual value
        else:
            raise NameError(f"Variable '{var_name}' not found in attributes.")
    elif isinstance(node, ast.Attribute):
        if isinstance(node.attr, str):
            return attributes[node.attr][0]
    else:
        raise NotImplementedError(f"Unsupported AST node type: {type(node)}")


def update_methods(module_dict: dict, function_defs: list) -> None:
    """
    Update module method dictionaries with new function definitions.

    Searches *module_dict* for nested dictionaries containing a
    ``"methods"`` entry and adds or replaces method definitions
    using the supplied AST function definitions.

    Existing methods with the same name are overwritten by the
    corresponding entries in *function_defs*.

    Parameters
    ----------
    module_dict : dict
        Module metadata structure containing one or more nested
        dictionaries with a ``"methods"`` key.
    function_defs : list[ast.FunctionDef]
        Function definitions to add to the target method dictionary.

    Notes
    -----
    - Only objects of type :class:`ast.FunctionDef` are processed.
    - Method definitions are indexed by their function name.
    - The update is performed in-place and no copy of
      *module_dict* is created.
    """
    # Loop over the top-level module(s)
    for _, module_content in module_dict.items():
        # Search for the inner dict that contains 'methods'
        for _, instance_val in module_content.items():
            if isinstance(instance_val, dict) and "methods" in instance_val:
                methods_dict = instance_val["methods"]
                for func_def in function_defs:
                    if isinstance(func_def, ast.FunctionDef):
                        methods_dict[func_def.name] = func_def
                # Ensure the dict is updated
                instance_val["methods"] = methods_dict
                break


def collect_dependencies(node: ast.AST) -> set[str]:
    """
    Collect variable dependencies referenced within an AST node.

    Traverses *node* and extracts all variable names referenced
    within the AST subtree. Both standalone variable references
    (:class:`ast.Name`) and attribute references
    (:class:`ast.Attribute`) are included in the dependency set.

    Parameters
    ----------
    node : ast.AST
        AST node to analyze.

    Returns
    -------
    set[str]
        Unique variable and attribute names referenced within
        *node*.

    Notes
    -----
    - Variable references contribute their ``id`` attribute.
    - Attribute references contribute their ``attr`` attribute.
    - Dependencies are collected recursively using
      :func:`ast.walk`.
    - Duplicate references are automatically removed by the
      returned set.
    """
    deps = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            deps.add(child.id)
        elif isinstance(child, ast.Attribute):
            deps.add(child.attr)
    return deps


def order_assignments(assign_nodes: list, diff: list) -> list:
    """
    Order assignment variables according to their dependency relationships.

    Builds a dependency graph from *assign_nodes* and performs a
    topological sort using Kahn's algorithm so that each variable
    appears after all variables it depends on. When *diff* is
    provided, only variables present in *diff* and their
    interdependencies are considered.

    If cyclic dependencies are detected, any remaining variables
    are appended to the result in arbitrary order after the
    topological sort completes.

    Parameters
    ----------
    assign_nodes : list
        Collection of assignment AST nodes whose dependencies
        should be analyzed.
    diff : list
        Optional subset of variable names used to restrict
        dependency analysis and ordering.

    Returns
    -------
    list
        Variable names sorted in dependency-respecting order.

    Notes
    -----
    - Dependencies are extracted from assignment right-hand-side
      expressions using :func:`collect_dependencies`.
    - Kahn's algorithm is used to compute the topological ordering.
    - Cyclic dependencies are not resolved; remaining nodes are
      appended to the output after sorting.
    """
    # Build dependency graph
    graph = defaultdict(set)
    indegree = defaultdict(int)

    for assign in assign_nodes:
        if not assign.targets:
            continue
        target = assign.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        else:
            continue

        if diff and name not in diff:
            continue

        if diff:
            deps = collect_dependencies(assign.value) & set(diff)
        else:
            deps = collect_dependencies(assign.value)
        graph[name].update(deps)
        # Create the dependecies graph for attributes
        # mostly between scalars with intialized values

    # Compute indegrees
    for var, deps in graph.items():
        for _ in deps:
            indegree[var] += 1

    # THe graphs in our case is the dependant dependeee type elements
    # https://www.interviewcake.com/concept/java/topological-sort,
    # thus we first go from the dependedant then the dependee
    # Kahn’s algorithm for topological sort code found here:
    # https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
    queue = deque([v for v in graph if indegree[v] == 0])
    result = []

    while queue:
        v = queue.popleft()
        result.append(v)
        for u in graph:
            if v in graph[u]:
                indegree[u] -= 1
                if indegree[u] == 0:
                    queue.append(u)

    # If cycles remain, just append the rest arbitrarily
    for v in graph:
        if v not in result:
            result.append(v)

    return result


def search_convar_dependencies(conv_vars: list[str], node: ast.AST) -> set[str]:
    """
    Find variables that depend on a set of conventional variables.

    Traverses *node* and identifies assignment statements whose
    right-hand-side expressions reference any variable listed in
    *conv_vars*. Returns the names of variables assigned from
    those dependent expressions.

    Assignments to names, attributes, and simple subscripts are
    supported.

    Parameters
    ----------
    conv_vars : list[str]
        Conventional variable names to search for within
        assignment expressions.
    node : ast.AST
        Root AST node to traverse.

    Returns
    -------
    set[str]
        Names of variables whose assigned values depend on one
        or more variables in *conv_vars*.

    Notes
    -----
    Dependency detection is delegated to
    :meth:`_find_conv_vars_in_expr`.
    """
    adjusted_vars = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            value = child.value
            if _find_conv_vars_in_expr(value, conv_vars):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        adjusted_vars.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        adjusted_vars.add(target.attr)
                    elif isinstance(target, ast.Subscript):
                        # Handles cases like x[i] = ...
                        if isinstance(target.value, ast.Name):
                            adjusted_vars.add(target.value.id)

    return adjusted_vars


def _find_conv_vars_in_expr(child: ast.AST, conv_vars: list[str]) -> bool:
    """
    Determine whether an expression references any conventional variables.

    Recursively inspects an AST expression node and checks whether
    it contains a reference to any variable listed in *conv_vars*.
    This method is primarily used to identify assignments that
    depend on conventional variables.

    Parameters
    ----------
    child : ast.AST
        Expression node to inspect.
    conv_vars : list[str]
        Variable names to search for within the expression.

    Returns
    -------
    bool
        ``True`` if the expression contains a reference to any
        variable in *conv_vars*, otherwise ``False``.

    Notes
    -----
    The current implementation supports detection within:

    - :class:`ast.Name`
    - :class:`ast.Attribute`
    - :class:`ast.BinOp`
    - :class:`ast.Compare`

    More complex expression types may require additional handling.
    """
    if isinstance(child, ast.Name) and child.id in conv_vars:
        return True
    elif isinstance(child, ast.Attribute):
        if isinstance(child.value, ast.Name) and child.value.id in conv_vars:
            return True
        if child.attr in conv_vars:
            return True
    elif isinstance(child, ast.BinOp):
        name = None
        if isinstance(child.left, ast.Name | ast.Attribute):
            name = (
                child.left.id if isinstance(child.left, ast.Name) else child.left.attr
            )
        elif isinstance(child.right, ast.Name | ast.Attribute):
            name = (
                child.right.id
                if isinstance(child.right, ast.Name)
                else child.right.attr
            )

        if name in conv_vars:
            return True
    elif isinstance(child, ast.Compare):
        # Check all parts of the comparison
        if isinstance(child.left, ast.Name) and child.left.id in conv_vars:
            return True
        for comparator in child.comparators:
            if isinstance(comparator, ast.Name) and comparator.id in conv_vars:
                return True

    return False


def python_parser(code: str) -> ast.Module | None:
    """
    Parse a Python source string into an abstract syntax tree (AST).

    Attempts to parse *code* using :func:`ast.parse`. Logs an
    informational message when parsing succeeds and returns the
    resulting AST. If the source contains invalid Python syntax,
    logs the error and returns ``None``.

    Parameters
    ----------
    code : str
        Python source code to parse.

    Returns
    -------
    ast.Module or None
        The parsed AST module if *code* is syntactically valid,
        otherwise ``None``.

    Raises
    ------
    None
        All :class:`SyntaxError` exceptions are handled internally
        and converted into a ``None`` return value.
    """
    try:
        tree = ast.parse(code)
        logging.info("INFO: Parsed python template is valid")
        return tree
    except SyntaxError as e:
        logging.error(f"ERROR: Syntax error: {e}")
        return None


def load_code_templates(config_path: str) -> dict | None:
    """
    Load code templates from a YAML configuration file.

    Opens the YAML file located at *config_path* and deserializes its
    contents using :func:`yaml.safe_load`. Returns the resulting
    template dictionary on success. If the file cannot be found,
    contains invalid YAML, or another unexpected error occurs,
    logs the error and returns ``None``.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing code templates.

    Returns
    -------
    dict or None
        Dictionary containing the loaded code templates if the file
        is successfully parsed, otherwise ``None``.

    Raises
    ------
    None
        All exceptions are handled internally. File access, YAML
        parsing, and unexpected errors are logged and result in a
        ``None`` return value.
    """
    try:
        with open(config_path) as file:
            templates = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Error: The file '{config_path}' was not found.")
        templates = None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        templates = None
    except Exception:
        logging.exception("An unexpected error occurred in load_code_templates")
        templates = None

    return templates
