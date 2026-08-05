# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
JAX transformation utilities for FGPT.

This module provides AST-level transformations and helper functions
to adapt Fortran-derived Python code for execution under JAX.

It includes:
- function and method rewriting for JAX compatibility
- main-function patching utilities
- array wrapping and conversion helpers

These tools are primarily used during the backend conversion phase
when migrating NumPy-style code to JAX-compatible representations.
"""

import ast
import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallEdge:
    caller: str
    callee: str
    call_node: ast.Call
    arg_shapes: dict[str, Any] = field(default_factory=dict)
    func_args: list = field(default_factory=list)


def topo_sort(class_dep: dict) -> list:
    """
    Perform a depth-first topological traversal of a dependency graph.

    The graph is represented as a mapping from a node to a collection of
    dependency edges. Each edge is expected to expose a ``callee`` attribute
    identifying the dependent node. Nodes are added to the output ordering
    after all of their dependencies have been visited, producing a
    child-before-parent ordering.

    Parameters
    ----------
    class_dep : dict
        Mapping from node names to iterables of dependency edges. Each edge
        must provide a ``callee`` attribute corresponding to another node
        in the dependency graph.

    Returns
    -------
    list
        Nodes in topological order such that dependencies appear before the
        nodes that depend on them.

    Notes
    -----
    This implementation assumes the dependency graph is acyclic. Cycles are
    not explicitly detected and may produce an invalid ordering.
    """
    visited = set()
    order = []

    def dfs(fn):
        if fn in visited:
            return
        visited.add(fn)
        for edge in class_dep.get(fn, []):
            dfs(edge.callee)
        order.append(fn)

    for fn in class_dep:
        dfs(fn)

    return order  # order[::-1] no reverse order children before parents


def get_name(node: ast.AST) -> str | None:
    """
    Extract the identifier name from a supported AST node.

    For attribute access expressions, the attribute name is returned.
    For variable references, the identifier is returned. For subscript
    expressions, the name is recursively extracted from the underlying
    value being indexed.

    Parameters
    ----------
    node : ast.AST
        AST node from which to extract a name.

    Returns
    -------
    str or None
        Extracted identifier name if the node is one of the supported
        types (`ast.Name`, `ast.Attribute`, or `ast.Subscript`),
        otherwise ``None``.
    """
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return get_name(node.value)
    return None


def contains_name(node: ast.AST, name: str) -> bool:
    """
    Check whether *name* appears as an ``ast.Name`` node anywhere in *node*.

    Parameters
    ----------
    node : ast.AST
        Root of the AST subtree to search.
    name : str
        Identifier to look for.

    Returns
    -------
    bool
        ``True`` if any ``ast.Name`` with ``id == name`` is found.

    Raises
    ------
    Exception
        Re-raises any unexpected error after logging.
    """
    try:
        return any(
            isinstance(sub, ast.Name) and sub.id == name for sub in ast.walk(node)
        )
    except Exception as e:
        raise e


def convert_np_jnp(node) -> ast.AST:
    """
    Recursively rewrite NumPy references to their JAX NumPy equivalents.

    This transformation performs two rewrites:

    1. Replaces attribute accesses of the form ``np.<attr>`` with
       ``jnp.<attr>``.
    2. Replaces calls to Python scalar constructors such as ``int(...)`` and
       ``float(...)`` with the corresponding JAX NumPy dtypes
       (``jnp.int32(...)`` and ``jnp.float64(...)``).

    The transformation is applied recursively to all descendant AST nodes.

    Parameters
    ----------
    node : ast.AST
        Root AST node to transform.

    Returns
    -------
    ast.AST
        The transformed AST node with NumPy references rewritten to JAX
        NumPy equivalents.

    Notes
    -----
    Source location information is preserved for rewritten ``np.<attr>``
    expressions using ``ast.copy_location``.

    Examples
    --------
    >>> expr = ast.parse("np.sin(x)", mode="eval")
    >>> new_expr = convert_np_jnp(expr)
    >>> ast.unparse(new_expr)
    'jnp.sin(x)'

    >>> expr = ast.parse("float(x)", mode="eval")
    >>> new_expr = convert_np_jnp(expr)
    >>> ast.unparse(new_expr)
    'jnp.float64(x)'
    """
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            attr = ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()), attr=node.attr, ctx=ast.Load()
            )
            return ast.copy_location(attr, node)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ["int", "float"]:
            NP_TO_JNP = {"int": "int32", "float": "float64"}
            node.func = ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr=NP_TO_JNP.get(node.func.id),
                ctx=ast.Load(),
            )

    for field_name, value in ast.iter_fields(node):
        if isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, ast.AST):
                    new_list.append(convert_np_jnp(item))
                else:
                    new_list.append(item)
            setattr(node, field_name, new_list)
        elif isinstance(value, ast.AST):
            setattr(node, field_name, convert_np_jnp(value))

    return node


def get_class_info_from_ast(cls_ast: ast.AST) -> dict:
    """
    Extract class attribute and method metadata from a class AST.

    This function analyzes one or more ``ast.ClassDef`` nodes contained
    within the supplied AST and builds a structured description of:

    * Instance attributes assigned through ``self.<attr>``.
    * JAX array attributes initialized with constructors such as
      ``jnp.zeros`` and ``jnp.array``.
    * Scalar attributes initialized with JAX scalar types
      (e.g. ``jnp.int32`` and ``jnp.float64``).
    * Method signatures and locally allocated arrays.

    Array metadata includes shape and dtype information when it can be
    statically determined from the AST.

    Parameters
    ----------
    cls_ast : ast.AST
        AST node containing one or more class definitions to analyze.
        Typically an ``ast.Module`` or ``ast.ClassDef``.

    Returns
    -------
    dict
        Mapping from class names to extracted metadata. The returned
        structure has the form::

            {
                "ClassName": {
                    "attributes": {
                        "attr_name": {
                            "type": str,
                            ...
                        }
                    },
                    "methods": {
                        "method_name": {
                            "args": list[str],
                            "local_arr": dict
                        }
                    }
                }
            }

    Raises
    ------
    NotImplementedError
        If an instance attribute is initialized from another attribute
        whose metadata has not yet been discovered.

    Notes
    -----
    The analysis currently recognizes:

    * ``jnp.zeros(...)``
    * ``jnp.array(...)``
    * ``jnp.int32(...)``
    * ``jnp.float32(...)``
    * ``jnp.float64(...)``
    * ``jnp.bool(...)``

    Shape information is inferred when dimensions can be statically
    extracted from tuples, constants, attributes, or simple expressions.

    Examples
    --------
    >>> tree = ast.parse(source_code)
    >>> info = get_class_info_from_ast(tree)
    >>> info["MyModel"]["attributes"]
    {'weights': {'type': 'jnp.ndarray',
                 'dimensions': [10, 20],
                 'dtype': 'float32'}}
    """
    if not isinstance(cls_ast, ast.AST):
        raise TypeError(f"Expected ast.AST, got {type(cls_ast).__name__}")
    class_info = {
        "attributes": {},
        "methods": {},
    }

    class_defs = []
    for node in ast.walk(cls_ast):
        if isinstance(node, ast.ClassDef):
            class_defs.append(node)
    type_map = {"int32": "int", "float64": "float", "float32": "float"}

    cls_info = {}
    for class_def in class_defs:
        for node in ast.walk(class_def):
            # Check if it's a class attribute (scalar or jax numpy array)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        attr_name = target.attr
                        value = node.value

                        if isinstance(value, ast.Call) and isinstance(
                            value.func, ast.Attribute
                        ):
                            # in the case of attributes scalars being modified then
                            # we need to ensure that it's not applied as static
                            # Check if the assignment is a jax numpy array (using jnp.array)
                            if value.func.attr in ["zeros", "array"]:
                                # Check the dimensions of the array (shape info)
                                dimensions = []
                                if isinstance(value.args[0], ast.Tuple):
                                    for dim in value.args[0].elts:
                                        if isinstance(dim, ast.Attribute):
                                            dimensions.append(dim.attr)
                                        elif isinstance(dim, ast.Constant):
                                            dimensions.append(dim.value)

                                        elif isinstance(dim, ast.BinOp):
                                            dimensions.append(ast.unparse(dim))
                                            # JUST put the binop itself directly.
                                    class_info["attributes"][attr_name] = {
                                        "type": "jnp.ndarray",
                                        "dimensions": dimensions,
                                        "dtype": value.keywords[0].value.attr,
                                        # We retrieve it's dtype for the arrays.
                                    }
                                elif value.func.attr == "array":

                                    def extract_list(node):
                                        if isinstance(node, ast.List):
                                            return [
                                                extract_list(elt) for elt in node.elts
                                            ]
                                        elif isinstance(node, ast.Constant):
                                            return node.value
                                        return None

                                    def get_shape(lst):
                                        if isinstance(lst, list):
                                            if len(lst) == 0:
                                                return [0]
                                            return [len(lst)] + get_shape(lst[0])
                                        return []

                                    array_data = extract_list(value.args[0])
                                    dimensions = get_shape(array_data)

                                    dtype = None
                                    for kw in value.keywords:
                                        if kw.arg == "dtype":
                                            if isinstance(kw.value, ast.Attribute):
                                                dtype = kw.value.attr
                                            elif isinstance(kw.value, ast.Name):
                                                dtype = kw.value.id

                                    class_info["attributes"][attr_name] = {
                                        "type": "jnp.ndarray",
                                        "dimensions": dimensions,
                                        "dtype": dtype,
                                    }

                            elif value.func.attr in ["int32", "float64"]:
                                if isinstance(value.args[0], ast.Constant):
                                    class_info["attributes"][attr_name] = {
                                        "type": type_map[value.func.attr],
                                        "value": value.args[0].value,
                                        "dtype": value.func.attr,
                                    }
                                elif isinstance(value.args[0], ast.Attribute):
                                    class_info["attributes"][attr_name] = {
                                        "type": type_map[value.func.attr],
                                        "value": value.args[0].attr,
                                        "dtype": value.func.attr,
                                    }
                                elif isinstance(value.args[0], ast.BinOp):
                                    class_info["attributes"][attr_name] = {
                                        "type": type_map[value.func.attr],
                                        "value": ast.unparse(value.args[0]),
                                        "dtype": value.func.attr,
                                    }

                            elif value.func.attr in ["bool"] and isinstance(
                                value.args[0], ast.Constant
                            ):
                                class_info["attributes"][attr_name] = {
                                    "type": value.func.attr,
                                    "value": value.args[0].value,
                                    "dtype": value.func.attr,
                                }
                        elif isinstance(value, ast.Attribute):
                            if value.attr in class_info["attributes"]:
                                resolved = class_info["attributes"][value.attr]

                                class_info["attributes"][attr_name] = {
                                    "type": resolved["type"],
                                    "value": resolved.get("value"),
                                    "dtype": resolved.get("dtype"),
                                    "dep_value": value.attr,
                                }
                            else:
                                raise NotImplementedError(
                                    f"The instance of where the value is not \
                                    already present in the cls_info: {ast.unparse(ast.fix_missing_locations(value))}"
                                )
            # Check if it's a method definition (excluding __init__)
            if isinstance(node, ast.FunctionDef):
                if node.name not in ["__init__", "wrapper", "timer"]:
                    method_args = [
                        arg.arg for arg in node.args.args if arg.arg != "self"
                    ]

                    # Find local arrays
                    local_arrays = {}
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):  # local variable
                                    var_name = target.id
                                    value = stmt.value
                                    if isinstance(value, ast.Call) and isinstance(
                                        value.func, ast.Attribute
                                    ):
                                        if value.func.attr in [
                                            "zeros",
                                            "ones",
                                            "array",
                                        ]:
                                            dimensions = []
                                            if value.args:
                                                if isinstance(value.args[0], ast.Tuple):
                                                    for dim in value.args[0].elts:
                                                        if isinstance(
                                                            dim, ast.Constant
                                                        ):
                                                            dimensions.append(dim.value)
                                                        elif isinstance(
                                                            dim, ast.Attribute
                                                        ):
                                                            dimensions.append(dim.attr)
                                                        elif isinstance(dim, ast.BinOp):
                                                            dimensions.append(
                                                                ast.unparse(dim)
                                                            )
                                                elif isinstance(
                                                    value.args[0], ast.Constant
                                                ):
                                                    dimensions.append(
                                                        value.args[0].value
                                                    )
                                                elif isinstance(
                                                    value.args[0], ast.Attribute
                                                ):
                                                    dimensions.append(
                                                        value.args[0].attr
                                                    )

                                            dtype = None
                                            if value.keywords:
                                                for kw in value.keywords:
                                                    if kw.arg == "dtype":
                                                        if isinstance(
                                                            kw.value, ast.Attribute
                                                        ):
                                                            dtype = kw.value.attr
                                                        elif isinstance(
                                                            kw.value, ast.Name
                                                        ):
                                                            dtype = kw.value.id

                                            local_arrays[var_name] = {
                                                "dimensions": dimensions,
                                                "dtype": dtype,
                                                "type": "jnp.ndarray",
                                            }

                    class_info["methods"][node.name] = {
                        "args": method_args,
                        "local_arr": local_arrays,
                    }

        cls_info[class_def.name] = class_info

    return cls_info


def collect_reads_before_def(
    stmts: list, already_defined: set[str], loop_carried: bool | None = False
) -> set[str]:
    """
    Identify variables that require state promotion across control-flow
    boundaries.

    The analysis walks a sequence of statements and detects variables whose
    values must be preserved across branches or loop iterations. Such
    variables typically need special handling during lowering to functional
    frameworks such as JAX, where mutable state is represented explicitly.

    A variable is classified as stateful if one or more of the following
    conditions hold:

    1. The variable is read before it is definitely assigned in the current
       scope.
    2. The variable is assigned in only some branches of a conditional
       nested within a loop and subsequently read within the loop body
       (loop-carried state).
    3. The variable is written inside a loop body and read after the loop
       exits (accumulator or loop-produced value pattern).

    Parameters
    ----------
    stmts : list[ast.stmt]
        Sequence of statements to analyze.
    already_defined : set[str]
        Variables known to be defined upon entry to the analyzed scope.
    loop_carried : bool, optional
        Indicates whether the current analysis is occurring within a loop
        body. This parameter is propagated during recursive calls and may
        be used by future extensions of the analysis. Default is ``False``.

    Returns
    -------
    set[str]
        Names of variables that require state promotion due to read-before-
        definition, loop-carried dependencies, or writes that escape loop
        scope.

    Notes
    -----
    The analysis performs a conservative approximation using AST traversal
    and does not construct a full control-flow graph(CFG).

    Built-in names and framework-specific identifiers such as ``np``,
    ``jnp``, and ``self`` are excluded from consideration.

    The analysis recognizes:

    * Standard assignments (``Assign``)
    * Augmented assignments (``AugAssign``)
    * Conditional branches (``If``)
    * ``for`` loops
    * ``while`` loops
    * Return statements
    * Expression statements

    Examples
    --------
    A read-before-definition pattern:

    >>> x = y + 1
    >>> collect_reads_before_def(...)
    {'y'}

    A loop-carried accumulator:

    >>> total = 0
    >>> for i in range(n):
    ...     total = total + values[i]
    >>> collect_reads_before_def(...)
    {'total'}

    A value produced inside a loop and consumed afterward:

    >>> for i in range(n):
    ...     result = values[i]
    >>> print(result)
    >>> collect_reads_before_def(...)
    {'result'}
    """
    stateful: set[str] = set()
    defined: set[str] = set(already_defined)

    BUILTINS = {
        "range",
        "len",
        "int",
        "float",
        "bool",
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "print",
        "enumerate",
        "zip",
        "map",
        "filter",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "True",
        "False",
        "None",
        "np",
        "jnp",
        "self",
    }

    def is_user_var(name: str) -> bool:
        return name not in BUILTINS and not name.startswith("_")

    def record_reads(expr, defined_at_point: set[str]):
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if is_user_var(node.id) and node.id not in defined_at_point:
                    stateful.add(node.id)

    def definitely_assigned_by(stmts: list) -> set[str]:
        """Variables definitely assigned on ALL paths through stmts."""
        defs = set()
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defs.add(t.id)
            elif isinstance(stmt, ast.If):
                body_defs = definitely_assigned_by(stmt.body)
                orelse_defs = (
                    definitely_assigned_by(stmt.orelse) if stmt.orelse else set()
                )
                defs |= body_defs & orelse_defs
        return defs

    def collect_written_in_loop(body: list) -> set[str]:
        """
        Collect every variable name that appears as an assignment target
        anywhere inside the loop body (including nested scopes).
        """
        written: set[str] = set()
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and is_user_var(t.id):
                        written.add(t.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and is_user_var(node.target.id):
                    written.add(node.target.id)
        return written

    def process_stmts(stmts: list, defined: set[str]) -> set[str]:
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Assign):
                record_reads(stmt.value, defined)
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined = defined | {t.id}

            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    record_reads(stmt.target, defined)
                record_reads(stmt.value, defined)
                if isinstance(stmt.target, ast.Name):
                    defined = defined | {stmt.target.id}

            elif isinstance(stmt, ast.If):
                record_reads(stmt.test, defined)
                body_defined = process_stmts(stmt.body, set(defined))
                orelse_defined = (
                    process_stmts(stmt.orelse, set(defined))
                    if stmt.orelse
                    else set(defined)
                )
                defined = body_defined & orelse_defined

            elif isinstance(stmt, ast.For):
                record_reads(stmt.iter, defined)

                loop_vars: set[str] = set()
                if isinstance(stmt.target, ast.Name):
                    loop_vars.add(stmt.target.id)
                elif isinstance(stmt.target, ast.Tuple):
                    for elt in stmt.target.elts:
                        if isinstance(elt, ast.Name):
                            loop_vars.add(elt.id)

                defined_at_loop_entry = defined | loop_vars
                body_certain = definitely_assigned_by(stmt.body)

                # Recursive read-before-def inside the loop body
                inner_stateful = collect_reads_before_def(
                    stmt.body, defined_at_loop_entry, loop_carried=True
                )
                stateful.update(inner_stateful)

                # Partial-branch
                # Vars assigned in SOME but not ALL conditional branches inside
                # the loop body are loop-carried if also read anywhere in the body.
                for node in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
                    if isinstance(node, ast.If):
                        body_defs = definitely_assigned_by(node.body)
                        orelse_defs = (
                            definitely_assigned_by(node.orelse)
                            if node.orelse
                            else set()
                        )
                        all_defs = body_defs & orelse_defs
                        some_defs = body_defs | orelse_defs
                        partial = some_defs - all_defs
                        for var in partial:
                            if is_user_var(var) and var in defined_at_loop_entry:
                                if _is_read_in(stmt.body, var):
                                    stateful.add(var)

                # Write-inside-loop
                subsequent_stmts = stmts[i + 1 :]
                if subsequent_stmts:
                    written_in_loop = collect_written_in_loop(stmt.body)

                    # Guard A: collect variables that are re-defined as a for-loop
                    # target in subsequent statements — those are definitions, not reads.
                    redefined_as_loop_var: set[str] = set()
                    for subsequent_stmt in subsequent_stmts:
                        if isinstance(subsequent_stmt, ast.For):
                            if isinstance(subsequent_stmt.target, ast.Name):
                                redefined_as_loop_var.add(subsequent_stmt.target.id)
                            elif isinstance(subsequent_stmt.target, ast.Tuple):
                                for elt in subsequent_stmt.target.elts:
                                    if isinstance(elt, ast.Name):
                                        redefined_as_loop_var.add(elt.id)

                    # Guard B: check if the variable's assigned value inside the loop
                    # depends on the loop variable — if not, it's loop-invariant and
                    # does not need promotion.
                    loop_var_ids: set[str] = set()
                    if isinstance(stmt.target, ast.Name):
                        loop_var_ids.add(stmt.target.id)
                    elif isinstance(stmt.target, ast.Tuple):
                        for elt in stmt.target.elts:
                            if isinstance(elt, ast.Name):
                                loop_var_ids.add(elt.id)

                    def _depends_on_loop_var(
                        body: list, var: str, loop_vars: set[str]
                    ) -> bool:
                        """
                        Returns True if any assignment to `var` in the loop body has a RHS
                        that references one of the loop_vars (e.g. ji), meaning its value
                        actually varies per iteration.
                        """
                        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
                            if isinstance(node, ast.Assign):
                                for t in node.targets:
                                    if isinstance(t, ast.Name) and t.id == var:
                                        for rhs_node in ast.walk(node.value):
                                            if (
                                                isinstance(rhs_node, ast.Name)
                                                and rhs_node.id in loop_vars
                                            ):
                                                return True
                        return False

                    for var in written_in_loop:
                        if not is_user_var(var):
                            continue
                        # Guard A: skip if subsequent for-loop redefines it as its own iterator
                        if var in redefined_as_loop_var:
                            continue
                        # Guard B: skip if its assigned value never depends on the loop variable
                        if not _depends_on_loop_var(stmt.body, var, loop_var_ids):
                            continue
                        if _is_read_in(subsequent_stmts, var):
                            stateful.add(var)

                defined = defined | body_certain

            elif isinstance(stmt, ast.While):
                record_reads(stmt.test, defined)
                inner_stateful = collect_reads_before_def(
                    stmt.body, defined, loop_carried=True
                )
                stateful.update(inner_stateful)

            elif isinstance(stmt, ast.Return):
                if stmt.value:
                    record_reads(stmt.value, defined)

            elif isinstance(stmt, ast.Expr):
                record_reads(stmt.value, defined)

        return defined

    process_stmts(stmts, defined)
    return stateful


def _is_read_in(stmts_or_node: list[ast.AST] | ast.AST, var: str) -> bool:
    """
    Determine whether a variable is read within an AST subtree.

    A read is defined as an occurrence of an ``ast.Name`` node whose context
    is ``ast.Load`` and whose identifier matches the specified variable.

    Parameters
    ----------
    stmts_or_node : list[ast.AST] or ast.AST
        Either a list of statements or a single AST node to search.
    var : str
        Variable name to look for.

    Returns
    -------
    bool
        ``True`` if the variable is read anywhere within the supplied AST
        subtree, otherwise ``False``.

    Notes
    -----
    When a list of statements is provided, the statements are wrapped in a
    temporary ``ast.Module`` before traversal.

    Examples
    --------
    >>> tree = ast.parse("y = x + 1")
    >>> _is_read_in(tree, "x")
    True

    >>> _is_read_in(tree, "z")
    False
    """
    if isinstance(stmts_or_node, list):
        root = ast.Module(body=stmts_or_node, type_ignores=[])
    else:
        root = stmts_or_node
    for node in ast.walk(root):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == var
        ):
            return True
    return False


REDUCTION_FUNCS = {"sum", "mean", "prod"}
SPECIAL_REDUCTIONS = {"argmax", "argmin", "amax", "amin"}

ELEMENTWISE_FUNCS = {
    "abs",
    "sqrt",
    "log",
    "exp",
    "sin",
    "cos",
    "tan",
    "clip",
    "maximum",
    "minimum",
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    "equal",
    "not_equal",
    "where",
}


class ReductionHandler:
    """
    Infer and normalize reduction operations in transformed JAX ASTs.

    This analysis tracks tensor shapes, vectorization context, and parent
    expression relationships in order to automatically determine the
    appropriate reduction axes for operations such as ``sum``, ``mean``,
    ``max``, and ``min``.

    The handler performs several tasks:

    * Infers reduction axes from array shapes and slicing operations.
    * Propagates dimensionality through elementwise expressions.
    * Accounts for vectorized dimensions introduced during lowering.
    * Inserts or rewrites ``axis=`` arguments when necessary.
    * Determines whether ``keepdims=True`` is required to preserve
      broadcasting semantics in downstream consumers.

    Parameters
    ----------
    cls_info : dict
        Class metadata produced by :func:`get_class_info_from_ast`.
    cls_name : str
        Name of the class currently being transformed.
    func_name : str, optional
        Name of the method currently being analyzed.
    vectorization_axis : dict, optional
        Mapping of loop variables to vectorized tensor axes.

    Notes
    -----
    The handler maintains a parent stack during AST traversal so that
    reduction consumers can be inspected when determining whether
    dimensionality-preserving reductions are required.
    """

    def __init__(
        self,
        cls_info: dict,
        cls_name: str,
        func_name: str | None = None,
        vectorization_axis: dict | None = None,
    ) -> None:
        self.vectorization_axis = vectorization_axis or {}
        self.loop_info = {}
        self._parent_stack = []
        self.cls_info = cls_info
        self.cls_name = cls_name
        self.func_name = func_name
        self.func_input_dim = None
        self.dynamic_variable_lift = None

    def process_call(self, node: ast.Call) -> ast.AST:
        """
        Analyse and rewrite a reduction call.

        Determines whether the supplied call represents a reduction via
        :meth:`_is_reduction_call`. If so, reduction axes are inferred using
        :meth:`_infer_axes` and injected into the call through
        :meth:`_inject_axis`.

        Parameters
        ----------
        node : ast.Call
            Call expression to analyse.

        Returns
        -------
        ast.AST
            The original node if no reduction is detected or no axes can be
            inferred, otherwise a modified call node containing inferred
            ``axis`` and possibly ``keepdims`` arguments.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            is_reduction, _ = self._is_reduction_call(node)
            if not is_reduction:
                return node

            axes = self._infer_axes(node)

            if axes is None:
                return node

            return self._inject_axis(node, axes)
        except Exception:
            raise

    def push_parent(self, node: ast.AST) -> None:
        """
        Push a parent node onto the traversal stack.

        The parent stack stored in :attr:`_parent_stack` is later consulted by
        :meth:`_get_consumer_shapes` and :meth:`_needs_keepdims` to determine
        how reduction results are consumed.

        Parameters
        ----------
        node : ast.AST
            AST node to register as the current parent context.
        """
        self._parent_stack.append(node)

    def pop_parent(self) -> None:
        """
        Remove the most recent parent node from the traversal stack.

        Updates :attr:`_parent_stack` after traversal exits the corresponding
        parent expression.
        """
        self._parent_stack.pop()

    def _remap_axes_after_slice(self, node: ast.AST, axes: set) -> set:
        """
        Remap tensor axes after slicing.

        Converts axis indices from the original tensor space into the axis
        numbering of the sliced tensor. Dimensions removed by scalar indexing
        are discarded, while dimensions preserved by slicing are renumbered.

        Parameters
        ----------
        node : ast.AST
            Slice expression to analyse.
        axes : set
            Axes defined in the original tensor coordinate system.

        Returns
        -------
        set
            Axes expressed in the coordinate system of the sliced tensor.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if not isinstance(node, ast.Subscript):
                return axes

            slice_node = node.slice
            if not isinstance(slice_node, ast.Tuple):
                return axes

            new_axes = set()
            new_pos = 0

            for i, idx in enumerate(slice_node.elts):
                if isinstance(idx, ast.Slice):
                    # dimension preserved
                    if i in axes:
                        new_axes.add(new_pos)
                    new_pos += 1
                else:
                    # dimension removed -> skip
                    continue

            return new_axes
        except Exception:
            raise

    def _infer_axes(self, node: ast.AST) -> set | None:
        """
        Infer reduction axes for a reduction operand.

        Axes are extracted from the first reduction argument via
        :meth:`_extract_axes`. Any vectorized dimensions recorded in
        :attr:`vectorization_axis` or :attr:`dynamic_variable_lift` are removed
        from the inferred reduction set.

        Parameters
        ----------
        node : ast.AST
            Reduction call whose operand should be analysed.

        Returns
        -------
        set or None
            Inferred reduction axes, or ``None`` if no valid reduction axes can
            be determined.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if not node.args:
                return None

            arg0 = node.args[0]
            axes = set()
            axes = self._extract_axes(arg0)

            # Vectorization handling
            vectorized_axes = set()
            for axes_set in self.vectorization_axis.values():
                vectorized_axes |= set(axes_set)

            if self.dynamic_variable_lift:
                for info in self.dynamic_variable_lift.values():
                    vectorized_axes |= set(info["batched_axis"])

            axes -= vectorized_axes

            return axes or None
        except Exception:
            raise

    def _get_array_info(self, var_name: str) -> list | None:
        """
        Resolve shape information for an array variable.

        Array metadata is searched in method-local variables, function inputs,
        and finally class attributes using information stored in
        :attr:`cls_info`.

        Parameters
        ----------
        var_name : str
            Variable name to resolve.

        Returns
        -------
        list or None
            Dimension list for the array if available, otherwise ``None``.

        Raises
        ------
        ValueError
            If the same variable name is found in multiple methods with
            conflicting dimension metadata.
        Exception
            Re-raises any unexpected error.
        """
        try:
            cls = self.cls_info.get(self.cls_name, {})
            methods = cls.get("methods", {})
            found_dims = None

            if self.func_name and self.func_name in methods:
                # Fast path: known method
                local_arr = methods[self.func_name].get("local_arr", {})
                if var_name in local_arr:
                    info = local_arr[var_name]
                    if info.get("type") == "jnp.ndarray":
                        return info.get("dimensions", [])

                if var_name in self.func_input_dim:
                    dims = self.func_input_dim.get(var_name, [])
                    if dims != []:
                        return dims

            else:
                for _, method in methods.items():
                    local_arr = method.get("local_arr", {})
                    if var_name in local_arr:
                        info = local_arr[var_name]
                        if info.get("type") == "jnp.ndarray":
                            dims = info.get("dimensions", [])
                            if found_dims is None:
                                found_dims = dims
                            elif found_dims != dims:
                                raise ValueError(
                                    f"Ambiguous array '{var_name}' found in multiple methods "
                                    f"with different dimensions: {found_dims} vs {dims}"
                                )

                if found_dims is not None:
                    return found_dims

            attrs = cls.get("attributes", {})
            if var_name in attrs:
                info = attrs[var_name]
                if info.get("type") == "jnp.ndarray":
                    return info.get("dimensions", [])

            return None
        except Exception:
            raise

    def _extract_axes(self, node: ast.AST) -> set:
        """
        Extract candidate reduction axes from an expression.

        Traverses array references, slices, elementwise operations, and
        arithmetic expressions to determine which tensor dimensions remain
        available for reduction.

        Parameters
        ----------
        node : ast.AST
            Expression to analyse.

        Returns
        -------
        set
            Set of inferred tensor axes.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            axes = set()

            if isinstance(node, ast.Subscript):
                sub = node.slice
                raw_axes = set()
                if isinstance(sub, ast.Tuple):
                    for i, idx in enumerate(sub.elts):
                        if isinstance(idx, ast.Slice):
                            raw_axes.add(i)
                elif isinstance(sub, ast.Slice):
                    if sub.lower is None and sub.upper is None:
                        raw_axes.add(0)

                # Remap raw_axes (original tensor positions) → sliced tensor positions
                if isinstance(sub, ast.Tuple):
                    new_axes = set()
                    new_pos = 0
                    for i, idx in enumerate(sub.elts):
                        if isinstance(idx, ast.Slice):
                            if i in raw_axes:
                                new_axes.add(new_pos)
                            new_pos += 1
                        elif isinstance(idx, ast.Constant) and idx.value is None:
                            # newaxis: adds a dimension but we don't want to reduce over it
                            new_pos += 1
                        # scalar index: dimension removed, new_pos not incremented
                    return new_axes

                return raw_axes

            if isinstance(node, ast.BinOp):
                left_axes = self._extract_axes(node.left)
                right_axes = self._extract_axes(node.right)
                axes |= left_axes | right_axes
                return axes

            if isinstance(node, ast.UnaryOp):
                return self._extract_axes(node.operand)

            if isinstance(node, ast.Call):
                name = self._get_func_name(node.func)
                if name in ELEMENTWISE_FUNCS:
                    axes = set()
                    for arg in node.args:
                        axes |= self._extract_axes(arg)
                    return axes

            if isinstance(node, ast.Name):
                dims = self._get_array_info(node.id)
                if dims:
                    return set(range(len(dims)))

            if isinstance(node, ast.Attribute):
                dims = self._get_array_info(node.attr)
                if dims:
                    return set(range(len(dims)))

            return axes
        except Exception:
            raise

    def _axis_matches(self, axis_node: ast.AST, axes: set) -> bool:
        """
        Check whether an axis argument matches an inferred axis set.

        Parameters
        ----------
        axis_node : ast.AST
            Existing ``axis`` argument from a reduction call.
        axes : set
            Inferred reduction axes.

        Returns
        -------
        bool
            ``True`` if the supplied axis specification exactly matches
            ``axes``, otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if isinstance(axis_node, ast.Constant):
                return axis_node.value in axes

            if isinstance(axis_node, ast.Tuple):
                vals = {
                    elt.value for elt in axis_node.elts if isinstance(elt, ast.Constant)
                }
                return vals == axes

            return False
        except Exception:
            raise

    def _contains_node(self, tree: ast.AST, target: ast.AST) -> bool:
        """
        Determine whether a subtree contains a target node.

        Parameters
        ----------
        tree : ast.AST
            AST subtree to search.
        target : ast.AST
            Node to locate.

        Returns
        -------
        bool
            ``True`` if ``target`` is present within ``tree``, otherwise
            ``False``.
        """
        for n in ast.walk(tree):
            if n is target:
                return True
        return False

    def _needs_keepdims(self, reduction_node: ast.AST, axes: set) -> bool:
        """
        Determine whether a reduction requires ``keepdims=True``.

        Consumer expressions are inspected through
        :meth:`_get_consumer_shapes`. If removing the reduction dimensions would
        break broadcasting compatibility, the reduction is marked as requiring
        dimension preservation.

        Parameters
        ----------
        reduction_node : ast.AST
            Reduction call being analysed.
        axes : set
            Reduction axes.

        Returns
        -------
        bool
            ``True`` if ``keepdims=True`` is required to preserve broadcasting
            semantics, otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            arg = reduction_node.args[0]
            shape = self._get_tensor_shape(arg)

            if not shape:
                return False

            no_keep, keep = self._compute_reduction_shapes(shape, axes)
            consumer_shapes = self._get_consumer_shapes(reduction_node)

            for cshape in consumer_shapes:
                if not self._broadcastable(no_keep, cshape):
                    return True

            return False
        except Exception:
            raise

    def _get_tensor_shape(self, node: ast.AST) -> list | None:
        """
        Infer the tensor shape represented by an expression.

        Supports direct array references, attributes, slices, elementwise
        operations, and reductions whose output shape can be propagated from
        their inputs.

        Parameters
        ----------
        node : ast.AST
            Expression whose shape should be inferred.

        Returns
        -------
        list or None
            Inferred shape dimensions, or ``None`` if the shape cannot be
            determined.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if isinstance(node, ast.Name):
                dims = self._get_array_info(node.id) or []
                return dims

            if isinstance(node, ast.Attribute):
                dims = self._get_array_info(node.attr) or []
                return dims

            if isinstance(node, ast.Subscript):
                # we might have three cases,
                # 1. arr[j][:, None] here we need add upon the exisiting dimension
                # + the broadcasted(info in the vectorization_context)
                # 2. arr[:, None] same as here
                # 3. arr[i,j] or arr <-- easy to handle through _get_tensor_shape
                # return self._get_tensor_shape(node.value)
                base_shape = self._get_tensor_shape(node.value)
                if not base_shape:
                    return None

                if isinstance(node.slice, ast.Tuple):
                    new_shape = []
                    for dim, idx in zip(base_shape, node.slice.elts):
                        if isinstance(idx, ast.Slice):
                            new_shape.append(dim)
                        # scalar index -> removed
                    return new_shape

                return base_shape

            if isinstance(node, ast.Call):
                name = self._get_func_name(node.func)

                if name in ELEMENTWISE_FUNCS or name in REDUCTION_FUNCS and node.args:
                    return self._get_tensor_shape(node.args[0])

            return None
        except Exception:
            raise

    def _compute_reduction_shapes(self, shape: list, axes: set) -> tuple[list, list]:
        """
        Compute output shapes with and without dimension preservation.

        Parameters
        ----------
        shape : list
            Original tensor shape.
        axes : set
            Reduction axes.

        Returns
        -------
        tuple[list, list]
            A pair ``(no_keep, keep)`` where ``no_keep`` is the shape produced
            by a standard reduction and ``keep`` is the shape produced when
            ``keepdims=True``.
        """
        no_keep = [d for i, d in enumerate(shape) if i not in axes]
        keep = [1 if i in axes else d for i, d in enumerate(shape)]

        return no_keep, keep

    def _get_consumer_shapes(self, reduction_node: ast.AST) -> list[list]:
        """
        Collect shapes of expressions consuming a reduction result.

        Walks the parent contexts stored in :attr:`_parent_stack` to identify
        operations that use the reduction output. These shapes are later used by
        :meth:`_needs_keepdims` when evaluating broadcasting requirements.

        Parameters
        ----------
        reduction_node : ast.AST
            Reduction expression whose consumers should be inspected.

        Returns
        -------
        list[list]
            Shapes of downstream consumer expressions.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            shapes = []

            for parent in reversed(self._parent_stack[:-1]):
                # elementwise binary op
                if isinstance(parent, ast.BinOp):
                    if self._contains_node(parent.left, reduction_node):
                        other = parent.right
                    elif self._contains_node(parent.right, reduction_node):
                        other = parent.left
                    else:
                        continue

                    shape = self._get_tensor_shape(other)

                    if shape:
                        shapes.append(shape)

                # comparisons (>, <, ==)
                elif isinstance(parent, ast.Compare):
                    for operand in [parent.left] + parent.comparators:
                        if operand is reduction_node:
                            continue

                        shape = self._get_tensor_shape(operand)
                        if shape:
                            shapes.append(shape)

                # elementwise call (greater, maximum, etc.)
                elif isinstance(parent, ast.Call):
                    name = self._get_func_name(parent.func)

                    if name in ELEMENTWISE_FUNCS:
                        for arg in parent.args:
                            if arg is reduction_node:
                                continue

                            shape = self._get_tensor_shape(arg)

                            if shape:
                                shapes.append(shape)

                # assignment determines final tensor
                elif isinstance(parent, ast.Assign):
                    target = parent.targets[0]

                    shape = self._get_tensor_shape(target)

                    if shape:
                        shapes.append(shape)

                    else:
                        # infer shape from RHS expression
                        rhs_shape = self._get_tensor_shape(parent.value)

                        if rhs_shape:
                            shapes.append(rhs_shape)

                    break
            return shapes
        except Exception:
            raise

    def _broadcastable(self, a: list, b: list) -> bool:
        """
        Determine whether two shapes are broadcast-compatible.

        Applies NumPy broadcasting rules by left-padding the shorter shape with
        singleton dimensions before comparison.

        Parameters
        ----------
        a : list
            First shape.
        b : list
            Second shape.

        Returns
        -------
        bool
            ``True`` if the shapes can be broadcast together, otherwise
            ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            a = list(a)
            b = list(b)

            while len(a) < len(b):
                a.insert(0, 1)

            while len(b) < len(a):
                b.insert(0, 1)

            for x, y in zip(a, b):
                if x == y or x == 1 or y == 1:
                    continue
                return False

            return True
        except Exception:
            raise

    def _inject_axis(self, node: ast.AST, axes: set) -> ast.AST:
        """
        Insert or rewrite reduction axis arguments.

        Updates the reduction call so that its ``axis`` argument matches the
        axes inferred by :meth:`_infer_axes`. If required by
        :meth:`_needs_keepdims`, a ``keepdims=True`` argument is also added.

        Parameters
        ----------
        node : ast.AST
            Reduction call to modify.
        axes : set
            Inferred reduction axes.

        Returns
        -------
        ast.AST
            Modified reduction call node.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        # Design rule:
        # - If axis is not explicitly provided, treat reduction as scalar (axis=None)
        # - Inferred axes are only used to validate or rewrite explicit axis arguments

        # TODO: Need to correct the case when the axis has the same shape as the argument of the
        # of the reduction node

        try:
            # Probably means that keyword = []
            if not any(kw.arg == "axis" for kw in node.keywords) and axes is None:
                node.keywords.append(
                    ast.keyword(arg="axis", value=ast.Constant(value=None))
                )
                return node

            keepdims = self._needs_keepdims(node, axes)
            has_axis = False
            # if we already have a matching axis value we resend
            for kw in node.keywords:
                if kw.arg == "axis" and self._axis_matches(kw.value, axes):
                    return node

            if len(axes) == 1:
                axis_value = ast.Constant(next(iter(axes)))
            else:
                axis_value = ast.Tuple(
                    elts=[ast.Constant(a) for a in sorted(axes)],
                    ctx=ast.Load(),
                )

            for kw in node.keywords:
                if kw.arg == "axis":
                    has_axis = True

                    # only update if different
                    if not self._axis_matches(kw.value, axes):
                        kw.value = axis_value
                    break

            if not has_axis:
                node.keywords.append(ast.keyword(arg="axis", value=axis_value))

            if keepdims:
                has_keepdims = any(kw.arg == "keepdims" for kw in node.keywords)

                if not has_keepdims:
                    node.keywords.append(
                        ast.keyword(arg="keepdims", value=ast.Constant(True))
                    )

            return node
        except Exception:
            raise

    def _is_reduction_call(self, node: ast.AST) -> tuple[bool, bool]:
        """
        Determine whether a call represents a reduction operation.

        Checks the function name against :data:`REDUCTION_FUNCS` and
        :data:`SPECIAL_REDUCTIONS`.

        Parameters
        ----------
        node : ast.AST
            Call expression to inspect.

        Returns
        -------
        tuple[bool, bool]
            A pair ``(is_reduction, is_special)`` indicating whether the call
            is a reduction and whether it belongs to the special reduction
            category.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            if not isinstance(node, ast.Call):
                return False, False

            name = self._get_func_name(node.func)
            if name in SPECIAL_REDUCTIONS:
                return True, True
            if name in REDUCTION_FUNCS:
                return True, False

            return False, False
        except Exception:
            raise

    def _get_func_name(self, func: ast.AST) -> str | None:
        """
        Extract the name of a callable expression.

        Supports both attribute access and direct name references.

        Parameters
        ----------
        func : ast.AST
            Callable expression.

        Returns
        -------
        str or None
            Function name if it can be resolved, otherwise ``None``.
        """
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None


class MaybeAddIndexTransformer(ast.NodeTransformer):
    """
    Insert broadcast and indexing operations required for vectorized
    expressions.

    This transformer analyzes tensor ranks and vectorization metadata to
    determine whether operands participating in an expression require
    additional indexing (``:``, ``None``) to achieve compatible shapes.

    The transformation is primarily used after vectorization analysis to
    reconcile mismatched ranks and broadcasting semantics introduced by
    lifted scalars, batched dimensions, and loop-to-vector conversions.

    Parameters
    ----------
    cls_info : dict
        Class metadata extracted from the source AST.
    cls_name : str
        Name of the class currently being transformed.
    func_name : str
        Name of the method currently being transformed.
    ranks : dict
        Mapping from AST nodes to inferred tensor ranks.
    target_rank : int
        Desired rank after promotion.
    vect_context : dict
        Vectorization analysis context containing loop and axis metadata.
    local_defined_variables : dict
        Locally promoted scalar variables and their inferred dimensions.
    func_input_dim : dict
        Shape metadata for function arguments.
    dynamic_variable_lift : dict
        Variables promoted from scalar state to vectorized state.
    inferred_ranks : dict
        Additional rank information inferred during transformation.

    Notes
    -----
    The transformer preserves source locations and only inserts indexing
    operations when rank promotion is required. Existing vectorized axes
    are respected when constructing broadcast dimensions.
    """

    def __init__(
        self,
        cls_info: dict,
        cls_name: str,
        func_name: str,
        ranks: dict,
        target_rank: int,
        vect_context: dict,
        local_defined_variables: dict,
        func_input_dim: dict,
        dynamic_variable_lift: dict,
        inferred_ranks: dict,
    ) -> None:
        super().__init__()
        self.cls_info = cls_info
        self.cls_name = cls_name
        self.func_name = func_name
        self.ranks = ranks
        self.target_rank = target_rank
        self.vect_context = vect_context
        self.local_defined_var = local_defined_variables
        self.elts_list = None
        self.func_input_dim = func_input_dim
        self.dynamic_variable_lift = dynamic_variable_lift
        self.inferred_ranks = inferred_ranks

    def visit(self, node: ast.AST) -> ast.AST:
        """
        Visit an AST node and apply rank-promotion indexing.

        If *node* has a rank lower than :attr:`target_rank`, and its active
        dimensions participate in vectorized loop axes, a subscript of the
        form ``[:, None, ...]`` is inserted to expose the appropriate
        broadcast dimensions.

        Parameters
        ----------
        node : ast.AST
            Node being visited.

        Returns
        -------
        ast.AST
            Either the original node or a transformed
            :class:`ast.Subscript` introducing broadcast dimensions.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        See Also
        --------
        :meth:`get_active_dims`
            Determines which dimensions remain active after indexing.
        :meth:`_is_lifted`
            Detects variables already promoted to vectorized state.
        """
        try:
            node = super().visit(node)
            rank = self.ranks.get(node)

            if not rank or rank >= self.target_rank or self._is_lifted(node):
                return node

            dimensions = self.get_active_dims(node)
            if not dimensions:
                return node

            loop_info = self.vect_context["loop_info"]
            vectorization_axis = self.vect_context.get("vectorization_axis", {})

            if not vectorization_axis:
                return node
            # Start with no-op indexing
            elts = [ast.Constant(value=None)] * self.target_rank
            modified = False

            for loop_dim, loop_var in loop_info.items():
                if loop_dim not in dimensions:
                    continue

                dim_index = dimensions.index(loop_dim)
                vect_axis = vectorization_axis.get(loop_var, [])

                if dim_index not in vect_axis:
                    continue

                elts[dim_index] = ast.Slice()
                modified = True

            if modified:
                self.elts_list = elts
            if not modified:
                return node

            sub_expr = ast.Subscript(
                value=node,
                slice=ast.Tuple(elts=elts, ctx=ast.Load()),
                ctx=ast.Load(),
            )

            return ast.copy_location(sub_expr, node)

        except Exception:
            raise

    def is_full_slice(self, s: ast.Slice) -> bool:
        """
        Determine whether a slice represents ``:``.

        Parameters
        ----------
        s : ast.Slice
            Slice node to inspect.

        Returns
        -------
        bool
            ``True`` if the slice has no lower bound, upper bound,
            or step specification; otherwise ``False``.
        """
        return (
            isinstance(s, ast.Slice)
            and s.lower is None
            and s.upper is None
            and s.step is None
        )

    def _is_lifted(self, node: ast.AST) -> bool:
        """
        Determine whether an expression refers to a lifted variable.

        Checks whether *node* corresponds to a variable recorded in
        :attr:`dynamic_variable_lift`, including direct names,
        ``self`` attributes, and indexed accesses derived from lifted
        variables.

        Parameters
        ----------
        node : ast.AST
            Expression node to inspect.

        Returns
        -------
        bool
            ``True`` if the expression originates from a dynamically
            lifted variable; otherwise ``False``.

        Notes
        -----
        Lifted variables already carry vectorization semantics and
        therefore should not receive additional promotion from
        :meth:`visit`.
        """
        if isinstance(node, ast.Name):
            return node.id in self.dynamic_variable_lift

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr in self.dynamic_variable_lift

        if isinstance(node, ast.Subscript):
            return self._is_lifted(node.value)

        return False

    # NOTE: If the ranks is None, which means that it's either a scalar that
    # get got vectorized and has a new dimsension thus we can use the local_defined_var
    # TO search for it check it's vectorized along the vectorized axis
    # If it's directly zero which means it's an attribute scalar, thus we don't modify
    # them contrary to the scalars of that are created a temporarly during the
    # code and might get vectorized.
    def visit_BinOp(self, node: ast.BinOp):
        """
        Promote operands of a binary expression to compatible ranks.

        Analyzes both operands of *node* to determine their inferred rank,
        vectorization status, and active vectorized axes. When required,
        additional indexing expressions are inserted so that broadcasting
        produces the intended result shape.

        Examples include:

        * Aligning operands vectorized along different axes.
        * Promoting lower-rank operands to match higher-rank operands.
        * Constructing outer-product style broadcasts.
        * Preserving vectorized dimensions introduced by lifted scalars.

        Parameters
        ----------
        node : ast.BinOp
            Binary operation to transform.

        Returns
        -------
        ast.BinOp
            The transformed binary operation with any required
            broadcasting indices inserted.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        Vectorization metadata is obtained from
        :attr:`vect_context`, :attr:`local_defined_var`,
        :attr:`dynamic_variable_lift`, and :attr:`inferred_ranks`.

        See Also
        --------
        :meth:`get_active_dims`
            Computes active dimensions for arrays.
        :meth:`extract_names`
            Extracts dimension-variable dependencies.
        :meth:`_is_lifted`
            Detects lifted scalar variables.
        """
        try:
            node = self.generic_visit(node)

            if not hasattr(node, "left") or not hasattr(node, "right"):
                return None

            if not self.vect_context:
                return node

            loop_info = self.vect_context["loop_info"]
            vectorization_axis = self.vect_context["vectorization_axis"]
            # metadata = self.vect_context.get('metadata')

            def analyze_operand(operand):
                if self._is_lifted(operand):
                    rank = self.ranks.get(operand, 1)
                    name = get_name(operand)

                    axis = None
                    if name:
                        data = self.dynamic_variable_lift.get(name)

                        if data:
                            axis = data.get("batched_axis")

                    return rank, True, axis
                # Recursive handling for nested BinOps
                if isinstance(operand, ast.BinOp):
                    l_rank, l_vec, l_axis = analyze_operand(operand.left)
                    r_rank, r_vec, r_axis = analyze_operand(operand.right)

                    rank = max(l_rank, r_rank)
                    vectorized = l_vec or r_vec
                    axis = l_axis if l_axis is not None else r_axis

                    return rank, vectorized, axis

                if isinstance(operand, ast.UnaryOp):
                    rank, vectorized, axis = analyze_operand(operand.operand)
                    return rank, vectorized, axis

                if isinstance(operand, ast.Compare):
                    l_rank, l_vec, l_axis = analyze_operand(operand.left)

                    rank = l_rank
                    vectorized = l_vec
                    axis = l_axis

                    for comp in operand.comparators:
                        r_rank, r_vec, r_axis = analyze_operand(comp)
                        rank = max(rank, r_rank)
                        vectorized = vectorized or r_vec
                        if axis is None:
                            axis = r_axis

                    return rank, vectorized, axis

                if isinstance(operand, ast.Call):
                    rank = 0
                    vectorized = False
                    axis = None

                    func_name = None

                    if isinstance(operand.func, ast.Attribute):
                        func_name = operand.func.attr
                    elif isinstance(operand.func, ast.Name):
                        func_name = operand.func.id

                    for arg in operand.args:
                        a_rank, a_vec, a_axis = analyze_operand(arg)
                        rank = max(rank, a_rank)
                        vectorized = vectorized or a_vec
                        if axis is None:
                            axis = a_axis

                    if func_name in {"sum", "mean", "max", "min"}:
                        # Find axis argument
                        reduction_axis = None

                        for kw in operand.keywords:
                            if kw.arg == "axis":
                                if isinstance(kw.value, ast.Constant):
                                    reduction_axis = kw.value.value

                        keepdims = False

                        for kw in operand.keywords:
                            if kw.arg == "keepdims":
                                if isinstance(kw.value, ast.Constant):
                                    keepdims = kw.value.value

                        if reduction_axis is None:
                            if keepdims:
                                # rank unchanged, but all reduced dims become size 1
                                # axis info becomes irrelevant for vectorization
                                vectorized = False
                                axis = None
                            else:
                                rank = 0
                                vectorized = False
                                axis = None

                        if reduction_axis is not None and rank > 0:
                            if not keepdims:
                                rank -= 1  # one dimension removed

                                # adjust axis_set
                                if axis is not None:
                                    new_axis = set()
                                    for a in axis:
                                        if a < reduction_axis:
                                            new_axis.add(a)
                                        elif a > reduction_axis:
                                            new_axis.add(a - 1)
                                    axis = new_axis if new_axis else None
                            else:
                                # keepdims=True -> rank unchanged
                                # BUT axis positions stay the same
                                # HOWEVER: reduction axis is now size=1 -> no longer vectorized

                                if axis is not None:
                                    axis = {a for a in axis if a != reduction_axis}
                                    axis = axis if axis else None

                            # recompute vectorization
                            vectorized = axis is not None
                    elif func_name == "where":
                        # jnp.where(cond, x, y) -> shape of x/y, not cond
                        # Re-analyze only the value arguments args[1] but not the args[2]
                        # since in most cases is the default value
                        if len(operand.args) >= 3:
                            rank, vectorized, axis = analyze_operand(operand.args[1])

                    return rank, vectorized, axis

                if isinstance(operand, ast.Subscript):
                    # rank = self.ranks.get(operand, 0)

                    array_name = get_name(operand)
                    inferred_present = array_name in self.inferred_ranks
                    if inferred_present:
                        rank = self.inferred_ranks[array_name]
                    else:
                        rank = self.ranks.get(operand, 0)

                    dims = self.get_active_dims(operand)  # THis retrieves the
                    # active dims meaning where the slice are happening thus full slice(:) or
                    # lower:upper type will also be present
                    axis_set = set()
                    for dim_idx, dim in enumerate(dims):
                        # dim may be a string expression; extract variable names from it
                        names = (
                            self.extract_names(dim) if isinstance(dim, str) else {dim}
                        )
                        for loop_dim in names & set(loop_info.keys()):
                            loop_var = loop_info[loop_dim]
                            vect_axes = vectorization_axis.get(loop_var, [])
                            if dim_idx in vect_axes:
                                axis_set.add(dim_idx)

                    vectorized = bool(axis_set)

                    return rank, vectorized, (axis_set or None)

                # Leaf cases
                if isinstance(operand, ast.Name | ast.Attribute):
                    name = (
                        operand.id
                        if isinstance(operand, ast.Name)
                        else operand.attr
                        if isinstance(operand, ast.Attribute)
                        else None
                    )

                    rank = self.ranks.get(operand, 0)
                    vectorized = False
                    axis_set = set()

                    if (
                        rank == 0
                        and self.local_defined_var
                        and name in self.local_defined_var
                    ):
                        dims = self.local_defined_var[name]
                    elif rank > 0:
                        dims = self.get_active_dims(operand)
                    else:
                        return 0, False, None

                    for dim_idx, dim in enumerate(dims):
                        # dim is a dimension name like 'kjpindex'; check if any loop iterates over it
                        names = (
                            self.extract_names(dim) if isinstance(dim, str) else {dim}
                        )
                        for loop_dim, loop_var in loop_info.items():
                            if loop_dim in names:
                                vect_axes = vectorization_axis.get(loop_var, [])
                                for vect_axis in vect_axes:
                                    # The axis in the resulting vector corresponds to dim_idx
                                    # but for a lifted scalar (rank was 0), the new dim is axis 0
                                    actual_axis = dim_idx if rank > 0 else vect_axis
                                    vectorized = True
                                    axis_set.add(actual_axis)

                    if vectorized:
                        if rank == 0:
                            rank += 1
                        axis_set = {a for a in axis_set if a < rank}

                    return rank, vectorized, (axis_set if axis_set else None)

                # Everything else treated as scalar
                return 0, False, None

            left_rank, left_vec, left_axis = analyze_operand(node.left)

            right_rank, right_vec, right_axis = analyze_operand(node.right)

            if left_rank == 0 or right_rank == 0:
                return node

            if left_rank == right_rank and left_vec and right_vec:

                def make_slice(rank, axis_set):
                    elts = [ast.Constant(value=None)] * rank
                    for idx in axis_set:
                        elts[idx] = ast.Slice()
                    return ast.Tuple(elts=elts, ctx=ast.Load())

                if (right_axis and left_axis) and right_axis != left_axis:
                    final_rank = max(left_rank, right_rank) + 1
                    node.left = ast.copy_location(
                        ast.Subscript(
                            value=node.left,
                            slice=make_slice(final_rank, left_axis),
                            ctx=ast.Load(),
                        ),
                        node.left,
                    )
                    node.right = ast.copy_location(
                        ast.Subscript(
                            value=node.right,
                            slice=make_slice(final_rank, right_axis),
                            ctx=ast.Load(),
                        ),
                        node.right,
                    )
                else:
                    return node

            # NEW BRANCH: same rank but operands are vectorized along *different*
            # independent dimensions (e.g. (N,) result vs (M,) array)
            # -> promote both to rank+1 with their own axis, broadcasting gives (N, M)
            if left_rank == right_rank:
                # Case: left is a lifted scalar (axis=0), right is an array along a
                # different dim (not detected as vectorized but has its own dimension)
                # We need: left[: , None] * right[None, :] → (N, M)
                def make_slice_explicit(axis_set, final_rank):
                    elts = [ast.Constant(value=None)] * final_rank
                    for idx in axis_set:
                        if idx < final_rank:
                            elts[idx] = ast.Slice()
                    return ast.Tuple(elts=elts, ctx=ast.Load())

                if left_vec and not right_vec and left_axis:
                    # right is an array whose dimension is NOT the vectorization axis
                    # -> it lives in a dimension orthogonal to the loop
                    # Promote: left stays (:,) -> (:, None), right becomes (None, :)
                    final_rank = max(left_rank, right_rank) + 1

                    # right has no detected axis — place it on the remaining axis
                    right_axis_inferred = {
                        i for i in range(final_rank) if i not in left_axis
                    }

                    node.left = ast.copy_location(
                        ast.Subscript(
                            value=node.left,
                            slice=make_slice_explicit(left_axis, final_rank),
                            ctx=ast.Load(),
                        ),
                        node.left,
                    )
                    node.right = ast.copy_location(
                        ast.Subscript(
                            value=node.right,
                            slice=make_slice_explicit(right_axis_inferred, final_rank),
                            ctx=ast.Load(),
                        ),
                        node.right,
                    )
                    return node

                elif right_vec and not left_vec and right_axis:
                    final_rank = max(left_rank, right_rank) + 1
                    left_axis_inferred = {
                        i for i in range(final_rank) if i not in right_axis
                    }

                    node.right = ast.copy_location(
                        ast.Subscript(
                            value=node.right,
                            slice=make_slice_explicit(right_axis, final_rank),
                            ctx=ast.Load(),
                        ),
                        node.right,
                    )
                    node.left = ast.copy_location(
                        ast.Subscript(
                            value=node.left,
                            slice=make_slice_explicit(left_axis_inferred, final_rank),
                            ctx=ast.Load(),
                        ),
                        node.left,
                    )
                    return node

            if left_rank != right_rank and (left_vec and right_vec):

                def promote(node, node_rank, target_rank, axis_set):
                    diff = target_rank - node_rank
                    if diff <= 0:
                        return node

                    # Build slicing like [:, None] or [None, :]
                    elts = []

                    # Simple heuristic: append new axes at the end
                    for i in range(target_rank):
                        if axis_set and i in axis_set:
                            elts.append(ast.Slice())  # keep vectorized dim
                        elif len(elts) < node_rank:
                            elts.append(ast.Slice())
                        else:
                            elts.append(ast.Constant(value=None))  # add new axis

                    return ast.Subscript(
                        value=node,
                        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
                        ctx=ast.Load(),
                    )

                if left_rank < right_rank:
                    node.left = ast.copy_location(
                        promote(node.left, left_rank, right_rank, left_axis),
                        node.left,
                    )
                else:
                    node.right = ast.copy_location(
                        promote(node.right, right_rank, left_rank, right_axis),
                        node.right,
                    )

                return node

            return node

        except Exception:
            raise

    def extract_names(self, expr_str: str) -> set | set[str]:
        """
        Extract variable names from a dimension expression.

        Parses a string expression and returns every variable name
        referenced within the resulting AST.

        Parameters
        ----------
        expr_str : str
            Expression string to analyze.

        Returns
        -------
        set[str]
            Set of variable names appearing in the expression.

        Notes
        -----
        Returns an empty set if the expression cannot be parsed.
        """
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError:
            return set()
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def get_declared_dims(self, node: ast.AST) -> list | None:
        """
        Retrieve declared dimensions associated with an array expression.

        Looks up dimension metadata for local arrays, class attributes,
        function arguments, or indexed array expressions.

        Parameters
        ----------
        node : ast.AST
            Array expression to inspect.

        Returns
        -------
        list or None
            Declared dimension names for the expression, or ``None``
            if the expression is scalar or dimension metadata is
            unavailable.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        For :class:`ast.Subscript` nodes, dimensions are inherited from
        the base array expression.
        """
        try:
            # Local arrays
            local_arr = self.cls_info[self.cls_name]["methods"][self.func_name].get(
                "local_arr", {}
            )

            # Class attributes
            attributes = self.cls_info[self.cls_name]["attributes"]
            if isinstance(node, ast.Name):
                if node.id in local_arr:
                    return local_arr[node.id].get("dimensions")
                elif (
                    node.id in attributes
                ):  # THis is just in case that the self was removed
                    return attributes[node.id].get("dimensions")
                elif self.func_input_dim and node.id in list(
                    self.func_input_dim.keys()
                ):
                    return self.func_input_dim.get(node.id)

            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    attr = node.attr
                    if attr in attributes:
                        return attributes[attr].get("dimensions")
                    elif self.func_input_dim and node.id in list(
                        self.func_input_dim.keys()
                    ):
                        return self.func_input_dim.get(node.id)

            if isinstance(node, ast.Subscript):
                # Recursively get base array dimensions
                base_dims = self.get_declared_dims(node.value)
                if base_dims is None:
                    return None  # unknown or scalar
                else:
                    return base_dims

            return None  # scalar or unknown
        except Exception:
            raise

    def _is_arange_over_dim(self, node: ast.Call) -> bool:
        """
        Determine whether a call represents a dimension-generating
        ``arange``.

        Recognizes expressions such as ``jnp.arange(self.kjpindex)``
        or ``np.arange(kjpindex)`` where the argument corresponds to
        a known dimension variable.

        Parameters
        ----------
        node : ast.Call
            Call expression to inspect.

        Returns
        -------
        bool
            ``True`` if the call generates indices over a declared
            dimension; otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        Used by :meth:`get_active_dims` when determining whether a
        dimension remains active after indexing.
        """
        try:
            if not isinstance(node, ast.Call):
                return False

            # Accept jnp.arange / np.arange / arange
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr != "arange":
                    return False
            elif isinstance(func, ast.Name):
                if func.id != "arange":
                    return False
            else:
                return False

            if len(node.args) != 1:
                return False

            arg = node.args[0]
            # Unwrap self.kjpindex  ->  Attribute(value=Name('self'), attr='kjpindex')
            if isinstance(arg, ast.Attribute):
                return arg.attr in self.cls_info[self.cls_name]["attributes"]
            # Also handle plain kjpindex (rare but possible)
            if isinstance(arg, ast.Name):
                return arg.id in self.cls_info[self.cls_name]["attributes"]

            return False
        except Exception:
            raise

    def get_active_dims(self, node: ast.AST) -> list:
        """
        Determine the active dimensions of an array expression.

        Starting from the dimensions returned by
        :meth:`get_declared_dims`, removes dimensions eliminated by
        scalar indexing while preserving dimensions referenced by
        slice operations, range-based indexing, or full-array access.

        Parameters
        ----------
        node : ast.AST
            Array expression or subscript operation.

        Returns
        -------
        list
            Ordered list of dimensions that remain active in the
            resulting expression.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        For indexed arrays, dimensions selected by scalar indices are
        removed, whereas dimensions accessed through slices or
        dimension-preserving indexing remain active.

        See Also
        --------
        :meth:`get_declared_dims`
            Retrieves declared dimension metadata.
        :meth:`_is_arange_over_dim`
            Detects dimension-preserving ``arange`` indexing.
        """
        try:
            declared = self.get_declared_dims(node)

            if not declared:
                return []

            # Subscript case
            if isinstance(node, ast.Subscript):
                sl = node.slice
                active = []

                if isinstance(sl, ast.Tuple):
                    for dim, idx in zip(declared, sl.elts):
                        # indexed -> dimension removed
                        if isinstance(idx, ast.Name | ast.Attribute | ast.BinOp):
                            continue

                        if self._is_arange_over_dim(idx):
                            active.append(dim)
                            continue

                        active.append(dim)
                else:
                    # single slice
                    if isinstance(sl, ast.Slice):
                        active = declared
                    elif self._is_arange_over_dim(sl):
                        active = declared
                    else:
                        active = declared[1:]

                return active

            # No subscript -> full array
            return declared

        except Exception:
            raise


class Control:
    """
    Store metadata describing a transformed control-flow construct.

    Instances of this class are used to record information about loops,
    conditionals, and other control-flow structures encountered during
    AST transformation. The stored metadata allows downstream passes to
    determine how a construct was classified and what lowering strategy
    should be applied.

    Parameters
    ----------
    kind : str
        Type of control-flow construct. Typical values include
        ``"if"`` and ``"loop"``.
    loop_info : dict, optional
        Loop-specific information. For loops, this typically maps loop
        bounds or iteration spaces to loop variables.
    transform_type : str, optional
        Classification assigned during analysis. Examples include:

        * ``"scalar"``
        * ``"index_loop"``
        * ``"vector"``
        * ``"masked"``

    vectorization_axis : str or dict, optional
        Description of the vectorization strategy or promoted axes.
        Examples include ``":None"``, ``"None:"``, ``"row"``, and
        ``"col"``.
    metadata : dict, optional
        Additional transformation-specific information. This may contain
        analysis results, inferred shapes, masking information, reduction
        metadata, or any other data required by downstream passes.

    Attributes
    ----------
    kind : str
        Control-flow construct type.
    loop_info : dict or None
        Loop metadata associated with the construct.
    transform_type : str or None
        Classification assigned by analysis.
    vectorization_axis : str or dict or None
        Vectorization information associated with the construct.
    metadata : dict
        Additional transformation metadata.

    Notes
    -----
    This class acts as a lightweight container for transformation state
    and analysis results during the conversion pipeline.
    """

    def __init__(
        self,
        kind,
        loop_info=None,
        transform_type=None,
        vectorization_axis=None,
        metadata=None,
    ):
        self.kind = kind
        self.loop_info = loop_info

        # transformation info
        self.transform_type = transform_type
        self.vectorization_axis = vectorization_axis

        # additional arbitrary information
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Return a dict representation of the control."""
        return {
            "kind": self.kind,
            "loop_info": self.loop_info,
            "transform_type": self.transform_type,
            "vectorization_axis": self.vectorization_axis,
            "metadata": self.metadata,
        }


SAFE_MATH_FUNCS = {
    # common numpy / jax functions
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "abs",
    "maximum",
    "minimum",
    "sum",
    "prod",
    "mean",
    "dot",
    "clip",
}


class VectorizationAnalyzer(ast.NodeVisitor):
    """
    Classify loops and conditional constructs for vectorization.

    This analysis determines whether control-flow structures can be
    converted into vectorized JAX operations, masked assignments, or must
    remain as scalar control flow.

    The analyzer identifies patterns such as:

    * Vectorizable loops.
    * Index-based loops.
    * Loops containing dynamic ``while`` constructs.
    * Masked assignments convertible to ``jnp.where``.
    * Control-flow conditionals requiring ``lax.cond``.
    * Branches operating on vectorized tensor expressions.

    Attributes
    ----------
    loop_stack : list[set[str]]
        Stack of active loop variables used when classifying nested
        control-flow structures.

    Notes
    -----
    Classification results are used by downstream lowering passes to
    select the appropriate transformation strategy. The analyzer does not
    modify the AST and only performs structural inspection.
    """

    def __init__(self, vectorize: list[str]) -> None:
        # stack of loop variable sets (outer -> inner)
        self.loop_stack: list[set[str]] = []
        self.vectorize = vectorize

    def classify_if(
        self, node: ast.If, surrounding_stmts: list[ast.AST] | None = None
    ) -> str | None:
        """
        Classify a conditional statement for vectorization lowering.

        Determines whether *node* represents scalar control flow, a masked
        assignment, an index-dependent branch, or a vectorized expression.
        Loop-variable context is inferred from *surrounding_stmts* via
        :meth:`_collect_loop_vars`. If no loop variables are discovered,
        the currently active loop context stored in :attr:`loop_stack`
        is used instead.

        Parameters
        ----------
        node : ast.If
            Conditional statement to classify.
        surrounding_stmts : list[ast.AST], optional
            Additional statements used to infer loop-variable context.

        Returns
        -------
        str or None
            One of:

            * ``"scalar"``
            * ``"vector"``
            * ``"index_loop"``
            * ``"masked"``
            * ``"masked_where"``

            Returns ``None`` if no classification rule applies.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        See Also
        --------
        :meth:`_classify_if_body`
            Performs the actual classification logic.
        :meth:`_collect_loop_vars`
            Extracts loop variables from surrounding statements.
        """
        try:
            stmts = surrounding_stmts or []
            # THe stmts contains the the body and the node.orelse of the
            # if statement and in the case if the loop vars is empty we
            # can simply check use self.loop_stack
            loop_vars = self._collect_loop_vars(stmts)
            if not loop_vars:
                loop_vars = [item for group in self.loop_stack for item in group]
            return self._classify_if_body(node, loop_vars)
        except Exception:
            raise

    def _collect_loop_vars_from_for(self, node: ast.For) -> set[str]:
        """
        Extract vectorization loop variables from a ``for`` loop.

        Examines the iterator expression of *node* and returns loop-target
        variables only when the loop iterates over a recognized
        vectorization construct.

        Parameters
        ----------
        node : ast.For
            Loop node to inspect.

        Returns
        -------
        set[str]
            Set of loop-variable names associated with the vectorized loop.
            Returns an empty set if the loop is not recognized as a
            iteration.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        Only loops matching the expected iterator pattern participate in
        vectorization analysis performed by :meth:`classify_for`.
        """
        try:
            vars: set[str] = set()

            # Ensure iterator is a function call
            if not isinstance(node.iter, ast.Call):
                return vars

            # Ensure the call has at least 2 arguments
            if len(node.iter.args) < 2:
                return vars

            # vect_variable = node.iter.args[1]
            # TODO: Perhaps need to handle the cases when we have
            # one or more range arguments.

            # Collect loop target variables
            target = node.target
            if isinstance(target, ast.Name):
                vars.add(target.id)
            elif isinstance(target, ast.Tuple | ast.List):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        vars.add(elt.id)

            return vars
        except Exception:
            raise

    def _is_vectorizable_loop(self, node: ast.For) -> bool:
        if not isinstance(node.iter, ast.Call):
            return False
        if len(node.iter.args) < 2:
            return False
        vect_variable = node.iter.args[1]
        return (
            isinstance(vect_variable, ast.Name) and vect_variable.id in self.vectorize
        ) or (
            isinstance(vect_variable, ast.Attribute)
            and vect_variable.attr in self.vectorize
        )

    def _collect_loop_vars(self, stmts: list[ast.AST]) -> set[str]:
        """
        Collect loop variables from a statement list.

        Traverses all nested :class:`ast.For` nodes contained within
        *stmts* and aggregates loop-variable names discovered via
        :meth:`_collect_loop_vars_from_for`.

        Parameters
        ----------
        stmts : list[ast.AST]
            Statements to scan.

        Returns
        -------
        set[str]
            Union of all recognized loop-variable names found in the
            supplied statement list.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        See Also
        --------
        :meth:`_collect_loop_vars_from_for`
            Extracts loop variables from individual ``for`` loops.
        """
        try:
            vars: set[str] = set()
            for s in stmts:
                for node in ast.walk(s):
                    if isinstance(node, ast.For):
                        vars |= self._collect_loop_vars_from_for(node)
            return vars
        except Exception:
            raise

    def _name_used(self, node: ast.AST, name: str) -> bool:
        """
        Determine whether a variable name appears in an AST subtree.

        Parameters
        ----------
        node : ast.AST
            Root node of the subtree to inspect.
        name : str
            Variable name to search for.

        Returns
        -------
        bool
            ``True`` if a matching :class:`ast.Name` node is found;
            otherwise ``False``.
        """
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == name:
                return True
        return False

    def _index_uses_loop_vars(self, idx: ast.AST, loop_vars: set[str]) -> bool:
        """
        Determine whether an index expression depends on loop variables.

        Recursively traverses *idx* and checks whether any component of the
        index expression references a variable contained in *loop_vars*.

        Parameters
        ----------
        idx : ast.AST
            Index expression to inspect.
        loop_vars : set[str]
            Active loop-variable names.

        Returns
        -------
        bool
            ``True`` if the index expression depends on at least one loop
            variable; otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        This helper is used by :meth:`_subscript_uses_loop_vars`,
        :meth:`_is_index_selection_if`, and
        :meth:`_is_static_or_indexed_by_loop`.
        """
        try:
            if idx is None:
                return False
            if isinstance(idx, ast.Name):
                return idx.id in loop_vars
            if isinstance(idx, ast.Tuple):
                return any(
                    self._index_uses_loop_vars(elt, loop_vars) for elt in idx.elts
                )
            if isinstance(idx, ast.BinOp):
                return self._index_uses_loop_vars(
                    idx.left, loop_vars
                ) or self._index_uses_loop_vars(idx.right, loop_vars)
            if isinstance(idx, ast.UnaryOp):
                return self._index_uses_loop_vars(idx.operand, loop_vars)
            if isinstance(idx, ast.Subscript):
                return self._subscript_uses_loop_vars(idx, loop_vars)
            # function calls: arr[f(ji, something)]
            if isinstance(idx, ast.Call):
                # check function name itself
                if self._index_uses_loop_vars(idx.func, loop_vars):
                    return True

                # check all positional args
                for arg in idx.args:
                    if self._index_uses_loop_vars(arg, loop_vars):
                        return True

            # Other node types (Constant, Attribute) generally don't reference loop vars directly
            for child in ast.iter_child_nodes(idx):
                if self._index_uses_loop_vars(child, loop_vars):
                    return True
            return False
        except Exception:
            raise

    def _subscript_uses_loop_vars(self, node: ast.AST, loop_vars: set[str]) -> bool:
        """
        Determine whether any subscript access depends on loop variables.

        Recursively scans *node* for :class:`ast.Subscript` expressions and
        checks whether their indexing expressions reference variables in
        *loop_vars*.

        Parameters
        ----------
        node : ast.AST
            AST subtree to inspect.
        loop_vars : set[str]
            Active loop-variable names.

        Returns
        -------
        bool
            ``True`` if any subscript index depends on a loop variable;
            otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        See Also
        --------
        :meth:`_index_uses_loop_vars`
            Evaluates individual index expressions.
        """
        try:
            if isinstance(node, ast.Subscript):
                # every Python AST version differs slightly: node.slice may be ast.Index or direct expr
                slice_node = node.slice
                # handle extended slice such as A[i, j]
                if self._index_uses_loop_vars(slice_node, loop_vars):
                    return True
            for child in ast.iter_child_nodes(node):
                if self._subscript_uses_loop_vars(child, loop_vars):
                    return True
            return False
        except Exception:
            raise

    def _is_pure_call(self, node: ast.Call) -> bool:
        """
        Determine whether a function call is considered pure.

        Classifies calls to known mathematical functions as side-effect-free
        when all arguments satisfy :meth:`_is_static_expr`.

        Parameters
        ----------
        node : ast.Call
            Call expression to inspect.

        Returns
        -------
        bool
            ``True`` if the call appears to be a safe mathematical operation;
            otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        This check is intentionally conservative and only recognizes
        functions listed in ``SAFE_MATH_FUNCS``.
        """
        try:
            # If function is a bare name like `sin(...)` or an attribute like `jnp.sin(...)`
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in SAFE_MATH_FUNCS:
                    return all(self._is_static_expr(arg) for arg in node.args)
                return False
            if isinstance(func, ast.Attribute):
                # attribute name ( jnp.sin). Check the attr part
                if func.attr in SAFE_MATH_FUNCS:
                    return all(self._is_static_expr(arg) for arg in node.args)
                return False
            return False
        except Exception:
            raise

    def _is_static_expr(self, node: ast.AST) -> bool:
        """
        Determine whether an expression is statically evaluable.

        Recognizes constants, names, attributes, simple arithmetic
        expressions, and pure mathematical calls as static expressions.

        Parameters
        ----------
        node : ast.AST
            Expression node to inspect.

        Returns
        -------
        bool
            ``True`` if the expression is considered static or side-effect
            free; otherwise ``False``.

        See Also
        --------
        :meth:`_is_pure_call`
            Evaluates call expressions.
        """
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Attribute):
            # obj.field (we conservatively allow attributes)
            return True
        if isinstance(node, ast.BinOp):
            return self._is_static_expr(node.left) and self._is_static_expr(node.right)
        if isinstance(node, ast.UnaryOp):
            return self._is_static_expr(node.operand)
        if isinstance(node, ast.Call):
            return self._is_pure_call(node)
        # Subscript or other constructs are not static in this conservative check
        return False

    def check_for_while(self, node: ast.For) -> bool:
        """
        Determine whether a loop contains nested ``while`` statements.

        Parameters
        ----------
        node : ast.For
            Loop node to inspect.

        Returns
        -------
        bool
            ``True`` if a nested :class:`ast.While` node is found anywhere
            within the loop body; otherwise ``False``.

        Notes
        -----
        Used by :meth:`classify_for` to distinguish vectorizable loops from
        loops requiring dynamic control-flow lowering.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.While):
                return True
        return False

    def classify_for(self, node: ast.For) -> str:
        """
        Classify a ``for`` loop for vectorization lowering.

        Determines whether *node* represents a vectorizable loop, an
        index-based loop, or a loop containing dynamic ``while`` control
        flow. Loop-variable context is recorded in :attr:`loop_stack`
        for subsequent branch analysis.

        Parameters
        ----------
        node : ast.For
            Loop node to classify.

        Returns
        -------
        str
            One of:

            * ``"vector"``
            * ``"index_loop"``
            * ``"vector_while"``

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        See Also
        --------
        :meth:`check_for_while`
            Detects nested ``while`` constructs.
        :meth:`_collect_loop_vars_from_for`
            Extracts vectorization loop variables.
        """
        try:
            loop_vars = self._collect_loop_vars_from_for(node)
            self.loop_stack.append(loop_vars)

            # check if a while loop is present inside
            is_vectorizable = self._is_vectorizable_loop(node)
            while_present = self.check_for_while(node)
            if not is_vectorizable:
                return "index_loop"
            elif is_vectorizable and while_present:
                return "vector_while"

            return "vector"
        except Exception:
            raise

    def _targets_match(self, t1: ast.AST, t2: ast.AST, loop_vars: set[str]) -> bool:
        """
        Determine whether two assignment targets refer to the "same slot".

        Extracted from the original inline logic in
        :meth:`_is_index_selection_if` so it can be reused when comparing
        every branch of an ``if/elif/.../else`` chain against a single
        reference target, not just an isolated ``if``/``else`` pair.

        Parameters
        ----------
        t1 : ast.AST
            First assignment target.
        t2 : ast.AST
            Second assignment target.
        loop_vars : set[str]
            Active loop-variable names.

        Returns
        -------
        bool
            ``True`` if *t1* and *t2* are considered the same assignment
            slot (same name, same attribute, or same array subscripted by
            a loop-variable-dependent index); otherwise ``False``.
        """
        # Case 1: simple scalar name
        if isinstance(t1, ast.Name) and isinstance(t2, ast.Name):
            return t1.id == t2.id

        # Case 2: attribute assignment
        if isinstance(t1, ast.Attribute) and isinstance(t2, ast.Attribute):
            return ast.dump(t1) == ast.dump(t2)  # safer full match

        # Case 3: subscript assignment
        if isinstance(t1, ast.Subscript) and isinstance(t2, ast.Subscript):
            # Must be same array
            if ast.dump(t1.value) != ast.dump(t2.value):
                return False

            # indices MUST use loop vars, or else it's not maskable
            if not (
                self._index_uses_loop_vars(t1.slice, loop_vars)
                or self._index_uses_loop_vars(t2.slice, loop_vars)
            ):
                return False

            return True

        return False

    def _collect_chain_targets(self, node: ast.If) -> list[ast.AST] | None:
        """
        Collect assignment targets across an entire ``if/elif/.../else`` chain.

        Walks the ``orelse`` of *node*: as long as it is exactly one nested
        :class:`ast.If` (i.e. an ``elif``), recurse into it; once it
        bottoms out at a single terminal :class:`ast.Assign` (the final
        ``else``), collect its target. Every branch along the way must
        consist of a single ``Assign`` statement, and the chain must end
        in an explicit ``else`` — otherwise coverage isn't guaranteed and
        the chain does not qualify as a pure selection.

        Parameters
        ----------
        node : ast.If
            Head of the (possibly single) ``if``/``elif`` chain.

        Returns
        -------
        list[ast.AST] or None
            Ordered list of assignment targets, one per branch (including
            the final ``else``), or ``None`` if any branch fails to
            qualify (multiple statements, non-``Assign`` statement, or a
            chain with no terminal ``else``).
        """
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Assign):
            return None

        targets: list[ast.AST] = [node.body[0].targets[0]]

        if not node.orelse:
            # No else -> branches don't fully cover the condition space,
            # so we can't treat this as a pure selection pattern.
            return None

        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            nested = self._collect_chain_targets(node.orelse[0])
            if nested is None:
                return None
            targets.extend(nested)
            return targets

        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Assign):
            targets.append(node.orelse[0].targets[0])
            return targets

        return None

    def _is_index_selection_if(self, node: ast.If, loop_vars: set[str]) -> bool:
        """
        Determine whether a conditional represents a masked selection.

        Identifies patterns where every branch of an ``if``/``elif``/
        ``else`` chain assigns to the same target and differs only in the
        value selected. Such constructs are potential candidates for
        lowering to masked assignments or ``jnp.where``/``jnp.select``
        operations.

        Parameters
        ----------
        node : ast.If
            Conditional statement to inspect (head of the chain).
        loop_vars : set[str]
            Active loop-variable names.

        Returns
        -------
        bool
            ``True`` if every branch in the chain matches a maskable
            selection pattern; otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        Subscript assignments must reference loop-variable indices in
        order to qualify. Unlike the original implementation, this now
        looks *through* nested ``elif`` branches instead of bailing out
        as soon as ``orelse`` isn't a literal ``Assign`` -- so every
        branch of a multi-way ``if/elif/.../else`` ladder is classified
        consistently rather than depending on its position in the chain.
        """
        try:
            targets = self._collect_chain_targets(node)
            if not targets or len(targets) < 2:
                return False

            reference = targets[0]
            return all(
                self._targets_match(reference, other, loop_vars)
                for other in targets[1:]
            )
        except Exception:
            raise

    def _classify_if_body(self, node: ast.If, loop_vars: set[str]) -> str | None:
        """
        Classify a conditional relative to loop context.

        Performs the core branch analysis used by
        :meth:`classify_if`, distinguishing scalar control flow,
        vectorized expressions, masked assignments, and index-dependent
        branches.

        Parameters
        ----------
        node : ast.If
            Conditional statement to classify.
        loop_vars : set[str]
            Loop-variable names currently in scope.

        Returns
        -------
        str or None
            One of:

            * ``"scalar"``
            * ``"vector"``
            * ``"index_loop"``
            * ``"masked"``
            * ``"masked_where"``

            Returns ``None`` if no classification can be determined.

        Raises
        ------
        Exception
            Re-raises any unexpected exception encountered during
            analysis.

        See Also
        --------
        :meth:`classify_if`
            Public classification entry point.
        :meth:`_is_masked_where_if`
            Detects mask-based update patterns.
        :meth:`_is_index_selection_if`
            Detects branch-selection patterns.
        """
        try:
            if not isinstance(node, ast.If) or not node.body:
                return None

            if self._is_masked_where_if(node):
                return "masked_where"

            if self._is_index_selection_if(node, loop_vars):
                return "masked"

            # 2. If test itself uses loop vars -> index_loop
            if self._subscript_uses_loop_vars(node.test, loop_vars):
                # if body assigns to subscript with same index -> masked
                if self._is_index_selection_if(node, loop_vars):
                    return "masked"

                return "index_loop"

            assigns = [s for s in node.body if isinstance(s, ast.Assign)]
            # IfStatement (scalar) -> control-flow wrapper,
            # ignore for vectorization, mostly linked to boolean conditions
            # CONTROL-FLOW IF (scalar predicate guarding a block)
            if isinstance(
                node.test, ast.Name | ast.Attribute
            ) and not self._subscript_uses_loop_vars(node.test, loop_vars):
                # if body contains loops, scans, or multiple statements -> control flow
                if len(node.body) > 1 or any(
                    isinstance(s, ast.For | ast.While) for s in node.body
                ):
                    return "scalar"

            # Otherwise body does not depend on loop index via test;
            # inspect assigns if there are more than one then we need to ensure that the take the
            # correct one.
            # 4. Inspect each assignment
            for stmt in assigns:
                target = stmt.targets[0]
                # Scalar assignment with static RHS -> scalar
                if isinstance(
                    target, ast.Name | ast.Attribute
                ) and self._is_static_expr(stmt.value):
                    return "scalar"
                # Assignment involves loop-indexed subscripts or RHS depends on loop -> index_loop
                if self._subscript_uses_loop_vars(stmt.value, loop_vars) or (
                    isinstance(target, ast.Subscript)
                    and self._index_uses_loop_vars(target.slice, loop_vars)
                ):
                    return "index_loop"

            # 5. Check if RHS is vectorized (not static or not purely indexed by loop)
            rhs_vectorized = any(
                not self._is_static_or_indexed_by_loop(a.value, loop_vars)
                for a in assigns
            )

            # Need to check for masked where statements
            if rhs_vectorized:
                return "vector"

            # 6. Default
            return "scalar"
        except Exception:
            raise

    def _is_static_or_indexed_by_loop(self, node: ast.AST, loop_vars: set[str]) -> bool:
        """
        Determine whether an expression is static or loop-indexed.

        An expression is considered valid if it is either classified as
        static by :meth:`_is_static_expr` or represents a subscript access
        whose indices depend on loop variables.

        Parameters
        ----------
        node : ast.AST
            Expression node to inspect.
        loop_vars : set[str]
            Active loop-variable names.

        Returns
        -------
        bool
            ``True`` if the expression is static or indexed by loop
            variables; otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected exception encountered during
            analysis.
        """
        try:
            if self._is_static_expr(node):
                return True
            if isinstance(node, ast.Subscript) and self._index_uses_loop_vars(
                node.slice, loop_vars
            ):
                return True
            if isinstance(node, ast.Call) and self._is_pure_call(node):
                return all(
                    self._is_static_or_indexed_by_loop(a, loop_vars) for a in node.args
                )

            return False
        except Exception:
            raise

    def _is_index_loop_lhs(self, target: ast.AST, loop_vars: set[str]) -> bool:
        """
        Determine whether an assignment target is loop-indexed.

        Parameters
        ----------
        target : ast.AST
            Assignment target expression.
        loop_vars : set[str]
            Active loop-variable names.

        Returns
        -------
        bool
            ``True`` if *target* is a subscript whose index expression
            references at least one loop variable; otherwise ``False``.

        See Also
        --------
        :meth:`_index_uses_loop_vars`
            Evaluates index expressions.
        """
        return isinstance(target, ast.Subscript) and self._index_uses_loop_vars(
            target.slice, loop_vars
        )

    def _is_masked_where_if(self, node: ast.If) -> bool:
        """
        Detect masked-update conditionals.

        Identifies branch patterns that can be lowered to masked array
        updates, such as ``A[mask] = value`` guarded by a condition of the
        form ``cond.any()``.

        Parameters
        ----------
        node : ast.If
            Conditional statement to inspect.

        Returns
        -------
        bool
            ``True`` if the conditional matches a masked-update pattern;
            otherwise ``False``.

        Raises
        ------
        Exception
            Re-raises any unexpected error.

        Notes
        -----
        Both inline masks and intermediate mask variables are supported.

        See Also
        --------
        :meth:`_structurally_equal`
            Compares AST expressions for equivalence.
        """
        try:
            test = node.test
            cond_expr = None

            if not (
                isinstance(test, ast.Call)
                and isinstance(test.func, ast.Attribute)
                and test.func.attr == "any"
            ):
                return False

            cond_expr = (
                test.func.value
                if isinstance(test.func.value, ast.Compare | ast.BinOp)
                else test.args[0]
            )
            mask_names = set()
            found_masked_write = False

            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue

                target = stmt.targets[0]

                # Case: mask = cond
                if isinstance(target, ast.Name):
                    if self._structurally_equal(stmt.value, cond_expr):
                        mask_names.add(target.id)

                # Case: A[mask]
                elif isinstance(target, ast.Subscript):
                    # Inline mask: A[cond] = ...
                    if self._structurally_equal(target.slice, cond_expr):
                        found_masked_write = True

                    # Mask variable: A[mask] = ...
                    elif isinstance(target.slice, ast.Name):
                        if target.slice.id in mask_names:
                            found_masked_write = True

            return found_masked_write
        except Exception:
            raise

    def _structurally_equal(self, a: ast.AST, b: ast.AST) -> bool:
        """
        Determine whether two AST nodes are structurally equivalent.

        Compares *a* and *b* using :func:`ast.compare` when available,
        falling back to a normalized :func:`ast.dump` representation.

        Parameters
        ----------
        a : ast.AST
            First AST node.
        b : ast.AST
            Second AST node.

        Returns
        -------
        bool
            ``True`` if both nodes represent the same AST structure;
            otherwise ``False``.

        Notes
        -----
        Source-location attributes are ignored during comparison.
        """
        return (
            ast.compare(a, b, compare_attributes=False)
            if hasattr(ast, "compare")
            else ast.dump(a, include_attributes=False)
            == ast.dump(b, include_attributes=False)
        )


class RemoveLogging(ast.NodeTransformer):
    """
    Remove logging and debugging statements from an AST.

    This transformer removes standalone expression statements that invoke
    logging or debugging functions such as ``logging.info(...)`` or
    ``print(...)``. Logging calls embedded within larger expressions are
    preserved.

    Supported removals include calls of the form::

        logging.info(...)
        logging.debug(...)
        logging.warning(...)
        logging.error(...)
        logging.exception(...)
        logging.log(...)
        print(...)

    Notes
    -----
    Only standalone expression statements (``ast.Expr``) are removed.
    Calls appearing inside assignments, conditionals, return statements,
    or other expressions are left unchanged.

    Examples
    --------
    The statement::

        logging.info("iteration=%d", i)

    is removed completely, while::

        x = logging.info("msg")

    is preserved because the call participates in a larger expression.
    """

    LOGGING_NAMES = {"logging", "print"}

    def _is_logging_call(self, node: ast.AST) -> bool:
        """
        Determine whether a statement is a removable logging expression.

        Parameters
        ----------
        node : ast.AST
            AST node to inspect.

        Returns
        -------
        bool
            ``True`` if the node is a standalone expression statement whose
            value is a call to a recognized logging or debugging function,
            otherwise ``False``.
        """
        if not isinstance(node, ast.Expr):
            return False

        call = node.value
        if not isinstance(call, ast.Call):
            return False

        func = call.func

        # logging.<level>(...)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self.LOGGING_NAMES
        ):
            return True

        return False

    def visit_If(self, node: ast.If) -> ast.If | None:
        """
        Remove logging statements from conditional branches.

        Parameters
        ----------
        node : ast.If
            Conditional statement to transform.

        Returns
        -------
        ast.If or None
            The transformed conditional. Returns ``None`` if both the body
            and ``else`` branch become empty after logging statements are
            removed.
        """
        self.generic_visit(node)

        node.body = [s for s in node.body if not self._is_logging_call(s)]
        node.orelse = [s for s in node.orelse if not self._is_logging_call(s)]

        if not node.body and not node.orelse:
            return None

        return node

    def visit_For(self, node: ast.For) -> ast.For | None:
        """
        Remove logging statements from loop bodies.

        Parameters
        ----------
        node : ast.For
            Loop node to transform.

        Returns
        -------
        ast.For or None
            The transformed loop. Returns ``None`` if the loop body becomes
            empty after logging statements are removed.
        """
        self.generic_visit(node)

        node.body = [s for s in node.body if not self._is_logging_call(s)]

        if not node.body:
            return None

        return node


class ReplaceSelfRef(ast.NodeTransformer):
    """
    Replace self-references to a vectorized variable with a scalar
    accumulator variable.

    This transformer is used when converting vectorized update patterns
    into scalar loop-carried state. Any subscripted access to a specified
    variable is replaced by a scalar replacement variable.

    Parameters
    ----------
    varname : str
        Original vectorized variable name.
    new_name : str
        Replacement scalar variable name.
    """

    def __init__(self, varname: str, new_name: str):
        self.varname = varname
        self.new_name = new_name

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """
        Replace subscripted accesses to the tracked variable.

        Parameters
        ----------
        node : ast.Subscript
            Subscript expression to inspect.

        Returns
        -------
        ast.AST
            Replacement variable reference if the subscript refers to the
            tracked variable, otherwise the recursively transformed node.
        """
        if isinstance(node.value, ast.Name) and node.value.id == self.varname:
            return ast.Name(id=self.new_name, ctx=ast.Load())
        return self.generic_visit(node)


class WhileVectorToScalar(ast.NodeTransformer):
    """
    Convert vectorized array-update operations inside while loops into
    scalar accumulator updates.

    This transformer is used when lowering vectorized loop constructs
    into scalar control-flow representations. It removes vectorization
    axes from indexed expressions, tracks loop-dependent variables, and
    rewrites JAX-style array update operations into scalar assignments.

    In particular, updates of the form::

        x = x.at[idx].set(expr)

    are rewritten into::

        x_scalar = expr

    and updates such as::

        x = x.at[idx].add(expr)

    become::

        x_scalar = x_scalar + expr

    The transformer also records variables written inside while loops so
    that downstream passes can determine which values must be threaded
    through loop state.

    Attributes
    ----------
    vector_arrays : set[str]
        Arrays participating in vectorized computation.
    vectorization_axis : dict
        Mapping from loop variables to vectorized axes.
    var_to_replace : dict
        Mapping from vectorized variables to scalar replacements.
    ji_dependent_vars : set[str]
        Variables whose values depend on the active loop index.
    while_used_vars : dict
        Variables assigned within while loops.
    loop_index : str or None
        Active vectorized loop index variable.
    """

    def __init__(self):
        self.vector_arrays: set = set()
        self.vectorization_axis: dict = {}
        self.var_to_replace: dict = {}
        self.ji_dependent_vars: set[str] = set()
        self.while_used_vars: dict = {}
        self.loop_index: str = None  # Set when visiting a vectorized for-loop
        self._in_while: bool = False
        self.op_map = {
            "add": ast.Add,
            "subtract": ast.Sub,
            "multiply": ast.Mult,
            "divide": ast.Div,
            "power": ast.Pow,
        }

    def _get_name(self, node: ast.AST) -> str | None:
        """
        Extract a symbolic name from a variable or attribute expression.

        Parameters
        ----------
        node : ast.AST
            AST node representing a variable reference.

        Returns
        -------
        str or None
            Variable or attribute name if one can be extracted,
            otherwise ``None``.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value is not None:
                return node.attr
        return None

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """
        Remove vectorization axes from indexed expressions.

        The transformation tracks dependencies on the active loop index and
        rewrites subscripts by eliminating dimensions associated with
        vectorized axes.

        Parameters
        ----------
        node : ast.Subscript
            Subscript expression to transform.

        Returns
        -------
        ast.AST
            Transformed subscript expression.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            node = self.generic_visit(node)
            # TODO: Added these
            if isinstance(node.value, ast.Attribute) and node.value.attr == "at":
                return node
            # Internal subscripts get's transformed
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in self.var_to_replace
            ):
                return node

            # Track dependency on loop index
            indices = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            for idx in indices:
                if isinstance(idx, ast.Name) and idx.id == self.loop_index:
                    base_name = self._get_name(node.value)
                    if base_name:
                        self.ji_dependent_vars.add(base_name)

            # Remove vectorization axes
            axes_to_remove = set()
            for i, idx in enumerate(indices):
                if isinstance(idx, ast.Name):
                    var = idx.id
                    if var in self.vectorization_axis:
                        expected_axes = self.vectorization_axis[var]
                        if i in expected_axes:
                            axes_to_remove.add(i)
                elif isinstance(idx, ast.Slice):
                    for axes in self.vectorization_axis.values():
                        if i in axes:
                            axes_to_remove.add(i)

            if axes_to_remove:
                new_indices = [
                    idx for i, idx in enumerate(indices) if i not in axes_to_remove
                ]
                if not new_indices:
                    return ast.Name(id=f"{node.value.id}", ctx=ast.Load())
                if len(new_indices) == 1:
                    new_slice = new_indices[0]
                else:
                    new_slice = ast.Tuple(elts=new_indices, ctx=ast.Load())
                new_value = node.value
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in self.vector_arrays
                ):
                    new_value = ast.Name(id=f"{node.value.id}", ctx=ast.Load())
                return ast.Subscript(value=new_value, slice=new_slice, ctx=node.ctx)

            # Rename array if needed
            if isinstance(node.value, ast.Name) and node.value.id in self.vector_arrays:
                node.value = ast.Name(id=f"{node.value.id}", ctx=ast.Load())
            return node
        except Exception:
            raise

    def visit_Name(self, node: ast.AST) -> ast.AST:
        """
        Rewrite references to vectorized arrays.

        Loaded references to variables registered in
        :attr:`vector_arrays` are normalized so that subsequent scalarization
        passes operate on a consistent representation. Store contexts are left
        unchanged.

        Parameters
        ----------
        node : ast.Name
            Variable reference to transform.

        Returns
        -------
        ast.AST
            The original name node or a rewritten name node referring to the
            vectorized array representation.
        """
        node = self.generic_visit(node)
        # Rename vector arrays when loaded
        if isinstance(node.ctx, ast.Load) and node.id in self.vector_arrays:
            return ast.Name(id=f"{node.id}", ctx=node.ctx)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        """
        Enter a while-loop context during traversal.

        Parameters
        ----------
        node : ast.While
            While-loop node.

        Returns
        -------
        ast.While
            Transformed while-loop node.
        """
        prev_flag = self._in_while
        self._in_while = True
        node = self.generic_visit(node)
        self._in_while = prev_flag
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """
        Rewrite vectorized array updates into scalar assignments.

        This method detects JAX array-update expressions such as
        ``x.at[idx].set(...)`` and ``x.at[idx].add(...)`` and converts them
        into scalar updates suitable for scalarized loop bodies.

        Parameters
        ----------
        node : ast.Assign
            Assignment statement to transform.

        Returns
        -------
        ast.Assign
            Original or transformed assignment node.

        Raises
        ------
        Exception
            Re-raises any unexpected error.
        """
        try:
            # Track assigned variables inside while loop
            if self._in_while:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in self.while_used_vars:
                            self.while_used_vars[target.id] = copy.deepcopy(node)
                    elif isinstance(target, ast.Subscript):
                        base_name = self._get_name(target.value)
                        if base_name not in self.while_used_vars:
                            self.while_used_vars[base_name] = copy.deepcopy(node)

            node = self.generic_visit(node)

            # Transform x.at[...].set(...) to scalar accumulation
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return node

            target = node.targets[0]
            if target.id not in self.var_to_replace:
                if target.id in self.while_used_vars:
                    self.while_used_vars.pop(target.id, None)
                return node

            varname = target.id
            new_name = self.var_to_replace[varname]
            value = node.value

            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                method = value.func.attr

                if method != "set" and method not in self.op_map:
                    return node

                sub = value.func.value
                if not isinstance(sub, ast.Subscript):
                    return node

                at_attr = sub.value
                if not (
                    isinstance(at_attr, ast.Attribute)
                    and at_attr.attr == "at"
                    and isinstance(at_attr.value, ast.Name)
                    and at_attr.value.id == varname
                ):
                    return node

                if len(value.args) != 1:
                    return node

                expr = value.args[0]
                replacer = ReplaceSelfRef(varname, new_name)
                new_expr = replacer.visit(expr)

                if method == "set":
                    new_value = new_expr
                else:
                    new_value = ast.BinOp(
                        left=ast.Name(id=new_name, ctx=ast.Load()),
                        op=self.op_map[method](),
                        right=new_expr,
                    )

                return ast.Assign(
                    targets=[ast.Name(id=new_name, ctx=ast.Store())],
                    value=new_value,
                )

            return node

        except Exception:
            raise
