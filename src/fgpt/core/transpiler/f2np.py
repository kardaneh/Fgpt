# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import ast
import copy
import re

from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

from fgpt.core.common.logger import Logger
from fgpt.core.common.utils import ast_walk
from fgpt.core.frontend.extractor import Extractor
from fgpt.core.transpiler.intrinsic import (
    intrinsic_signatures,
    normalize_intrinsic_call,
)


class F2NP:
    """
    Translate a single Fortran subroutine/function body into a Python AST.

    Where :class:`~transformer.Transformer` works above the level of a whole
    program (declarations, classes, modules, file I/O), ``F2NP`` operates at
    the statement and expression level: it walks the Fortran AST produced by
    ``fparser`` for one subroutine/function and incrementally builds the
    equivalent Python :mod:`ast` tree, statement by statement, expression by
    expression.

    Parameters
    ----------
    extractor : Extractor, optional
        Instance providing extracted Fortran metadata (array shapes, loop
        variable groupings, allowed external subroutines, etc.) needed to
        disambiguate constructs such as array references vs. function calls
        in :meth:`handle_part_ref`, or to apply array-masking semantics in
        :meth:`apply_mask_to_rhs`. If ``None``, behavior that depends on this
        metadata falls back to conservative defaults (e.g. treating an
        ambiguous reference as an array rather than a function call).

    Attributes
    ----------
    extractor : Extractor or None
        Reference to the extractor instance, or ``None`` if not supplied.
    replacements : dict
        Mapping of Fortran relational/logical operator tokens and keywords
        (``.LT.``, ``.AND.``, ``IF``, ``THEN``, etc.) to their Python textual
        equivalents. Primarily used as a lookup table when resolving operator
        tokens to entries in :attr:`conditional_ops_map`.
    intrinsic_replacements : dict
        Mapping of Fortran intrinsic function names (``ABS``, ``SQRT``,
        ``MAXVAL``, ...) to their NumPy/Python equivalents, consulted by
        :meth:`handle_intrinsic_function_reference`.
    conditional_ops_map : dict
        Mapping from Python operator symbols/keywords (``'>'``, ``'and'``,
        ``'not'``, ...) to their corresponding :mod:`ast` operator node
        instances (:class:`ast.Gt`, :class:`ast.And`, :class:`ast.Not`,
        etc.).
    logger : Logger
        Logger instance used for structured logging and exception reporting.
    func_name : str or None
        Name of the subroutine/function currently being translated; set in
        :meth:`recursive_ast` when a ``Subroutine_Stmt``/``Function_Stmt`` is
        encountered and cleared at the corresponding end statement. Used to
        look up per-function array metadata (see :meth:`_get_func_arrays`).
    arg_list : list of str
        Dummy argument names of the subroutine/function currently being
        translated; populated by :meth:`handle_subroutine_stmt`.

    Notes
    -----
    The main entry point is :meth:`recursive_ast`, which performs a
    depth-first walk of the Fortran AST and dispatches each statement type
    (assignments, ``DO``/``IF``/``WHERE``/``SELECT CASE`` constructs, ``CALL``
    and ``PRINT``/``WRITE`` statements, ``CYCLE``/``EXIT``, ``RETURN``, etc.)
    to a dedicated ``handle_*`` method. Control-flow constructs are tracked
    via an explicit stack (``control_stack``) and per-construct counters
    (``counters``) rather than relying on Python's call stack, since Fortran's
    block-closing statements (``END IF``, ``END DO``, ``END SELECT``) must be
    matched against possibly nested and chained (``ELSE IF``) constructs.

    Expression-level translation is centralized in :meth:`handle_expr`, which
    dispatches literals, binary/unary operations, array part references
    (:meth:`handle_part_ref`), and intrinsic function calls
    (:meth:`handle_intrinsic_function_reference`) to their respective
    handlers and returns the corresponding :mod:`ast` node.

    Fortran ``WHERE`` constructs are lowered to ``if mask.any(): ...``
    blocks combined with boolean-mask subscripting on the left-hand side,
    implemented via :meth:`handle_where_stmt` and :meth:`apply_mask_to_rhs`.
    """

    def __init__(self, extractor: Extractor | None = None):
        self.extractor = extractor
        self.replacements = {
            r"\bELSE IF\b": "elif",
            r"\bIF\b": "if",
            r"\bELSE\b": "else:",
            r"\.LT\.": "<",
            r"\.LE\.": "<=",
            r"\.GT\.": ">",
            r"\.GE\.": ">=",
            r"\.NE\.": "!=",
            r"\.EQ\.": "==",
            r"\.AND\.": "and",
            r"\.OR\.": "or",
            r"\.NOT\.": "not",
            r"\bTHEN\b": ":",
        }
        self.intrinsic_replacements = {
            r"\bINT\b": "int",
            r"\bREAL\b": "float",
            r"\bMIN\b": "np.minimum",
            r"\bMAX\b": "np.maximum",  # https://medium.com/@amit25173/understanding-element-wise-maximum-in-numpy-43916b1c2002
            r"\bMAXVAL\b": "np.max",
            r"\bMINVAL\b": "np.min",
            r"\bMINLOC\b": "np.argmin",
            r"\bMAXLOC\b": "np.argmax",
            r"\bABS\b": "np.abs",
            r"\bSQRT\b": "np.sqrt",
            r"\bEXP\b": "np.exp",
            r"\bLOG\b": "np.log",
            r"\bSIN\b": "np.sin",
            r"\bCOS\b": "np.cos",
            r"\bTAN\b": "np.tan",
            r"\bASIN\b": "np.arcsin",
            r"\bACOS\b": "np.arccos",
            r"\bATAN\b": "np.arctan",
            r"\bATAN2\b": "np.arctan2",
            r"\bAINT\b": "np.trunc",
            r"\bMOD\b": "np.mod",
            r"\bCEILING\b": "np.ceil",
            r"\bFLOOR\b": "np.floor",
            r"\bSUM\b": "np.sum",
            r"\bPRODUCT\b": "np.prod",
            r"\bDOT_PRODUCT\b": "np.dot",
            r"\bMATMUL\b": "np.matmul",
            r"\bRESHAPE\b": "np.reshape",
            r"\bALLOCATE\b": "np.empty",
            r"\bSIZE\b": "np.size",
        }

        self.conditional_ops_map = {
            ">": ast.Gt(),
            ">=": ast.GtE(),
            "<": ast.Lt(),
            "<=": ast.LtE(),
            "!=": ast.NotEq(),
            "==": ast.Eq(),
            "not": ast.Not(),
            "and": ast.And(),
            "or": ast.Or(),
        }

        self.logger = Logger()
        self.logger.show_header("F2NP")

        self.func_name = None

    def append_to_current_parent(self, stmt: ast.AST, control_stack: list) -> None:
        """
        Append *stmt* to the body of the current control-flow parent.

        The current parent is the top of *control_stack*, which may be an
        ``ast.If``/``ast.For``/``ast.While`` node (appended to its
        ``.body``), a bare Python list (an ``orelse`` list pushed
        directly), or a dict carrying an ``'if_chain'`` key (from
        :attr:`recursive_ast`'s ``SELECT CASE`` handling). If
        *control_stack* is empty, *stmt* is pushed onto it directly,
        becoming the new top-level container for subsequent appends.

        Parameters
        ----------
        stmt : ast.AST
            The statement to attach to the current control-flow scope.
        control_stack : list
            The stack tracking currently open loop/conditional/select
            blocks, mutated in place.

        Raises
        ------
        RuntimeError
            If the top of *control_stack* is neither an object with a
            ``.body`` list, a bare list, nor a recognised dict.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not control_stack:
                return
            current_parent = None
            if control_stack and len(control_stack) > 0:
                current_parent = control_stack[-1]
            else:
                current_parent = control_stack
            if current_parent is not None and (
                hasattr(current_parent, "body") or isinstance(current_parent, list)
            ):
                if hasattr(current_parent, "body") and isinstance(
                    current_parent.body, list
                ):
                    current_parent.body.append(stmt)

                elif isinstance(current_parent, list):
                    current_parent.append(stmt)

                else:
                    raise RuntimeError(
                        "Expected parent with 'body' attribute for nested control block"
                    )

            elif current_parent is not None and isinstance(current_parent, dict):
                if_chain = current_parent["if_chain"]
                if hasattr(if_chain, "body") and isinstance(if_chain.body, list):
                    if_chain.body.append(stmt)
            else:
                control_stack.append(stmt)
        except Exception:
            raise

    def recursive_ast(
        self,
        block,
        control_stack: list | None = None,
        counters: dict | None = None,
        module_stack: list | None = None,
    ) -> tuple[list, dict, list]:
        """
        Recursively walk a Fortran AST block and build the equivalent
        Python AST.

        The main entry point of :class:`F2NP`. Iterates ``block.content``
        and dispatches each child node by type to a dedicated handler,
        threading three pieces of mutable state through the recursion:

        - *control_stack* — tracks open loop/conditional/select-case
          blocks so that ``END IF``/``END DO``/``END SELECT`` can be
          matched against possibly nested and chained (``ELSE IF``)
          constructs, since Fortran's block-closing statements don't carry
          explicit nesting depth the way Python's indentation does.
        - *counters* — per-construct-kind nesting counts (``'do'``,
          ``'if'``, ``'elif'``, ``'ifwhere'``, ``'elifwhere'``, ``'case'``)
          used to decide whether a freshly built statement belongs at
          module level or inside the currently open block.
        - *module_stack* — the top-level container statements are flushed
          into once their enclosing block closes.

        Dispatch table (non-exhaustive, by Fortran construct):

        - ``Nonlabel_Do_Stmt`` → :meth:`handle_do_stmt`, pushed onto
          *control_stack*.
        - ``If_Then_Stmt`` → :meth:`handle_if_condition`, pushed onto
          *control_stack*; ``If_Stmt`` (single-line ``IF ... THEN``
          without a block) is built inline.
        - ``Assignment_Stmt`` → :meth:`handle_assignment`; when inside a
          ``WHERE``/``ELSEWHERE`` region (``counters['ifwhere']`` or
          ``counters['elifwhere']`` > 0), the LHS is additionally wrapped
          in a ``[mask]`` subscript and the RHS routed through
          :meth:`apply_mask_to_rhs`.
        - ``Else_If_Stmt`` / ``Else_Stmt`` → attached to the ``orelse`` of
          the parent ``ast.If`` on *control_stack*, using
          :meth:`handle_if_condition` for the new ``elif`` branch.
          - ``End_Do_Stmt`` / ``End_If_Stmt`` → pops the matching frames from
          *control_stack* (including unwinding chained ``elif`` frames),
          flushing into *module_stack* once both ``'do'`` and ``'if'``
          counters return to zero.
        - ``Print_Stmt`` / ``Write_Stmt`` → :meth:`handle_print_stmt`.
        - ``Call_Stmt`` → :meth:`handle_call_stmt`.
        - ``Where_Stmt`` / ``Where_Construct_Stmt`` /
          ``Masked_Elsewhere_Stmt`` / ``Elsewhere_Stmt`` / ``End_Where_Stmt``
          → lowered to ``if mask.any(): ...`` chains via
          :meth:`handle_where_stmt`, mirroring the ``If``/``Else``
          bookkeeping above but tracked through ``counters['ifwhere']``
          and ``counters['elifwhere']``.
        - ``Subroutine_Stmt`` / ``Function_Stmt`` → :meth:`handle_subroutine_stmt`,
          setting :attr:`func_name`; the matching ``End_Function_Stmt`` /
          ``End_Subroutine_Stmt`` folds all of *module_stack* into the
          function body and appends a ``return`` node if the parsed suffix
          indicates a function result variable.
        - ``Type_Declaration_Stmt`` → :meth:`handle_type_declaration_stmt`,
          skipped for names already present in :attr:`arg_list` (dummy
          arguments don't need local declarations).
        - ``Cycle_Stmt`` / ``Exit_Stmt`` → ``ast.Continue()`` /
          ``ast.Break()``, requiring an open ``DO`` loop.
        - ``Select_Case_Stmt`` / ``Case_Stmt`` / ``End_Select_Stmt`` →
          built into a chained ``ast.If``/``orelse`` structure via an
          internal dict frame (``{'type': 'select_case', 'switch_expr':
          ..., 'if_chain': ...}``) pushed onto *control_stack*.
        - ``Return_Stmt`` → ``ast.Return()``, with explicit values resolved
          through :meth:`handle_expr`.
        - Any other node type → recurses into it directly via a nested
          :meth:`recursive_ast` call, propagating the same three state
          objects.

        Parameters
        ----------
        block : object
            A node or block from the Fortran AST (must expose a
            ``.content`` list) to translate.
        control_stack : list, optional
            Stack tracking open control-flow constructs. A fresh list is
            created if ``None``.
        counters : dict, optional
            Per-construct nesting counters. A fresh dict with all counts
            at zero is created if ``None``.
        module_stack : list, optional
            Top-level statement container. A fresh list is created if
            ``None``.

        Returns
        -------
        tuple[list, dict, list]
            ``(control_stack, counters, module_stack)`` — the same objects
            passed in (or freshly created), mutated to reflect the final
            state after traversal.

        Raises
        ------
        AttributeError
            If *block* has no ``.content`` attribute.
        Exception
            Re-raises any unexpected error after logging.

        Notes
        -----
        This method recursively walks the Fortran AST, transforming nodes into their Python AST
        equivalents. The stacks and counters assist in maintaining contextual information throughout
        the traversal, supporting accurate translation of control flow and modular constructs.
        """
        if (
            control_stack is None
        ):  # THis will now be used for the loops and conditional elements
            control_stack = []

        if (
            module_stack is None
        ):  # This is the primary stack in which we will contain all the converted/transformed ast code
            module_stack = []

        if counters is None:
            counters = {
                "do": 0,
                "if": 0,
                "elif": 0,
                "ifwhere": 0,
                "elifwhere": 0,
                "case": 0,
            }

        if hasattr(block, "content"):
            idx = 0
            while idx < len(block.content):
                try:
                    child = block.content[idx]
                    if isinstance(child, F23.Nonlabel_Do_Stmt):
                        for_loop = self.handle_do_stmt(child)
                        self.append_to_current_parent(for_loop, control_stack)
                        control_stack.append(for_loop)
                        counters["do"] += 1

                    # Handle IF-THEN
                    elif isinstance(child, F23.If_Then_Stmt):
                        if walk(child, F23.Part_Ref):
                            child = self.handle_assignment(child)

                        if_stmt = self.handle_if_condition(child)
                        self.append_to_current_parent(if_stmt, control_stack)
                        control_stack.append(if_stmt)
                        counters["if"] += 1

                    elif isinstance(child, F23.If_Stmt):
                        if_condition = child.children[0]
                        condition_stmt = child.children[1]
                        if_condition_ast = self.handle_expr(if_condition)
                        condition_stmt_ast = self.handle_expr(condition_stmt)
                        if_stmt = ast.If(
                            test=if_condition_ast, body=[condition_stmt_ast], orelse=[]
                        )
                        if counters["if"] == 0 and counters["do"] == 0:
                            module_stack.append(if_stmt)
                        else:
                            self.append_to_current_parent(if_stmt, control_stack)

                    elif isinstance(child, F23.Assignment_Stmt):
                        stmt = self.handle_assignment(child)

                        if (
                            counters["if"] == 0
                            and counters["do"] == 0
                            and counters["case"] == 0
                        ):
                            if counters["ifwhere"] > 0 or counters["elifwhere"] > 0:
                                # Need to create a deepcopy if not they will share the same address
                                stmt_copy = copy.deepcopy(stmt)
                                # Now we need to modify the stmt itself
                                stmt = ast.Assign(
                                    targets=[
                                        ast.Subscript(
                                            value=stmt_copy.targets[0],
                                            slice=ast.Name(id="mask", ctx=ast.Load()),
                                            ctx=ast.Store(),
                                        )
                                    ],
                                    value=self.apply_mask_to_rhs(stmt_copy.value)
                                    if getattr(self, "extractor", None)
                                    else stmt_copy.value,
                                )

                                if module_stack and isinstance(
                                    module_stack[-1], ast.If | list
                                ):
                                    self.append_to_current_parent(
                                        stmt, control_stack=module_stack
                                    )
                                else:
                                    module_stack.append(stmt)
                            else:
                                module_stack.append(stmt)
                        else:
                            if counters["ifwhere"] > 0:
                                # Need to create a deepcopy if not they will share the same address
                                stmt_copy = copy.deepcopy(stmt)

                                stmt = ast.Assign(
                                    targets=[
                                        ast.Subscript(  # LHS SIDE adjustemeent
                                            value=stmt_copy.targets[0],
                                            slice=ast.Name(id="mask", ctx=ast.Load()),
                                            ctx=ast.Store(),
                                        )
                                    ],
                                    value=self.apply_mask_to_rhs(stmt_copy.value)
                                    if getattr(self, "extractor", None)
                                    else stmt_copy.value,
                                )

                                if control_stack and isinstance(
                                    control_stack[-1], ast.If | list
                                ):
                                    self.append_to_current_parent(
                                        stmt, control_stack=control_stack
                                    )
                                else:
                                    control_stack.append(stmt)
                            else:
                                self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, F23.Else_If_Stmt | F23.Else_Stmt):
                        if not control_stack or not isinstance(
                            control_stack[-1], ast.If
                        ):
                            raise RuntimeError("Else/Else If without a preceding If")

                        parent_if = control_stack[
                            -1
                        ]  # We go back to the parent if of the current else/else if statement

                        if isinstance(child, F23.Else_If_Stmt):
                            # Create new ast.If node for Else If
                            if isinstance(child, F23.Else_If_Stmt) and walk(
                                child, F23.Part_Ref
                            ):
                                child = self.handle_assignment(child)

                            elif_node = self.handle_if_condition(child)
                            while control_stack and not isinstance(
                                control_stack[-1], ast.If
                            ):
                                control_stack.pop()
                            # Attach to orelse of previous If the new instance IF
                            parent_if.orelse = [elif_node]

                            # But we move on to the newly created elif_node
                            control_stack.append(elif_node)
                            counters["elif"] += 1

                        if isinstance(child, F23.Else_Stmt):
                            # https://stackoverflow.com/questions/44728436/difference-between-nested-if-else-and-elif
                            if not control_stack or not isinstance(
                                control_stack[-1], ast.If
                            ):
                                raise RuntimeError("Else without a preceding If")

                            control_stack.append(parent_if.orelse)

                    elif isinstance(child, F23.End_Do_Stmt | F23.End_If_Stmt):
                        if control_stack:
                            if (
                                isinstance(child, F23.End_Do_Stmt)
                                and counters["do"] != 0
                            ):
                                counters["do"] -= 1
                            elif (
                                isinstance(child, F23.End_If_Stmt)
                                and counters["if"] != 0
                            ):
                                counters["if"] -= 1

                            if len(control_stack) > 1:
                                # NOTE: Special handling for ELSE and ELSE IF:
                                # In these cases, we temporarily pushed the `orelse` list (a Python list)
                                # onto the stack so `popped` may be a list instead of an AST node.
                                # If the popped element is a list and the current top of stack is an ast.If node,
                                # then we are finishing an ELSE/ELSE IF(which is basically a IF) block,
                                # and we may also need to pop the corresponding IF.

                                if isinstance(control_stack[-1], list) and isinstance(
                                    control_stack[-2], ast.If
                                ):
                                    control_stack.pop()
                                if counters["elif"] > 0:
                                    # NOTE: Case: nested IF/ELIF chains
                                    # Fortran uses a single END IF to close a chain of IF / ELSE IF / ELSE,
                                    # whereas Python AST uses nested `if` statements in `orelse`.
                                    # We need to pop all the nested `ast.If` nodes representing the
                                    # ELSE IF chain and thanks to the elif statement
                                    # we can ensure that we remove only the corresponding the number of if(elif).
                                    while counters["elif"] > 0:
                                        if isinstance(control_stack[-1], ast.If):
                                            control_stack.pop()
                                            counters["elif"] -= 1
                                        else:
                                            break

                                if len(control_stack) > 1:
                                    control_stack.pop()

                        if counters["do"] == 0 and counters["if"] == 0:
                            module_stack.extend(control_stack)
                            control_stack.clear()

                    elif isinstance(child, F23.Comment):
                        # https://pypi.org/project/ast-comments/
                        pass

                    elif isinstance(child, F23.Print_Stmt | F23.Write_Stmt):
                        # NOTE: FOR NOW WE will treat it as a print since numout = 6 in the class
                        # https://docs.oracle.com/cd/E19957-01/805-4940/6j4m1u7oh/index.html
                        # https://stackoverflow.com/questions/28620899/difference-between-write-and-write6-in-fortran
                        # Since 1363 is to write into the files

                        if not any(
                            walk(
                                walk(child, F23.Io_Control_Spec),
                                F23.Int_Literal_Constant,
                            )
                        ):
                            stmt = self.handle_print_stmt(child)
                            if counters["do"] == 0 and counters["if"] == 0:
                                module_stack.append(stmt)
                            else:
                                self.append_to_current_parent(stmt, control_stack)
                        else:
                            raise NotImplementedError(
                                "When 1363 is present, the approach hasn't be implemented yet"
                            )

                    elif isinstance(child, F23.Call_Stmt):
                        stmt = self.handle_call_stmt(child)
                        if stmt is None:
                            raise ValueError(
                                "Call statement is None due to prior error"
                            )
                        if (
                            counters["do"] == 0
                            and counters["if"] == 0
                            and counters["case"] == 0
                        ):
                            if not isinstance(stmt, ast.Pass):
                                module_stack.append(stmt)
                        else:
                            if not isinstance(stmt, ast.Pass):
                                self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, F23.Where_Stmt):
                        condition_stmt_ast = self.handle_expr(child.children[0])
                        value_stmt_ast = self.handle_assignment(child.children[1])

                        stmt_copy = copy.deepcopy(value_stmt_ast)
                        stmt = ast.Assign(
                            targets=[
                                ast.Subscript(
                                    value=stmt_copy.targets[0],
                                    slice=condition_stmt_ast,
                                    ctx=ast.Store(),
                                )
                            ],
                            value=stmt_copy.value,
                        )

                        if counters["do"] == 0 and counters["if"] == 0:
                            module_stack.append(stmt)
                        else:
                            self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, F23.Where_Construct_Stmt):
                        # NOTE: Where statements are transformed into If/else of Python
                        stmt = self.handle_where_stmt(child)
                        counters["ifwhere"] += 1
                        if counters["do"] == 0 and counters["if"] == 0:
                            self.append_to_current_parent(stmt, module_stack)
                            module_stack.append(stmt)
                        else:
                            self.append_to_current_parent(stmt, control_stack)
                            control_stack.append(stmt)

                    elif isinstance(
                        child, F23.Masked_Elsewhere_Stmt | F23.Elsewhere_Stmt
                    ):
                        if counters["do"] == 0 and counters["if"] == 0:
                            stack_to_check = module_stack
                        else:
                            stack_to_check = control_stack

                        if not stack_to_check or not isinstance(
                            stack_to_check[-1], ast.If
                        ):
                            raise RuntimeError(
                                "Else/Else If for where stmt without a preceding If"
                            )

                        parent_if = stack_to_check[-1]
                        if isinstance(child, F23.Masked_Elsewhere_Stmt):
                            # Create new ast.If node for Else If
                            if isinstance(child, F23.Masked_Elsewhere_Stmt) and walk(
                                child, F23.Part_Ref
                            ):
                                child = self.handle_assignment(child)

                            elif_node = self.handle_where_stmt(child)
                            # Attach to orelse of previous If the new instance IF
                            parent_if.orelse.append(elif_node)
                            # But we move on to the newly created elif_node
                            stack_to_check.append(elif_node)
                            counters["elifwhere"] += 1

                        if isinstance(child, F23.Elsewhere_Stmt):
                            stack_to_check.append(parent_if.orelse)

                    elif isinstance(child, F23.End_Where_Stmt):
                        if counters["do"] == 0 and counters["if"] == 0:
                            stack_to_check = module_stack
                        else:
                            stack_to_check = control_stack

                        if stack_to_check and counters["ifwhere"] != 0:
                            counters["ifwhere"] -= 1

                        if stack_to_check:
                            if len(stack_to_check) > 1:
                                if isinstance(stack_to_check[-1], list) and isinstance(
                                    stack_to_check[-2], ast.If
                                ):
                                    stack_to_check.pop()
                                if counters["elifwhere"] > 0:
                                    while counters["elifwhere"] > 0:
                                        if isinstance(stack_to_check[-1], ast.If):
                                            stack_to_check.pop()
                                            counters["elifwhere"] -= 1
                                        else:
                                            break
                                if len(stack_to_check) > 1:
                                    stack_to_check.pop()

                    elif isinstance(child, F23.Implicit_Stmt):
                        pass

                    elif isinstance(child, F23.Subroutine_Stmt):
                        self.func_name = child.items[1].string
                        stmt = self.handle_subroutine_stmt(child)
                        if stmt is None:
                            raise ValueError(
                                "AST subroutine function statement is None due to prior error"
                            )
                        module_stack.append(stmt)

                    elif isinstance(child, F23.Type_Declaration_Stmt):
                        if walk(child, F23.Explicit_Shape_Spec) or walk(
                            child, F23.Attr_Spec
                        ):
                            if (
                                walk(walk(child, F23.Entity_Decl), F23.Name)[0].string
                                not in self.arg_list
                            ):
                                stmt = self.handle_type_declaration_stmt(child)
                                if counters["do"] == 0 and counters["if"] == 0:
                                    module_stack.append(stmt)
                                else:
                                    self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(
                        child, F23.End_Function_Stmt | F23.End_Subroutine_Stmt
                    ):
                        func_def = module_stack[0]
                        if hasattr(func_def, "body"):
                            func_body = func_def.body
                            for node in module_stack[1:]:
                                func_body.append(node)

                            module_stack[:] = [func_def]
                            if isinstance(child, F23.End_Function_Stmt):
                                # Try to check if the element SUFFIX is present or not, which usually means that we have a function and not a subroutine
                                return_stmt = walk(
                                    walk(child.parent, F23.Suffix), F23.Name
                                )[0]
                                return_node = ast.Return()
                                if return_stmt:
                                    return_node.value = ast.Name(
                                        id=return_stmt.string, ctx=ast.Load()
                                    )

                                    func_def.body.append(return_node)
                            self.func_name = None
                        else:
                            raise AttributeError(
                                "Function definition does not have a 'body' attribute"
                            )

                    elif isinstance(child, F23.Function_Stmt):
                        self.func_name = child.items[1].string
                        stmt = self.handle_subroutine_stmt(child)
                        if stmt is None:
                            raise ValueError(
                                "AST function statement is None due to prior error"
                            )

                        module_stack.append(stmt)

                    elif isinstance(
                        child, F23.Cycle_Stmt
                    ):  # The equivalent of this in Python is 'continue'
                        stmt = ast.Continue()

                        if counters["do"] == 0:
                            raise ValueError(
                                "Continue(CYCLE in Fortran) stmt needs to be placed inside a For loop"
                            )
                        else:
                            self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, F23.Exit_Stmt):
                        stmt = ast.Break()

                        if counters["do"] == 0:
                            raise ValueError(
                                "Break(EXIT in Fortran) stmt needs to be placed inside a For loop"
                            )
                        else:
                            self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, F23.Select_Case_Stmt):
                        switch_expr = self.handle_expr(child.children[0])
                        case_stack = {
                            "type": "select_case",
                            "switch_expr": switch_expr,
                            "if_chain": None,
                        }
                        control_stack.append(case_stack)
                        counters["case"] = counters.get("case", 0) + 1

                    elif isinstance(child, F23.Case_Stmt):
                        # Each CASE (...) or CASE DEFAULT
                        select_info = next(
                            (
                                s
                                for s in reversed(control_stack)
                                if isinstance(s, dict)
                                and s.get("type") == "select_case"
                            ),
                            None,
                        )
                        if not select_info:
                            raise RuntimeError(
                                "CASE statement found without an enclosing SELECT CASE"
                            )

                        switch_expr = select_info["switch_expr"]

                        # Extract the selector (which can be None for DEFAULT)
                        selector_node = child.children[0]
                        # This may have Case_Value_Range_List or None
                        value_list = getattr(selector_node, "children", [None])[0]

                        if value_list is None:
                            # CASE DEFAULT
                            prev_if = select_info["if_chain"]
                            if prev_if is None:
                                raise RuntimeError(
                                    "CASE DEFAULT without any preceding CASE"
                                )
                            # Default -> orelse list of the last if
                            prev_if.orelse = []
                            control_stack.append(prev_if.orelse)
                        else:
                            # Extract value from Case_Value_Range_List
                            case_value_node = walk(selector_node, F23.Name)[0]
                            case_value = self.handle_expr(case_value_node)

                            case_if = ast.If(
                                test=ast.Compare(
                                    left=switch_expr,
                                    ops=[ast.Eq()],
                                    comparators=[case_value],
                                ),
                                body=[],
                                orelse=[],
                            )

                            if select_info["if_chain"] is None:
                                # First CASE — attach to module or enclosing control
                                select_info["if_chain"] = case_if
                                if counters["do"] == 0 and counters["if"] == 0:
                                    module_stack.append(case_if)
                                else:
                                    self.append_to_current_parent(
                                        case_if, control_stack
                                    )
                            else:
                                # Subsequent CASE — attach to orelse of previous one
                                prev_if = select_info["if_chain"]
                                while prev_if.orelse and isinstance(
                                    prev_if.orelse[0], ast.If
                                ):
                                    prev_if = prev_if.orelse[0]
                                prev_if.orelse = [case_if]
                                select_info["if_chain"] = case_if

                    elif isinstance(child, F23.End_Select_Stmt):
                        if counters.get("case", 0) > 0:
                            counters["case"] -= 1

                        while control_stack and not (
                            isinstance(control_stack[-1], dict)
                            and control_stack[-1].get("type") == "select_case"
                        ):
                            control_stack.pop()
                        if control_stack and isinstance(control_stack[-1], dict):
                            control_stack.pop()

                    elif isinstance(child, F23.Return_Stmt):
                        args = []
                        for child in child.children:
                            if child:
                                args.append(self.handle_expr(child))

                        if args:
                            if len(args) == 1:
                                stmt = ast.Return(value=args[0])
                            else:
                                stmt = ast.Return(value=ast.Tuple(elts=args))
                        else:
                            stmt = ast.Return()

                        if counters["do"] == 0 and counters["if"] == 0:
                            module_stack.append(stmt)
                        else:
                            self.append_to_current_parent(stmt, control_stack)
                    elif isinstance(child, F23.Intrinsic_Stmt):
                        pass
                    else:
                        self.recursive_ast(
                            child,
                            control_stack=control_stack,
                            counters=counters,
                            module_stack=module_stack,
                        )

                except Exception as e:
                    self.logger.exception(
                        f"Exception in recursive block at index {idx}, block type: {type(child).__name__}",
                        e,
                    )
                    raise

                idx += 1
        else:
            raise AttributeError(
                f"Block doesn't have the `content` attribute for the block : {block}, {type(block)}"
            )

        return control_stack, counters, module_stack

    def handle_subroutine_stmt(
        self,
        stmt: F23.Subroutine_Stmt | F23.Function_Stmt,
    ) -> ast.FunctionDef:
        """
        Convert a Fortran ``SUBROUTINE``/``FUNCTION`` statement into a
        Python ``ast.FunctionDef`` shell.

        Extracts the routine name and its dummy argument list, populating
        :attr:`arg_list` so that later declarations
        (:meth:`handle_type_declaration_stmt`, dispatched from
        :meth:`recursive_ast`) can skip names that are already parameters.
        The returned function has an empty body; statements are appended
        to it later as :meth:`recursive_ast` processes the routine's
        content, and the body is folded in when the matching
        ``End_Function_Stmt``/``End_Subroutine_Stmt`` is reached.

        Parameters
        ----------
        stmt : F23.Subroutine_Stmt or F23.Function_Stmt
            The Fortran subroutine/function statement to convert.

        Returns
        -------
        ast.FunctionDef
            Function definition with name and arguments populated, empty
            body.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            self.arg_list = []
            for child in stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Name):
                    subroutine_name = child.tostr()
                elif isinstance(child, F23.Dummy_Arg_List):
                    # arg_list = child.tostr()
                    for gchild in child.children:
                        self.arg_list.append(gchild.tostr())

            args = [ast.arg(arg) for arg in self.arg_list]
            function_def = ast.FunctionDef(
                name=subroutine_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[],
                decorator_list=[],
            )
            return function_def
        except Exception as e:
            self.logger.exception("Exception in handle_subroutine_stmt", e)
            raise

    def handle_call_stmt(self, stmt: F23.Call_Stmt) -> ast.Expr:
        """
        Convert a Fortran ``CALL`` statement into a Python call expression.

        Two cases are handled based on whether the called routine appears
        in :attr:`extractor`'s ``allowed_external_subroutines``: ordinary
        calls are built via :meth:`_build_regular_call`; calls to routines
        in that set are instead treated as logging calls and built via
        :meth:`_build_call_logging` (used for routines like XIOS field
        send/receive that have no meaningful Python translation but whose
        invocation is worth recording).

        Parameters
        ----------
        stmt : F23.Call_Stmt
            The Fortran ``CALL`` statement to convert.

        Returns
        -------
        ast.Expr
            An expression statement wrapping either a regular function
            call or a logging call.

        Raises
        ------
        AttributeError
            If *stmt* has no ``children`` attribute.
        ValueError
            If *stmt* does not have exactly two children (function name
            and argument spec list).
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            if not hasattr(stmt, "children"):
                raise AttributeError("stmt has no children")

            if len(stmt.children) != 2:
                raise ValueError(
                    "Expected two children: function_name and args_spec_list"
                )

            function_node, args_spec_list = stmt.children
            func_name = function_node.string

            # Case 1: regular function call
            if func_name not in self.extractor.allowed_external_subroutines:
                return ast.Expr(
                    value=self._build_regular_call(func_name, args_spec_list)
                )

            # Case 2: special/logging call
            return self._build_call_logging(func_name, args_spec_list)

        except Exception:
            self.logger.exception("Exception in handle_call_stmt")
            raise

    def _build_regular_call(self, func_name: str, args_spec_list: F23.Base) -> ast.Call:
        """
        Build a plain Python function call AST node.

        Each argument in *args_spec_list* is converted via
        :meth:`handle_expr`. Used both by :meth:`handle_call_stmt` for
        ordinary ``CALL`` statements and by :meth:`handle_part_ref` when a
        part reference resolves to a function call rather than an array
        access.

        Parameters
        ----------
        func_name : str
            Name of the function to call.
        args_spec_list : F23.Base
            Fortran node containing the call's argument list.

        Returns
        -------
        ast.Call
            The constructed call expression.
        """
        args = [self.handle_expr(arg) for arg in args_spec_list.children]

        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()), args=args, keywords=[]
        )

    def _build_call_logging(self, func_name: str, args_spec_list: F23.Base) -> ast.Expr:
        """
        Convert a special Fortran ``CALL`` into a Python ``logging`` call.

        Routines in a fixed set (XIOS field send/receive) are logged at
        ``INFO`` level with an ``"INFO: <name>:"`` prefix; any other
        routine in :attr:`extractor`'s allowed-external set is logged at
        ``ERROR`` level with an ``"Exception:"`` prefix. String literal
        arguments are extracted via :meth:`_extract_string_args` and
        joined into the log message, then passed to
        :meth:`_build_logging_call` as a raw message.

        Parameters
        ----------
        func_name : str
            Name of the function being called.
        args_spec_list : F23.Base
            Fortran node representing the call's arguments.

        Returns
        -------
        ast.Expr
            Expression statement wrapping the logging call.
        """

        special_skip_functions = {
            "xios_orchidee_send_field",
            "xios_orchidee_recv_field",
        }

        string_args = self._extract_string_args(args_spec_list)

        is_info = func_name in special_skip_functions
        level = "info" if is_info else "error"

        prefix = f"INFO: {func_name}:" if is_info else "Exception:"

        message = " ".join([prefix] + string_args)
        call = self._build_logging_call(level=level, raw_message=message)

        return ast.Expr(value=call)

    def _extract_string_args(self, args_spec_list: F23.Base) -> list[str]:
        """
        Extract string literal values from a call's argument list.

        Only ``F23.Char_Literal_Constant`` arguments are extracted; other
        argument kinds are ignored. Used by :meth:`_build_call_logging` to
        assemble the text of a logging message.

        Parameters
        ----------
        args_spec_list : F23.Base
            Fortran node containing the call's argument list.

        Returns
        -------
        list[str]
            String literal values, with surrounding quotes stripped.
        """
        values = []

        for arg in args_spec_list.children:
            if isinstance(arg, F23.Char_Literal_Constant):
                # safer strip (only quotes, not spaces inside)
                value = arg.items[0].strip().strip("'").strip('"')
                values.append(value)

        return values

    def handle_type_declaration_stmt(
        self, stmt: F23.Type_Declaration_Stmt
    ) -> ast.Assign:
        """
        Convert a Fortran type declaration into a NumPy-backed Python
        assignment.

        Dispatches to one of three builders depending on the declaration
        shape, determined via :meth:`_extract_shapes`,
        :meth:`_extract_decl_name`, and :meth:`_extract_dtype`:

        - An array with an explicit constructor (``walk(stmt,
          F23.Array_Constructor)`` non-empty)
          → :meth:`_build_array_from_constructor`.
        - An array without a constructor → :meth:`_build_zeros_array`.
        - A scalar declaration → :meth:`_handle_scalar_declaration`.

        Parameters
        ----------
        stmt : F23.Type_Declaration_Stmt
            The Fortran type declaration statement to convert.

        Returns
        -------
        ast.Assign
            Assignment initialising the declared variable.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            func_arrays = self._extract_shapes(stmt)

            name = self._extract_decl_name(stmt)
            dtype_name, dtype_attr = self._extract_dtype(stmt)

            # Case 1: Array declaration
            if func_arrays:
                if bool(walk(stmt, F23.Array_Constructor)):
                    value = self._build_array_from_constructor(
                        stmt, dtype_name, dtype_attr
                    )
                else:
                    value = self._build_zeros_array(func_arrays, dtype_name, dtype_attr)

                return ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())], value=value
                )

            # Case 2: Scalar declaration
            return self._handle_scalar_declaration(stmt, dtype_name, dtype_attr)

        except Exception:
            self.logger.exception("Exception in handle_type_declaration_stmt")
            raise

    def _extract_shapes(self, stmt: F23.Base) -> list[ast.AST]:
        """
        Extract per-dimension size expressions from an array declaration's
        explicit shape specification.

        For each ``F23.Explicit_Shape_Spec`` found, builds the Python
        expression for that dimension's extent: when both lower and upper
        bounds are present, ``upper - lower + 1``; when only one bound is
        present, that bound alone (matching Fortran's implicit
        lower-bound-1 convention). Each bound is converted via
        :meth:`handle_expr`. Consumed by :meth:`handle_type_declaration_stmt`
        to decide between array and scalar handling, and by
        :meth:`_build_zeros_array` as the shape tuple.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing the array declaration.

        Returns
        -------
        list[ast.AST]
            One size expression per declared dimension.

        Raises
        ------
        ValueError
            If a dimension has neither a lower nor an upper bound.
        """
        shape = []

        for dim in walk(stmt, F23.Explicit_Shape_Spec):
            lb, ub = dim.children

            if lb and ub:
                # This for the formula: ub - lb + 1
                lower = self.handle_expr(lb)
                upper = self.handle_expr(ub)

                shape.append(
                    ast.BinOp(
                        left=ast.BinOp(left=upper, op=ast.Sub(), right=lower),
                        op=ast.Add(),
                        right=ast.Constant(1),
                    )
                )
            elif lb:
                shape.append(self.handle_expr(lb))
            elif ub:
                shape.append(self.handle_expr(ub))
            else:
                raise ValueError("Invalid shape specification")

        return shape

    def _extract_decl_name(self, stmt: F23.Base) -> str:
        """
        Extract the declared variable's name from an entity declaration.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing the entity declaration.

        Returns
        -------
        str
            The declared variable's name.
        """
        return walk(walk(stmt, F23.Entity_Decl), F23.Name)[0].string

    def _extract_dtype(self, stmt: F23.Base) -> tuple[str, str]:
        """
        Resolve a Fortran intrinsic type to its NumPy module/attribute
        pair.

        Maps ``REAL`` → ``np.float64``, ``INTEGER`` → ``np.int32``,
        ``LOGICAL`` → ``np.bool``. Used by
        :meth:`handle_type_declaration_stmt` to determine the ``dtype``
        keyword passed to :meth:`_build_array_from_constructor`,
        :meth:`_build_zeros_array`, or :meth:`_handle_scalar_declaration`.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node containing the type specification.

        Returns
        -------
        tuple[str, str]
            ``(module_name, type_name)``, e.g. ``('np', 'float64')``.

        Raises
        ------
        KeyError
            If the Fortran intrinsic type has no NumPy mapping.
        """
        TYPE = {"REAL": "np.float64", "INTEGER": "np.int32", "LOGICAL": "np.bool"}

        fdtype = walk(stmt, F23.Intrinsic_Type_Spec)[0].children[0]

        np_dtype = TYPE.get(fdtype)
        if np_dtype is None:
            raise KeyError(f"Unsupported dtype: {fdtype}")

        idx, attr = np_dtype.split(".")
        return idx, attr

    def _build_array_from_constructor(
        self,
        stmt: F23.Base,
        idx: str = "np",
        attr: str = "float64",
    ) -> ast.Call:
        """
        Build a ``np.array([...], dtype=...)`` call from a Fortran array
        constructor.

        Each element of the constructor's value list is converted via
        :meth:`handle_expr`. Invoked from
        :meth:`handle_type_declaration_stmt` when the declaration includes
        an explicit ``F23.Array_Constructor``.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing the array constructor declaration.
        idx : str, optional
            NumPy module alias, by default ``'np'``.
        attr : str, optional
            NumPy dtype attribute name, by default ``'float64'``.

        Returns
        -------
        ast.Call
            The constructed ``np.array(...)`` call.
        """

        array_list = walk(walk(stmt, F23.Array_Constructor), F23.Ac_Value_List)[0]

        elements = [self.handle_expr(val) for val in array_list.children]

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()
            ),
            args=[ast.List(elts=elements, ctx=ast.Load())],
            keywords=[
                ast.keyword(
                    arg="dtype",
                    value=ast.Attribute(
                        value=ast.Name(id=idx, ctx=ast.Load()),
                        attr=attr,
                        ctx=ast.Load(),
                    ),
                )
            ],
        )

    def _build_zeros_array(
        self,
        shape: list[ast.expr],
        idx: str = "np",
        attr: str = "float64",
    ) -> ast.Call:
        """
        Build a ``np.zeros((...), dtype=...)`` call for an array
        declaration with no explicit constructor.

        Invoked from :meth:`handle_type_declaration_stmt` with the shape
        expressions produced by :meth:`_extract_shapes`.

        Parameters
        ----------
        shape : list[ast.expr]
            Per-dimension size expressions.
        idx : str, optional
            NumPy module alias, by default ``'np'``.
        attr : str, optional
            NumPy dtype attribute name, by default ``'float64'``.

        Returns
        -------
        ast.Call
            The constructed ``np.zeros(...)`` call.
        """
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=idx, ctx=ast.Load()), attr="zeros", ctx=ast.Load()
            ),
            args=[ast.Tuple(elts=shape, ctx=ast.Load())],
            keywords=[
                ast.keyword(
                    arg="dtype",
                    value=ast.Attribute(
                        value=ast.Name(id=idx, ctx=ast.Load()),
                        attr=attr,
                        ctx=ast.Load(),
                    ),
                )
            ],
        )

    def _handle_scalar_declaration(
        self,
        stmt: F23.Base,
        idx: str = "np",
        attr: str = "float64",
    ) -> ast.Assign | None:
        """
        Build an initialising assignment for a scalar variable
        declaration.

        Handles ``REAL``/``INTEGER`` declarations by wrapping the
        initial-value expression (resolved via :meth:`handle_expr`) in the
        matching NumPy scalar constructor, and ``LOGICAL`` declarations by
        parsing the Fortran boolean literal directly into
        ``np.bool(True/False)``. Invoked from
        :meth:`handle_type_declaration_stmt` when the declaration is not
        an array.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing the scalar declaration.
        idx : str, optional
            NumPy module alias, by default ``'np'``.
        attr : str, optional
            NumPy dtype attribute name, by default ``'float64'``.

        Returns
        -------
        Optional[ast.Assign]
            The initialising assignment, or ``None`` if the declaration
            has no initial value.
        """
        intrinsic_type_spec, _, entity_decl_list = stmt.children

        for entity_decl in entity_decl_list.children:
            var_name, _, _, initialization = entity_decl.children

            target = ast.Name(id=var_name.string, ctx=ast.Store())

            if initialization is None:
                return None  # or default value if needed

            _, value_node = initialization.children
            value_ast = self.handle_expr(value_node)

            # Numeric types
            if intrinsic_type_spec.children[0] in ["REAL", "INTEGER"]:
                return ast.Assign(
                    targets=[target],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=idx, ctx=ast.Load()),
                            attr=attr,
                            ctx=ast.Load(),
                        ),
                        args=[value_ast],
                        keywords=[],
                    ),
                )

            # Logical
            if intrinsic_type_spec.children[0] == "LOGICAL":
                bool_val = value_node.string.strip(".").upper() == "TRUE"

                return ast.Assign(
                    targets=[target],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="bool",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=bool_val)],
                        keywords=[],
                    ),
                )

    def simplify_limits(self, expression) -> str:
        """
        Simplifies expressions for loop bounds or array dimensions, combining constants
        and variables in the expression.

        Parameters
        ----------
        expression
            The expression representing loop bounds or dimensions.

        Returns
        -------
        str
            Simplified version of the expression.
        """
        terms = re.split(r"\s*([+\-])\s*", expression)
        numbers = []
        variables = []
        for i in range(len(terms)):
            if i % 2 == 0:
                if terms[i].isdigit() or (
                    terms[i].startswith("-") and terms[i][1:].isdigit()
                ):
                    if i > 0 and terms[i - 1] == "-":
                        numbers.append(-int(terms[i]))
                    else:
                        numbers.append(int(terms[i]))
                elif terms[i].strip():
                    variables.append(terms[i].strip())

        total = sum(numbers)
        new_expression = " + ".join(variables)
        if total != 0:
            sign = "+" if total > 0 else "-"
            new_expression += f" {sign} {abs(total)}"
        return new_expression.lstrip("+ ").strip()

    def handle_where_stmt(self, stmt: F23.Where_Construct_Stmt) -> ast.If:
        """
        Convert a Fortran ``WHERE`` construct into an ``if mask.any():``
        block.

        Extracts the mask condition via :meth:`_extract_where_masks`
        (currently only single-condition ``WHERE`` blocks are supported),
        wraps it in an ``.any()``/``np.any()`` test via
        :meth:`_build_any_call`, and assigns the raw mask expression to a
        ``mask`` variable inside the block body via
        :meth:`_build_mask_assignment`. The body's actual masked
        assignments are filled in separately by :meth:`recursive_ast` as
        it processes the construct's statements.

        Parameters
        ----------
        stmt : F23.Where_Construct_Stmt
            The Fortran ``WHERE`` construct statement.

        Returns
        -------
        ast.If
            ``if <mask>.any(): mask = <mask_expr>`` with an empty
            ``orelse``.

        Raises
        ------
        NotImplementedError
            If the ``WHERE`` statement has more than one mask condition.
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            masks = self._extract_where_masks(stmt)

            if len(masks) != 1:
                raise NotImplementedError(
                    "handle_where_stmt with multiple conditions is not implemented"
                )

            mask_ast = masks[0]
            test_expr = self._build_any_call(mask_ast)

            return ast.If(
                test=test_expr, body=[self._build_mask_assignment(mask_ast)], orelse=[]
            )

        except Exception:
            self.logger.exception("Exception in handle_where_stmt")
            raise

    def _extract_where_masks(self, stmt: F23.Base) -> list[ast.AST]:
        """
        Extract and convert each non-``None`` child of a ``WHERE``
        statement into a Python expression.

        Each child is resolved via :meth:`handle_expr`. Used by
        :meth:`handle_where_stmt`.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing the ``WHERE`` statement.

        Returns
        -------
        list[ast.AST]
            One expression per mask condition found.
        """
        return [self.handle_expr(child) for child in stmt.children if child is not None]

    def _build_any_call(self, expr: ast.AST) -> ast.Call:
        """
        Build a ``.any()`` or ``np.any()`` call wrapping *expr*.

        Subscript expressions (array references) use the free-function
        form ``np.any(expr)``; any other expression uses the method form
        ``expr.any()``. Used by :meth:`handle_where_stmt` to build the
        ``WHERE`` block's guard condition.

        Parameters
        ----------
        expr : ast.AST
            The mask expression to wrap.

        Returns
        -------
        ast.Call
            The ``.any()``/``np.any()`` call.
        """
        if isinstance(expr, ast.Subscript):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()), attr="any", ctx=ast.Load()
                ),
                args=[expr],
                keywords=[],
            )
        return ast.Call(
            func=ast.Attribute(value=expr, attr="any", ctx=ast.Load()),
            args=[],
            keywords=[],
        )

    def _build_mask_assignment(self, mask_ast: ast.AST) -> ast.Assign:
        """
        Build the ``mask = <mask_ast>`` assignment placed inside a
        ``WHERE``-derived ``if`` block.

        Parameters
        ----------
        mask_ast : ast.AST
            The mask expression to assign.

        Returns
        -------
        ast.Assign
            Assignment to the variable named ``mask``.
        """
        return ast.Assign(
            targets=[ast.Name(id="mask", ctx=ast.Store())], value=mask_ast
        )

    def get_conventional_var(self, candidate_var: str, upper_key: str) -> str | None:
        """
        Resolve a loop variable name to its "conventional" counterpart for
        a given dimension key.

        Builds a mapping from :attr:`extractor`'s ``loop_dict`` (which
        groups loop-variable names by the array dimension they index) and
        looks up *candidate_var* under the normalised *upper_key*. For
        dimensions with a single associated variable, that variable maps
        to itself; for dimensions with two associated variables, both map
        to the second (sorted) one — used to canonicalise inconsistent
        Fortran loop-index naming across nested loops indexing the same
        dimension.

        Parameters
        ----------
        candidate_var : str
            The loop variable name to look up.
        upper_key : str
            The dimension key (loop bound expression text) to resolve
            against; whitespace is normalised before lookup.

        Returns
        -------
        Optional[str]
            The conventional variable name, or ``None`` if no mapping
            exists for the given key/variable pair.
        """
        normalized_upper_key = upper_key.replace(" ", "")

        mapping = {}
        for key, var_set in self.extractor.loop_dict.items():
            normalized_key = key.replace(" ", "")  # normalize keys from loop_dict

            if len(var_set) == 1:
                only_var = next(iter(var_set))
                mapping[(normalized_key, only_var)] = only_var
            elif len(var_set) == 2:
                var1, var2 = sorted(var_set)
                mapping[(normalized_key, var1)] = var2
                mapping[(normalized_key, var2)] = var2

        return mapping.get((normalized_upper_key, candidate_var))

    def handle_do_stmt(self, stmt: F23.Nonlabel_Do_Stmt) -> ast.For | ast.While:
        """
        Convert a Fortran ``DO`` loop into an ``ast.For`` or ``ast.While``.

        Loop control with no iteration elements (a ``DO WHILE`` loop) is
        converted to an ``ast.While`` with its test resolved via
        :meth:`handle_expr` and an empty body. Counted ``DO`` loops are
        converted to ``ast.For`` over ``range(start, end, stride)``: the
        start bound is shifted to Python's 0-based indexing via
        :meth:`adjust_start`, the end bound is wrapped in ``int(...)`` when
        it is itself a ``Part_Ref`` (since Fortran array-size expressions
        may be inexact), and the end bound is further corrected for
        negative strides via :meth:`adjust_end_for_stride`.

        Parameters
        ----------
        stmt : F23.Nonlabel_Do_Stmt
            The Fortran ``DO`` statement to convert.

        Returns
        -------
        Union[ast.For, ast.While]
            The Python loop equivalent, with an empty body to be
            populated by the caller (:meth:`recursive_ast`).

        Raises
        ------
        ValueError
            If a ``DO WHILE`` loop's control expression cannot be
            resolved.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            loop_control = walk(stmt, F23.Loop_Control)[0]
            elements = loop_control.children[1]

            # Handle DO WHILE loops
            if not elements:
                loop_expr = next(
                    (self.handle_expr(c) for c in loop_control.items if c), None
                )
                if loop_expr is None:
                    raise ValueError("Empty DO WHILE loop control")
                return ast.While(test=loop_expr, body=[], orelse=[])

            # Handle DO loops
            loop_var, start_end_stride_values = elements[0].string, elements[1]
            start, end = start_end_stride_values[0], start_end_stride_values[1]

            # Adjust start to Python 0-based indexing
            start = self.adjust_start(start)
            stride = (
                start_end_stride_values[2] if len(start_end_stride_values) == 3 else 1
            )

            # Convert bounds and stride to AST nodes
            lower_bound = self.to_ast_constant_or_expr(start)
            if isinstance(end, F23.Part_Ref):
                end_ast = ast.Call(
                    func=ast.Name(id="int", ctx=ast.Load()),
                    args=[self.handle_expr(end)],
                    keywords=[],
                )
            else:
                end_ast = self.handle_expr(end)
            stride_ast = self.to_ast_constant_or_expr(stride)

            # Adjust end if stride is negative
            end_ast = self.adjust_end_for_stride(end_ast, stride_ast)

            return ast.For(
                target=ast.Name(id=loop_var, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[lower_bound, end_ast, stride_ast],
                    keywords=[],
                ),
                body=[],
                orelse=[],
            )
        except Exception:
            self.logger.exception("Exception in handle_do_stmt")
            raise

    def adjust_start(
        self, start: F23.Base
    ) -> F23.Int_Literal_Constant | F23.Level_2_Expr | ast.AST:
        """
        Shift a Fortran loop start bound to Python's 0-based indexing.

        For integer literal starts, the shift is folded directly into a
        new literal via :meth:`simplify_limits`. For starts that are already
        compound expressions (``Level_2_Expr``, e.g. ``nslm - 1``), the
        subtraction is built directly as a Python ``ast.BinOp`` node via
        :meth:`handle_expr` to avoid re-feeding a parsed expression back
        into fparser's constructor. For all other non-literal starts
        (simple variable or other node), a ``Level_2_Expr`` representing
        ``(start) - 1`` is constructed as a string for fparser to parse.

        Parameters
        ----------
        start : F23.Base
            The Fortran loop's start-bound node.

        Returns
        -------
        Union[F23.Int_Literal_Constant, F23.Level_2_Expr, ast.AST]
            The 0-based-adjusted start bound. Returns an
            ``F23.Int_Literal_Constant`` for integer literals, an
            ``ast.BinOp`` for compound ``Level_2_Expr`` inputs, or an
            ``F23.Level_2_Expr`` for simple variable/expression inputs.
            The ``ast.BinOp`` case is handled transparently by
            :meth:`to_ast_constant_or_expr`.
        """
        if isinstance(start, F23.Int_Literal_Constant):
            value = self.simplify_limits(start.tostr() + "-1")
            return F23.Int_Literal_Constant(value or "0")
        if isinstance(start, F23.Level_2_Expr):
            # Already a complex expression — build the subtraction directly in Python AST
            return ast.BinOp(
                left=self.handle_expr(start), op=ast.Sub(), right=ast.Constant(value=1)
            )
        # Simple variable or other node
        return F23.Level_2_Expr(f"({start.tostr()}) - 1")

    def to_ast_constant_or_expr(self, value: int | F23.Base | ast.AST) -> ast.AST:
        """
        Convert a plain Python integer, a Fortran expression node, or an
        already-converted Python AST node into a Python AST node.

        The ``ast.AST`` input case arises when :meth:`adjust_start` returns
        an ``ast.BinOp`` directly (for ``Level_2_Expr`` start bounds) rather
        than a Fortran node, allowing the result to pass through without a
        redundant :meth:`handle_expr` call.

        Parameters
        ----------
        value : int, F23.Base, or ast.AST
            Either a literal Python integer, a Fortran expression node,
            or an already-converted Python AST node.

        Returns
        -------
        ast.AST
            An ``ast.Constant`` for integer input, the node itself for
            ``ast.AST`` input, or the result of :meth:`handle_expr` for
            Fortran node input.
        """
        if isinstance(value, int):
            return ast.Constant(value=value)
        if isinstance(value, ast.AST):
            return value
        return self.handle_expr(value)

    def adjust_end_for_stride(self, end_ast: ast.AST, stride_ast: ast.AST) -> ast.AST:
        """
        Negate a constant loop end bound when the stride is negative.

        ``range()`` requires a negative-direction end bound to itself be
        negative-going for descending loops; this corrects an end bound
        that was parsed as a plain positive constant when the accompanying
        stride is an ``ast.UnaryOp`` (i.e. a negative literal).

        Parameters
        ----------
        end_ast : ast.AST
            The loop's end-bound expression.
        stride_ast : ast.AST
            The loop's stride expression.

        Returns
        -------
        ast.AST
            *end_ast*, with its value negated in place if the stride is
            negative and the end bound is a positive constant; otherwise
            unchanged.
        """
        if (
            isinstance(stride_ast, ast.UnaryOp)
            and isinstance(end_ast, ast.Constant)
            and end_ast.value > 0
        ):
            end_ast.value *= -1
        return end_ast

    def handle_if_condition(
        self, condition: F23.If_Then_Stmt | F23.Else_If_Stmt
    ) -> ast.If:
        """
        Convert a Fortran ``IF``/``ELSE IF`` condition into an ``ast.If``
        node with an empty body.

        Three input shapes are handled: an already-converted Python
        ``ast.AST`` test (used when :meth:`recursive_ast` has pre-processed
        the condition via :meth:`handle_assignment` for part-reference
        conditions); a bare logical name condition (``IF (flag) THEN``),
        resolved via :meth:`handle_expr`; and a general condition requiring
        full assignment-style parsing via :meth:`handle_assignment`.

        Parameters
        ----------
        condition : F23.If_Then_Stmt or F23.Else_If_Stmt
            The Fortran ``IF``/``ELSE IF`` condition statement.

        Returns
        -------
        ast.If
            ``ast.If`` node with the resolved test and empty
            ``body``/``orelse``.

        Raises
        ------
        ValueError
            If *condition* is ``None``.
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            if condition is None:
                raise ValueError("condition argument is None")
            condition_stmt = None
            if isinstance(condition, ast.AST):
                condition_stmt = ast.If(test=condition, body=[], orelse=[])
            else:
                if (
                    hasattr(condition, "children") and condition.children[0] is not None
                ):  # These cases corresponds to the IF/ELSE IF
                    if len(condition.children) == 1 and isinstance(
                        condition.children[0], F23.Name
                    ):  # This is for the logical case
                        condition = self.handle_expr(condition.children[0])
                        condition_stmt = ast.If(test=condition, body=[], orelse=[])
                    else:
                        stmt = self.handle_assignment(condition)
                        condition_stmt = ast.If(test=stmt, body=[], orelse=[])

            return condition_stmt
        except Exception:
            self.logger.exception("Exception in handle_if_condition")
            raise

    def handle_print_stmt(self, stmt: F23.Base) -> ast.Expr:
        """
        Convert a Fortran ``PRINT`` statement into a ``logging.info(...)``
        call.

        Separates string literals from value expressions via
        :meth:`_extract_print_items`, then builds the logging call (as a
        structured f-string when value parts are present) via
        :meth:`_build_logging_call`.

        Parameters
        ----------
        stmt : F23.Base
            The Fortran ``PRINT`` statement.

        Returns
        -------
        ast.Expr
            Expression statement wrapping the logging call.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            string_parts, value_parts = self._extract_print_items(stmt)
            call = self._build_logging_call(
                level="info", string_parts=string_parts, value_parts=value_parts
            )
            return ast.Expr(value=call)

        except Exception:
            self.logger.exception("Exception in handle_print_statement")
            raise

    def _extract_print_items(
        self, stmt: F23.Base
    ) -> tuple[list[ast.AST], list[ast.AST]]:
        """
        Separate a ``PRINT`` statement's output items into string literals
        and value expressions.

        Each item is resolved via :meth:`handle_expr`; character literal
        items are classified as string parts, everything else as value
        parts. Used by :meth:`handle_print_stmt`.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing the ``PRINT`` statement.

        Returns
        -------
        tuple[list[ast.AST], list[ast.AST]]
            ``(string_parts, value_parts)``.
        """

        string_parts = []
        value_parts = []

        for child in stmt.children:
            if isinstance(child, F23.Output_Item_List):
                for elem in child.children:
                    expr = self.handle_expr(elem)

                    if isinstance(elem, F23.Char_Literal_Constant):
                        string_parts.append(expr)
                    else:
                        value_parts.append(expr)

        return string_parts, value_parts

    def _build_logging_call(
        self,
        level: str,
        string_parts: list[ast.AST] = None,
        value_parts: list[ast.AST] = None,
        raw_message: str = None,
    ) -> ast.Call:
        """
        Build a ``logging.<level>(...)`` call covering three message
        shapes.

        When *raw_message* is given, it is logged directly as a single
        string constant (used by :meth:`_build_call_logging`). When
        *value_parts* is non-empty, the message is built as an f-string via
        :meth:`_build_fstring_values` (used by :meth:`handle_print_stmt`
        for ``PRINT`` statements with variables). Otherwise, *string_parts*
        are passed as positional arguments directly.

        Parameters
        ----------
        level : str
            Logging level method name, e.g. ``'info'``, ``'error'``.
        string_parts : list[ast.AST], optional
            String-literal expressions to log.
        value_parts : list[ast.AST], optional
            Variable-value expressions to interpolate.
        raw_message : str, optional
            A raw string message to log directly, bypassing the
            string/value-parts logic.

        Returns
        -------
        ast.Call
            The constructed ``logging.<level>(...)`` call.
        """
        func = ast.Attribute(
            value=ast.Name(id="logging", ctx=ast.Load()), attr=level, ctx=ast.Load()
        )

        # Case 1: raw message (CALL)
        if raw_message is not None:
            return ast.Call(
                func=func, args=[ast.Constant(value=raw_message)], keywords=[]
            )

        # Case 2: structured (PRINT with variables)
        if value_parts:
            values = self._build_fstring_values(string_parts, value_parts)

            return ast.Call(func=func, args=[ast.JoinedStr(values=values)], keywords=[])

        # Case 3: only strings
        return ast.Call(func=func, args=string_parts or [], keywords=[])

    def _build_fstring_values(
        self,
        string_parts: list[ast.AST],
        value_parts: list[ast.AST],
    ) -> list[ast.AST]:
        """
        Interleave string literals and formatted values into an f-string's
        component list.

        Pairs each string part with the following value part (formatted
        via ``ast.FormattedValue``), inserting a ``", "`` separator between
        successive value/string pairs. Used by :meth:`_build_logging_call`
        to construct an ``ast.JoinedStr``.

        Parameters
        ----------
        string_parts : list[ast.AST]
            String-literal expressions.
        value_parts : list[ast.AST]
            Variable-value expressions to format.

        Returns
        -------
        list[ast.AST]
            Components suitable for ``ast.JoinedStr(values=...)``.
        """
        values = []

        max_len = max(len(string_parts), len(value_parts))

        for i in range(max_len):
            if i < len(string_parts) and string_parts[i] is not None:
                values.append(string_parts[i])

            if i < len(value_parts) and value_parts[i] is not None:
                values.append(ast.FormattedValue(value=value_parts[i], conversion=-1))
                # Add separator if not last element
                if i < max_len - 1:
                    values.append(ast.Constant(value=", "))

        return values

    def handle_intrinsic_function_reference(
        self,
        intrinsic_function_reference: F23.Intrinsic_Function_Reference | list,
    ) -> ast.Call:
        """
        Convert a Fortran intrinsic function call into its NumPy/Python
        equivalent.

        Resolves the intrinsic name against :attr:`intrinsic_replacements`;
        intrinsics with no direct mapping fall back to a small hardcoded
        exception table (currently ``EPSILON`` → ``np.finfo(np.float64).eps``).
        Arguments are split into positional and keyword form via
        :meth:`handle_expr`, then normalised against a known call signature
        (via ``normalize_intrinsic_call``/``intrinsic_signatures``) when
        one exists for the intrinsic — this handles intrinsics whose
        Fortran keyword names differ from their NumPy equivalents.
        ``MIN``/``MAX`` with more than two operands are folded into a chain
        of binary ``np.minimum``/``np.maximum`` calls, since Python's
        NumPy equivalents only accept two array arguments at a time.

        Parameters
        ----------
        intrinsic_function_reference : F23.Intrinsic_Function_Reference or List
            The Fortran intrinsic function reference to convert.

        Returns
        -------
        ast.Call
            The constructed Python/NumPy equivalent call.

        Raises
        ------
        NotImplementedError
            If the intrinsic has no NumPy mapping and is not covered by
            the hardcoded exception table.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            intrinsic_name = intrinsic_function_reference.items[0].string.upper()
            pattern = rf"\b{intrinsic_name}\b"
            func_name = self.intrinsic_replacements.get(pattern, None)
            intrinsic_func = None
            if func_name is None:
                # NOTE: https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2025-0/epsilon.html
                # it says that the x that enters must be real for epsilon
                # but for some pythonic functions you might not need to send
                # an argument or any element such the case of epsilon, in python we
                # don't need argument as such, we can already predine the python AST for such case
                instrinsic_exception = {
                    r"\bEPSILON\b": ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="np", ctx=ast.Load()),
                                attr="finfo",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Attribute(
                                    value=ast.Name(id="np", ctx=ast.Load()),
                                    attr="float64",
                                    ctx=ast.Load(),
                                )
                            ],
                            keywords=[],
                        ),
                        attr="eps",
                        ctx=ast.Load(),
                    )
                }
                intrinsic_func = instrinsic_exception.get(pattern, None)
                if not intrinsic_func:
                    raise NotImplementedError(
                        f"Not implemented intrinsic exception:{pattern}"
                    )

            if func_name:
                # Once we retieve the function name we need to identify if the
                # it's just normal instrinics or numpy based intrinsic function
                func = None
                if len(func_name.split(".")) > 1:
                    mod, attr = func_name.split(".")
                    func = ast.Attribute(
                        value=ast.Name(id=mod, ctx=ast.Load()),
                        attr=attr,
                        ctx=ast.Load(),
                    )
                else:
                    func = ast.Name(id=func_name.lower(), ctx=ast.Load())

                # Find the intrinsic arguments, as such it would allow us
                # to retrieve the the actual arguments inside the intrinsic parameter
                intrinsic_args = walk(
                    intrinsic_function_reference, F23.Actual_Arg_Spec_List
                )[0]
                positional_args = []
                keyword_args = {}
                for arg in intrinsic_args.children:
                    if isinstance(arg, F23.Actual_Arg_Spec):
                        kw = self.handle_expr(arg)
                        keyword_args[kw.arg.lower()] = kw.value
                    else:
                        positional_args.append(self.handle_expr(arg))

                signature = intrinsic_signatures.get(intrinsic_name)
                if signature:
                    normalized = normalize_intrinsic_call(
                        signature, positional_args, keyword_args
                    )
                    final_args = []
                    final_keywords = []
                    if "array" in normalized:
                        final_args.append(normalized["array"])
                    # keywords via arg_map
                    for key, value in normalized.items():
                        if key == "array" or value is None:
                            continue
                        py_name = signature.arg_map.get(key, key)
                        if value is not None:
                            final_keywords.append(ast.keyword(arg=py_name, value=value))
                    if signature.name in ("MIN", "MAX"):
                        # NOTE: Fortran constructs may contain an arbitrary number of nested
                        # elements or operands. In contrast, Python's AST often represents
                        # binary operations using two operands at a time. Recursion is therefore
                        # used to traverse and process all nested elements while preserving the
                        # original Fortran semantics.
                        values = normalized["values"]
                        func = ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="minimum" if signature.name == "MIN" else "maximum",
                            ctx=ast.Load(),
                        )
                        expr = values[0]
                        for v in values[1:]:
                            expr = ast.Call(func=func, args=[expr, v], keywords=[])
                        return expr
                    intrinsic_func = ast.Call(
                        func=func, args=final_args, keywords=final_keywords
                    )
                else:
                    intrinsic_func = ast.Call(
                        func=func,
                        args=positional_args,
                        keywords=[
                            ast.keyword(arg=k, value=v) for k, v in keyword_args.items()
                        ],
                    )

            return intrinsic_func
        except Exception:
            self.logger.exception("Exception in handle_intrinsic_function_reference")
            raise

    def handle_real_literal_constant(
        self,
        real_literal_constant: F23.Real_Literal_Constant | list,
    ) -> ast.Constant:
        """
        Convert a Fortran real literal constant into a Python float
        constant.

        Parameters
        ----------
        real_literal_constant : F23.Real_Literal_Constant or list
            The Fortran real literal constant node.

        Returns
        -------
        ast.Constant
            Constant node holding the float value.
        """
        return ast.Constant(value=float(real_literal_constant.items[0]))

    def handle_part_ref(self, part_ref: list | F23.Part_Ref) -> ast.AST:
        """
        Convert a Fortran part reference into either a function call or an
        array subscript.

        Resolves the leading name via :meth:`_extract_name`, then uses
        :meth:`_is_function_call` (consulting :attr:`extractor`'s array
        metadata when available) to decide between the two. Function calls
        are built via :meth:`_build_regular_call`. Array references with a
        single nested ``Part_Ref`` go through :meth:`_handle_single_part_ref`;
        references with multiple nested ``Part_Ref`` levels (e.g. derived
        types or chained indexing) go through :meth:`_handle_nested_part_ref`.

        Parameters
        ----------
        part_ref : list or F23.Part_Ref
            The Fortran part reference to convert.

        Returns
        -------
        ast.AST
            Either an ``ast.Call`` (function call) or an ``ast.Subscript``
            (array reference).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """

        try:
            name = self._extract_name(part_ref)

            if self._is_function_call(name):
                _, args_spec_list = part_ref.children
                return self._build_regular_call(
                    func_name=name, args_spec_list=args_spec_list
                )

            part_refs = walk(part_ref, F23.Part_Ref)

            if len(part_refs) == 1:
                return self._handle_single_part_ref(part_refs[0])

            return self._handle_nested_part_ref(part_ref)

        except Exception:
            self.logger.exception("Exception in handle_part_ref")
            raise

    def _extract_name(self, part_ref: F23.Base) -> str:
        """
        Extract the first variable/function name from a part reference.

        Parameters
        ----------
        part_ref : F23.Base
            Fortran node representing a part reference.

        Returns
        -------
        str
            The first name found.

        Raises
        ------
        ValueError
            If no name is found in *part_ref*.
        """

        names = walk(part_ref, F23.Name)
        if not names:
            raise ValueError("No name found in part_ref")
        return names[0].string

    def _is_function_call(self, name: str) -> bool:
        """
        Determine whether *name* refers to a function call rather than an
        array reference.

        Consults :attr:`extractor`'s ``all_array_info`` (a per-routine map
        of known array names); if *name* does not appear in any routine's
        array set, it is treated as a function call. When no
        :attr:`extractor` is configured, the conservative default is to
        treat *name* as an array (``False``).

        Parameters
        ----------
        name : str
            The candidate name to classify.

        Returns
        -------
        bool
            ``True`` if *name* is judged to be a function call, ``False``
            if it is an array reference.
        """
        extractor = getattr(self, "extractor", None)
        if not extractor:
            return False  # default: treat as array

        return not any(
            name in elements for elements in extractor.all_array_info.values()
        )

    def _handle_single_part_ref(self, part_ref) -> ast.Subscript:
        """
        Convert a simple (non-nested) part reference into a Python
        subscript.

        Dimensions are extracted via :meth:`_extract_dimensions` and each
        converted to a slice or index expression via
        :meth:`_build_slice_or_index`, then assembled into the final
        subscript via :meth:`_make_subscript`.

        Parameters
        ----------
        part_ref : F23.Base
            Fortran node representing a single part reference.

        Returns
        -------
        ast.Subscript
            The subscripted variable reference.
        """

        name = part_ref.children[0].tostr()

        dims = self._extract_dimensions(part_ref)
        args = [self._build_slice_or_index(dim) for dim in dims]

        return self._make_subscript(name, args)

    def _extract_dimensions(self, part_ref: F23.Base) -> list[tuple]:
        """
        Extract and pre-parse each dimension of a part reference's
        subscript list.

        Each dimension is parsed via :meth:`_parse_dimension`. Used by
        :meth:`_handle_single_part_ref`.

        Parameters
        ----------
        part_ref : F23.Base
            Fortran node representing a part reference.

        Returns
        -------
        list[tuple]
            One parsed-dimension tuple (see :meth:`_parse_dimension`) per
            subscript dimension.
        """
        dims = []

        for child in part_ref.children:
            if isinstance(child, F23.Section_Subscript_List):
                for dim in child.children:
                    dims.append(self._parse_dimension(dim))

        return dims

    def _parse_dimension(
        self, dim: F23.Base
    ) -> tuple[str, str | None, str | None, F23.Base]:
        """
        Classify a single subscript dimension as a slice or a plain index.

        Slice dimensions (``F23.Subscript_Triplet``) have their bounds
        text-adjusted via :meth:`simplify_limits` (the lower bound shifted
        by ``-1`` for 0-based indexing); index dimensions are returned with
        their raw text. Used by :meth:`_extract_dimensions`.

        Parameters
        ----------
        dim : F23.Base
            Fortran node representing a single dimension.

        Returns
        -------
        tuple[str, Optional[str], Optional[str], F23.Base]
            ``('slice', lower, upper, node)`` for slice dimensions, or
            ``('index', value, None, node)`` for plain index dimensions.
        """
        text = dim.tostr()
        limits = text.split(":")

        # Slice case
        if isinstance(dim, F23.Subscript_Triplet):
            lb = limits[0].strip() if limits[0] else None
            ub = limits[1].strip() if len(limits) > 1 else None

            if lb:
                lb = self.simplify_limits(lb + "-1")
            if ub:
                ub = self.simplify_limits(ub)

            return ("slice", lb, ub, dim)

        # Index case
        return ("index", text.strip(), None, dim)

    def _build_slice_or_index(self, parsed_dim: tuple) -> ast.AST:
        """
        Convert a pre-parsed dimension tuple into an ``ast.Slice`` or a
        plain index expression.

        Bound text is resolved through :meth:`handle_expr` on the original
        Fortran child nodes. Used by :meth:`_handle_single_part_ref`.

        Parameters
        ----------
        parsed_dim : tuple
            A dimension tuple as produced by :meth:`_parse_dimension`.

        Returns
        -------
        ast.AST
            An ``ast.Slice`` for slice dimensions, or the resolved index
            expression otherwise.
        """

        kind, lb, ub, node = parsed_dim

        if kind == "slice":
            return ast.Slice(
                lower=self.handle_expr(node.children[0]) if lb else None,
                upper=self.handle_expr(node.children[1]) if ub else None,
            )

        # index
        return self.handle_expr(node)

    def _make_subscript(self, name: str, args: list[ast.AST]) -> ast.Subscript:
        """
        Assemble a name and a list of index/slice expressions into a
        Python subscript.

        A single index collapses to a bare slice/index; multiple indices
        are wrapped in an ``ast.Tuple``. Used by
        :meth:`_handle_single_part_ref`.

        Parameters
        ----------
        name : str
            The base variable name.
        args : list[ast.AST]
            Per-dimension index/slice expressions.

        Returns
        -------
        ast.Subscript
            The assembled subscript expression.
        """

        slice_node = args[0] if len(args) == 1 else ast.Tuple(elts=args, ctx=ast.Load())

        return ast.Subscript(
            value=ast.Name(id=name, ctx=ast.Load()), slice=slice_node, ctx=ast.Load()
        )

    def _handle_nested_part_ref(self, part_ref: F23.Base) -> ast.Subscript:
        """
        Convert a part reference with nested structure (e.g. chained or
        derived-type indexing) into a Python subscript.

        Each subscript-list child is resolved via :meth:`handle_expr`;
        results that are themselves subscripts with no slice component
        (i.e. a fully-indexed sub-reference, used as an index into the
        outer reference) are wrapped in ``int(...)`` to coerce them to a
        scalar index. Used as the fallback path in :meth:`handle_part_ref`
        when more than one nested ``Part_Ref`` level is present.

        Parameters
        ----------
        part_ref : F23.Base
            Fortran node representing the nested part reference.

        Returns
        -------
        ast.Subscript
            The assembled (possibly multi-index) subscript expression.
        """

        name_node = None
        elts = []

        for element in part_ref.children:
            if isinstance(element, F23.Name):
                name_node = ast.Name(id=element.tostr(), ctx=ast.Load())

            elif isinstance(element, F23.Section_Subscript_List):
                for child in element.children:
                    node = self.handle_expr(child)

                    # If it's a subscript without slices -> wrap in int()
                    if isinstance(node, ast.Subscript) and not any(
                        ast_walk(node, ast.Slice)
                    ):
                        node = ast.Call(
                            func=ast.Name(id="int", ctx=ast.Load()),
                            args=[node],
                            keywords=[],
                        )

                    if isinstance(node, list):
                        elts.extend(node)
                    else:
                        elts.append(node)

        slice_node = elts[0] if len(elts) == 1 else ast.Tuple(elts=elts, ctx=ast.Load())

        return ast.Subscript(value=name_node, slice=slice_node, ctx=ast.Load())

    def handle_level_4expr(self, stmt) -> ast.Compare:
        """
        Convert a Fortran level-4 (relational) expression into a Python
        comparison.

        Resolves the left and right operands via :meth:`handle_expr`, maps
        the Fortran relational operator token (``.LT.``, ``.EQ.``, etc.)
        through :attr:`replacements` to its textual symbol and then through
        :attr:`conditional_ops_map` to the corresponding ``ast`` operator
        instance.

        Parameters
        ----------
        stmt
            A parsed Fortran level-4 expression node.

        Returns
        -------
        ast.Compare
            The resolved comparison expression.

        Raises
        ------
        ValueError
            If either operand fails to resolve.
        KeyError
            If the relational operator has no entry in
            :attr:`conditional_ops_map`.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            level4_expr = walk(stmt, F23.Level_4_Expr)[0].children
            left_node, op, right_node = level4_expr

            left_ast = self.handle_expr(left_node)
            right_ast = self.handle_expr(right_node)

            if left_ast is None or right_ast is None:
                raise ValueError(
                    f"Either left or right part of handle_or_and_operand, left:{left_ast}, right:{right_ast}"
                )

            pattern = rf"\.{op.strip('.')}\."

            operator = self.replacements.get(pattern, None)
            # We will check if the conditional operator is present in the self.replacements
            # dict if it's None then we check directly into the conditional_op_map
            # which contains the ast format of each conditional operators directly.
            if operator is not None:
                ast_op = self.conditional_ops_map.get(operator, None)
            else:
                ast_op = self.conditional_ops_map.get(op, None)

            if ast_op is None:
                raise KeyError(
                    f"Error in ast_mapping: {op} isn't available in the ast_map"
                )

            ast_stmt = ast.Compare(left=left_ast, ops=[ast_op], comparators=[right_ast])

            return ast_stmt

        except Exception:
            self.logger.exception("Exception in handle_level_4expr")
            raise

    def handle_OR_AND_Operand(self, stmt) -> ast.AST | None:
        """
        Convert a Fortran ``.AND.``/``.OR.``/``.NOT.`` expression into its
        Python equivalent.

        For binary forms (``a .AND. b``), resolves both operands via
        :meth:`handle_expr` and the operator via
        :meth:`_map_logical_operator`. If both operands are comparisons
        over sliced (array) subscripts (detected via
        :meth:`_contains_slice_subscript`), the result is built as an
        element-wise ``ast.BinOp`` with a bitwise operator (via
        :meth:`_to_bitwise_operator`) instead of an ``ast.BoolOp``, since
        Python's ``and``/``or`` cannot be overloaded for NumPy arrays. For
        unary forms (``.NOT. a``), the operator is resolved via
        :meth:`_map_unary_operator` and the operand via :meth:`handle_expr`.

        Parameters
        ----------
        stmt
            A parsed Fortran ``.AND.``/``.OR.``/``.NOT.`` operand
            expression.

        Returns
        -------
        Optional[ast.AST]
            ``ast.BinOp``, ``ast.BoolOp``, or ``ast.UnaryOp`` depending on
            the input shape; ``None`` if *stmt* matches neither the binary
            nor unary pattern.

        Raises
        ------
        ValueError
            If an operand fails to resolve.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            or_and_stmt = self._extract_logical_operand(stmt)

            # Binary logical case (e.g., a .AND. b)
            if len(stmt.children) > 2:
                left_ast = self.handle_expr(or_and_stmt.items[0])
                op_token = or_and_stmt.items[1]
                right_ast = self.handle_expr(or_and_stmt.items[2])

                if left_ast is None or right_ast is None:
                    raise ValueError(
                        f"Invalid operands: left={left_ast}, right={right_ast}"
                    )

                op = self._map_logical_operator(op_token)

                # Decide between BoolOp and BinOp (array-wise logic)
                if (
                    isinstance(left_ast, ast.Compare)
                    and isinstance(right_ast, ast.Compare)
                    and self._contains_slice_subscript(left_ast)
                    and self._contains_slice_subscript(right_ast)
                ):
                    op = self._to_bitwise_operator(op)
                    return ast.BinOp(left=left_ast, op=op, right=right_ast)

                return ast.BoolOp(op=op, values=[left_ast, right_ast])

            # Unary case (e.g., .NOT. a)
            elif len(stmt.children) == 2:
                op_token, operand = stmt.children

                ast_op = self._map_unary_operator(op_token)
                operand_ast = self.handle_expr(operand)

                if operand_ast is None:
                    raise ValueError("Operand is None in unary operation")

                return ast.UnaryOp(op=ast_op, operand=operand_ast)

            return None

        except Exception:
            self.logger.exception("Exception in handle_OR_AND_Operand")
            raise

    def _extract_logical_operand(
        self,
        stmt: F23.Base,
    ) -> F23.Or_Operand | F23.And_Operand | F23.Equiv_Operand:
        """
        Find the first logical operand node within a statement.

        Searches in order for ``F23.Or_Operand``, ``F23.And_Operand``, then
        ``F23.Equiv_Operand``. Used by :meth:`handle_OR_AND_Operand`.

        Parameters
        ----------
        stmt : F23.Base
            Fortran node representing a logical expression.

        Returns
        -------
        F23.Or_Operand or F23.And_Operand or F23.Equiv_Operand
            The first matching logical operand node.

        Raises
        ------
        ValueError
            If no logical operand is found.
        """

        for cls in (F23.Or_Operand, F23.And_Operand, F23.Equiv_Operand):
            results = walk(stmt, cls)
            if results:
                return results[0]
        raise ValueError("No logical operand found")

    def _map_logical_operator(self, op_token: str) -> ast.boolop:
        """
        Map a Fortran ``.AND.``/``.OR.`` token to its ``ast`` operator
        instance.

        Parameters
        ----------
        op_token : str
            The Fortran logical operator token.

        Returns
        -------
        ast.boolop
            ``ast.And()`` or ``ast.Or()``.

        Raises
        ------
        NotImplementedError
            If *op_token* is not ``AND`` or ``OR``.
        """

        op_str = op_token.strip().strip(".").upper()

        op_map = {
            "AND": ast.And(),
            "OR": ast.Or(),
        }

        op = op_map.get(op_str)
        if op is None:
            raise NotImplementedError(f"Logical operator {op_token} not supported")

        return op

    def _map_unary_operator(self, op_token: str) -> ast.unaryop:
        """
        Map a Fortran unary operator token (e.g. ``.NOT.``) to its ``ast``
        operator instance via :attr:`replacements` and
        :attr:`conditional_ops_map`.

        Parameters
        ----------
        op_token : str
            The Fortran unary operator token.

        Returns
        -------
        ast.unaryop
            The resolved unary operator instance.

        Raises
        ------
        KeyError
            If *op_token* has no entry in :attr:`conditional_ops_map`.
        """
        pattern = rf"\.{op_token.strip('.').upper()}\."
        operator = self.replacements.get(pattern, op_token)

        ast_op = self.conditional_ops_map.get(operator)
        if ast_op is None:
            raise KeyError(f"{op_token} not found in conditional_ops_map")

        return ast_op

    def _to_bitwise_operator(self, op: ast.And | ast.Or) -> ast.operator:
        """
        Convert a logical ``ast`` operator to its bitwise counterpart.

        Used by :meth:`handle_OR_AND_Operand` when both operands of a
        logical expression are array comparisons, since element-wise
        boolean combination on NumPy arrays requires ``&``/``|`` rather
        than ``and``/``or``.

        Parameters
        ----------
        op : ast.And or ast.Or
            The logical operator to convert.

        Returns
        -------
        ast.operator
            ``ast.BitAnd()`` for ``ast.And``, ``ast.BitOr()`` for
            ``ast.Or``, or *op* unchanged if it is neither.
        """
        if isinstance(op, ast.And):
            return ast.BitAnd()
        if isinstance(op, ast.Or):
            return ast.BitOr()
        return op

    def _contains_slice_subscript(self, node: ast.AST) -> bool:
        """
        Return ``True`` if *node* contains a subscript indexed by a slice
        anywhere in its subtree.

        Used by :meth:`handle_OR_AND_Operand` to detect array-valued
        comparison operands.

        Parameters
        ----------
        node : ast.AST
            Subtree to inspect.

        Returns
        -------
        bool
            ``True`` if any descendant ``ast.Subscript`` has an
            ``ast.Slice`` index.
        """
        return any(
            isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice)
            for sub in ast.walk(node)
        )

    def handle_assignment(self, stmt: F23.Assignment_Stmt | F23.Part_Ref) -> ast.AST:
        """
        Convert a Fortran assignment statement into a Python ``ast.Assign``.

        For statements that don't have exactly three children (LHS,
        ``=``, RHS) — for example a bare boolean expression used as an
        ``IF`` condition — the node is instead resolved directly via
        :meth:`handle_expr` and returned without wrapping in an
        ``ast.Assign``. Otherwise, the LHS is built via :meth:`_build_lhs`,
        the RHS via :meth:`_build_rhs`, and array-copy broadcasting
        semantics (``a = b`` → ``a[:] = b`` when both are known arrays) are
        applied via :meth:`_apply_array_copy_semantics`, using array
        metadata from :meth:`_get_func_arrays`.

        Parameters
        ----------
        stmt : F23.Assignment_Stmt or F23.Part_Ref
            The Fortran assignment statement (or bare condition
            expression) to convert.

        Returns
        -------
        ast.AST
            Either an ``ast.Assign`` node, or a bare expression node when
            *stmt* was not a true three-part assignment.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if len(stmt.children) != 3:
                # This is for the case that might have a and b type elements
                # ex: humrel[ji, jv] > min_sechiba and soiltile[ji, jst] * vegtot[ji] > min_sechiba
                ast_stmt = self.handle_expr(stmt.items[0])
                if ast_stmt is None:
                    raise ValueError("ast_stmt for handle assignement is None")
                return ast_stmt

            lhs_node, _, rhs_node = stmt.children
            func_arrays = self._get_func_arrays()

            lhs_ast = self._build_lhs(lhs_node, rhs_node, func_arrays)
            rhs_ast = self._build_rhs(rhs_node)

            lhs_ast = self._apply_array_copy_semantics(lhs_ast, rhs_ast, func_arrays)

            return ast.Assign(targets=[lhs_ast], value=rhs_ast)

        except Exception:
            self.logger.exception(f"Exception in handle_assignment: {stmt}")
            raise

    def _get_func_arrays(self) -> dict:
        """
        Look up the array-name → dimensions mapping for the routine
        currently being translated.

        Consults :attr:`extractor`'s ``all_array_info`` keyed by
        :attr:`func_name`. Returns an empty dict if no extractor is
        configured or no routine is currently set.

        Returns
        -------
        dict
            Mapping of array names to their dimension metadata for the
            current routine, or an empty dict.
        """
        if getattr(self, "extractor", None) and self.func_name:
            return self.extractor.all_array_info.get(self.func_name, {})
        return {}

    def _build_lhs(
        self, lhs_node: F23.Base, rhs_node: F23.Base, func_arrays: dict
    ) -> ast.AST:
        """
        Build the left-hand side AST node for an assignment.

        Plain-name LHS nodes are routed through :meth:`_handle_name_lhs`
        (which detects implicit array-broadcast assignment); ``Part_Ref``
        LHS nodes (already-subscripted targets) are resolved directly via
        :meth:`handle_expr`. Used by :meth:`handle_assignment`.

        Parameters
        ----------
        lhs_node : F23.Base
            Fortran node representing the assignment's LHS.
        rhs_node : F23.Base
            Fortran node representing the assignment's RHS, passed through
            for the implicit-broadcast check in
            :meth:`_handle_name_lhs`.
        func_arrays : dict
            Array metadata for the current routine, from
            :meth:`_get_func_arrays`.

        Returns
        -------
        ast.AST
            The resolved LHS target node.

        Raises
        ------
        TypeError
            If *lhs_node* is neither an ``F23.Name`` nor an
            ``F23.Part_Ref``.
        ValueError
            If conversion of a ``Part_Ref`` LHS yields ``None``.
        """
        if isinstance(lhs_node, F23.Name):
            return self._handle_name_lhs(lhs_node, rhs_node, func_arrays)

        if isinstance(lhs_node, F23.Part_Ref):
            lhs_ast = self.handle_expr(lhs_node)
            if lhs_ast is None:
                raise ValueError("lhs_ast is None")
            return lhs_ast

        raise TypeError(f"Unsupported LHS node type: {type(lhs_node)}")

    def _handle_name_lhs(
        self,
        lhs_node: F23.Name,
        rhs_node: F23.Base,
        func_arrays: dict,
    ) -> ast.AST:
        """
        Build the LHS target for a plain-name assignment, detecting
        implicit array broadcasting.

        When the target name is a known array (per *func_arrays*) and the
        RHS is a scalar-shaped literal or name (``a = 1`` where ``a`` is
        actually declared as an array), the target is rewritten to a
        full-slice subscript (``a[:]``/``a[:, :]``, via
        :meth:`_build_full_slice`) so the assignment broadcasts correctly
        under NumPy semantics. Otherwise, a plain ``ast.Name`` store target
        is returned. Used by :meth:`_build_lhs`.

        Parameters
        ----------
        lhs_node : F23.Name
            Fortran node for the assignment target name.
        rhs_node : F23.Base
            Fortran node for the RHS expression, used to detect the
            broadcast-assignment pattern.
        func_arrays : dict
            Array metadata for the current routine.

        Returns
        -------
        ast.AST
            An ``ast.Subscript`` (full-slice) target if broadcast
            semantics apply, otherwise a plain ``ast.Name`` target.
        """
        name = lhs_node.string

        # Detect case like: a = 1 or a = TRUE where a is actually an array
        if name in func_arrays and isinstance(
            rhs_node,
            F23.Name
            | F23.Logical_Literal_Constant
            | F23.Real_Literal_Constant
            | F23.Int_Literal_Constant,
        ):
            ndim = len(func_arrays[name])
            slice_node = self._build_full_slice(ndim)

            return ast.Subscript(
                value=ast.Name(id=name, ctx=ast.Load()),
                slice=slice_node,
                ctx=ast.Store(),
            )

        return ast.Name(id=name, ctx=ast.Store())

    def _build_rhs(self, rhs_node: F23.Base) -> ast.AST:
        """
        Resolve an assignment's right-hand side via :meth:`handle_expr`.

        Parameters
        ----------
        rhs_node : F23.Base
            Fortran node for the RHS expression.

        Returns
        -------
        ast.AST
            The resolved RHS expression.

        Raises
        ------
        ValueError
            If resolution yields ``None``.
        """

        rhs_ast = self.handle_expr(rhs_node)
        if rhs_ast is None:
            raise ValueError("rhs_ast is None")
        return rhs_ast

    def _apply_array_copy_semantics(
        self,
        lhs_ast: ast.AST,
        rhs_ast: ast.AST,
        func_arrays: dict,
    ) -> ast.AST:
        """
        Rewrite a name-to-name array copy assignment to use full-slice
        broadcasting.

        When both *lhs_ast* and *rhs_ast* are plain ``ast.Name`` nodes
        whose identifiers are both known arrays (per *func_arrays*), the
        LHS is rewritten to a full-slice subscript via
        :meth:`_build_full_slice` (``a = b`` → ``a[:] = b``), matching
        Fortran's whole-array assignment semantics. Used by
        :meth:`handle_assignment`.

        Parameters
        ----------
        lhs_ast : ast.AST
            The already-built LHS node.
        rhs_ast : ast.AST
            The already-built RHS node.
        func_arrays : dict
            Array metadata for the current routine.

        Returns
        -------
        ast.AST
            The (possibly rewritten) LHS node.
        """
        if not isinstance(lhs_ast, ast.Name) or not isinstance(rhs_ast, ast.Name):
            return lhs_ast

        left_name = lhs_ast.id
        right_name = rhs_ast.id

        if left_name in func_arrays and right_name in func_arrays:
            ndim = len(func_arrays[left_name])
            slice_node = self._build_full_slice(ndim)

            return ast.Subscript(
                value=ast.Name(id=left_name, ctx=ast.Load()),
                slice=slice_node,
                ctx=ast.Store(),
            )

        return lhs_ast

    def _build_full_slice(self, ndim: int) -> ast.Slice | ast.Tuple:
        """
        Build a full-slice index expression for an array of *ndim*
        dimensions.

        Parameters
        ----------
        ndim : int
            Number of array dimensions.

        Returns
        -------
        Union[ast.Slice, ast.Tuple]
            A bare ``ast.Slice()`` for 1-D arrays, or an ``ast.Tuple`` of
            ``ast.Slice()`` nodes for higher dimensions.
        """
        if ndim == 1:
            return ast.Slice()

        return ast.Tuple(elts=[ast.Slice() for _ in range(ndim)], ctx=ast.Load())

    def build_binop(self, left: ast.AST, op_token: str, right: ast.AST) -> ast.BinOp:
        """
        Build a Python binary operation from a Fortran arithmetic operator
        token.

        Parameters
        ----------
        left : ast.AST
            Left operand.
        op_token : str
            Operator token (``'+'``, ``'-'``, ``'*'``, ``'/'``, ``'**'``).
        right : ast.AST
            Right operand.

        Returns
        -------
        ast.BinOp
            The constructed binary operation.

        Raises
        ------
        NotImplementedError
            If *op_token* has no mapping to an ``ast`` operator.
        """
        # Get operator symbol from token or string
        op_str = str(op_token).strip()

        op_map = {
            "+": ast.Add(),
            "-": ast.Sub(),
            "*": ast.Mult(),
            "/": ast.Div(),
            "**": ast.Pow(),
        }

        op = op_map.get(op_str)
        if not op:
            raise NotImplementedError(f"Operator {op_str} not supported.")

        return ast.BinOp(left=left, op=op, right=right)

    def _binop_from_items(self, items: list) -> ast.BinOp:
        """
        Build a binary operation from a three-element
        ``[left, operator, right]`` item list.

        Both operands are resolved via :meth:`handle_expr`; the operator is
        passed to :meth:`build_binop`. Used by :meth:`handle_expr` for
        ``Level_2_Expr``, ``Mult_Operand``, and bare three-element tuple
        nodes.

        Parameters
        ----------
        items : list
            ``[left_expr, operator_token, right_expr]``.

        Returns
        -------
        ast.BinOp
            The constructed binary operation.
        """
        left = self.handle_expr(items[0])
        op_token = items[1]
        right = self.handle_expr(items[2])
        return self.build_binop(left, op_token, right)

    def handle_expr(self, expr_node) -> ast.AST:
        """
        Dispatch a Fortran expression node to its dedicated handler and
        return the equivalent Python AST node.

        The central expression-level dispatcher, mirroring
        :meth:`recursive_ast`'s role at the statement level. Routes by
        node type to (non-exhaustive):

        - ``Real_Literal_Constant`` → :meth:`handle_real_literal_constant`.
        - ``Int_Literal_Constant`` → a direct ``ast.Constant(int(...))``.
        - ``Logical_Literal_Constant`` → a direct ``ast.Constant(bool)``.
        - ``Char_Literal_Constant`` → a direct ``ast.Constant(str)`` with
          quotes stripped.
        - ``Part_Ref`` → :meth:`handle_part_ref`.
        - ``Intrinsic_Function_Reference`` → :meth:`handle_intrinsic_function_reference`,
          except ``MINLOC``/``MAXLOC`` nested inside a ``Level_2_Expr``,
          which is unwrapped and re-dispatched directly.
        - ``Level_2_Expr`` / ``Mult_Operand`` / bare 3-tuples →
          :meth:`_binop_from_items`.
        - ``Add_Operand`` → re-dispatched on its ``.items``.
        - ``Level_2_Unary_Expr`` → ``ast.UnaryOp`` with ``+``/``-`` mapped
          to ``ast.UAdd``/``ast.USub``.
        - ``Level_4_Expr`` → :meth:`handle_level_4expr`.
        - ``Parenthesis`` → re-dispatched on the contained expression.
        - ``Name`` → a direct ``ast.Name`` load reference.
        - ``And_Operand`` / ``Or_Operand`` → :meth:`handle_OR_AND_Operand`.
        - ``Equiv_Operand`` → a direct ``ast.BoolOp`` built inline from its
          ``AND``/``OR`` token.
        - ``Actual_Arg_Spec`` → an ``ast.keyword`` for named call
          arguments.
        - ``Write_Stmt`` / ``Print_Stmt`` → :meth:`handle_print_stmt`.
        - ``Call_Stmt`` → :meth:`handle_call_stmt`.
        - ``Subscript_Triplet`` → a list of slice/index AST nodes built
          inline (used when resolving multi-dimension subscript lists).
        - ``Assignment_Stmt`` → :meth:`handle_assignment`.

        Parameters
        ----------
        expr_node : object
            A Fortran expression-level AST node, or in some recursive
            cases a plain ``tuple`` of sub-items.

        Returns
        -------
        ast.AST
            The translated Python expression node. Exact type depends on
            the dispatched branch (``ast.Constant``, ``ast.Name``,
            ``ast.BinOp``, ``ast.BoolOp``, ``ast.UnaryOp``,
            ``ast.keyword``, etc.).

        Raises
        ------
        NotImplementedError
            If *expr_node* is of an unsupported type, or matches a
            specifically unimplemented sub-case (e.g. a tuple of length
            other than 1 or 3).
        """

        if isinstance(expr_node, F23.Real_Literal_Constant):
            return self.handle_real_literal_constant(expr_node)

        elif isinstance(expr_node, F23.Part_Ref):
            return self.handle_part_ref(expr_node)

        elif isinstance(expr_node, F23.Intrinsic_Function_Reference):
            return self.handle_intrinsic_function_reference(expr_node)

        elif isinstance(expr_node, F23.Level_2_Expr):
            # Composite expression, which contains tuples of different other expressions
            if isinstance(
                expr_node.items[0], F23.Intrinsic_Function_Reference
            ) and expr_node.items[0].items[0].string in ["MINLOC", "MAXLOC"]:
                return self.handle_expr(expr_node.items[0])
            else:
                return self._binop_from_items(expr_node.items)

        elif isinstance(expr_node, F23.Add_Operand):
            return self.handle_expr(expr_node.items)

        elif isinstance(expr_node, tuple):
            # These are mostly used for the assignement task used
            # inside the intrinsic_arg_spec list or that of
            # Level_1_expr or Level_3_expr

            if len(expr_node) == 1:
                return self.handle_expr(expr_node[0])

            elif len(expr_node) == 3:
                return self._binop_from_items(expr_node)
            else:
                raise NotImplementedError(
                    "Not implemented for tuple with a size not equal to 1 or 3"
                )

        elif isinstance(expr_node, F23.Int_Literal_Constant):
            return ast.Constant(value=int(expr_node.string))

        elif isinstance(expr_node, F23.Level_2_Unary_Expr):
            op_token, operand_node = expr_node.children
            operand_ast = self.handle_expr(operand_node)

            op_map = {
                "-": ast.USub(),
                "+": ast.UAdd(),
            }

            op = op_map.get(op_token)
            if not op:
                raise NotImplementedError(f"Unary operator {op_token} not supported.")

            return ast.UnaryOp(op=op, operand=operand_ast)

        elif isinstance(expr_node, F23.Level_4_Expr):
            return self.handle_level_4expr(expr_node)

        elif isinstance(expr_node, F23.Parenthesis):
            # we directly send the element inside the paranthesis
            return self.handle_expr(expr_node.items[1])

        elif isinstance(expr_node, F23.Name):
            return ast.Name(id=expr_node.string, ctx=ast.Load())

        elif isinstance(expr_node, F23.Mult_Operand):
            return self._binop_from_items(expr_node.items)

        elif isinstance(expr_node, F23.And_Operand | F23.Or_Operand):
            return self.handle_OR_AND_Operand(expr_node)

        elif isinstance(expr_node, F23.Equiv_Operand):
            left = self.handle_expr(expr_node.items[0])
            op_token = expr_node.items[1]
            op_str = op_token.strip().strip(".").upper()
            op_map = {
                "AND": ast.And(),
                "OR": ast.Or(),
            }
            op = op_map.get(op_str.upper())
            right = self.handle_expr(expr_node.items[2])
            values = [left, right]
            return ast.BoolOp(op=op, values=values)

        elif isinstance(expr_node, F23.Logical_Literal_Constant):
            bool_val, _ = expr_node.children
            return ast.Constant(
                value=False if bool_val.strip().strip(".").upper() == "FALSE" else True
            )

        elif isinstance(expr_node, F23.Char_Literal_Constant):
            expr_node = expr_node.string.strip(" ' ").strip('"')
            return ast.Constant(value=expr_node)

        elif isinstance(expr_node, F23.Actual_Arg_Spec):
            if len(expr_node.children) == 2:
                name_node, value_node = expr_node.children

                if not isinstance(name_node, F23.Name):
                    raise NotImplementedError(
                        f"Unsupported arg name node: {type(name_node)}"
                    )
                arg_name = name_node.string.lower()
                value_ast = self.handle_expr(value_node)

                return ast.keyword(arg=arg_name, value=value_ast)

            else:
                raise NotImplementedError(f"Unsupported Actual_Arg_Spec: {expr_node}")

        elif isinstance(expr_node, F23.Write_Stmt | F23.Print_Stmt):
            if not any(
                walk(walk(expr_node, F23.Io_Control_Spec), F23.Int_Literal_Constant)
            ):
                stmt = self.handle_print_stmt(expr_node)
                return stmt

        elif isinstance(expr_node, F23.Call_Stmt):
            return self.handle_call_stmt(expr_node)

        elif isinstance(expr_node, F23.Subscript_Triplet):
            shape = []
            limits = expr_node.tostr().split(":")
            lb = limits[0]
            if len(limits) > 1:
                ub = limits[1]
                if lb:
                    lb = lb + "-1"
                lb = self.simplify_limits(lb)
                ub = self.simplify_limits(ub)

                shape.append((f"{lb}:{ub}", expr_node))
            elif len(limits) == 1:
                shape.append((f"{lb}", expr_node))
            args = []
            for sh, node in shape:
                if ":" in sh and isinstance(node, F23.Subscript_Triplet):
                    # It's a slice
                    if sh == ":":
                        # Simple ':' slice
                        slice_node = ast.Slice()
                    else:
                        # Possibly lb:ub
                        lb_ub = sh.split(":")
                        lb = lb_ub[0].strip() or None
                        ub = lb_ub[1].strip() if len(lb_ub) > 1 else None

                        slice_node = ast.Slice(
                            lower=self.handle_expr(node.children[0]) if lb else None,
                            upper=self.handle_expr(node.children[1]) if ub else None,
                        )
                    args.append(slice_node)
                else:
                    # it's a direct index
                    if isinstance(node, ast.AST):
                        expr_node = node
                    else:
                        expr_node = self.handle_expr(node)
                    args.append(expr_node)

            return args

        elif isinstance(expr_node, F23.Assignment_Stmt):
            return self.handle_assignment(expr_node)

        else:
            raise NotImplementedError(
                f"Unsupported node type: {type(expr_node).__name__}\n"
                f"Node content: {repr(expr_node)}"
            )

    def apply_mask_to_rhs(self, node: ast.AST) -> ast.AST:
        """
        Recursively apply a ``[mask]`` subscript to every array reference
        in *node*.

        Walks *node* and, for any ``ast.Name``, ``ast.Subscript``, or
        ``ast.Attribute`` whose base name is a known array (per
        :attr:`extractor`'s ``all_array_info``), wraps it in
        ``...[mask]``. For already-subscripted references, the existing
        slice is itself recursively masked first (so ``array[:]`` becomes
        ``array[:][mask]`` rather than double-applying the mask), with the
        mask only added when the (possibly tupled) slice actually contains
        a real ``ast.Slice`` component. All other node kinds are walked
        field-by-field and rebuilt with recursively masked children. Used
        by :meth:`recursive_ast` for the RHS of assignments found inside a
        ``WHERE``/``ELSEWHERE`` region.

        Parameters
        ----------
        node : ast.AST
            The expression subtree to mask.

        Returns
        -------
        ast.AST
            The mask-applied subtree (mutated and/or rebuilt as needed).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            # Handle variable names like: array
            if isinstance(node, ast.Name):
                for elements in self.extractor.all_array_info.values():
                    if node.id in elements.keys():
                        return ast.Subscript(
                            value=ast.Name(id=node.id, ctx=ast.Load()),
                            slice=ast.Name(id="mask", ctx=ast.Load()),
                            ctx=ast.Load(),
                        )

            # Handle subscript access like: array[:]
            elif isinstance(node, ast.Subscript):
                base = node.value
                if isinstance(base, ast.Name):
                    for elements in self.extractor.all_array_info.values():
                        if base.id in elements.keys():
                            # Recursively apply masking to subscript slice
                            node.slice = self.apply_mask_to_rhs(node.slice)

                            # Case 1: It's a full slice like [:]
                            if isinstance(node.slice, ast.Slice):
                                return ast.Subscript(
                                    value=node,
                                    slice=ast.Name(id="mask", ctx=ast.Load()),
                                    ctx=ast.Load(),
                                )

                            # Case 2: It's a tuple like [:, 1]
                            elif isinstance(node.slice, ast.Tuple):
                                # Only apply mask if any element in the tuple is a slice
                                if any(
                                    isinstance(elt, ast.Slice)
                                    for elt in node.slice.elts
                                ):
                                    return ast.Subscript(
                                        value=node,
                                        slice=ast.Name(id="mask", ctx=ast.Load()),
                                        ctx=ast.Load(),
                                    )

                            else:
                                return node

            # Optionally handle attribute access like obj.attr (if you want to mask those too)
            elif isinstance(node, ast.Attribute):
                # Check if this attribute is a known array (only if it's in tracking attributes in all_array_info)
                attr_str = node.attr  # For example: "obj.attr"
                for elements in self.extractor.all_array_info.values():
                    if attr_str in elements.keys():
                        return ast.Subscript(
                            value=node,
                            slice=ast.Name(id="mask", ctx=ast.Load()),
                            ctx=ast.Load(),
                        )

            # Recursively walk all child nodes
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    new_values = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            new_values.append(self.apply_mask_to_rhs(item))
                        else:
                            new_values.append(item)
                    setattr(node, field, new_values)
                elif isinstance(value, ast.AST):
                    setattr(node, field, self.apply_mask_to_rhs(value))

            return node
        except Exception as e:
            self.logger.exception("Exception in apply_mask_to_rhs:", e)
            raise
