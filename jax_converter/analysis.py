import ast
from typing import Dict, List, Optional, Set, Tuple

class _BranchAnalysis:
    """
    Static analysis helpers over branch and loop statement lists.

    Composes onto ``JaxConverter`` to answer questions about a block of
    statements (an ``if`` branch, a ``for``/``while`` body) before it is
    lowered: which names are assigned, which are read, which loop
    targets are locally bound, and whether a statement is a removable
    logging call.

    These methods are read-only with respect to AST structure (they do
    not rewrite nodes) except where noted; :meth:`_collect_assigned`
    and :meth:`_assigned_name_or_attr` additionally record ``self.attr``
    writes into :attr:`_mutated_attrs`, since attribute mutation must be
    tracked globally across the whole function transform.
    """

    def _is_logging_call(self, stmt: ast.AST) -> bool:
        """
        Return ``True`` if *stmt* is a bare ``logging.xxx(...)``
        expression.

        Used to strip logging calls from branch and loop bodies before
        JAX lowering, since ``logging`` calls have side effects that
        are incompatible with JAX tracing.

        Parameters
        ----------
        stmt : ast.AST
            Statement to check — only ``ast.Expr`` nodes can match.

        Returns
        -------
        bool
            ``True`` if *stmt* is an expression statement calling a
            ``logging.*`` method.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(stmt, ast.Expr):
                return False
            value = stmt.value
            return (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == 'logging'
            )
        except Exception as e:
            self.logger.exception('Exception in _is_logging_call:', e)
            raise

    def _collect_loop_vars(self, stmts: List[ast.stmt]) -> Set[str]:
        """
        Collect all ``for``-loop target names from *stmts*.

        Handles both simple targets (``for i in ...``) and
        tuple-unpacking targets (``for i, j in ...``). Used by
        :meth:`_first_reads` to exclude loop-index names from the
        "first seen as load" analysis, since they are always defined by
        the loop itself.

        Parameters
        ----------
        stmts : List[ast.stmt]
            Statement list to walk — typically a branch or loop body.

        Returns
        -------
        Set[str]
            Names used as ``for``-loop targets anywhere in *stmts*.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            loop_vars: Set[str] = set()
            for stmt in stmts:
                for node in ast.walk(stmt):
                    if not isinstance(node, ast.For):
                        continue
                    target = node.target
                    if isinstance(target, ast.Name):
                        loop_vars.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                loop_vars.add(elt.id)
            return loop_vars
        except Exception as e:
            self.logger.exception('Exception in _collect_loop_vars:', e)
            raise

    def _first_reads(self, stmts: List[ast.stmt]) -> List[str]:
        """
        Return variable names in the order they are first read (loaded).

        Walks *stmts* in true source order, recording for each name
        whether its first occurrence was a load or a store. Names whose
        first occurrence is a store are excluded from the result, since
        they are locally defined rather than read from an outer scope.

        Special-cased handling:

        - ``ast.Assign`` — the RHS (``node.value``) is visited before
          the LHS (``node.targets``), so that ``a[i] = a[i] + 1``
          correctly records ``a`` as first seen via a LOAD rather than
          a STORE.
        - ``self.attr`` — recorded under the key ``"self.attr"`` so
          that class-attribute reads are distinguished from
          local-variable reads.
        - Loop-index names (from :meth:`_collect_loop_vars`) are
          excluded entirely, since they are always loop-defined.

        Parameters
        ----------
        stmts : List[ast.stmt]
            Statement list to walk — typically a branch or loop body.

        Returns
        -------
        List[str]
            Names whose first occurrence was a LOAD, in ascending order
            of first appearance.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            first_seen: Dict[str, Tuple[str, int]] = {}
            order = 0
            loop_vars = self._collect_loop_vars(stmts)

            def visit_in_order(node):
                nonlocal order

                if isinstance(node, ast.Name) and node.id not in loop_vars:
                    if node.id not in first_seen:
                        kind = 'load' if isinstance(node.ctx, ast.Load) else 'store'
                        first_seen[node.id] = (kind, order)
                        order += 1

                if isinstance(node, ast.Assign):
                    # Visit RHS before LHS so `a[i] = a[i] + 1` records a LOAD first
                    if isinstance(node.value, ast.AST):
                        visit_in_order(node.value)
                    for t in node.targets:
                        visit_in_order(t)
                    for field, value in ast.iter_fields(node):
                        if field in ('value', 'targets'):
                            continue
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, ast.AST):
                                    visit_in_order(item)
                        elif isinstance(value, ast.AST):
                            visit_in_order(value)
                    return

                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == 'self'
                ):
                    full_name = f'self.{node.attr}'
                    if full_name not in first_seen and isinstance(node.ctx, ast.Load):
                        first_seen[full_name] = ('load', order)
                        order += 1

                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.AST):
                                visit_in_order(item)
                    elif isinstance(value, ast.AST):
                        visit_in_order(value)

            for s in stmts:
                visit_in_order(s)

            loads = [
                (name, idx) for name, (kind, idx) in first_seen.items()
                if kind == 'load'
            ]
            loads.sort(key=lambda x: x[1])
            return [name for name, _ in loads]

        except Exception as e:
            self.logger.exception('Exception in _first_reads:', e)
            raise

    def _collect_rhs_uses(self, stmts: List[ast.stmt]) -> Set[str]:
        """
        Collect all variable names read on the right-hand side of
        assignments in *stmts*.

        Unlike :meth:`_first_reads`, this only inspects RHS expressions
        of ``Assign``/``AugAssign`` nodes (never LHS targets) and
        ignores statement order — it returns the full set of names
        used, which is combined with :meth:`_collect_assigned` in
        callers to compute ``used_after`` (names both assigned and
        later read).

        Parameters
        ----------
        stmts : List[ast.stmt]
            Statement list to walk — typically a branch or loop body.

        Returns
        -------
        Set[str]
            Names loaded anywhere in an RHS expression.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            uses: Set[str] = set()

            class RHSVisitor(ast.NodeVisitor):
                def visit_Assign(self, node):
                    self.visit(node.value)

                def visit_AugAssign(self, node):
                    self.visit(node.value)

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        uses.add(node.id)

                def visit_Subscript(self, node):
                    self.visit(node.value)
                    self.visit(node.slice)

                def visit_Attribute(self, node):
                    self.visit(node.value)

            for s in stmts:
                RHSVisitor().visit(s)

            return uses

        except Exception as e:
            self.logger.exception('Exception in _collect_rhs_uses:', e)
            raise

    def _collect_assigned(self, stmts: List[ast.AST]) -> List[str]:
        """
        Return the names of all variables assigned anywhere in *stmts*.

        Handles three assignment forms:

        - ``ast.Assign`` — plain and subscript targets.
        - ``ast.AugAssign`` — in-place operators (``+=``, ``*=``, …).
        - ``ast.NamedExpr`` — walrus operator (``:=``).

        ``self.attr`` targets are recorded in both the returned list
        and :attr:`_mutated_attrs`.

        Parameters
        ----------
        stmts : List[ast.AST]
            Statement list to walk — typically a branch body or loop
            body.

        Returns
        -------
        List[str]
            Unique variable names (plain or attribute) that are written
            to.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            assigned = []
            for stmt in stmts:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            assigned.extend(
                                self._assigned_name_or_attr(target, assigned)
                            )
                    elif isinstance(node, ast.AugAssign):
                        assigned.extend(
                            self._assigned_name_or_attr(node.target, assigned)
                        )
                    elif isinstance(node, ast.NamedExpr):
                        assigned.extend(
                            self._assigned_name_or_attr(node.target, assigned)
                        )
            return assigned
        except Exception as e:
            self.logger.exception('Exception in _collect_assigned:', e)
            raise

    def _assigned_name_or_attr(
        self,
        target: ast.AST,
        assigned: List[str],
    ) -> List[str]:
        """
        Extract assignable names from a single assignment target node.

        Resolves four target patterns:

        - ``ast.Name`` — plain local variable.
        - ``ast.Attribute`` on ``self`` — class attribute; also
          recorded in :attr:`_mutated_attrs`.
        - ``ast.Subscript`` on ``self.attr`` or a plain ``Name`` — the
          base name is extracted and the attribute added to
          :attr:`_mutated_attrs` where applicable.
        - Nested ``ast.Subscript`` — handled recursively.

        Parameters
        ----------
        target : ast.AST
            The LHS target node from an ``Assign``, ``AugAssign``, or
            ``NamedExpr``.
        assigned : List[str]
            Already-collected names for the current statement; used to
            avoid duplicates.

        Returns
        -------
        List[str]
            New names to append to the collected-assigned list.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            names = []

            if isinstance(target, ast.Name):
                if target.id not in assigned:
                    names.append(target.id)

            elif (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'self'
            ):
                attr = target.attr
                if attr not in assigned:
                    names.append(attr)
                    self._mutated_attrs.add(attr)

            elif isinstance(target, ast.Subscript):
                inner = target.value

                # self.attr[i]
                if (
                    isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == 'self'
                ):
                    attr = inner.attr
                    if attr not in assigned:
                        names.append(attr)
                        self._mutated_attrs.add(attr)

                # plain_var[i]
                elif isinstance(inner, ast.Name):
                    if inner.id not in assigned:
                        names.append(inner.id)

                # nested subscript: a[i][j]
                elif isinstance(inner, ast.Subscript):
                    nested = self._assigned_name_or_attr(inner, assigned)
                    if not set(nested).issubset(names):
                        names.extend(nested)

            return names

        except Exception as e:
            self.logger.exception('Exception in _assigned_name_or_attr:', e)
            raise

    def _subscript_uses_loop_vars(self, node: ast.AST, loop_vars: List[str]) -> bool:
        """
        Recursively check whether any subscript index in *node*
        references a name in *loop_vars*.

        Used to detect whether an array assignment's LHS or RHS depends
        on the current vectorisation loop index, which determines
        whether the assignment must be lifted/vectorised.

        Parameters
        ----------
        node : ast.AST
            Subtree to search — typically an assignment target or
            value.
        loop_vars : List[str]
            Names of the active loop indices to check against.

        Returns
        -------
        bool
            ``True`` if any ``ast.Subscript`` within *node* indexes
            using a name or tuple element present in *loop_vars*.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Name) and node.slice.id in loop_vars:
                    return True
                if isinstance(node.slice, ast.Tuple):
                    for elt in node.slice.elts:
                        if isinstance(elt, ast.Name) and elt.id in loop_vars:
                            return True

            for child in ast.iter_child_nodes(node):
                if self._subscript_uses_loop_vars(child, loop_vars):
                    return True

            return False

        except Exception as e:
            self.logger.exception('Exception in _subscript_uses_loop_vars:', e)
            raise

    def _unwrap_stmt_list(
        self,
        stmts,
    ) -> Tuple[List[ast.AST], Optional[ast.AST]]:
        """
        Split a list of statements (as returned by ``visit()``) into a
        preamble and a single terminal assignment.

        When ``visit()`` returns a list rather than a single node (e.g.
        a masked subscript assignment that expanded into several
        statements), the masked-``if`` lowering needs to identify which
        statement is the "real" array update so it can be merged into a
        ``jnp.where``. Terminal detection rules, in priority order:

        1. LHS is ``ast.Subscript`` (direct array index update).
        2. RHS is a ``.at[...].<op>(...)`` call.
        3. Fallback — the last element in the list.

        Parameters
        ----------
        stmts : list or ast.AST
            Either a single statement or a list of statements.

        Returns
        -------
        Tuple[List[ast.AST], Optional[ast.AST]]
            ``(preamble_stmts, terminal_assign)``. If *stmts* is not a
            list, returns ``([], stmts)`` unchanged.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(stmts, list):
                return [], stmts

            def _is_at_op_call(node):
                return (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {
                        'set', 'add', 'mul', 'divide', 'multiply',
                        'subtract', 'power', 'min', 'max',
                    }
                    and isinstance(node.func.value, ast.Subscript)
                    and isinstance(node.func.value.value, ast.Attribute)
                    and node.func.value.value.attr == 'at'
                )

            def _is_terminal(stmt):
                if not isinstance(stmt, ast.Assign):
                    return False
                if isinstance(stmt.targets[0], ast.Subscript):
                    return True
                if _is_at_op_call(stmt.value):
                    return True
                return False

            preamble = []
            terminal = None

            for stmt in reversed(stmts):
                if terminal is None and _is_terminal(stmt):
                    terminal = stmt
                else:
                    preamble.append(stmt)

            preamble.reverse()

            if terminal is None and stmts:
                *pre, terminal = stmts
                preamble = list(pre)

            return preamble, terminal

        except Exception as e:
            self.logger.exception('Exception in _unwrap_stmt_list:', e)
            raise