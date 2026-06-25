import ast

from fgpt.jax_utils import Control


class _ConditionalLowering:
    """
    Lowers Python ``if`` statements into JAX-compatible operations.

    Composes onto ``JaxConverter`` to implement the three conditional
    lowering strategies dispatched from ``visit_If`` (defined on the
    masking/vectorisation classes for the ``'masked'``/``'vector'``/
    ``'index_loop'`` branches; the general fallback lives here):

    - **Value-select** (:meth:`_is_pure_select_cond`,
      :meth:`_handle_scalar_select`) — pure ``jnp.where`` chains for
      branches that only assign plain local variables with no side
      effects. Cheapest lowering, preferred whenever it applies.
    - **``lax.cond``** (:meth:`_handle_lax_cond`) — synthesises
      ``_if_true_N`` / ``_if_false_N`` helper functions for branches
      that mutate ``self`` attributes, use array subscripts, or carry
      state across iterations.
    - **Early-return hoisting** (:meth:`_lift_if_return`,
      :meth:`_rewrite_if_return`,
      :meth:`_replace_returns`) — a pre-pass run once per function
      (from ``visit_FunctionDef``) that restructures
      ``if cond: return x`` / ``<rest of body>`` into a symmetric
      ``if/else`` so the conditional lowering above can treat both
      branches uniformly.

    Also owns the boolean-expression rewriting used by every conditional path
    (:meth:`_transform_if_test`,
    :meth:`_transform_boolop`, :meth:`_transform_compare`,
    :meth:`_transform_binop`), converting Python ``and``/``or``/``not``/
    comparison operators into their ``jnp.logical_*`` equivalents, and
    the ``self.attr`` ↔ bare-name naming helpers used when building
    helper-function operands (:meth:`to_arg`, :meth:`make_operand`).
    """

    def _lift_if_return(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        """
        Hoist the first ``if`` block containing a ``return`` into a
        unified branch structure.

        When a function body contains an early-return pattern such as::

            if cond:
                return val
            <rest of body>

        the conditional lowering pass cannot treat the two branches
        symmetrically.  This method rewrites the structure so that the
        ``if`` body and the remaining statements become sibling branches
        of a single ``if`` node, enabling uniform ``lax.cond`` emission.

        Parameters
        ----------
        fn : ast.FunctionDef
            Function node to inspect and potentially rewrite in place.

        Returns
        -------
        ast.FunctionDef
            The (possibly rewritten) function node.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            for i, stmt in enumerate(fn.body):
                if isinstance(stmt, ast.If) and any(
                    isinstance(s, ast.Return) for s in ast.walk(stmt)
                ):
                    return self._rewrite_if_return(fn, stmt, i)
            return fn
        except Exception as e:
            self.logger.exception("Exception in _lift_if_return:", e)
            raise

    def _rewrite_if_return(
        self,
        fn: ast.FunctionDef,
        if_stmt: ast.If,
        idx: int,
    ) -> ast.FunctionDef:
        """
        Merge an early-return ``if`` with the statements that follow it.

        Replaces the pattern::

            <preamble>
            if cond:          # idx
                <true body>
                return ...
            <rest of body>    # idx + 1 …

        with::

            <preamble>
            if cond:
                <true body without return>
            else:
                <rest of body without return>

        so that both branches are structurally equivalent for
        ``lax.cond`` lowering.

        Parameters
        ----------
        fn : ast.FunctionDef
            The enclosing function whose body is rewritten in place.
        if_stmt : ast.If
            The ``if`` node that contains a ``return``.
        idx : int
            Position of *if_stmt* inside ``fn.body``.

        Returns
        -------
        ast.FunctionDef
            The rewritten function node.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            true_body = [s for s in if_stmt.body if not isinstance(s, ast.Return)]
            false_body = [
                s for s in fn.body[idx + 1 :] if not isinstance(s, ast.Return)
            ]

            new_if = ast.If(
                test=if_stmt.test,
                body=true_body,
                orelse=false_body,
            )
            fn.body = fn.body[:idx] + [new_if]
            return fn

        except Exception as e:
            self.logger.exception("Exception in _rewrite_if_return:", e)
            raise

    def _replace_returns(self, stmts: list[ast.AST]) -> list[ast.AST]:
        """
        Strip ``return`` statements from a statement list.

        Used by :meth:`_rewrite_if_return` to produce branch bodies
        that can be emitted as ``lax.cond`` helper functions without a
        premature ``return`` terminating the helper early.

        Parameters
        ----------
        stmts : list[ast.AST]
            Statement list to filter.

        Returns
        -------
        list[ast.AST]
            A new list with all ``ast.Return`` nodes removed.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            return [s for s in stmts if not isinstance(s, ast.Return)]
        except Exception as e:
            self.logger.exception("Exception in _replace_returns:", e)
            raise

    def to_arg(self, n: str) -> str:
        """
        Strip a leading ``"self."`` prefix from a name, if present.

        Used throughout helper-function construction to convert a
        class-attribute reference into the bare parameter name the
        helper will receive.

        Parameters
        ----------
        n : str
            Name, optionally prefixed with ``"self."``.

        Returns
        -------
        str
            *n* with the ``"self."`` prefix removed, or *n* unchanged
            if no such prefix exists.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            return n.split(".", 1)[1] if n.startswith("self.") else n
        except Exception as e:
            self.logger.exception("Exception in to_arg:", e)
            raise

    def make_operand(self, n: str) -> ast.AST:
        """
        Build the AST node used to pass *n* as a ``lax.cond`` /
        ``lax.scan`` operand.

        If *n* is a ``"self.attr"`` name that has **not** been mutated
        (i.e. still safe to read directly off ``self``), an
        ``ast.Attribute`` node (``self.attr``) is returned. Otherwise a
        plain ``ast.Name`` node is returned, since the value has
        already been unpacked into a local variable.

        Parameters
        ----------
        n : str
            Name to convert, optionally ``"self."``-prefixed.

        Returns
        -------
        ast.AST
            Either an ``ast.Attribute`` (``self.attr``) or an
            ``ast.Name`` load node.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if n.startswith("self.") and n not in self._mutated_attrs:
                attr = n.split(".", 1)[1]
                return ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                )
            return ast.Name(id=n, ctx=ast.Load())
        except Exception as e:
            self.logger.exception("Exception in make_operand:", e)
            raise

    def _is_pure_select_cond(self, node: ast.If) -> bool:
        """
        Return ``True`` if *node* is safe to lower as a value-select
        (``jnp.where``) rather than ``lax.cond``.

        A conditional qualifies when, across all branches (including
        any ``elif`` chain):

        - every statement is a plain ``ast.Assign`` to a single
          ``Name`` target (no ``self.attr``, no subscripts),
        - the RHS contains no function calls (pure expressions only),
          and
        - all branches assign exactly the same set of local variable
          names.

        This is the cheapest and most XLA-friendly lowering, since it
        avoids the helper-function machinery required by ``lax.cond``.

        Parameters
        ----------
        node : ast.If
            The conditional to classify.

        Returns
        -------
        bool
            ``True`` if the value-select lowering applies.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:

            def collect_body(stmts):
                assigned = set()
                for stmt in stmts:
                    if not isinstance(stmt, ast.Assign):
                        return assigned, False
                    if len(stmt.targets) != 1:
                        return assigned, False
                    target = stmt.targets[0]
                    if not isinstance(target, ast.Name):
                        return assigned, False
                    for n in ast.walk(stmt.value):
                        if isinstance(n, ast.Call):
                            return assigned, False
                    assigned.add(target.id)
                return assigned, True

            branch_assigned_sets = []
            current = node

            while isinstance(current, ast.If):
                body_assigned, ok = collect_body(current.body)
                if not ok or not body_assigned:
                    return False
                branch_assigned_sets.append(body_assigned)

                if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    current = current.orelse[0]
                else:
                    if current.orelse:
                        else_assigned, ok = collect_body(current.orelse)
                        if not ok or not else_assigned:
                            return False
                        branch_assigned_sets.append(else_assigned)
                    break

            if not branch_assigned_sets:
                return False

            return all(s == branch_assigned_sets[0] for s in branch_assigned_sets)

        except Exception as e:
            self.logger.exception("Exception in _is_pure_select_cond:", e)
            raise

    def _handle_scalar_select(
        self,
        node: ast.If,
        assigned: list[str],
    ) -> list[ast.Assign]:
        """
        Lower a pure value-select conditional (see
        :meth:`_is_pure_select_cond`) into nested ``jnp.where``
        assignments.

        This is an SSA-inspired value-select lowering, preferred over
        ``lax.cond`` whenever it applies, since it avoids
        helper-function overhead and is more amenable to vectorising
        compilers. Safe when:

        - all branches assign the same plain local variables (no
          ``self.*``, no subscripts),
        - RHS values are pure expressions (no side-effecting calls),
          and
        - no variable is read before being written (no carried state).

        For each branch's test, a named condition variable
        (``_cond_0``, ``_cond_1``, …) is emitted first. Then for each
        output variable, a right-to-left fold builds a chain of
        ``jnp.where(cond_i, val_i, expr_so_far)`` so the first matching
        condition takes priority.

        Parameters
        ----------
        node : ast.If
            The conditional to lower; must satisfy
            :meth:`_is_pure_select_cond`.
        assigned : list[str]
            Names assigned across the branches (unused directly here
            but kept for call-site symmetry with the ``lax.cond``
            path).

        Returns
        -------
        list[ast.Assign]
            Condition assignments followed by one ``jnp.where``-chain
            assignment per output variable.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            result_stmts = []
            branches = []
            current = node

            while isinstance(current, ast.If):
                branch_vals = {}
                for stmt in current.body:
                    if isinstance(stmt, ast.Assign) and isinstance(
                        stmt.targets[0], ast.Name
                    ):
                        branch_vals[stmt.targets[0].id] = stmt.value
                branches.append((current.test, branch_vals))

                if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    current = current.orelse[0]
                else:
                    if current.orelse:
                        else_vals = {}
                        for stmt in current.orelse:
                            if isinstance(stmt, ast.Assign) and isinstance(
                                stmt.targets[0], ast.Name
                            ):
                                else_vals[stmt.targets[0].id] = stmt.value
                        branches.append((None, else_vals))
                    break

            cond_names = []
            for i, (test, _) in enumerate(branches):
                if test is None:
                    cond_names.append(None)
                    continue
                cond_name = f"_cond_{i}"
                cond_names.append(cond_name)
                transformed_test = self._transform_if_test(test)
                result_stmts.append(
                    ast.Assign(
                        targets=[ast.Name(id=cond_name, ctx=ast.Store())],
                        value=transformed_test,
                    )
                )

            all_vars = branches[0][1].keys()

            for var in all_vars:
                reversed_branches = list(reversed(list(zip(cond_names, branches))))
                first_cond_name, (_, first_vals) = reversed_branches[0]

                if first_cond_name is None:
                    expr = first_vals[var]
                    reversed_branches = reversed_branches[1:]
                else:
                    expr = ast.Constant(value=0)

                for cond_name, (_, vals) in reversed_branches:
                    expr = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="where",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=cond_name, ctx=ast.Load()), vals[var], expr],
                        keywords=[],
                    )

                result_stmts.append(
                    ast.Assign(
                        targets=[ast.Name(id=var, ctx=ast.Store())],
                        value=expr,
                    )
                )

            return [ast.fix_missing_locations(s) for s in result_stmts]

        except Exception as e:
            self.logger.exception("Exception in _handle_scalar_select:", e)
            raise

    def _handle_lax_cond(
        self,
        node: ast.If,
        assigned: list[str],
        read_before_write: list[str],
        used_after: set[str],
    ) -> list[ast.AST]:
        """
        Lower a general (non-value-select) conditional into
        ``lax.cond``.

        Synthesises one or two helper functions (``_if_true_N`` and,
        when ``node.orelse`` exists, ``_if_false_N``) whose inputs are
        the names read before being written in the branches and whose
        outputs are the names assigned. The ``lax.cond`` call site
        receives those inputs as a packed tuple operand and unpacks the
        helper's returned tuple into the output names.

        When no ``orelse`` exists, the false branch is synthesised as
        an identity ``lambda`` that returns its inputs unchanged,
        preserving ``self.attr`` naming where required.

        Parameters
        ----------
        node : ast.If
            The conditional to lower.
        assigned : list[str]
            Names assigned anywhere in the branches.
        read_before_write : list[str]
            Names read before being written, in first-read order —
            becomes the helper input list.
        used_after : set[str]
            Names assigned and later read — passed through to
            ``_mask_vector_assign`` for vectorised mask propagation.

        Returns
        -------
        list[ast.AST]
            ``[true_fn, (false_fn,) assign]`` — the helper function(s)
            followed by the ``lax.cond`` call-site assignment.

        Raises
        ------
        NotImplementedError
            If neither inputs nor outputs can be determined for the
            conditional (degenerate ``if`` with no reads or writes).
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            inputs = []
            inside_helper = len(self._context_stack) > 0 and any(
                self.to_arg(name) in self._context_stack[-1]["helper_args"]
                for name in self._mutated_attrs
            )

            start_idx = len(self._modified_ret_stack)
            self._modified_ret_stack.append([])

            for name in read_before_write:
                if name.startswith("self."):
                    attr = self.to_arg(name)
                    if attr in self._mutated_attrs:
                        if inside_helper:
                            if attr in assigned:
                                inputs.append(attr)
                        else:
                            if (
                                self._func_arg_stack
                                and attr
                                in [self.to_arg(n) for n in self._func_arg_stack[-1]]
                            ) or attr in self._var_modif["attr"]:
                                inputs.append(attr)
                            else:
                                inputs.append(name)
                elif name not in self._always_exclude:
                    inputs.append(name)

            if self._local_defined_stack and self._local_defined_stack[-1]:
                inputs.extend(self._local_defined_stack[-1].keys())
                inputs = list(set(inputs))

            vectorization_context = (
                self._control_stack[-1].to_dict() if self._control_stack else None
            )

            mask_name = None
            if vectorization_context:
                metadata = vectorization_context["metadata"]
                if metadata.get("current_mask_assign"):
                    mask_assign = metadata.get("current_mask_assign")
                    mask_name = mask_assign.targets[0].id
                    inputs.append(mask_name)

                for element in inputs:
                    if element in vectorization_context.get("vectorization_axis"):
                        inputs.remove(element)

            inputs.sort()
            arg_stack = set()
            for name in inputs:
                if (
                    "." in name
                    and name.split(".", 1)[1] in self._mutated_attrs
                    and inside_helper
                ):
                    arg_stack.add(name.split(".", 1)[1])
                else:
                    arg_stack.add(name)
            arg_stack = sorted(arg_stack)
            self._func_arg_stack.append(arg_stack)

            outputs = set()
            arg_set = {self.to_arg(arg) for arg in arg_stack}
            if assigned and arg_set:
                outputs = (
                    set(assigned) & arg_set
                    if set(assigned) & arg_set
                    else set(assigned)
                )

            true_name, false_name = self._fresh_names()

            def make_fn(name, body_stmts, inputs, outputs):
                arg_node = ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="arg")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                )
                if self._local_defaults:
                    self._local_defaults.popitem()

                new_body = []
                for stmt in body_stmts:
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        for v in visited:
                            if isinstance(v, ast.Assign) and mask_name:
                                new_body.extend(
                                    self._mask_vector_assign(
                                        v, mask_name, assigned, used_after
                                    )
                                )
                            elif isinstance(v, ast.Continue):
                                continue
                            else:
                                new_body.append(v)
                    elif isinstance(visited, ast.Continue):
                        pass
                    elif isinstance(visited, ast.Assign) and mask_name:
                        new_body.extend(
                            self._mask_vector_assign(
                                visited, mask_name, assigned, used_after
                            )
                        )
                    elif visited is not None:
                        new_body.append(visited)

                body_stmts = new_body

                collected = []
                while len(self._modified_ret_stack) > start_idx:
                    collected.extend(self._modified_ret_stack.pop())
                local_modified = list(set(collected))
                if self._modified_ret_stack:
                    self._modified_ret_stack[-1].extend(local_modified)

                new_inputs = []
                if local_modified:
                    diff_inputs = set(local_modified) - set(
                        self.to_arg(i) for i in inputs
                    )
                    if self._scan_stack:
                        carry = set()
                        for frame in self._scan_stack:
                            carry.update(
                                frame.get("introduced", set())
                                | frame.get("mutated", set())
                            )
                        for var in diff_inputs:
                            if var == "self":
                                new_inputs.append("self")
                            elif var in carry:
                                new_inputs.append(var)
                            else:
                                new_inputs.append(f"self.{var}")
                        inputs.extend([self.to_arg(n) for n in new_inputs])
                    else:
                        new_inputs = [
                            var if var == "self" else f"self.{var}"
                            for var in diff_inputs
                        ]
                        inputs.extend([self.to_arg(n) for n in new_inputs])

                input_args = [
                    ast.Name(id=self.to_arg(n), ctx=ast.Store()) for n in inputs
                ]
                args_assign = ast.Assign(
                    targets=[ast.Tuple(elts=input_args, ctx=ast.Store())],
                    value=ast.Name(id="arg", ctx=ast.Load()),
                )

                if not body_stmts and not outputs:
                    if not inputs:
                        raise NotImplementedError(
                            f"No inputs for visit_If: "
                            f"{ast.unparse(ast.fix_missing_locations(node))}"
                        )
                    for element in inputs:
                        if element not in self._func_arg_stack[-1]:
                            outputs.add(element)

                if local_modified:
                    outputs = (
                        (outputs | set(local_modified))
                        if outputs
                        else set(local_modified)
                    )

                if outputs:
                    ret = ast.Return(
                        value=ast.Tuple(
                            elts=[ast.Name(id=o, ctx=ast.Load()) for o in outputs],
                            ctx=ast.Load(),
                        )
                    )
                else:
                    ret = ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))

                body = [args_assign] + list(body_stmts) + [ret]
                fn = ast.FunctionDef(
                    name=name, args=arg_node, body=body, decorator_list=[]
                )
                return ast.fix_missing_locations(fn), outputs, new_inputs

            true_fn, outputs, new_inputs = make_fn(
                true_name, node.body, inputs, outputs
            )
            false_fn = None
            if node.orelse:
                false_fn, _, _ = make_fn(false_name, node.orelse, inputs, outputs)

            self._pending_helpers.extend([true_fn, false_fn] if false_fn else [true_fn])

            arg_stack.extend(new_inputs)
            tuple_in = (
                ast.Tuple(
                    elts=[self.make_operand(n) for n in arg_stack], ctx=ast.Load()
                )
                if inputs
                else ast.Constant(value=None)
            )
            test = self._transform_if_test(node.test)

            if false_fn:
                false_case = ast.Name(id=false_name, ctx=ast.Load())
            else:
                self_vars = {
                    v.split(".", 1)[1] for v in inputs if v.startswith("self.")
                }
                corrected_outputs = [
                    ("self." + out) if out in self_vars else out for out in outputs
                ]
                false_case = ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="_")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=ast.Tuple(
                        elts=[
                            ast.Name(id=o, ctx=ast.Load()) for o in corrected_outputs
                        ],
                        ctx=ast.Load(),
                    ),
                )

            cond_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="lax", ctx=ast.Load()),
                    attr="cond",
                    ctx=ast.Load(),
                ),
                args=[
                    test,
                    ast.Name(id=true_name, ctx=ast.Load()),
                    false_case,
                    tuple_in,
                ],
                keywords=[],
            )

            if outputs:
                assign = ast.Assign(
                    targets=[
                        ast.Tuple(
                            elts=[ast.Name(id=o, ctx=ast.Store()) for o in outputs],
                            ctx=ast.Store(),
                        )
                    ],
                    value=cond_call,
                )
            else:
                assign = ast.Assign(
                    targets=[ast.Name(id="result", ctx=ast.Store())],
                    value=cond_call,
                )

            if false_fn:
                return [true_fn, false_fn, ast.copy_location(assign, node)]
            return [true_fn, ast.copy_location(assign, node)]

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _handle_lax_cond:", e)
            raise

    def _transform_if_test(self, test: ast.AST) -> ast.AST:
        """
        Dispatch a boolean test expression to the appropriate ``jnp``
        rewrite.

        Entry point for converting Python conditional expressions
        (``Compare``, ``BoolOp``, ``UnaryOp(Not)``, ``BinOp`` with
        bitwise operators) into their ``jnp.logical_*`` /
        ``jnp.greater`` / etc. equivalents so the expression can be
        evaluated element-wise under JAX tracing.

        Parameters
        ----------
        test : ast.AST
            The test expression to rewrite — typically ``node.test``
            from an ``ast.If``.

        Returns
        -------
        ast.AST
            The rewritten expression, or *test* unchanged if it matches
            none of the recognised patterns.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(test, ast.Compare):
                return self._transform_compare(test)
            elif isinstance(test, ast.BoolOp):
                return self._transform_boolop(test)
            elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                operand = self._transform_if_test(test.operand)
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_not",
                        ctx=ast.Load(),
                    ),
                    args=[operand],
                    keywords=[],
                )
            elif isinstance(test, ast.BinOp):
                return self._transform_binop(test)
            return test
        except Exception as e:
            self.logger.exception("Exception in _transform_if_test:", e)
            raise

    def _transform_boolop(self, node: ast.BoolOp) -> ast.AST:
        """
        Rewrite ``and``/``or`` chains into nested ``jnp.logical_and`` /
        ``jnp.logical_or`` calls.

        Example::

            a < b and c > d
            -> jnp.logical_and(jnp.less(a, b), jnp.greater(c, d))

        Each value in the ``BoolOp`` is first recursively transformed
        via :meth:`_transform_if_test`, then reduced left-to-right into
        a single nested call.

        Parameters
        ----------
        node : ast.BoolOp
            The boolean operation node to rewrite.

        Returns
        -------
        ast.AST
            The reduced ``jnp.logical_and``/``jnp.logical_or`` call
            chain.

        Raises
        ------
        NotImplementedError
            If the ``BoolOp`` uses an operator other than ``And`` or
            ``Or``.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node.op, ast.And):
                op_name = "logical_and"
            elif isinstance(node.op, ast.Or):
                op_name = "logical_or"
            else:
                raise NotImplementedError(f"Unsupported BoolOp: {node.op}")

            transformed_values = [self._transform_if_test(v) for v in node.values]

            result = transformed_values[0]
            for nxt in transformed_values[1:]:
                result = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr=op_name,
                        ctx=ast.Load(),
                    ),
                    args=[result, nxt],
                    keywords=[],
                )

            return ast.copy_location(result, node)

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _transform_boolop:", e)
            raise

    def _transform_compare(self, test: ast.Compare) -> ast.AST:
        """
        Rewrite a (possibly chained) comparison into ``jnp`` comparison
        calls.

        Each comparison operator is mapped to its ``jnp`` function
        equivalent (``Gt`` → ``greater``, ``Eq`` → ``equal``, etc.).
        Chained comparisons (``a < b < c``) are expanded into a
        conjunction of pairwise comparisons joined by
        ``jnp.logical_and``, matching Python's chained-comparison
        semantics.

        Parameters
        ----------
        test : ast.Compare
            The comparison node to rewrite.

        Returns
        -------
        ast.AST
            A single ``jnp.<op>`` call, or a ``jnp.logical_and`` chain
            for multi-operator comparisons.

        Raises
        ------
        NotImplementedError
            If a comparison operator has no ``jnp`` equivalent in the
            mapping.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            left = test.left
            ops = test.ops
            comparators = test.comparators

            op_map = {
                ast.Gt: "greater",
                ast.GtE: "greater_equal",
                ast.Lt: "less",
                ast.LtE: "less_equal",
                ast.Eq: "equal",
                ast.NotEq: "not_equal",
            }
            comparisons = []

            for i, op in enumerate(ops):
                right = comparators[i]
                op_type = type(op)
                if op_type not in op_map:
                    raise NotImplementedError(f"Unsupported comparison operator: {op}")

                func_name = op_map[op_type]
                call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr=func_name,
                        ctx=ast.Load(),
                    ),
                    args=[left, right],
                    keywords=[],
                )
                comparisons.append(call)
                left = right

            result = comparisons[0]
            for comp in comparisons[1:]:
                result = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_and",
                        ctx=ast.Load(),
                    ),
                    args=[result, comp],
                    keywords=[],
                )

            return ast.copy_location(result, test)

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _transform_compare:", e)
            raise

    def _transform_binop(self, node: ast.AST) -> ast.AST:
        """
        Recursively rewrite bitwise ``BinOp``, ``UnaryOp(Not)``,
        ``Compare``, and ``BoolOp`` nodes into ``jnp`` equivalents.

        Unlike :meth:`_transform_if_test`, this method recurses into
        ``BinOp`` children explicitly, converting ``a | b`` to
        ``jnp.logical_or(a, b)`` and ``a & b`` to
        ``jnp.logical_and(a, b)`` — needed because Python's ``and``/
        ``or`` cannot be overloaded on arrays, so element-wise boolean
        logic is conventionally written with ``|``/``&`` instead, which
        ``_transform_boolop`` does not handle directly.

        Parameters
        ----------
        node : ast.AST
            Expression to rewrite.

        Returns
        -------
        ast.AST
            The rewritten expression, or *node* unchanged if it matches
            none of the recognised patterns.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.BinOp):
                left = self._transform_binop(node.left)
                right = self._transform_binop(node.right)

                if isinstance(node.op, ast.BitOr):
                    return ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="logical_or",
                            ctx=ast.Load(),
                        ),
                        args=[left, right],
                        keywords=[],
                    )
                elif isinstance(node.op, ast.BitAnd):
                    return ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="logical_and",
                            ctx=ast.Load(),
                        ),
                        args=[left, right],
                        keywords=[],
                    )
                else:
                    node.left = left
                    node.right = right
                    return node

            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                operand = self._transform_binop(node.operand)
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_not",
                        ctx=ast.Load(),
                    ),
                    args=[operand],
                    keywords=[],
                )

            elif isinstance(node, ast.Compare):
                return self._transform_compare(node)

            elif isinstance(node, ast.BoolOp):
                return self._transform_boolop(node)

            return node

        except Exception as e:
            self.logger.exception("Exception in _transform_binop:", e)
            raise

    def visit_If(self, node: ast.If):
        """
        Convert a Python ``if`` statement into JAX-compatible operations.

        Strips logging statements from both branches, then classifies the
        conditional via ``self.analyzer.classify_if`` and dispatches to one
        of five lowering strategies based on the result:

        - **``'masked'``** → :meth:`_lower_masked_branch_pair`
        - **``'vector'``** → :meth:`_lower_vector_condition`
        - **``'index_loop'``** → :meth:`_lower_index_loop_condition`, or
        :meth:`_emit_convergence_break` when either branch contains a
        ``break``
        - **``'masked_where'``** → :meth:`handle_masked_where`
        - **fallback** → :meth:`_handle_scalar_select` when
        :meth:`_is_pure_select_cond` holds, otherwise
        :meth:`_handle_lax_cond`

        Outer-to-inner (pre-order) traversal is preserved: nested
        conditionals and loops inside a branch are visited as part of
        lowering that branch, not before.

        Parameters
        ----------
        node : ast.If
            The conditional node to lower.

        Returns
        -------
        ast.AST or list[ast.AST] or None
            A single replacement node, a list of replacement statements, or
            ``None`` if both branches were empty after logging-call removal
            (signalling deletion to the enclosing ``NodeTransformer``).

        Raises
        ------
        NotImplementedError
            Propagated from any of the dispatched lowering methods.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            node.body = [stmt for stmt in node.body if not self._is_logging_call(stmt)]
            node.orelse = [
                stmt for stmt in node.orelse if not self._is_logging_call(stmt)
            ]
            if not node.body and not node.orelse:
                return None

            branch_stmts = list(node.body) + list(node.orelse)
            assigned = self._collect_assigned(branch_stmts)
            rhs_uses = self._collect_rhs_uses(branch_stmts)
            used_after = set(assigned) & rhs_uses
            read_before_write = self._first_reads(branch_stmts)

            cond_type = self.analyzer.classify_if(node)

            # The test may reference arrays that need axis/shape correction
            self.generic_visit(node.test)

            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            if cond_type in ["masked"]:
                return self._lower_masked_branch_pair(node, vectorization_context)

            elif cond_type == "vector":
                return self._lower_vector_condition(
                    node, assigned, used_after, vectorization_context
                )

            elif cond_type == "index_loop":
                has_break_in_body = any(isinstance(s, ast.Break) for s in node.body)
                has_break_in_orelse = any(isinstance(s, ast.Break) for s in node.orelse)
                if has_break_in_body or has_break_in_orelse:
                    return self._emit_convergence_break(
                        node, vectorization_context, assigned, used_after
                    )
                return self._lower_index_loop_condition(
                    node, assigned, used_after, vectorization_context
                )

            elif cond_type == "masked_where":
                return self.handle_masked_where(node)

            else:
                if self._is_pure_select_cond(node):
                    return self._handle_scalar_select(node, assigned)
                return self._handle_lax_cond(
                    node,
                    assigned=assigned,
                    read_before_write=read_before_write,
                    used_after=used_after,
                )

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception("Exception in visit_If:", e)
            raise

    def _lower_masked_branch_pair(
        self,
        node: ast.If,
        vectorization_context: dict | None,
    ) -> ast.AST | list[ast.AST]:
        """
        Lower an ``if``/``else`` pair where both branches assign the same
        targets, into ``jnp.where``-based updates.

        For each ``(true_assign, false_assign)`` pair (matched positionally
        by target), two cases are handled:

        - **Plain-name target** — both branch values are visited, optionally
        broadcast to match the target's known vectorisation axes (via
        :meth:`broadcast_scalar` using :attr:`var_deps`), then combined
        into a single ``new_var = jnp.where(test, true_val, false_val)``.
        - **Subscript target** — both branches are unwrapped (via
        :meth:`_unwrap_stmt_list`) to find their terminal ``.at[...].op(...)``
        assignment. A mask is computed from the condition, broadcast to
        the target's rank if needed, and the ``.set``/``.add``/etc.
        payloads are merged via ``jnp.where(mask, true_payload,
        false_payload)``. If both branches use the same array operation
        (e.g. both ``.add``), the merged value is wrapped in that same
        operation; otherwise an explicit ``old_val`` read is inserted and
        each branch's operation is reconstructed as a ``BinOp`` before
        merging, with the final array update forced to ``.set``.

        Parameters
        ----------
        node : ast.If
            The conditional to lower; must have one ``ast.Assign`` per
            target in each of ``node.body`` and ``node.orelse``.
        vectorization_context : Optional[dict]
            The active :class:`Control` context as a dict, or ``None`` if
            no vectorisation scope is active.

        Returns
        -------
        ast.AST or list[ast.AST]
            The merged assignment(s); a single node if only one target was
            found, otherwise a list (preamble statements followed by the
            merged assignments).

        Raises
        ------
        NotImplementedError
            If both branches independently produce a non-trivial broadcast
            index list for the same subscript target (unsupported: cannot
            determine which one should drive the final indexing).
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            assigns_true = [stmt for stmt in node.body if isinstance(stmt, ast.Assign)]
            assigns_false = [
                stmt for stmt in node.orelse if isinstance(stmt, ast.Assign)
            ]

            new_nodes = []
            for stmt_true, stmt_false in zip(assigns_true, assigns_false):
                target_true = stmt_true.targets[0]

                deps_true = self._expr_depends_on_axes(
                    stmt_true.value, vectorization_context["vectorization_axis"]
                )
                deps_false = self._expr_depends_on_axes(
                    stmt_false.value, vectorization_context["vectorization_axis"]
                )

                if isinstance(target_true, ast.Name):
                    stmts_true_list = (
                        [self.visit(s) for s in stmt_true]
                        if isinstance(stmt_true, list)
                        else [self.visit(stmt_true)]
                    )
                    stmts_false_list = (
                        [self.visit(s) for s in stmt_false]
                        if isinstance(stmt_false, list)
                        else [self.visit(stmt_false)]
                    )

                    true_value = stmts_true_list[-1].value
                    false_value = stmts_false_list[-1].value
                    if self.var_deps.get(target_true.id):
                        target_axes = self.var_deps[target_true.id]
                        true_value = self.broadcast_scalar(
                            true_value, deps_true, target_axes
                        )
                        false_value = self.broadcast_scalar(
                            false_value, deps_false, target_axes
                        )

                    jnp_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="where",
                            ctx=ast.Load(),
                        ),
                        args=[
                            self._transform_if_test(node.test),
                            true_value,
                            false_value,
                        ],
                        keywords=[],
                    )
                    final_nodes = stmts_true_list[:-1] + stmts_false_list[:-1]
                    new_nodes.extend(final_nodes)
                    new_assign = ast.Assign(targets=[target_true], value=jnp_call)

                elif isinstance(target_true, ast.Subscript):
                    stmt_true = self.visit(stmt_true)
                    stmt_false = self.visit(stmt_false)

                    true_preamble, stmt_true = self._unwrap_stmt_list(stmt_true)
                    false_preamble, stmt_false = self._unwrap_stmt_list(stmt_false)

                    if not isinstance(stmt_true, ast.Assign) or not isinstance(
                        stmt_false, ast.Assign
                    ):
                        new_nodes.extend(true_preamble)
                        if stmt_true is not None:
                            new_nodes.append(stmt_true)
                        new_nodes.extend(false_preamble)
                        if stmt_false is not None:
                            new_nodes.append(stmt_false)
                        return new_nodes if len(new_nodes) > 1 else new_nodes[0]

                    new_nodes.extend(true_preamble)
                    new_nodes.extend(false_preamble)

                    same_operation = False
                    if isinstance(stmt_true.value.func, ast.Attribute) and isinstance(
                        stmt_false.value.func, ast.Attribute
                    ):
                        operation_true = stmt_true.value.func.attr
                        operation_false = stmt_false.value.func.attr
                        if operation_false == operation_true:
                            same_operation = True

                    target_slice = target_true.slice
                    target_rank = self._target_rank(target_slice)

                    new_assign = None
                    true_args, false_args = None, None
                    if isinstance(stmt_true.value, ast.Call):
                        true_args = stmt_true.value.args[0]
                    if isinstance(stmt_false.value, ast.Call):
                        false_args = stmt_false.value.args[0]

                    true_args, true_elts_list = self.maybe_add_index(
                        true_args, target_rank, vectorization_context
                    )
                    false_args, false_elts_list = self.maybe_add_index(
                        false_args, target_rank, vectorization_context
                    )

                    mask_name = f"_mask_{self._mask_counter}"
                    self._mask_counter += 1
                    node.test = self.visit(node.test)
                    mask_assign = ast.Assign(
                        targets=[ast.Name(id=mask_name, ctx=ast.Store())],
                        value=node.test,
                    )
                    mask_assign = self._boolean_mask(mask_assign)
                    new_nodes.append(mask_assign)

                    if true_elts_list and false_elts_list:
                        raise NotImplementedError(
                            "Handling both true_elts_list and false_elts_list is not implemented yet."
                        )

                    selected_elts_list = (
                        true_elts_list if true_elts_list else false_elts_list
                    )
                    ranks = self.subscript_ranks(node.test)
                    mask_rank = ranks.get(node.test, 0)

                    if not selected_elts_list:
                        if mask_rank == target_rank:
                            mask_expr = ast.Name(id=mask_name, ctx=ast.Load())
                        elif mask_rank < target_rank:
                            dimensions = self.get_active_dims(self.visit(target_true))
                            loop_info = vectorization_context["loop_info"]
                            vectorization_axis = vectorization_context[
                                "vectorization_axis"
                            ]

                            elts = [ast.Constant(value=None)] * target_rank
                            for loop_dim, loop_var in loop_info.items():
                                if loop_dim not in dimensions:
                                    continue
                                dim_index = dimensions.index(loop_dim)
                                vect_axis = vectorization_axis.get(loop_var, [])
                                if dim_index not in vect_axis:
                                    continue
                                elts[dim_index] = ast.Slice()

                            mask_expr = ast.Subscript(
                                value=ast.Name(id=mask_name, ctx=ast.Load()),
                                slice=ast.Tuple(elts=elts, ctx=ast.Load()),
                                ctx=ast.Load(),
                            )
                    else:
                        mask_expr = ast.Subscript(
                            value=ast.Name(id=mask_name, ctx=ast.Load()),
                            slice=ast.Tuple(elts=selected_elts_list, ctx=ast.Load()),
                            ctx=ast.Load(),
                        )

                    jnp_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="where",
                            ctx=ast.Load(),
                        ),
                        args=[
                            mask_expr if mask_expr else mask_name,
                            true_args,
                            false_args,
                        ],
                        keywords=[],
                    )
                    jnp_call = self.visit(ast.fix_missing_locations(jnp_call))

                    if same_operation:
                        value = ast.Call(
                            func=stmt_true.value.func, args=[jnp_call], keywords=[]
                        )
                        new_assign = ast.Assign(
                            targets=[stmt_true.targets[0]], value=value
                        )
                    else:
                        target_stmt = self.visit_Subscript(target_true)
                        target_stmt.ctx = ast.Load()
                        old_val_name = ast.Name(id="old_val", ctx=ast.Load())
                        old_val = ast.Assign(
                            targets=[ast.Name(id="old_val", ctx=ast.Store())],
                            value=target_stmt,
                        )

                        ops = {
                            "add": ast.Add,
                            "subtract": ast.Sub,
                            "multiply": ast.Mult,
                            "divide": ast.Div,
                            "power": ast.Pow,
                        }

                        if operation_true in ops:
                            true_expr = ast.BinOp(
                                left=old_val_name,
                                op=ops[operation_true](),
                                right=true_args,
                            )
                        else:
                            true_expr = true_args

                        if operation_false in ops:
                            false_expr = ast.BinOp(
                                left=old_val_name,
                                op=ops[operation_false](),
                                right=false_args,
                            )
                        else:
                            false_expr = false_args

                        jnp_call = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()),
                                attr="where",
                                ctx=ast.Load(),
                            ),
                            args=[
                                mask_expr if mask_expr else mask_name,
                                true_expr,
                                false_expr,
                            ],
                            keywords=[],
                        )
                        jnp_call = self.visit(ast.fix_missing_locations(jnp_call))

                        stmt_true.value.func.attr = "set"
                        value = ast.Call(
                            func=stmt_true.value.func, args=[jnp_call], keywords=[]
                        )
                        new_assign = ast.Assign(
                            targets=[stmt_true.targets[0]], value=value
                        )
                        new_nodes.append(old_val)

                new_assign = ast.fix_missing_locations(new_assign)
                new_nodes.append(new_assign)

            return new_nodes if len(new_nodes) > 1 else new_nodes[0]

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _lower_masked_branch_pair:", e)
            raise

    def _lower_vector_condition(
        self,
        node: ast.If,
        assigned: list[str],
        used_after: set[str],
        vectorization_context: dict | None,
    ) -> list[ast.AST]:
        """
        Lower an ``if`` whose condition depends on the active vectorisation
        axis into a masked-update sequence.

        Computes a boolean mask from ``node.test`` (with subscripted
        comparisons reduced to plain names via
        :meth:`_replace_subscript_with_name` for element-wise evaluation),
        pushes a new ``if``-scoped :class:`Control` carrying that mask onto
        :attr:`_control_stack`, then visits the body with each resulting
        assignment routed through :meth:`_mask_vector_assign`.

        If ``node.orelse`` is non-empty, a negated mask
        (``jnp.logical_not``) is computed, the control metadata is updated
        to reference it, and the ``orelse`` statements are visited under a
        fresh scope. Local variables are only retained in the enclosing
        scope if defined identically (same shape) in both branches.

        Parameters
        ----------
        node : ast.If
            The conditional to lower; ``node.test`` has already been
            ``generic_visit``-ed by the caller.
        assigned : list[str]
            Names assigned anywhere in the branches.
        used_after : set[str]
            Names assigned and later read, passed through to
            :meth:`_mask_vector_assign`.
        vectorization_context : Optional[dict]
            The active :class:`Control` context as a dict.

        Returns
        -------
        list[ast.AST]
            The mask assignment(s) followed by every masked statement
            produced by visiting the branches.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            mask_name = f"_mask_{self._mask_counter}"
            self._mask_counter += 1

            node.test = self.visit(node.test)
            ranks = self.subscript_ranks(node.test)
            mask_rank = ranks.get(node.test, 0)
            cond_copy = self._replace_subscript_with_name(
                ast.copy_location(node.test, node)
            )
            mask_assign = ast.Assign(
                targets=[ast.Name(id=mask_name, ctx=ast.Store())],
                value=self.visit(cond_copy),
            )
            mask_assign = self._boolean_mask(mask_assign)

            if vectorization_context:
                self._control_stack.append(
                    Control(
                        kind="if",
                        loop_info=vectorization_context["loop_info"],
                        transform_type="vector",
                        vectorization_axis=vectorization_context["vectorization_axis"],
                        metadata={
                            "current_mask_assign": mask_assign,
                            "current_mask_rank": mask_rank,
                        },
                    )
                )

            new_stmts = [mask_assign]
            self._push_scope()
            for stmt in node.body:
                visited = self.visit(stmt)
                if isinstance(visited, list):
                    for v in visited:
                        if isinstance(v, ast.Assign):
                            new_stmts.extend(
                                self._mask_vector_assign(
                                    v, mask_name, assigned, used_after
                                )
                            )
                        elif isinstance(v, ast.Continue):
                            continue
                        else:
                            new_stmts.append(v)
                elif isinstance(visited, ast.Assign):
                    new_stmts.extend(
                        self._mask_vector_assign(
                            visited, mask_name, assigned, used_after
                        )
                    )
                elif isinstance(visited, ast.Continue):
                    pass
                elif visited is not None:
                    new_stmts.append(visited)

            true_scope = self._local_defined_stack[-1].copy()
            self._pop_scope()

            if node.orelse != []:
                else_mask_name = f"_mask_{self._mask_counter}"
                self._mask_counter += 1
                negated_mask_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_not",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=mask_name, ctx=ast.Load())],
                    keywords=[],
                )
                else_mask_assign = ast.Assign(
                    targets=[ast.Name(id=else_mask_name, ctx=ast.Store())],
                    value=negated_mask_value,
                )
                else_mask_assign = self._boolean_mask(else_mask_assign)

                if vectorization_context:
                    self._control_stack[-1].metadata.update(
                        {
                            "current_mask_assign": else_mask_assign,
                            "current_mask_rank": mask_rank,
                        }
                    )

                self._push_scope()
                new_stmts.append(else_mask_assign)
                for stmt in node.orelse:
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        for v in visited:
                            if isinstance(v, ast.Assign):
                                new_stmts.extend(
                                    self._mask_vector_assign(
                                        v, else_mask_name, assigned, used_after
                                    )
                                )
                            elif isinstance(v, ast.Continue):
                                continue
                            else:
                                new_stmts.append(v)
                    elif isinstance(visited, ast.Assign):
                        new_stmts.extend(
                            self._mask_vector_assign(
                                visited, else_mask_name, assigned, used_after
                            )
                        )
                    elif isinstance(visited, ast.Continue):
                        pass
                    elif visited is not None:
                        new_stmts.append(visited)

                false_scope = self._local_defined_stack[-1].copy()
                self._pop_scope()

                merged = {
                    name: true_scope[name]
                    for name in true_scope
                    if name in false_scope and true_scope[name] == false_scope[name]
                }
                self._local_defined_stack[-1].update(merged)

            if self._control_stack and self._control_stack[-1].kind == "if":
                self._control_stack.pop()

            if self._local_defaults:
                self._local_defaults.popitem()

            return new_stmts

        except Exception as e:
            self.logger.exception("Exception in _lower_vector_condition:", e)
            raise

    def _lower_index_loop_condition(
        self,
        node: ast.If,
        assigned: list[str],
        used_after: set[str],
        vectorization_context: dict | None,
    ) -> list[ast.AST]:
        """
        Lower an ``if`` whose condition is tied to a sequential
        (index-carrying) loop into a masked-update sequence.

        Structurally mirrors :meth:`_lower_vector_condition` — computes a
        boolean mask, pushes a ``Control(transform_type='index_loop')``
        scope, masks every assignment in the body via
        :meth:`_mask_vector_assign`, and (if present) repeats the process
        for ``node.orelse`` under the negated mask. The two paths are kept
        separate (rather than unified) because ``'index_loop'`` carries
        different downstream implications for ``handle_scan`` carry
        tracking than ``'vector'`` does.

        This method assumes neither branch contains a ``break`` — that
        case is intercepted earlier in :meth:`visit_If` and routed to
        :meth:`_emit_convergence_break` instead.

        Parameters
        ----------
        node : ast.If
            The conditional to lower.
        assigned : list[str]
            Names assigned anywhere in the branches.
        used_after : set[str]
            Names assigned and later read, passed through to
            :meth:`_mask_vector_assign`.
        vectorization_context : Optional[dict]
            The active :class:`Control` context as a dict.

        Returns
        -------
        list[ast.AST]
            The mask assignment(s) followed by every masked statement
            produced by visiting the branches.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            node.test = self.visit(node.test)
            ranks = self.subscript_ranks(node.test)
            mask_rank = ranks.get(node.test, 0)
            cond_copy = self._replace_subscript_with_name(
                ast.copy_location(node.test, node)
            )

            mask_name = f"_mask_{self._mask_counter}"
            self._mask_counter += 1
            mask_assign = ast.Assign(
                targets=[ast.Name(id=mask_name, ctx=ast.Store())],
                value=self.visit(cond_copy),
            )
            mask_assign = self._boolean_mask(mask_assign)

            if vectorization_context:
                self._control_stack.append(
                    Control(
                        kind="if",
                        loop_info=vectorization_context["loop_info"],
                        transform_type="index_loop",
                        vectorization_axis=vectorization_context["vectorization_axis"],
                        metadata={
                            "current_mask_assign": mask_assign,
                            "current_mask_rank": mask_rank,
                        },
                    )
                )

            self._push_scope()
            new_stmts = [mask_assign]
            for stmt in node.body:
                visited = self.visit(stmt)
                if isinstance(visited, list):
                    for v in visited:
                        if isinstance(v, ast.Assign):
                            new_stmts.extend(
                                self._mask_vector_assign(
                                    v, mask_name, assigned, used_after
                                )
                            )
                        elif isinstance(v, ast.Continue):
                            continue
                        else:
                            new_stmts.append(v)
                elif isinstance(visited, ast.Assign):
                    new_stmts.extend(
                        self._mask_vector_assign(
                            visited, mask_name, assigned, used_after
                        )
                    )
                elif isinstance(visited, ast.Continue):
                    pass
                elif visited is not None:
                    new_stmts.append(visited)

            true_scope = self._local_defined_stack[-1].copy()
            self._pop_scope()

            if node.orelse != []:
                else_mask_name = f"_mask_{self._mask_counter}"
                self._mask_counter += 1
                negated_mask_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_not",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=mask_name, ctx=ast.Load())],
                    keywords=[],
                )
                else_mask_assign = ast.Assign(
                    targets=[ast.Name(id=else_mask_name, ctx=ast.Store())],
                    value=negated_mask_value,
                )
                else_mask_assign = self._boolean_mask(else_mask_assign)

                if vectorization_context:
                    self._control_stack[-1].metadata.update(
                        {
                            "current_mask_assign": else_mask_assign,
                            "current_mask_rank": mask_rank,
                        }
                    )

                new_stmts.append(else_mask_assign)
                self._push_scope()
                for stmt in node.orelse:
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        for v in visited:
                            if isinstance(v, ast.Assign):
                                new_stmts.extend(
                                    self._mask_vector_assign(
                                        v, else_mask_name, assigned, used_after
                                    )
                                )
                            elif isinstance(v, ast.Continue):
                                continue
                            else:
                                new_stmts.append(v)
                    elif isinstance(visited, ast.Assign):
                        new_stmts.extend(
                            self._mask_vector_assign(
                                visited, else_mask_name, assigned, used_after
                            )
                        )
                    elif isinstance(visited, ast.Continue):
                        pass
                    elif visited is not None:
                        new_stmts.append(visited)

                false_scope = self._local_defined_stack[-1].copy()
                self._pop_scope()

                merged = {
                    name: true_scope[name]
                    for name in true_scope
                    if name in false_scope and true_scope[name] == false_scope[name]
                }
                self._local_defined_stack[-1].update(merged)

            if self._control_stack and self._control_stack[-1].kind == "if":
                self._control_stack.pop()

            if self._local_defaults:
                self._local_defaults.popitem()

            return new_stmts

        except Exception as e:
            self.logger.exception("Exception in _lower_index_loop_condition:", e)
            raise
