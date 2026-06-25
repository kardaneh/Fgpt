import ast

from fgpt.jax_utils import Control, get_name


class _LoopLowering:
    """
    Lowers Python ``for`` and ``while`` loops into ``lax.scan`` /
    ``jax.vmap`` / ``eqx.internal.while_loop`` primitives.

    Composes onto ``JaxConverter`` to implement the three loop-lowering
    strategies dispatched from ``visit_For`` based on the loop's
    classification (``self.analyzer.classify_for``):

    - **Sequential index loops** — :meth:`handle_scan` synthesises a
      ``_scan_body_N`` helper function carrying a tuple state, and
      replaces the loop with ``lax.scan(body_fn, init_carry,
      jnp.arange(...))``. Range bounds are first normalised via
      :meth:`simplify_range_args` (dropping redundant ``start``/``step``
      arguments when they default to ``0``/``1``), which itself relies
      on :meth:`is_static_expr` and ``extract_constant_value`` to
      confirm the bounds are compile-time constants. Carry ordering is
      stabilised via :meth:`sort_carry` so that nested scans and
      surrounding ``lax.cond`` helpers agree on argument positions.
    - **Vectorisable loops** — handled inline in ``visit_For`` (axis
      setup via :meth:`_setup_vector_control`) by removing the loop
      entirely and broadcasting the body over the relevant array axis;
      a ``while`` nested inside such a loop is instead lowered via
      :meth:`handle_vmap`, which wraps the loop body in a synthetic
      function and calls ``jax.vmap`` over the loop-dependent
      variables.
    - **``while`` loops** (:meth:`visit_While`) — rewritten into
      ``eqx.internal.while_loop(cond_fn, body_fn, init_state)``, with
      the state tuple's initial values resolved by walking the
      enclosing statement stack (:meth:`get_initial_values_from_stack`).
      The ``kind`` keyword (``'lax'`` vs ``'checkpointed'``) is chosen
      from :attr:`mode` to balance runtime speed against the memory
      cost of reverse-mode differentiation through long loops.

    All three strategies share dependency-resolution helpers from
    :class:`_BranchAnalysisMixin` (``_first_reads``, ``_collect_assigned``,
    etc.) to determine which names must be threaded through as
    carry/state, and read/write :attr:`_control_stack`,
    :attr:`_scan_stack`, and :attr:`_func_arg_stack` to keep nested
    helper-function scoping consistent across loop levels.
    """

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.AST] | None:
        """
        Transform supported ``for`` loops into functional control-flow
        representations.

        This visitor classifies range-based loops and lowers them into the
        appropriate intermediate representation depending on their
        iteration semantics. Index-based loops are transformed into
        :func:`lax.scan` constructs, vectorized loops are rewritten into
        vectorized regions, and loops containing vectorized while-loop
        structures are lowered through :meth:`handle_vmap`.

        Unsupported loop forms are delegated to
        :meth:`ast.NodeTransformer.generic_visit`.

        Parameters
        ----------
        node : ast.For
            Loop node to transform.

        Returns
        -------
        ast.AST | list[ast.AST] | None
            Transformed node representation.

            * ``None`` if the loop body becomes empty after removing
            logging statements.
            * A list of generated AST nodes for lowered scan, vectorized,
            or vmap-based transformations.
            * The result of :meth:`generic_visit` for unsupported loop
            structures.

        Notes
        -----
        Before analysis, logging statements identified by
        :meth:`_is_logging_call` are removed from the loop body.

        Only loops of the form::

            for i in range(start, stop, step):

        are handled by this transformation pass.

        Loop classification is delegated to
        :attr:`analyzer` through
        :meth:`analyzer.classify_for`, which determines one of the
        following transformation strategies:

        ``"index_loop"``
            Lowered to :func:`lax.scan` through :meth:`handle_scan`.

        ``"vector"``
            Rewritten under an active vectorization context. Vectorization
            metadata is created by :meth:`_setup_vector_control` and stored
            in :attr:`_control_stack`.

        ``"vector_while"``
            Lowered through :meth:`handle_vmap` to support vectorized
            regions containing transformed while loops.

        Dynamic loop bounds, including subscript expressions, function
        calls, or stateful loop dimensions, are delegated to
        :meth:`handle_dynamic_loop`.

        During vectorized transformations, assignment targets may be
        adjusted using :meth:`maybe_add_index` to preserve shape
        consistency after index lifting.

        Scope management for vectorized regions is handled through
        :meth:`_push_scope` and :meth:`_pop_scope`.

        See Also
        --------
        :meth:`handle_scan`
            Lowers index-based loops into :func:`lax.scan`.

        :meth:`handle_vmap`
            Generates vectorized helper functions using
            :func:`jax.vmap`.

        :meth:`handle_dynamic_loop`
            Handles loops with dynamic iteration bounds.

        :meth:`_setup_vector_control`
            Creates vectorization metadata for loop transformations.

        :attr:`_control_stack`
            Stores active control-flow and vectorization contexts.

        :attr:`symbol_table`
            Records inferred types for loop indices.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during loop
            transformation.
        """
        try:
            node.body = [stmt for stmt in node.body if not self._is_logging_call(stmt)]
            if not node.body:
                return None

            range_present = (
                isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            )
            # Only supporting for i in range(a, b)
            if not range_present:
                return self.generic_visit(node)

            loop_index = node.target.id
            range_args = node.iter.args
            start, stop, step = tuple(range_args)

            if isinstance(stop, ast.Subscript | ast.Call):
                self.symbol_table[loop_index] = "int64"
                return self.handle_dynamic_loop(node, start, stop, step, loop_index)

            if any(
                isinstance(node, ast.Name) and node.id in self._stateful_vars
                for node in (start, stop, step)
                if node is not None
            ):
                self.symbol_table[loop_index] = "int64"
                return self.handle_dynamic_loop(node, start, stop, step, loop_index)

            cond_type = self.analyzer.classify_for(node)
            # First we need to verify if the range args has start loop at 0 then we just send the args[1],
            # but we also need to handle the cases of when we have args[1] isn't static
            # which means we need to ensure that attributes and binop cases that we don't have arrays present inside them
            if cond_type == "index_loop":
                self.symbol_table[loop_index] = "int64"
                return self.handle_scan(node, start, stop, step, loop_index=loop_index)

            elif cond_type == "vector":
                self._push_scope()
                try:
                    vectorization_axis = self._setup_vector_control(
                        loop_index, get_name(stop), cond_type, node
                    )
                    vectorization_context = None
                    if self._control_stack:
                        vectorization_context = self._control_stack[-1].to_dict()
                    new_stmts = []
                    for stmt in node.body:
                        visited = self.visit(stmt)
                        if isinstance(visited, list):
                            new_stmts.extend(visited)
                        elif isinstance(visited, ast.Assign):
                            # THis is done here due to the reason being that when the [:,:]
                            # disappears we need to tend to their shapes wihting the for loop
                            # since due to vectorization only the for loop body exist
                            if isinstance(stmt.targets[0], ast.Subscript):
                                target_rank = self._target_rank(stmt.targets[0].slice)
                                visited.value, _ = self.maybe_add_index(
                                    visited.value, target_rank, vectorization_context
                                )
                                new_stmts.append(visited)
                            else:
                                new_stmts.append(visited)
                        elif visited is not None:
                            new_stmts.append(visited)
                    if self._control_stack and self._control_stack[-1].kind == "loop":
                        self._control_stack.pop()

                    return new_stmts
                finally:
                    self._pop_scope()

            elif cond_type == "vector_while":
                self._push_scope()
                try:
                    self.symbol_table[loop_index] = "int64"
                    # NOTE: Since we have a different transformation strategy for the while
                    # within a for loop(vectorizable) thus we separate them
                    vectorization_axis = self._setup_vector_control(
                        loop_index, get_name(stop), cond_type, node
                    )
                    return self.handle_vmap(node, loop_index, vectorization_axis)
                finally:
                    self._pop_scope()

        except Exception as e:
            self.logger.exception("Exception in visit_For:", e)
            raise

    def handle_scan(
        self,
        node: ast.For,
        start: ast.AST,
        stop: ast.AST,
        step: ast.AST,
        loop_index: str,
    ) -> list[ast.AST]:
        """
        Lower a Python ``for`` loop into a :func:`lax.scan`
        transformation.

        This method converts an imperative loop into a functional scan by
        synthesizing a helper function representing a single loop iteration,
        computing loop-carried state, and generating the corresponding
        :func:`lax.scan` invocation.

        Variables that are read before modification are treated as scan
        inputs, while variables mutated within the loop body become scan
        carry values. Newly introduced state resulting from nested control
        flow or transformed function calls is propagated through
        :attr:`_scan_stack` and incorporated into the carry tuple when
        required.

        Parameters
        ----------
        node : ast.For
            Loop node being transformed.
        start : ast.AST
            Lower bound of the loop range.
        stop : ast.AST
            Upper bound of the loop range.
        step : ast.AST
            Step expression of the loop range.
        loop_index : str
            Name of the loop iteration variable.

        Returns
        -------
        list[ast.AST]
            Sequence of generated AST nodes consisting of the synthesized
            scan-body helper and the corresponding :func:`lax.scan`
            invocation.

        Notes
        -----
        Range bounds are normalized using
        :meth:`simplify_range_args`.

        The generated helper has the form::

            def _scan_body_N(carry, idx):
                ...
                return carry, None

        Carry-state analysis uses:

        * :meth:`_first_reads`
        * :meth:`_collect_assigned`
        * :meth:`_collect_rhs_uses`
        * :attr:`_scan_stack`
        * :attr:`_modified_ret_stack`

        Nested scan contexts propagate mutation and carry information
        through :attr:`_scan_stack` to ensure correct state threading across
        nested control-flow regions.

        Generated helper functions are registered in
        :attr:`_pending_helpers` and processed later by
        :meth:`process_helpers`.

        Vectorized assignments may be rewritten through
        :meth:`_mask_vector_assign` and
        :meth:`maybe_add_index` when an active vectorization context is
        present.

        See Also
        --------
        :meth:`process_helpers`
            Processes synthesized scan-body helper functions.

        :meth:`sort_carry`
            Determines carry ordering.

        :meth:`simplify_range_args`
            Normalizes range bounds.

        :attr:`_scan_stack`
            Tracks carry and mutation information for nested scans.

        Raises
        ------
        NotImplementedError
            If the loop range contains unsupported dynamic expressions.

        Exception
            Re-raises any unexpected error encountered during scan
            generation.
        """
        try:
            try:
                final_args = self.simplify_range_args(start, stop, step)
            except NotImplementedError as e:
                raise NotImplementedError(f"Dynamic loop range is not implemented: {e}")

            # Collect the assigned variables (outputs of IF transformations)
            branch_stmts = list(node.body)
            assigned = self._collect_assigned(branch_stmts)
            rhs_uses = self._collect_rhs_uses(branch_stmts)
            used_after = set(assigned) & rhs_uses
            read_before_write = self._first_reads(branch_stmts)
            inputs = []

            # These contain attributes that were newly mutated by function calls and
            # have already been marked as assigned, but have not yet been propagated
            # to `mutated_attrs`. This situation occurs for function calls nested
            # inside `lax.scan` and `lax.cond`, where the mutation information is
            # collected separately and has not yet been merged into the global
            # `mutated_attrs` set.
            prev_scan_ = []
            if self._scan_stack:
                prev_scan_ = list(
                    self._scan_stack[-1]["mutated"] | self._scan_stack[-1]["introduced"]
                )
            # This does region based aggregation
            start_idx = len(self._modified_ret_stack)
            self._modified_ret_stack.append([])

            inside_helper = len(self._context_stack) > 0 and any(
                self.to_arg(name) in self._context_stack[-1]["helper_args"]
                for name in self._mutated_attrs
            )
            for name in read_before_write:
                if name.startswith("self."):
                    attr = self.to_arg(name)

                    if attr in self._mutated_attrs:
                        if inside_helper:
                            # INSIDE helper:
                            # Use the local variable rather than `self.attr`. If the attribute has
                            # not been modified within the helper's scope, referencing the local name
                            # is sufficient. Otherwise, use the updated value corresponding to the
                            # modified attribute.
                            if attr in assigned:
                                inputs.append(attr)
                        else:
                            # OUTSIDE helper:
                            # Use `self.attr` when referring to the original instance attribute.
                            # After the attribute has been mutated and replaced by a tracked local
                            # variable, use that variable directly rather than accessing `self.attr`
                            # again.
                            if (
                                self._func_arg_stack
                                and attr
                                in [self.to_arg(n) for n in self._func_arg_stack[-1]]
                            ) or attr in self._var_modif["attr"]:
                                inputs.append(attr)
                            else:
                                inputs.append(name)
                    elif attr in prev_scan_:
                        inputs.append(attr)

                elif name not in self._always_exclude and name != loop_index:
                    # We don't need the loop index (mostly for the outer loop cases)
                    inputs.append(name)

            if self._local_defined_stack and self._local_defined_stack[-1]:
                inputs.extend(list(self._local_defined_stack[-1]))

            inputs.sort()  # sort the input

            # Even also need to remove the any vectorized loop index from the inputs since they won't be necessary
            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            if vectorization_context:
                for key in list(vectorization_context["vectorization_axis"].keys()):
                    if key in inputs:
                        inputs.remove(key)

            arg_stack = set()
            # NOTE:
            # This differs from the `arg_stack` used in `visit_If`. Here, the stack
            # serves as the source of argument information when packing and unpacking
            # values passed through the carry. Since JAX carry values must be explicit
            # data containers, they must not include any `self` references. Therefore,
            # this stack tracks only the arguments that should be extracted from or
            # inserted into the carry state.
            for names in inputs:
                if "." in names and names.split(".", 1)[1] in self._mutated_attrs:
                    arg_stack.add(names.split(".", 1)[1])
                else:
                    arg_stack.add(names)

            new_carry = self.sort_carry(inputs, arg_stack, prev_scan_)
            # THis is just to ensure that the carry statments has the same positions of that of the inputs assigned

            self._func_arg_stack.append(new_carry)
            # Removing this causes an error on the inner nested for loop transformations !!!

            # Build helper name
            body_id = f"_scan_body_{self.for_counter}"
            self.for_counter += 1

            # Create the helper function: (carry, idx) -> (carry, None)
            # since we don't need the array of the recursively accumaulated values
            helper_fn = ast.FunctionDef(
                name=body_id,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="carry"), ast.arg(arg=loop_index)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[],
                decorator_list=[],
            )
            # Insert the loop body (already containing if-rewrites)
            self._scan_stack.append(
                {
                    "carry": set(new_carry),
                    "mutated": set(),
                    "introduced": set(),
                    "parent_reads": set(inputs),
                }
            )
            for stmt in node.body:
                transformed = self.visit(stmt)
                if isinstance(transformed, list):
                    for v in transformed:
                        if isinstance(v, ast.Assign):
                            if vectorization_context and vectorization_context[
                                "metadata"
                            ].get("current_mask_assign"):
                                mask_name = vectorization_context["metadata"].get(
                                    "current_mask_assign"
                                )
                                helper_fn.body.extend(
                                    self._mask_vector_assign(
                                        v, mask_name.targets[0].id, assigned, used_after
                                    )
                                )
                            else:
                                helper_fn.body.append(v)
                        else:
                            helper_fn.body.append(v)
                elif isinstance(transformed, ast.Assign):
                    if vectorization_context and vectorization_context["metadata"].get(
                        "current_mask_assign"
                    ):
                        mask_name = vectorization_context["metadata"].get(
                            "current_mask_assign"
                        )
                        helper_fn.body.extend(
                            self._mask_vector_assign(
                                transformed,
                                mask_name.targets[0].id,
                                assigned,
                                used_after,
                            )
                        )
                    else:
                        if isinstance(stmt, ast.Assign) and isinstance(
                            stmt.targets[0], ast.Subscript
                        ):
                            target_rank = self._target_rank(stmt.targets[0].slice)
                            transformed.value, _ = self.maybe_add_index(
                                transformed.value, target_rank, vectorization_context
                            )
                            helper_fn.body.append(transformed)
                        else:
                            helper_fn.body.extend([transformed])
                elif transformed is not None:
                    helper_fn.body.append(transformed)

            ctx = self._scan_stack.pop()
            # Need to check if the the functions has modiifed any element
            # thus returned and needs to be added upon the carry
            collected = []
            while len(self._modified_ret_stack) > start_idx:
                collected.extend(self._modified_ret_stack.pop())

            local_modified = list(set(collected))

            if self._modified_ret_stack:
                self._modified_ret_stack[-1].extend(local_modified)

            local_modified = list(ctx["mutated"] & ctx["introduced"])

            if local_modified:
                seen = set(new_carry)
                for el in local_modified:
                    if el not in seen:
                        new_carry.append(el)
                        seen.add(el)

            # Unpack carry
            unpack = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=v, ctx=ast.Store()) for v in new_carry],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Name(id="carry", ctx=ast.Load()),
            )
            helper_fn.body.insert(0, unpack)

            # Repack updated carry
            repack = ast.Assign(
                targets=[ast.Name(id="carry", ctx=ast.Store())],
                value=ast.Tuple(
                    elts=[ast.Name(id=v, ctx=ast.Load()) for v in new_carry],
                    ctx=ast.Load(),
                ),
            )
            helper_fn.body.append(repack)

            # Return (carry, None)
            helper_fn.body.append(
                ast.Return(
                    value=ast.Tuple(
                        elts=[
                            ast.Name(id="carry", ctx=ast.Load()),
                            ast.Constant(value=None),
                        ],
                        ctx=ast.Load(),
                    )
                )
            )
            # Add helper to transformation pipeline
            self._pending_helpers.append(helper_fn)
            parent = {}
            if self._scan_stack:
                parent = self._scan_stack[-1]
                # propagate mutations upward
                parent["mutated"].update(ctx["mutated"])
                # ONLY propagate introduced vars if they escape
                escaping = ctx["mutated"] & parent["parent_reads"]
                parent["introduced"].update(escaping)

                parent["carry"].update(set(new_carry))

            # Construct the initial carry
            updated_inputs = []
            seen = set()
            # Account for attributes returned from nested function calls when
            # determining inputs and mutated state. This is especially important
            # inside `lax.scan` and `lax.cond`, where returned attributes may not be
            # directly visible in the surrounding scope but must still be propagated
            # through the transformed state.

            modified = set(local_modified) if local_modified else set()
            for name in inputs:
                if "." in name:
                    base = self.to_arg(name)

                    if base in modified:
                        updated_inputs.append(base)
                        seen.add(base)
                    else:
                        if parent and base in parent.get("carry", []):
                            updated_inputs.append(base)
                            seen.add(base)
                        else:
                            updated_inputs.append(name)
                            seen.add(base)
                else:
                    updated_inputs.append(name)
                    seen.add(name)

            diff = set()
            if local_modified:
                # These correspond to attributes that must be accessed via `self` because
                # they were not modified prior to this step, but are updated during the
                # current transformation process. As a result, they need to be passed and
                # referenced as instance (`self`) attributes to preserve correct state.
                diff = set(local_modified) - (
                    (self._mutated_attrs) | self._var_modif["attr"]
                )

            for var in modified:
                if var in seen:
                    continue

                if var == "self":
                    candidate = "self"
                elif var in diff:
                    if "converged_var" in ctx and var in ctx["converged_var"]:
                        candidate = var
                    else:
                        candidate = f"self.{var}"
                else:
                    candidate = var

                updated_inputs.append(candidate)
                seen.add(candidate)

            normalized_map = {}
            for val in updated_inputs:
                key = self.to_arg(val)
                normalized_map[key] = val

            # Rebuild in new_carry order
            ordered_inputs = []
            for var in new_carry:
                key = self.to_arg(var)
                if key in normalized_map:
                    ordered_inputs.append(normalized_map[key])

            updated_inputs = ordered_inputs

            if parent:  # the parent needs to have the updated inputs
                parent["parent_reads"].update(set(updated_inputs))

            elts = []
            for v in updated_inputs:
                if isinstance(v, str) and v.startswith("self."):
                    attr = v.split(".", 1)[1]
                    elts.append(
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr=attr,
                            ctx=ast.Load(),
                        )
                    )
                else:
                    elts.append(ast.Name(id=v, ctx=ast.Load()))

            init_carry = ast.Tuple(elts=elts, ctx=ast.Load())

            # Create the lax.scan call:
            loop_call = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=v, ctx=ast.Store()) for v in new_carry
                                ],
                                ctx=ast.Store(),
                            ),
                            ast.Name(id="_", ctx=ast.Store()),
                        ]
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="lax", ctx=ast.Load()),
                        attr="scan",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id=body_id, ctx=ast.Load()),
                        init_carry,
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()),
                                attr="arange",
                                ctx=ast.Load(),
                            ),
                            args=final_args,
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                ),
            )

            result = [helper_fn, loop_call]
            if "converged_init" in ctx:
                result.insert(1, ctx["converged_init"])

            return result

        except Exception as e:
            self.logger.exception("Exception in handle_scan:", e)
            raise

    def handle_vmap(
        self,
        node: ast.AST,
        loop_index: str,
        vectorization_axis: dict,
    ) -> list[ast.AST]:
        """
        Lower a vectorized loop region into a :func:`jax.vmap`
        transformation.

        This method extracts loop-body computations into a synthesized
        helper function and applies :func:`jax.vmap` over variables that
        depend on the active vectorization axis.

        The transformation is primarily used when vectorized control-flow
        regions contain constructs that cannot be directly represented by a
        standard scan transformation, such as transformed
        :func:`eqx.internal.while_loop` invocations.

        Parameters
        ----------
        node : ast.AST
            Loop or control-flow node representing the vectorized region.
        loop_index : str
            Name of the active vectorization index variable.
        vectorization_axis : dict
            Mapping describing active vectorized dimensions.

        Returns
        -------
        list[ast.AST]
            Generated helper function definition, corresponding
            :func:`jax.vmap` invocation, and any deferred update statements.

        Notes
        -----
        Dependency analysis is delegated to
        :attr:`while_transformer`, which records variables dependent on the
        vectorization index.

        The generated helper has the form::

            def vmap_<loop_index>(...):
                ...
                return ...

        Calls to :func:`eqx.internal.while_loop` are handled specially so
        that loop-carried values can be returned from the vectorized
        function and reassigned after vectorization.

        This method is typically invoked after vectorization metadata has
        been established through :meth:`_setup_vector_control`.

        See Also
        --------
        :meth:`_setup_vector_control`
            Creates vectorization metadata for control-flow regions.

        :attr:`while_transformer`
            Tracks vectorized dependencies and transforms while loops.

        :meth:`visit_While`
            Generates vectorizable while-loop representations.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during
            transformation.
        """
        try:
            # Before the generic_visit, we need to retrieve all the arrays and elements that is dependant or uses present inside the body
            # mostly subscripts, needs to be sent as arguments to the funciton and also to find teh while loop structure(variable used inside and loop index)
            self.while_transformer.vectorization_axis = vectorization_axis
            self.while_transformer.loop_index = loop_index  # the current loop index
            self.while_transformer.visit(node)
            # print(f'ji dependant vars : {transformer.ji_dependent_vars}')
            new_body = []
            ret_names = []

            def is_eqx_while_loop_call(node):
                if not isinstance(node, ast.Call):
                    return False

                func = node.func

                # Match eqx.internal.while_loop
                return (
                    isinstance(func, ast.Attribute)
                    and func.attr == "while_loop"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "internal"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "eqx"
                )

            updated_values = []
            for stmt in node.body:
                visited = self.visit(stmt)

                if isinstance(visited, list):
                    if not visited:
                        continue

                    eqx_pos = None
                    for i, last_node in enumerate(visited):
                        if isinstance(last_node, ast.Assign) and is_eqx_while_loop_call(
                            last_node.value
                        ):
                            targets = last_node.targets
                            for t in targets:
                                if isinstance(t, ast.Tuple):
                                    ret_names.extend(
                                        elt.id
                                        for elt in t.elts
                                        if isinstance(elt, ast.Name)
                                    )
                                elif isinstance(t, ast.Name):
                                    ret_names.append(t.id)
                            eqx_pos = i
                            break

                    if eqx_pos is not None:
                        new_body.extend(visited[: eqx_pos + 1])
                        updated_values.extend(visited[eqx_pos + 1 :])
                    else:
                        new_body.extend(visited)

                elif visited is not None:
                    new_body.append(visited)

            node.body = new_body
            # Create the function defintion
            fn_def = ast.FunctionDef(
                name=f"vmap_{loop_index}",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg=name, annotation=None)
                        for name in self.while_transformer.ji_dependent_vars
                    ]
                    if self.while_transformer.ji_dependent_vars
                    else [],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=list(node.body),  # copy of the loop body
                decorator_list=[],
                returns=None,
            )
            # Append the return statement
            if len(ret_names) == 1:
                return_value = ast.Name(id=ret_names[0], ctx=ast.Load())
                targets = [ast.Name(id=ret_names[0], ctx=ast.Store())]
            else:
                return_value = ast.Tuple(
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in ret_names],
                    ctx=ast.Load(),
                )
                targets = [
                    ast.Tuple(
                        elts=[ast.Name(id=name, ctx=ast.Store()) for name in ret_names],
                        ctx=ast.Store(),
                    )
                ]

            fn_def.body.append(ast.Return(value=return_value))
            # Now we create the call statement for the vmap function
            assign_stmt = ast.Assign(
                targets=targets,
                value=ast.Call(
                    func=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jax", ctx=ast.Load()),
                            attr="vmap",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=fn_def.name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    args=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in self.while_transformer.ji_dependent_vars
                    ],
                    keywords=[],
                ),
            )
            return [fn_def, assign_stmt] + updated_values

        except Exception as e:
            self.logger.exception("Exception in handle_vmap:", e)
            raise

    def _setup_vector_control(
        self,
        loop_index: str,
        stop: str,
        cond_type: str,
        node: ast.For,
    ) -> dict:
        """
        Create or update vectorization metadata for a control-flow region.

        This method computes active vectorization axes for a loop and
        records the resulting metadata in :attr:`_control_stack`. If a
        control context already exists, the new loop information is merged
        into the current context.

        Parameters
        ----------
        loop_index : str
            Loop iteration variable.
        stop : str
            Loop dimension associated with the iteration variable.
        cond_type : str
            Type of control-flow transformation being applied.
        node : ast.For
            Loop node used for vectorization analysis.

        Returns
        -------
        dict
            Mapping of vectorized loop variables to their associated axes.

        Notes
        -----
        Vectorization information is computed using
        :meth:`compute_vectorization_axes`.

        The resulting metadata is stored in :attr:`_control_stack`
        together with loop-dimension information and transformation type.

        Nested vectorized control-flow regions extend the currently active
        control context rather than creating a new one.

        See Also
        --------
        :meth:`compute_vectorization_axes`
            Determines vectorized dimensions within a loop body.

        :attr:`_control_stack`
            Stores active control-flow and vectorization metadata.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during analysis.
        """
        try:
            # If the cond_type is that of vector, we need to find the vectorization axis
            vectorization_axis = self.compute_vectorization_axes(
                node, loop_index=loop_index
            )

            # in the case if the control stack already has a previous element,
            # we can simply attach this information to the previous element
            if not self._control_stack:
                self._control_stack.append(
                    Control(
                        "loop",
                        loop_info={stop: loop_index},
                        transform_type=cond_type,
                        vectorization_axis=vectorization_axis,
                    )
                )
            else:
                prev_element = self._control_stack[-1]
                prev_element.loop_info[stop] = loop_index
                prev_element.vectorization_axis |= vectorization_axis

            return vectorization_axis
        except Exception as e:
            self.logger.exception("Exception in _setup_vector_control:", e)
            raise

    def get_initial_values_from_stack(
        self, inputs: list[str]
    ) -> dict[str, ast.AST | None]:
        """
        Retrieve initial values for loop-carried variables from enclosing
        scopes.

        This method walks the parent-node stack in reverse order and
        searches for assignments corresponding to the supplied variable
        names. Variables, object attributes, and subscript bases are
        supported.

        Parameters
        ----------
        inputs : list[str]
            Names of variables for which initial values are required.

        Returns
        -------
        dict[str, ast.AST | None]
            Mapping from variable name to the corresponding initializing
            expression.

        Notes
        -----
        The search is performed over :attr:`reduction._parent_stack`,
        beginning with the nearest enclosing scope.

        Supported assignment targets include:

        * Local variables.
        * ``self.<attr>`` attributes.
        * Array bases referenced through subscripts.

        The first matching assignment encountered while traversing outward
        through the parent stack is used.

        This method is primarily used by :meth:`visit_While` when
        constructing the initial state tuple for functional while-loop
        transformations.

        See Also
        --------
        :meth:`visit_While`
            Uses the returned values to build ``init_state``.

        :attr:`reduction._parent_stack`
            Maintains enclosing AST context during traversal.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during lookup.
        """
        try:
            initial_values = {name: None for name in inputs}

            # Walk the parent stack backward (skip current node)
            for ctx_node in reversed(self.reduction._parent_stack[:-1]):
                body = getattr(ctx_node, "body", None)
                if body is None:
                    continue

                for stmt in body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            # Plain variable
                            if isinstance(target, ast.Name):
                                var_name = target.id
                            # self attribute
                            elif (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                var_name = f"self.{target.attr}"
                            # Subscript: take the base name
                            elif isinstance(target, ast.Subscript):
                                base = target.value
                                if isinstance(base, ast.Name):
                                    var_name = base.id
                                elif (
                                    isinstance(base, ast.Attribute)
                                    and isinstance(base.value, ast.Name)
                                    and base.value.id == "self"
                                ):
                                    var_name = f"self.{base.attr}"
                                else:
                                    continue
                            else:
                                continue

                            # Save first occurrence
                            if (
                                var_name in initial_values
                                and initial_values[var_name] is None
                            ):
                                initial_values[var_name] = stmt.value

                if all(v is not None for v in initial_values.values()):
                    break

            return initial_values
        except Exception as e:
            self.logger.exception("Exception in get_initial_values_from_stack:", e)
            raise

    def visit_While(self, node: ast.While) -> list[ast.AST]:
        """
        Transform a Python ``while`` loop into an
        :func:`eqx.internal.while_loop` representation.

        Imperative loop state is converted into an explicit state tuple that
        is threaded through generated condition and body functions. The
        resulting transformation is compatible with JAX tracing and supports
        both scalar and vectorized loop conditions.

        Parameters
        ----------
        node : ast.While
            While-loop node to transform.

        Returns
        -------
        list[ast.AST]
            Generated condition function, body function, initial state
            construction, while-loop invocation, and any deferred state
            update statements.

        Notes
        -----
        The transformation generates functions equivalent to::

            def cond_fn(state):
                ...
                return condition

            def body_fn(state):
                ...
                return new_state

        which are passed to
        :func:`eqx.internal.while_loop`.

        Condition analysis is performed using
        :meth:`check_condition` to determine whether the loop condition is
        scalar or vectorized.

        For vectorized conditions, expression rewriting is delegated to
        :attr:`while_transformer` and may subsequently be lifted through
        :func:`jax.vmap`.

        Initial loop state is obtained from
        :meth:`get_initial_values_from_stack`.

        Depending on :attr:`mode`, the generated while loop executes in
        either ``"lax"`` or ``"checkpointed"`` mode.

        This method forms the core lowering pass for imperative while loops
        and integrates with vectorization, reduction, and state-propagation
        transformations.

        See Also
        --------
        :meth:`check_condition`
            Determines whether a condition is scalar or vectorized.

        :meth:`get_initial_values_from_stack`
            Retrieves initial state values.

        :meth:`handle_vmap`
            Lifts vectorized while-loop regions through :func:`jax.vmap`.

        :attr:`while_transformer`
            Performs while-specific expression rewriting.

        Raises
        ------
        ValueError
            If the initial loop state cannot be constructed.

        Exception
            Re-raises any unexpected error encountered during
            transformation.
        """
        try:
            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            # We first need to find if the conditon is that of vector or scalar
            is_vector, vector_arrays = self.check_condition(
                node.test, vectorization_context
            )
            node = self.generic_visit(node)

            # Condition state function
            branch_stmts = list(node.body)
            assigned = self._collect_assigned(branch_stmts)
            rhs_uses = self._collect_rhs_uses(branch_stmts)
            used_after = set(assigned) & rhs_uses
            read_before_write = self._first_reads(branch_stmts)

            inputs = []
            inside_helper = len(self._context_stack) > 0 and any(
                self.to_arg(name) in self._context_stack[-1]["helper_args"]
                for name in self._mutated_attrs
            )
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

            if self._local_defined_stack and self._local_defined_stack[-1]:
                inputs.extend(list(self._local_defined_stack[-1]))

            state_inputs = list(set(inputs) | used_after)
            state_inputs.sort()  # sort the input

            intial_values = self.get_initial_values_from_stack(state_inputs)
            # We need replace the rest of the inputs with value other than that of the loop variable of while
            var_to_replace = {}
            for i, var in enumerate(state_inputs):
                if var in (set(state_inputs) - set(inputs)):
                    state_inputs[i] = var + "_val"
                    var_to_replace[var] = state_inputs[i]

            # Now we need to create teh init_state
            if not all(list(intial_values.values())):
                raise ValueError("Intial_values dict is empty")
            init_state = ast.Assign(
                targets=[ast.Name(id="init_state", ctx=ast.Store())],
                value=ast.Tuple(elts=list(intial_values.values()), ctx=ast.Load()),
            )

            # NOTE: There are two cases for handling the while-loop transformation,
            # as required for compatibility with `eqx.internal.while_loop`.
            # Case 1: is_vector = True
            # The original Python for-loop is transformed using `vmap` to preserve
            # vectorized semantics and correctly represent the iteration behavior
            # of native Python loops.
            # Case 2: is_vector = False
            # The loop condition is a scalar boolean, so the condition can be used
            # directly without vectorization, and the `while_loop` is applied as-is.
            self.while_transformer.vector_arrays = vector_arrays
            self.while_transformer.var_to_replace = var_to_replace
            self.while_transformer._in_while = True
            if is_vector:
                # Vectorized case: condition depends on mapped axis -> will be lifted under vmap
                test_node = self.while_transformer.visit(node.test)
                test_node = ast.fix_missing_locations(test_node)
                test_node = self._transform_if_test(test_node)
                cond_fn = ast.FunctionDef(
                    name="cond_fn",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="state")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Tuple(
                                    elts=[
                                        ast.Name(id=el, ctx=ast.Store())
                                        for el in state_inputs
                                    ]
                                )
                            ],
                            value=ast.Name(id="state", ctx=ast.Load()),
                        ),
                        ast.Return(value=test_node),
                    ],
                    decorator_list=[],
                )

            else:
                test_node = self._transform_if_test(node.test)
                cond_fn = ast.FunctionDef(
                    name="cond_fn",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="state")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Tuple(
                                    elts=[
                                        ast.Name(id=el, ctx=ast.Store())
                                        for el in state_inputs
                                    ]
                                )
                            ],
                            value=ast.Name(id="state", ctx=ast.Load()),
                        ),
                        ast.Return(value=test_node),
                    ],
                    decorator_list=[],
                )

            body = []
            for n in node.body:
                new_node = self.while_transformer.visit(n)
                body.append(ast.copy_location(new_node, n))
            node.body = body

            # NOTE:
            # Replace array-based variables with their corresponding loop-state
            # representations (excluding the loop index). In functional
            # while-loops, only the final state after all iterations is returned,
            # so all loop-carried variables must be explicitly threaded through the
            # state tuple.
            #
            # We operate on scalarized or per-element values inside the loop body
            # to avoid repeated use of `.at[].set()`, which is expensive in JAX
            # because it produces a new array at each update. Although a buffer-based
            # approach could reduce overhead, in-place updates are not supported
            # (see: https://github.com/patrick-kidger/equinox/blob/74f5b5f1895fb1b2512de4e10424d73a7e1e6632/equinox/internal/loop/loop.py#L21).
            #
            # Only the updated RHS values are propagated forward as the new loop
            # state at each iteration.
            body_fn = ast.FunctionDef(
                name="body_fn",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="state")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=el, ctx=ast.Store())
                                    for el in state_inputs
                                ]
                            )
                        ],
                        value=ast.Name(id="state", ctx=ast.Load()),
                    ),
                ]
                + node.body
                + [
                    ast.Return(
                        value=ast.Tuple(
                            elts=[
                                ast.Name(id=el, ctx=ast.Load()) for el in state_inputs
                            ],
                            ctx=ast.Load(),
                        )
                    )
                ],
                decorator_list=[],
            )
            # Select execution mode (lax vs checkpointed)
            if self.mode in ["fwd", "jax"]:
                kind = ast.keyword(arg="kind", value=ast.Constant(value="lax"))
            else:
                kind = ast.keyword(arg="kind", value=ast.Constant(value="checkpointed"))
            input_elts = []
            for state_input in state_inputs:
                if state_input in inputs:
                    input_elts.append(ast.Name(id="_", ctx=ast.Store()))
                else:
                    input_elts.append(ast.Name(id=state_input, ctx=ast.Store()))

            while_cond = ast.Assign(
                targets=[ast.Tuple(elts=input_elts, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="eqx", ctx=ast.Load()),
                            attr="internal",
                            ctx=ast.Load(),
                        ),
                        attr="while_loop",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id="cond_fn", ctx=ast.Load()),
                        ast.Name(id="body_fn", ctx=ast.Load()),
                        ast.Name(id="init_state", ctx=ast.Load()),
                    ],
                    keywords=[kind],
                ),
            )

            self.while_transformer._in_while = False
            # Now we need to update the accumaulated values inside it's corresponding variable that was accumulated inside
            updated = []
            for key, values in self.while_transformer.while_used_vars.items():
                if isinstance(values, ast.Assign):
                    values.value = ast.Name(id=f"{key}_val", ctx=ast.Load())
                visited = self.visit(values)
                updated.append(visited)

            return [cond_fn, body_fn, init_state, while_cond] + updated

        except Exception as e:
            self.logger.exception("Exception in visit_While:", e)
            raise

    def sort_carry(self, inputs: list, carry: set, prev_scan: list):
        """
        Order scan carry variables according to their appearance in the
        original input sequence.

        Carry values are normalized before sorting by converting attribute
        references of the form ``self.<attr>`` into plain attribute names
        when the attribute is known to be mutable or was introduced by a
        previous scan transformation.

        The resulting ordering preserves consistency between generated scan
        carry tuples and the corresponding function inputs.

        Parameters
        ----------
        inputs : list
            Original input variable names used to establish ordering.
        carry : set
            Carry variables that must be threaded through a scan
            transformation.
        prev_scan : list
            Variables introduced by previously generated scan operations.

        Returns
        -------
        list
            Carry variables sorted according to their position in
            ``inputs``.

        Notes
        -----
        Attribute normalization uses :attr:`_mutated_attrs` to determine
        whether ``self.<attr>`` references should be treated as local state
        variables.

        Variable names are normalized using :meth:`to_arg` before
        determining their ordering.

        This method is primarily used during scan lowering to ensure stable
        carry tuple construction.

        See Also
        --------
        :meth:`to_arg`
            Converts AST values and attribute references into normalized
            argument names.

        :attr:`_mutated_attrs`
            Tracks attributes that have been rewritten into explicit state
            variables.
        """
        # Normalize names by removing 'self.' when needed
        temp = []
        for name in inputs:
            if "." in name:
                attr = name.split(".", 1)[1]
                if attr in self._mutated_attrs or attr in prev_scan:
                    temp.append(attr)
                    continue
            temp.append(name)

        # Map each input name to its index
        temp_dict = {name: i for i, name in enumerate(temp)}

        # Sort carry items by their position in the input list
        new_carry = sorted(carry, key=lambda v: temp_dict[self.to_arg(v)])

        return new_carry

    def is_static_expr(self, node: ast.AST) -> bool:
        """
        Determine whether an expression can be treated as statically known.

        Static expressions are expressions whose structure can be preserved
        without requiring runtime indexing or function evaluation.

        The following node types are considered static:

        * :class:`ast.Constant`
        * :class:`ast.Name`
        * :class:`ast.Attribute`
        * Simple arithmetic expressions composed entirely of static
        subexpressions

        Subscript operations and function calls are treated as dynamic and
        therefore are not considered static.

        Parameters
        ----------
        node : ast.AST
            Expression node to analyse.

        Returns
        -------
        bool
            ``True`` if the expression is considered static, otherwise
            ``False``.

        Notes
        -----
        This method is used by :meth:`simplify_range_args` when validating
        range bounds prior to simplification.

        Unary negation is supported when its operand is itself a static
        expression.

        See Also
        --------
        :meth:`simplify_range_args`
            Simplifies range arguments when all bounds are statically
            representable.

        :meth:`extract_constant_value`
            Extracts literal values from static expressions.
        """
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Attribute):
            return True
        if isinstance(node, ast.BinOp):
            return self.is_static_expr(node.left) and self.is_static_expr(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self.is_static_expr(node.operand)

        return False  # Reject Subscript, Call

    def extract_constant_value(self, node: ast.AST):
        """
        Extract a literal numeric value from an AST expression.

        Constant expressions are converted to their corresponding Python
        values. Unary negation applied to numeric constants is evaluated and
        returned as a signed value.

        Parameters
        ----------
        node : ast.AST
            Expression node from which to extract a constant value.

        Returns
        -------
        object or None
            The extracted constant value if the expression represents a
            supported literal, otherwise ``None``.

        Notes
        -----
        Negative numeric constants represented as
        :class:`ast.UnaryOp` nodes are normalized into a single Python
        value.

        This method is used by :meth:`simplify_range_args` to identify
        special cases such as ``range(stop)`` and
        ``range(start, stop)``.

        See Also
        --------
        :meth:`simplify_range_args`
            Simplifies range arguments using extracted constant values.
        """
        if isinstance(node, ast.Constant):
            return node.value

        # Handle negative constants
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self.extract_constant_value(node.operand)
            if isinstance(inner, int | float):
                return -inner

        return None

    def simplify_range_args(
        self, start: ast.AST, stop: ast.AST, step: ast.AST
    ) -> list[ast.AST]:
        """
        Simplify range bounds into a canonical argument list.

        Range expressions are normalized by removing redundant default
        values when possible. Static range bounds are analysed to determine
        whether the explicit ``start`` or ``step`` arguments can be omitted.

        The following simplifications are applied:

        * ``range(0, stop, 1)`` → ``[stop]``
        * ``range(start, stop, 1)`` → ``[start, stop]``
        * Any other step value is preserved.

        Parameters
        ----------
        start : ast.AST
            Range start expression.
        stop : ast.AST
            Range stop expression.
        step : ast.AST
            Range step expression.

        Returns
        -------
        list[ast.AST]
            Simplified range argument list.

        Notes
        -----
        All arguments must satisfy :meth:`is_static_expr`.

        Constant values are extracted using
        :meth:`extract_constant_value` to identify default range
        parameters.

        This method is used when generating canonical loop and iteration
        constructs during AST lowering.

        See Also
        --------
        :meth:`is_static_expr`
            Determines whether a range bound is statically representable.

        :meth:`extract_constant_value`
            Extracts literal values used during simplification.

        Raises
        ------
        NotImplementedError
            If one or more range arguments contain unsupported dynamic
            expressions.
        """
        if not (
            self.is_static_expr(start)
            and self.is_static_expr(stop)
            and self.is_static_expr(step)
        ):
            raise NotImplementedError("Dynamic or unsupported range expression")

        s0 = self.extract_constant_value(start)
        s2 = self.extract_constant_value(step)

        # Case A — start=0, step=1
        if s0 == 0 and s2 == 1:
            return [stop]

        # Case B — step=1
        if s2 == 1:
            return [start, stop]

        # Case C — any other step (including negative)
        return [start, stop, step]
