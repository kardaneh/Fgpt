import ast
import copy
from typing import Dict, List, Optional, Set, Tuple
from jax_utils import get_name, contains_name

class _DynamicLoop:
    """
    Lowers ``for`` loops whose bounds are not statically known
    (dynamic-depth / dynamic-start loops) into vectorised mask-based
    operations.

    Composes onto ``JaxConverter`` to handle the case where a Fortran
    ``DO`` loop's trip count depends on a runtime value rather than a
    fixed dimension (e.g. ``DO jj = 1, snow_layers(ji)``). Since
    ``lax.scan`` requires a static trip count, this mixin instead
    expands the loop body over the full statically-known maximum range
    (``layers = jnp.arange(0, max_depth)``) and masks out elements past
    each row's actual depth (``mask = layers < depth_idx``).

    The pipeline, orchestrated by :meth:`handle_dynamic_loop`:

    1. Build a ``depth_idx`` (and, if the start bound is also dynamic,
       a ``depth_idx_start``) array via :meth:`make_depth_index`.
    2. Classify every assignment/condition in the loop body in one pass
       to resolve a static upper bound
       (:meth:`_resolve_static_upper_bound`) and detect whether any
       target array carries an extra batch axis
       (:meth:`_detect_batch_dimension`).
    3. Build the ``layers``/``mask`` pair, broadcast to the correct
       rank (:meth:`_build_layers_mask`).
    4. Rewrite each body statement — elementwise updates
       (:meth:`_transform_elementwise`) for assignments whose LHS
       indexes with the loop variable, accumulations
       (:meth:`_transform_accumulation`, plus its helpers
       :meth:`_extract_set_payload`, :meth:`_extract_accum_base_from_payload`,
       :meth:`_extract_reduction_term`, :meth:`_rhs_is_accumulation`,
       :meth:`_substitute_loop_index_with_offset`) for reductions that
       don't index by the loop variable on the LHS, and conditional
       searches (:meth:`_transform_conditional_search`,
       :meth:`_vectorize_condition`, :meth:`_vectorize_compare`,
       :meth:`_emit_masked_search_assign`, :meth:`_substitute_name`)
       for ``if cond(loop_index): scalar = expr`` patterns lowered via
       ``jnp.argmax``/``jnp.where``.

    :meth:`_is_dynamic_bound` and :meth:`_lhs_contains_loop_index` are
    the classification predicates that route control into this mixin
    from ``visit_For`` and decide which transform applies per
    statement.
    """

    def make_depth_index(self, node: ast.AST) -> Tuple[Optional[ast.Assign], bool]:
        """
        Build the ``depth_idx`` assignment for a dynamic loop bound.

        Resolves *node* to a target expression depending on its shape:
        a direct ``ast.Subscript``, the first argument of an
        ``ast.Call`` (e.g. ``int(arr[i])``), or an ``ast.Name`` already
        known to be stateful (:attr:`_stateful_vars`). Any other shape
        yields ``(None, False)``.

        When the target is itself a vectorised subscript or array
        reference, *uses_vectorized_axis* is set by checking both the
        subscripted array's own lift metadata
        (:attr:`dynamic_variable_lift`) and its index names against the
        active ``vectorization_axis``.

        Registers the resolved source name into
        :attr:`dynamic_created_variables['depth_idx']` so
        :meth:`_resolve_dynamic_origin` can later trace it back.

        Parameters
        ----------
        node : ast.AST
            The loop bound expression (``start`` or ``stop``) to
            convert into a depth index.

        Returns
        -------
        Tuple[Optional[ast.Assign], bool]
            ``(depth_assign_node, uses_vectorized_axis)``.
            *depth_assign_node* is ``depth_idx = <target>.astype(jnp.int32)``
            (or a direct passthrough assignment for already-integer
            ``Name`` targets), or ``None`` if *node* did not resolve to
            a usable target.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Subscript):
                target_node = node
            elif isinstance(node, ast.Call):
                if node.args:
                    target_node = node.args[0]
                else:
                    return None, False
            else:
                if isinstance(node, ast.Name) and (node.id in self._stateful_vars):
                    target_node = node
                else:
                    return None, False

            uses_vectorized_axis = False
            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            if vectorization_context and isinstance(target_node, ast.Subscript):
                vectorization_axis = vectorization_context.get('vectorization_axis', {})
                loop_info = vectorization_context.get('loop_info', {})

                arr_name = get_name(target_node.value)
                if arr_name and arr_name in self.dynamic_variable_lift:
                    data = self.dynamic_variable_lift[arr_name]
                    vectorized_loop = data.get('vectorized_loop', [])
                    for loop_var in vectorized_loop:
                        loop_name = get_name(loop_var)
                        if loop_name and loop_name in loop_info:
                            dim_axis = loop_info.get(loop_name)
                            if dim_axis and dim_axis in vectorization_axis:
                                uses_vectorized_axis = True

                if not uses_vectorized_axis:
                    indices = (
                        target_node.slice.elts
                        if isinstance(target_node.slice, ast.Tuple)
                        else [target_node.slice]
                    )
                    for idx_node in indices:
                        idx_name = get_name(idx_node)
                        if idx_name and idx_name in vectorization_axis:
                            if vectorization_axis.get(idx_name):
                                uses_vectorized_axis = True
                                break

            elif vectorization_context and isinstance(target_node, ast.Name):
                self.dynamic_created_variables['depth_idx'] = get_name(target_node)
                return ast.Assign(
                    targets=[ast.Name(id='depth_idx', ctx=ast.Store())],
                    value=target_node,
                ), True

            self.dynamic_created_variables['depth_idx'] = get_name(target_node)
            depth_assign = ast.Assign(
                targets=[ast.Name(id='depth_idx', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=target_node, attr='astype', ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='int32', ctx=ast.Load())],
                    keywords=[],
                ),
            )
            return depth_assign, uses_vectorized_axis

        except Exception as e:
            self.logger.exception('Exception in make_depth_index:', e)
            raise

    def handle_dynamic_loop(
        self,
        node: ast.For,
        start: ast.AST,
        stop: ast.AST,
        step: ast.AST,
        loop_index: str,
    ) -> List[ast.AST]:
        """
        Lower a ``for`` loop with a dynamic (non-statically-known)
        bound into a masked, fully-unrolled-over-``layers`` statement
        sequence.

        Orchestrates the full dynamic-loop pipeline: resolves
        ``depth_idx`` (and ``depth_idx_start`` if the start bound is
        also dynamic per :meth:`_is_dynamic_bound`) via
        :meth:`make_depth_index`, classifies all assignments and
        conditions in a single body pass to find a static upper bound
        (:meth:`_resolve_static_upper_bound`), builds the
        ``layers``/``mask`` pair (:meth:`_build_layers_mask`), then
        rewrites each statement: assignments via
        :meth:`_transform_dynamic_assign`, ``if`` statements that
        reference *loop_index* via
        :meth:`_transform_conditional_search` (others get the active
        mask attached to context and are visited normally), and nested
        ``for`` loops visited normally with the mask attached to
        context.

        Parameters
        ----------
        node : ast.For
            The dynamic-bound loop to lower.
        start : ast.AST
            The loop's start-bound expression (``ast.Constant(0)`` if
            originally omitted).
        stop : ast.AST
            The loop's stop-bound expression.
        step : ast.AST
            The loop's step expression.
        loop_index : str
            The loop's target variable name.

        Returns
        -------
        List[ast.AST]
            The fully lowered statement sequence replacing the loop.

        Raises
        ------
        NotImplementedError
            If the loop body contains a statement type other than
            ``Assign``, ``If``, or ``For``.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if stop is None:
                stop = start
                start = ast.Constant(0)

            start_is_dynamic = self._is_dynamic_bound(start)
            depth_index, vect_present = self.make_depth_index(stop)
            start_depth_index, start_vect_present = (
                self.make_depth_index(start) if start_is_dynamic else (None, False)
            )

            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            all_assign_nodes = []
            all_condition_nodes = []

            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    all_assign_nodes.append(stmt)
                elif isinstance(stmt, ast.If):
                    all_condition_nodes.append(stmt.test)
                    all_assign_nodes.extend(s for s in stmt.body if isinstance(s, ast.Assign))
                elif isinstance(stmt, ast.For):
                    pass
                else:
                    raise NotImplementedError(
                        f'Dynamic loop body: unsupported statement type {type(stmt)}'
                    )

            upper_bound_static = self._resolve_static_upper_bound(
                all_assign_nodes, loop_index, vectorization_context,
                condition_nodes=all_condition_nodes,
            )

            layers_assign, mask_assign, loop_axis, vect_axes = self._build_layers_mask(
                start, stop, step,
                depth_index, vect_present,
                upper_bound_static,
                all_assign_nodes, loop_index, vectorization_context,
                start_depth_index=start_depth_index,
                start_vect_present=start_vect_present,
                condition_nodes=all_condition_nodes,
            )

            result_stmts = []

            if start_depth_index is not None:
                start_depth_index.targets[0] = ast.Name(id='depth_idx_start', ctx=ast.Store())
                self.dynamic_created_variables['depth_idx_start'] = get_name(start)
                ranks = self.subscript_ranks(start)
                if isinstance(start, ast.Call):
                    start = start.args[0]
                self._inferred_ranks['depth_idx_start'] = ranks.get(start)
                result_stmts.append(self.visit(start_depth_index))

            visited_depth_idx = self.visit(depth_index)
            ranks = self.subscript_ranks(stop)
            if isinstance(stop, ast.Call):
                stop = stop.args[0]
            self._inferred_ranks['depth_idx'] = ranks.get(stop)
            self._inferred_ranks['layers'] = 1

            result_stmts.append(visited_depth_idx)
            if layers_assign is not None:
                result_stmts.append(layers_assign)
            result_stmts.append(mask_assign)

            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    lhs_node = stmt.targets[0]
                    rhs_node = stmt.value
                    transformed = self._transform_dynamic_assign(
                        stmt, lhs_node, rhs_node,
                        loop_index, loop_axis, vect_axes,
                        start, upper_bound_static,
                        vectorization_context,
                        start_is_dynamic=start_is_dynamic,
                    )
                    result_stmts.extend(transformed)

                elif isinstance(stmt, ast.If):
                    if contains_name(stmt.test, loop_index):
                        transformed = self._transform_conditional_search(
                            stmt, loop_index, loop_axis, vect_axes,
                            start, upper_bound_static,
                            vectorization_context,
                            layer_value=layers_assign.value if layers_assign else None,
                            start_is_dynamic=start_is_dynamic,
                        )
                        result_stmts.extend(transformed)
                    else:
                        ranks = self.subscript_ranks(mask_assign.value)
                        mask_rank = ranks.get(mask_assign.value, 0)
                        if vectorization_context is not None:
                            vectorization_context['metadata'].update({
                                'current_mask_assign': mask_assign,
                                'current_mask_rank': mask_rank,
                            })
                        visited = self.visit(stmt)
                        if isinstance(visited, list):
                            result_stmts.extend(visited)
                        elif visited is not None:
                            result_stmts.append(visited)

                elif isinstance(stmt, ast.For):
                    ranks = self.subscript_ranks(mask_assign.value)
                    mask_rank = ranks.get(mask_assign.value, 0)
                    if vectorization_context is not None:
                        vectorization_context['metadata'].update({
                            'current_mask_assign': mask_assign,
                            'current_mask_rank': mask_rank,
                        })
                    visited = self.visit(stmt)
                    if isinstance(visited, list):
                        result_stmts.extend(visited)
                    elif visited is not None:
                        result_stmts.append(visited)

            return result_stmts

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception('Exception in handle_dynamic_loop:', e)
            raise

    def _transform_conditional_search(
        self,
        if_node: ast.If,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        start: ast.AST,
        upper_bound_static: Optional[ast.AST],
        vectorization_context: Optional[Dict],
        layer_value: Optional[ast.AST],
        start_is_dynamic: bool = False,
    ) -> List[ast.AST]:
        """
        Lower ``if <cond(loop_index)>: scalar_var = <expr>`` into a
        vectorised argmax-based search.

        Handles the common Fortran idiom of searching for the first
        loop index satisfying a condition (e.g. finding a snow layer
        boundary): vectorises the condition over ``layers``
        (:meth:`_vectorize_condition`), combines it with the existing
        loop mask into ``_loop_active``, then emits a masked search
        assignment (:meth:`_emit_masked_search_assign`) per statement
        in the ``if`` body.

        Parameters
        ----------
        if_node : ast.If
            The conditional whose test references *loop_index*.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis ``layers`` occupies in broadcast expressions.
        vect_axes : Set[int]
            Active vectorisation axes from :meth:`_build_layers_mask`.
        start : ast.AST
            The loop's start-bound expression.
        upper_bound_static : Optional[ast.AST]
            The resolved static upper bound, if any.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict.
        layer_value : Optional[ast.AST]
            The ``layers = jnp.arange(...)`` RHS expression, threaded
            through metadata so nested ``visit_Subscript`` calls can
            emit ``layers[+offset]`` indexing.
        start_is_dynamic : bool, optional
            Whether the loop's start bound is itself dynamic.

        Returns
        -------
        List[ast.AST]
            ``_loop_cond``, ``_loop_active`` assignments, followed by
            one masked-search assignment per statement in
            ``if_node.body``.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            result_stmts = []
            old_metadata = (vectorization_context or {}).get('metadata', {}).copy()

            if vectorization_context is not None:
                vectorization_context['metadata'].update({
                    'loop_index': loop_index,
                    'use_layers_index': layer_value,
                })

            cond_node = self._vectorize_condition(
                if_node.test, loop_index, loop_axis, vect_axes, vectorization_context
            )
            if vectorization_context is not None:
                vectorization_context['metadata'] = old_metadata

            result_stmts.append(ast.Assign(
                targets=[ast.Name(id='_loop_cond', ctx=ast.Store())],
                value=cond_node,
            ))

            result_stmts.append(ast.Assign(
                targets=[ast.Name(id='_loop_active', ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id='mask', ctx=ast.Load()),
                    op=ast.BitAnd(),
                    right=ast.Name(id='_loop_cond', ctx=ast.Load()),
                ),
            ))

            for assign in if_node.body:
                lhs = assign.targets[0]
                rhs = assign.value
                stmts = self._emit_masked_search_assign(
                    lhs, rhs, loop_index, loop_axis, vect_axes, vectorization_context
                )
                result_stmts.extend(stmts)

            return result_stmts

        except Exception as e:
            self.logger.exception('Exception in _transform_conditional_search:', e)
            raise

    def _vectorize_condition(
        self,
        test_node: ast.AST,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        vectorization_context: Optional[Dict],
    ) -> ast.AST:
        """
        Recursively rewrite a boolean condition into its ``jnp``
        vectorised equivalent over ``layers``.

        Handles ``ast.BoolOp`` (mapped to ``jnp.logical_and``/
        ``jnp.logical_or``), ``ast.Compare`` (delegated to
        :meth:`_vectorize_compare`), and ``ast.UnaryOp(Not)`` (mapped to
        ``jnp.logical_not``). Any other node is visited via the normal
        ``self.visit`` pipeline on a deep copy, relying on
        ``visit_Subscript``'s ``use_layers_index`` metadata (set by the
        caller) to substitute *loop_index* references with ``layers``.

        Parameters
        ----------
        test_node : ast.AST
            The condition expression to vectorise.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis ``layers`` occupies in broadcast expressions.
        vect_axes : Set[int]
            Active vectorisation axes.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict.

        Returns
        -------
        ast.AST
            The vectorised condition expression.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(test_node, ast.BoolOp):
                jnp_func = 'logical_and' if isinstance(test_node.op, ast.And) else 'logical_or'
                vectorized_values = [
                    self._vectorize_condition(v, loop_index, loop_axis, vect_axes, vectorization_context)
                    for v in test_node.values
                ]
                result = vectorized_values[0]
                for v in vectorized_values[1:]:
                    result = ast.Call(
                        func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr=jnp_func, ctx=ast.Load()),
                        args=[result, v],
                        keywords=[],
                    )
                return result

            elif isinstance(test_node, ast.Compare):
                return self._vectorize_compare(
                    test_node, loop_index, loop_axis, vect_axes, vectorization_context
                )

            elif isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
                inner = self._vectorize_condition(
                    test_node.operand, loop_index, loop_axis, vect_axes, vectorization_context
                )
                return ast.Call(
                    func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='logical_not', ctx=ast.Load()),
                    args=[inner],
                    keywords=[],
                )

            else:
                return self.visit(copy.deepcopy(test_node))

        except Exception as e:
            self.logger.exception('Exception in _vectorize_condition:', e)
            raise

    def _vectorize_compare(
        self,
        compare_node: ast.Compare,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        vectorization_context: Optional[Dict],
    ) -> ast.AST:
        """
        Vectorise a single (possibly chained) ``Compare`` node into
        ``jnp`` comparison calls, with operands rank-aligned.

        Visits each operand first (so ``loop_index`` references become
        ``layers`` via the ``use_layers_index`` metadata path), then
        applies :meth:`maybe_add_index` to every operand with
        ``target_rank = loop_axis + 1`` so shapes broadcast correctly
        before comparison. Chained comparisons (``a < b < c``) are
        expanded into a ``jnp.logical_and`` conjunction.

        Parameters
        ----------
        compare_node : ast.Compare
            The comparison to vectorise.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis ``layers`` occupies; determines the target rank
            operands are aligned to.
        vect_axes : Set[int]
            Active vectorisation axes (unused directly here but kept
            for call-site symmetry).
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict.

        Returns
        -------
        ast.AST
            A single ``jnp.<op>`` call, or a ``jnp.logical_and`` chain
            for multi-operator comparisons.

        Raises
        ------
        NotImplementedError
            If a comparison operator has no ``jnp`` equivalent.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            _CMP_OPS = {
                ast.Lt: 'less', ast.LtE: 'less_equal',
                ast.Gt: 'greater', ast.GtE: 'greater_equal',
                ast.Eq: 'equal', ast.NotEq: 'not_equal',
            }

            visited_left = self.visit(copy.deepcopy(compare_node.left))
            visited_comparators = [self.visit(copy.deepcopy(c)) for c in compare_node.comparators]

            target_rank = loop_axis + 1

            aligned_left, _ = self.maybe_add_index(visited_left, target_rank, vectorization_context)
            aligned_comparators = []
            for comp in visited_comparators:
                aligned_comp, _ = self.maybe_add_index(comp, target_rank, vectorization_context)
                aligned_comparators.append(aligned_comp)

            result = None
            left = aligned_left
            for op, comparator in zip(compare_node.ops, aligned_comparators):
                jnp_func = _CMP_OPS.get(type(op))
                if jnp_func is None:
                    raise NotImplementedError(f'Unsupported comparison op: {type(op)}')

                cmp_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr=jnp_func, ctx=ast.Load()),
                    args=[left, comparator],
                    keywords=[],
                )
                result = cmp_call if result is None else ast.Call(
                    func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='logical_and', ctx=ast.Load()),
                    args=[result, cmp_call],
                    keywords=[],
                )
                left = comparator

            return result

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception('Exception in _vectorize_compare:', e)
            raise

    def _emit_masked_search_assign(
        self,
        lhs_node: ast.AST,
        rhs_node: ast.AST,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        vectorization_context: Optional[Dict],
    ) -> List[ast.Assign]:
        """
        Emit the argmax-based masked search assignment for a single
        statement inside a conditional search.

        For an assignment of the form ``scalar_var = expr(loop_index)``,
        finds the first active layer via
        ``jnp.argmax(_loop_active, axis=loop_axis)``, substitutes that
        found index for *loop_index* in the RHS
        (:meth:`_substitute_name`), and guards the update with
        ``jnp.where(jnp.any(_loop_active, axis=loop_axis), new_val,
        old_val)`` so rows with no active layer keep their previous
        value rather than incorrectly picking index 0.

        Parameters
        ----------
        lhs_node : ast.AST
            The assignment target.
        rhs_node : ast.AST
            The assignment value, expected to reference *loop_index*.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis to reduce over when finding the active layer.
        vect_axes : Set[int]
            Active vectorisation axes (unused directly here but kept
            for call-site symmetry).
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict (unused
            directly here but kept for call-site symmetry).

        Returns
        -------
        List[ast.Assign]
            ``[_found_layer assignment, guarded target assignment]``.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            result_stmts = []

            found_idx = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='argmax', ctx=ast.Load()),
                args=[ast.Name(id='_loop_active', ctx=ast.Load())],
                keywords=[ast.keyword(arg='axis', value=ast.Constant(value=loop_axis))],
            )
            result_stmts.append(ast.Assign(
                targets=[ast.Name(id='_found_layer', ctx=ast.Store())],
                value=found_idx,
            ))

            rhs_with_found = self._substitute_name(copy.deepcopy(rhs_node), loop_index, '_found_layer')
            visited_rhs = self.visit(rhs_with_found)
            visited_lhs = self.visit(copy.deepcopy(lhs_node))

            any_active = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='any', ctx=ast.Load()),
                args=[ast.Name(id='_loop_active', ctx=ast.Load())],
                keywords=[ast.keyword(arg='axis', value=ast.Constant(value=loop_axis))],
            )

            guarded_rhs = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='where', ctx=ast.Load()),
                args=[any_active, visited_rhs, visited_lhs],
                keywords=[],
            )

            result_stmts.append(ast.Assign(targets=[visited_lhs], value=guarded_rhs))
            return result_stmts

        except Exception as e:
            self.logger.exception('Exception in _emit_masked_search_assign:', e)
            raise

    def _substitute_name(self, node: ast.AST, old_name: str, new_name: str) -> ast.AST:
        """
        Replace every ``ast.Name(id=old_name)`` in *node* with
        ``ast.Name(id=new_name)``.

        Parameters
        ----------
        node : ast.AST
            Subtree to rewrite.
        old_name : str
            Name to find.
        new_name : str
            Name to substitute.

        Returns
        -------
        ast.AST
            The rewritten subtree.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            class Substitutor(ast.NodeTransformer):
                def visit_Name(self, n):
                    if n.id == old_name:
                        return ast.Name(id=new_name, ctx=n.ctx)
                    return n

            return Substitutor().visit(node)
        except Exception as e:
            self.logger.exception('Exception in _substitute_name:', e)
            raise

    def _is_dynamic_bound(self, node: ast.AST) -> bool:
        """
        Return ``True`` if a loop bound expression references a lifted
        (per-element) variable rather than a fixed scalar.

        A dynamic bound varies per vectorised slice and therefore
        cannot be treated as a static ``arange`` start/stop; it instead
        requires the full ``depth_idx``-style masking treatment.

        Parameters
        ----------
        node : ast.AST
            The loop bound expression to classify.

        Returns
        -------
        bool
            ``True`` if *node* is a ``Name`` or ``Subscript`` resolving
            to an entry in :attr:`dynamic_variable_lift`; ``False`` for
            constants and unrecognised shapes.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Constant):
                return False
            if isinstance(node, ast.Name):
                return node.id in self.dynamic_variable_lift
            if isinstance(node, ast.Subscript):
                arr_name = get_name(node.value)
                if arr_name is not None and arr_name in self.dynamic_variable_lift:
                    return True
            return False
        except Exception as e:
            self.logger.exception('Exception in _is_dynamic_bound:', e)
            raise

    def _resolve_static_upper_bound(
        self,
        assign_nodes: List[ast.Assign],
        loop_index: str,
        vectorization_context: Optional[Dict],
        condition_nodes: Optional[List[ast.AST]] = None,
    ) -> Optional[ast.AST]:
        """
        Find a statically-known array dimension corresponding to
        *loop_index*, to use as the ``layers`` upper bound.

        Searches, in order, the LHS subscripts of *assign_nodes*, then
        their RHS subscripts, then any subscript inside
        *condition_nodes* — returning the declared dimension at the
        position where *loop_index* appears, as ``self.<dim>`` if it is
        a class attribute or a bare ``ast.Name`` otherwise.

        Parameters
        ----------
        assign_nodes : List[ast.Assign]
            All assignment statements collected from the loop body
            (including those nested inside ``if`` blocks).
        loop_index : str
            The dynamic loop's index variable name.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict (unused
            directly here but kept for call-site symmetry).
        condition_nodes : Optional[List[ast.AST]]
            Test expressions from ``if`` statements in the loop body,
            searched as a fallback after assignments.

        Returns
        -------
        Optional[ast.AST]
            The resolved dimension expression, or ``None`` if
            *loop_index* could not be matched to any declared
            dimension.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            for assign_node in assign_nodes:
                lhs = assign_node.targets[0]
                if isinstance(lhs, ast.Subscript):
                    node_dims = self.get_declared_dims(lhs)
                    subscript_indices = lhs.slice.elts if isinstance(lhs.slice, ast.Tuple) else [lhs.slice]
                    for pos, idx_node in enumerate(subscript_indices):
                        if contains_name(idx_node, loop_index) and pos < len(node_dims):
                            dim = node_dims[pos]
                            attributes = self.cls_info[self.cls_name].get('attributes', {})
                            if dim in attributes:
                                return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=dim, ctx=ast.Load())
                            return ast.Name(id=dim, ctx=ast.Load())

                for sub in ast.walk(assign_node.value):
                    if isinstance(sub, ast.Subscript):
                        node_dims = self.get_declared_dims(sub)
                        sub_indices = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
                        for pos, idx_node in enumerate(sub_indices):
                            if contains_name(idx_node, loop_index) and pos < len(node_dims):
                                dim = node_dims[pos]
                                attributes = self.cls_info[self.cls_name].get('attributes', {})
                                if dim in attributes:
                                    return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=dim, ctx=ast.Load())
                                return ast.Name(id=dim, ctx=ast.Load())

                for test_node in (condition_nodes or []):
                    for sub in ast.walk(test_node):
                        if isinstance(sub, ast.Subscript):
                            node_dims = self.get_declared_dims(sub)
                            sub_indices = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
                            for pos, idx_node in enumerate(sub_indices):
                                if contains_name(idx_node, loop_index) and pos < len(node_dims):
                                    dim = node_dims[pos]
                                    attributes = self.cls_info[self.cls_name].get('attributes', {})
                                    if dim in attributes:
                                        return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=dim, ctx=ast.Load())
                                    return ast.Name(id=dim, ctx=ast.Load())
            return None
        except Exception as e:
            self.logger.exception('Exception in _resolve_static_upper_bound:', e)
            raise

    def _detect_batch_dimension(
        self,
        assign_nodes: List[ast.Assign],
        loop_index: str,
        vectorization_context: Optional[Dict],
        condition_nodes: Optional[List[ast.AST]] = None,
    ) -> bool:
        """
        Detect whether any array touched by the dynamic loop carries an
        extra vectorised batch axis beyond the loop-index dimension
        itself.

        For each subscripted LHS in *assign_nodes* (and, as a
        fallback, each subscript inside *condition_nodes*), determines
        which declared dimensions correspond to *loop_index* and checks
        whether any *other* declared dimension maps to an active
        vectorisation axis — if so, the array has a batch dimension
        the lowering must account for (affecting ``loop_axis``
        placement in :meth:`_build_layers_mask`).

        Parameters
        ----------
        assign_nodes : List[ast.Assign]
            Assignment statements collected from the loop body.
        loop_index : str
            The dynamic loop's index variable name.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict.
        condition_nodes : Optional[List[ast.AST]]
            Test expressions from ``if`` statements in the loop body.

        Returns
        -------
        bool
            ``True`` if a batch dimension is detected.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            loop_info = (vectorization_context or {}).get('loop_info', {})
            vectorization_axis = (vectorization_context or {}).get('vectorization_axis', {})

            for assign_node in assign_nodes:
                lhs = assign_node.targets[0]
                if not isinstance(lhs, ast.Subscript):
                    continue

                node_dims = self.get_declared_dims(lhs)
                subscript_indices = lhs.slice.elts if isinstance(lhs.slice, ast.Tuple) else [lhs.slice]

                loop_positions = [
                    pos for pos, idx_node in enumerate(subscript_indices)
                    if contains_name(idx_node, loop_index)
                ]
                loop_dims = {node_dims[pos] for pos in loop_positions if pos < len(node_dims)}

                for dim in node_dims:
                    if dim in loop_dims:
                        continue
                    if dim in loop_info:
                        dim_axis = loop_info[dim]
                        vect_axes = vectorization_axis.get(dim_axis, set())
                        if vect_axes:
                            return True

            for test_node in (condition_nodes or []):
                for sub in ast.walk(test_node):
                    if not isinstance(sub, ast.Subscript):
                        continue

                    node_dims = self.get_declared_dims(sub)
                    if not node_dims:
                        continue

                    sub_indices = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
                    loop_positions = [
                        pos for pos, idx_node in enumerate(sub_indices)
                        if contains_name(idx_node, loop_index)
                    ]
                    if not loop_positions:
                        continue

                    loop_dims = {node_dims[pos] for pos in loop_positions if pos < len(node_dims)}
                    for dim in node_dims:
                        if dim in loop_dims:
                            continue
                        if dim in loop_info:
                            dim_axis = loop_info[dim]
                            vect_axes = vectorization_axis.get(dim_axis, set())
                            if vect_axes:
                                return True
                return False

        except Exception as e:
            self.logger.exception('Exception in _detect_batch_dimension:', e)
            raise

    def _build_layers_mask(
        self,
        start: ast.AST,
        stop: ast.AST,
        step: ast.AST,
        depth_index: Optional[ast.Assign],
        vect_present: bool,
        upper_bound_static: Optional[ast.AST],
        assign_nodes: List[ast.Assign],
        loop_index: str,
        vectorization_context: Optional[Dict],
        start_depth_index: Optional[ast.Assign] = None,
        start_vect_present: bool = False,
        condition_nodes: Optional[List[ast.AST]] = None,
    ) -> Tuple[Optional[ast.Assign], ast.Assign, int, Set[int]]:
        """
        Build the ``layers = jnp.arange(...)`` and ``mask = ...``
        assignments for a dynamic loop, with broadcast shapes resolved.

        Three cases are handled:

        1. **No vectorised axis on the bound** — ``layers`` is 1-D and
           the mask is a plain ``layers < depth_idx`` (optionally
           ``and layers >= depth_idx_start`` / ``>= start``). The
           direction of ``arange`` (ascending vs descending) follows
           the sign of *step* (an ``ast.UnaryOp`` signals a descending
           range).
        2. **Vectorised axis detected on the depth bound** — both
           ``layers`` and ``depth_idx``/``depth_idx_start`` are
           reshaped via ``None``/``Slice`` broadcasting indices so the
           comparison produces a correctly-ranked mask.
        3. **No usable loop index** (``loop_index in (None, '_')``) —
           the mask degenerates to the raw ``depth_idx`` itself and
           ``layers_assign`` is ``None``, signalling the caller to skip
           layer expansion entirely.

        Parameters
        ----------
        start, stop, step : ast.AST
            The loop's bound and step expressions.
        depth_index : Optional[ast.Assign]
            The ``depth_idx`` assignment from :meth:`make_depth_index`.
        vect_present : bool
            Whether the stop bound itself is vectorised.
        upper_bound_static : Optional[ast.AST]
            The resolved static upper bound from
            :meth:`_resolve_static_upper_bound`.
        assign_nodes : List[ast.Assign]
            Assignment statements from the loop body, used to detect a
            batch dimension via :meth:`_detect_batch_dimension`.
        loop_index : str
            The dynamic loop's index variable name.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict.
        start_depth_index : Optional[ast.Assign], optional
            The ``depth_idx_start`` assignment, if the start bound is
            dynamic.
        start_vect_present : bool, optional
            Whether the start bound itself is vectorised.
        condition_nodes : Optional[List[ast.AST]], optional
            Test expressions from ``if`` statements in the loop body.

        Returns
        -------
        Tuple[Optional[ast.Assign], ast.Assign, int, Set[int]]
            ``(layers_assign, mask_assign, loop_axis, vect_axes)``.
            *layers_assign* is ``None`` only in the degenerate
            no-loop-index case.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            loop_info = (vectorization_context or {}).get('loop_info', {})
            vectorization_axis = (vectorization_context or {}).get('vectorization_axis', {})

            has_batch_dimension = self._detect_batch_dimension(
                assign_nodes, loop_index=loop_index,
                vectorization_context=vectorization_context,
                condition_nodes=condition_nodes,
            )

            if isinstance(step, ast.UnaryOp):
                arange_start = ast.BinOp(left=upper_bound_static, op=ast.Sub(), right=step.operand)
                arange_stop = ast.Constant(value=-1)
            else:
                arange_start = start if has_batch_dimension and not start_vect_present else ast.Constant(0)
                arange_stop = upper_bound_static

            layers_assign = ast.Assign(
                targets=[ast.Name(id='layers', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='arange', ctx=ast.Load()),
                    args=[arange_start, arange_stop, step],
                    keywords=[],
                ),
            )

            vect_axes = set()
            target_node = None
            if vect_present and depth_index:
                if isinstance(stop, ast.Subscript):
                    target_node = stop
                elif isinstance(stop, ast.Call) and stop.args:
                    target_node = stop.args[0]
                elif isinstance(stop, ast.Name):
                    target_node = stop

            if target_node is not None:
                if isinstance(target_node, ast.Subscript):
                    arr_name = get_name(target_node.value)
                    if arr_name and arr_name in self.dynamic_variable_lift:
                        data = self.dynamic_variable_lift[arr_name]
                        vectorized_loop = data.get('vectorized_loop', [])
                        for loop_var in vectorized_loop:
                            loop_name = get_name(loop_var)
                            if loop_name and loop_name in loop_info:
                                dim_axis = loop_info.get(loop_name)
                                if dim_axis and dim_axis in vectorization_axis:
                                    vect_axes |= vectorization_axis.get(dim_axis, set())
                    if not vect_axes:
                        idxs = (
                            target_node.slice.elts
                            if isinstance(target_node.slice, ast.Tuple)
                            else [target_node.slice]
                        )
                        for idx_node in idxs:
                            idx_name = get_name(idx_node)
                            if idx_name and idx_name in vectorization_axis:
                                vect_axes |= vectorization_axis.get(idx_name, set())

                elif isinstance(target_node, ast.Name):
                    var_name = target_node.id
                    if target_node.id in self.dynamic_variable_lift:
                        data = self.dynamic_variable_lift.get(var_name, {})
                        vectorized_loop = data.get('vectorized_loop', [])
                        for loop_var in vectorized_loop:
                            loop_name = get_name(loop_var)
                            if loop_name and loop_name in loop_info:
                                dim_axis = loop_info.get(loop_name)
                                if dim_axis and dim_axis in vectorization_axis:
                                    vect_axes |= vectorization_axis.get(dim_axis, set())
                    elif target_node.id in self.var_state and self.var_state.get(var_name)[0] == 'stateful':
                        shape = self._infer_scalar_shape(vectorization_context)
                        for sh in shape:
                            if sh in loop_info:
                                dim_axis = loop_info.get(sh)
                                if dim_axis and dim_axis in vectorization_axis:
                                    vect_axes |= vectorization_axis.get(dim_axis, set())

            if not vect_axes:
                layers_node = ast.Name(id='layers', ctx=ast.Load())
                upper_cmp = ast.Compare(
                    left=layers_node, ops=[ast.Lt()],
                    comparators=[ast.Name(id='depth_idx', ctx=ast.Load())],
                )

                if start_depth_index is not None:
                    lower_cmp = ast.Compare(
                        left=layers_node, ops=[ast.GtE()],
                        comparators=[ast.Name(id='depth_idx_start', ctx=ast.Load())],
                    )
                elif isinstance(start, ast.Constant) and start.value != 0:
                    lower_cmp = ast.Compare(left=layers_node, ops=[ast.GtE()], comparators=[start])
                else:
                    lower_cmp = None

                mask_val = (
                    ast.BinOp(left=lower_cmp, op=ast.BitAnd(), right=upper_cmp)
                    if lower_cmp else upper_cmp
                )
                mask_assign = ast.Assign(targets=[ast.Name(id='mask', ctx=ast.Store())], value=mask_val)

                loop_axis = 1 if has_batch_dimension else 0
                return layers_assign, mask_assign, loop_axis, vect_axes

            max_vect_axis = max(vect_axes)
            loop_axis = max_vect_axis + 1
            ndim = loop_axis + 1

            elts = [ast.Constant(value=None)] * ndim
            for ax in vect_axes:
                elts[ax] = ast.Slice()

            layers_elts = [
                ast.Slice() if i == loop_axis else ast.Constant(value=None)
                for i in range(ndim)
            ]
            layers_subscript = ast.Subscript(
                value=ast.Name(id='layers', ctx=ast.Load()),
                slice=ast.Tuple(elts=layers_elts, ctx=ast.Load()),
                ctx=ast.Load(),
            )

            depth_index_subscript = ast.Subscript(
                value=ast.Name(id='depth_idx', ctx=ast.Load()),
                slice=ast.Tuple(elts=list(elts), ctx=ast.Load()),
                ctx=ast.Load(),
            )
            upper_cmp = ast.Compare(left=layers_subscript, ops=[ast.Lt()], comparators=[depth_index_subscript])

            if start_depth_index is not None:
                start_subscript = ast.Subscript(
                    value=ast.Name(id='depth_idx_start', ctx=ast.Load()),
                    slice=ast.Tuple(elts=list(elts), ctx=ast.Load()),
                    ctx=ast.Load(),
                )
                lower_cmp = ast.Compare(left=layers_subscript, ops=[ast.GtE()], comparators=[start_subscript])
            elif not has_batch_dimension and isinstance(start, ast.Constant):
                lower_cmp = ast.Compare(left=layers_subscript, ops=[ast.GtE()], comparators=[start])
            else:
                lower_cmp = None

            if loop_index is None or loop_index == '_':
                mask_assign = ast.Assign(
                    targets=[ast.Name(id='mask', ctx=ast.Store())],
                    value=ast.Name(id='depth_idx', ctx=ast.Load()),
                )
                return None, mask_assign, 0, vect_axes

            mask_val = (
                ast.BinOp(left=lower_cmp, op=ast.BitAnd(), right=upper_cmp)
                if lower_cmp else upper_cmp
            )
            mask_assign = ast.Assign(targets=[ast.Name(id='mask', ctx=ast.Store())], value=mask_val)

            return layers_assign, mask_assign, loop_axis, vect_axes

        except Exception as e:
            self.logger.exception('Exception in _build_layers_mask:', e)
            raise

    def _lhs_contains_loop_index(self, lhs_node: ast.AST, loop_index: str) -> bool:
        """
        Return ``True`` if *loop_index* appears anywhere in *lhs_node*'s
        subscript.

        Used by :meth:`_transform_dynamic_assign` to route between
        elementwise updates (LHS indexed by the loop variable) and
        accumulations (LHS not indexed by it).

        Parameters
        ----------
        lhs_node : ast.AST
            The assignment target to check.
        loop_index : str
            The dynamic loop's index variable name.

        Returns
        -------
        bool
            ``True`` if *lhs_node* is a subscript referencing
            *loop_index* in any index position.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(lhs_node, ast.Subscript):
                return False
            indices = lhs_node.slice.elts if isinstance(lhs_node.slice, ast.Tuple) else [lhs_node.slice]
            return any(contains_name(idx, loop_index) for idx in indices)
        except Exception as e:
            self.logger.exception('Exception in _lhs_contains_loop_index:', e)
            raise

    def _transform_dynamic_assign(
        self,
        assign_node: ast.Assign,
        lhs_node: ast.AST,
        rhs_node: ast.AST,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        start: ast.AST,
        upper_bound_static: Optional[ast.AST],
        vectorization_context: Optional[Dict],
        start_is_dynamic: bool = False,
    ) -> List[ast.AST]:
        """
        Route a single dynamic-loop-body assignment to the appropriate
        lowering.

        Parameters
        ----------
        assign_node : ast.Assign
            The full assignment statement.
        lhs_node : ast.AST
            The assignment target.
        rhs_node : ast.AST
            The assignment value.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis ``layers`` occupies in broadcast expressions.
        vect_axes : Set[int]
            Active vectorisation axes.
        start : ast.AST
            The loop's start-bound expression.
        upper_bound_static : Optional[ast.AST]
            The resolved static upper bound.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict.
        start_is_dynamic : bool, optional
            Whether the loop's start bound is itself dynamic.

        Returns
        -------
        List[ast.AST]
            The lowered statement(s): a plain visit when no usable loop
            index exists, otherwise the result of
            :meth:`_transform_elementwise` (when *lhs_node* indexes by
            *loop_index*) or :meth:`_transform_accumulation` (otherwise).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if loop_index is None or loop_index == '_':
                return [self.visit(assign_node)]

            if self._lhs_contains_loop_index(lhs_node, loop_index):
                return self._transform_elementwise(
                    assign_node, lhs_node, rhs_node,
                    loop_index, loop_axis, vect_axes,
                    start, upper_bound_static,
                    vectorization_context,
                )
            else:
                return self._transform_accumulation(
                    assign_node, lhs_node, rhs_node,
                    loop_index, loop_axis, vect_axes,
                    start, upper_bound_static,
                    vectorization_context,
                    start_is_dynamic=start_is_dynamic,
                )
        except Exception as e:
            self.logger.exception('Exception in _transform_dynamic_assign:', e)
            raise

    def _transform_elementwise(
        self,
        assign_node: ast.Assign,
        lhs_node: ast.AST,
        rhs_node: ast.AST,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        start: ast.AST,
        upper_bound_static: Optional[ast.AST],
        vectorization_context: Optional[Dict],
    ) -> List[ast.Assign]:
        """
        Lower an assignment whose LHS indexes by *loop_index* directly
        (e.g. ``arr[ji, jj] = expr(jj)``).

        Detects whether the target array carries an extra batch axis
        (:meth:`_detect_batch_dimension`-equivalent inline check) to
        decide whether ``n_layers`` bound metadata is needed downstream
        in ``visit_Subscript``. Visits the assignment normally, then
        wraps the RHS in ``jnp.where(mask, value, 0)`` so out-of-range
        layer positions are zeroed rather than left at whatever
        ``maybe_add_index`` broadcast produced.

        Parameters
        ----------
        assign_node : ast.Assign
            The assignment statement to lower.
        lhs_node : ast.AST
            The assignment target.
        rhs_node : ast.AST
            The assignment value.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis ``layers`` occupies in broadcast expressions
            (unused directly here but kept for call-site symmetry).
        vect_axes : Set[int]
            Active vectorisation axes (unused directly here but kept
            for call-site symmetry).
        start : ast.AST
            The loop's start-bound expression, threaded into
            ``n_layers`` metadata when a batch axis is present.
        upper_bound_static : Optional[ast.AST]
            The resolved static upper bound, threaded into
            ``n_layers`` metadata when a batch axis is present.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict; its
            ``metadata`` is temporarily mutated and restored.

        Returns
        -------
        List[ast.Assign]
            Single-element list containing the masked assignment.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            node_dims = self.get_declared_dims(lhs_node)
            array_name = get_name(lhs_node)
            local_arrays = (
                self.cls_info[self.cls_name].get('methods').get(self.func_name).get('local_arr', [])
            )
            is_local_array = array_name in local_arrays

            loop_info = (vectorization_context or {}).get('loop_info', {})
            vectorization_axis_map = (vectorization_context or {}).get('vectorization_axis', {})

            subscript_indices = (
                lhs_node.slice.elts if isinstance(lhs_node.slice, ast.Tuple) else [lhs_node.slice]
            )
            loop_positions = [
                pos for pos, idx_node in enumerate(subscript_indices)
                if contains_name(idx_node, loop_index)
            ]
            loop_dims_in_lhs = [node_dims[pos] for pos in loop_positions]

            has_batch_dimension = False
            for dim in node_dims:
                if dim in loop_dims_in_lhs:
                    continue
                if dim in loop_info:
                    dim_axis = loop_info[dim]
                    vect_ax = vectorization_axis_map.get(dim_axis, set())
                    if vect_ax:
                        has_batch_dimension = True
                        break

            array_has_batch_axis = has_batch_dimension and not is_local_array
            old_metadata = vectorization_context['metadata'].copy()
            if array_has_batch_axis:
                vectorization_context['metadata'].update({
                    'loop_index': loop_index,
                    'n_layers': [start, upper_bound_static],
                })
            else:
                vectorization_context['metadata'].update({'loop_index': loop_index})

            assign_node = self.visit(assign_node)
            target_rank = self._target_rank(lhs_node.slice)
            value, temp = self.maybe_add_index(rhs_node, target_rank, vectorization_context)

            mask_cond = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='where', ctx=ast.Load()),
                args=[ast.Name(id='mask', ctx=ast.Load()), value, ast.Constant(value=0)],
                keywords=[],
            )
            assign_node.value.args = [mask_cond]
            vectorization_context['metadata'] = old_metadata

            return [assign_node]

        except Exception as e:
            self.logger.exception('Exception in _transform_elementwise:', e)
            raise

    def _extract_set_payload(self, assign_node: ast.Assign, target_name: str) -> Optional[ast.AST]:
        """
        Extract the value argument from a ``.at[...].set(...)``-style
        assignment.

        Parameters
        ----------
        assign_node : ast.Assign
            The assignment to inspect.
        target_name : str
            Expected array name backing the ``.at[...]`` chain (passed
            to :meth:`is_arr_at_op_call`).

        Returns
        -------
        Optional[ast.AST]
            The payload argument, or ``None`` if *assign_node* is not a
            recognised ``.at[...]`` call.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            value = assign_node.value
            if self.is_arr_at_op_call(value, target_name) and value.args:
                return value.args[0]
            return None
        except Exception as e:
            self.logger.exception('Exception in _extract_set_payload:', e)
            raise

    def _transform_accumulation(
        self,
        assign_node: ast.Assign,
        lhs_node: ast.AST,
        rhs_node: ast.AST,
        loop_index: str,
        loop_axis: int,
        vect_axes: Set[int],
        start: ast.AST,
        upper_bound_static: Optional[ast.AST],
        vectorization_context: Optional[Dict],
        start_is_dynamic: bool = False,
    ) -> List[ast.Assign]:
        """
        Lower an assignment whose LHS does **not** index by
        *loop_index* — a reduction over the dynamic loop's range (e.g.
        ``total += arr[ji, jj]`` accumulating over ``jj``).

        Mirrors :meth:`_transform_elementwise`'s batch-axis detection
        to decide ``n_layers`` metadata, then visits the assignment,
        extracts the accumulation payload (:meth:`_extract_set_payload`
        for ``.at[].set()`` forms, else a fresh visit of *rhs_node*),
        isolates the reduction term from the accumulation base
        (:meth:`_extract_accum_base_from_payload`,
        :meth:`_extract_reduction_term`), substitutes a dynamic-start
        offset if needed (:meth:`_substitute_loop_index_with_offset`),
        and builds ``jnp.sum(jnp.where(mask, term, 0), axis=loop_axis)``
        — added to the accumulation base if one was identified
        (:meth:`_rhs_is_accumulation`), otherwise used directly.

        Parameters
        ----------
        assign_node : ast.Assign
            The assignment statement to lower.
        lhs_node : ast.AST
            The assignment target.
        rhs_node : ast.AST
            The assignment value.
        loop_index : str
            The dynamic loop's index variable name.
        loop_axis : int
            The axis to reduce over.
        vect_axes : Set[int]
            Active vectorisation axes (unused directly here but kept
            for call-site symmetry).
        start : ast.AST
            The loop's start-bound expression.
        upper_bound_static : Optional[ast.AST]
            The resolved static upper bound.
        vectorization_context : Optional[Dict]
            The active :class:`Control` context as a dict; its
            ``metadata`` is temporarily mutated and restored.
        start_is_dynamic : bool, optional
            Whether the loop's start bound is dynamic, triggering the
            offset substitution.

        Returns
        -------
        List[ast.Assign]
            Single-element list containing the reduced assignment.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            old_metadata = (vectorization_context or {}).get('metadata', {}).copy()

            loop_info = (vectorization_context or {}).get('loop_info', {})
            vectorization_axis_map = (vectorization_context or {}).get('vectorization_axis', {})

            local_arrays = (
                self.cls_info[self.cls_name].get('methods', {}).get(self.func_name, {}).get('local_arr', {})
            )

            node_dims = self.get_declared_dims(lhs_node)
            array_name = get_name(lhs_node)
            is_local_array = array_name in local_arrays

            if isinstance(lhs_node, ast.Subscript):
                subscript_indices = (
                    lhs_node.slice.elts if isinstance(lhs_node.slice, ast.Tuple) else [lhs_node.slice]
                )
            else:
                subscript_indices = []

            loop_positions = [
                pos for pos, idx_node in enumerate(subscript_indices)
                if contains_name(idx_node, loop_index)
            ]
            loop_dims_in_lhs = {node_dims[pos] for pos in loop_positions if pos < len(node_dims)}

            has_batch_dimension = False
            for dim in node_dims:
                if dim in loop_dims_in_lhs:
                    continue
                if dim in loop_info:
                    dim_axis = loop_info[dim]
                    vect_ax = vectorization_axis_map.get(dim_axis, set())
                    if vect_ax:
                        has_batch_dimension = True
                        break

            array_has_batch_axis = has_batch_dimension and not is_local_array

            if vectorization_context is not None:
                if array_has_batch_axis:
                    vectorization_context['metadata'].update({
                        'loop_index': loop_index,
                        'n_layers': [start, upper_bound_static],
                    })
                else:
                    vectorization_context['metadata'].update({'loop_index': loop_index})

            visited_assign = self.visit(copy.deepcopy(assign_node))
            lhs_visited = visited_assign.targets[0]
            target_name = get_name(lhs_visited)
            payload = self._extract_set_payload(visited_assign, target_name)

            if payload is None:
                payload = self.visit(copy.deepcopy(rhs_node))

            accum_base = self._extract_accum_base_from_payload(payload, lhs_visited)
            reduction_term = self._extract_reduction_term(payload, accum_base, loop_index)

            if start_is_dynamic:
                reduction_term = self._substitute_loop_index_with_offset(reduction_term, loop_index)

            where_call = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='where', ctx=ast.Load()),
                args=[ast.Name(id='mask', ctx=ast.Load()), reduction_term, ast.Constant(value=0)],
                keywords=[],
            )
            sum_call = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='sum', ctx=ast.Load()),
                args=[where_call],
                keywords=[ast.keyword(arg='axis', value=ast.Constant(value=loop_axis))],
            )

            new_rhs = (
                ast.BinOp(left=accum_base, op=ast.Add(), right=sum_call)
                if self._rhs_is_accumulation(payload, accum_base)
                else sum_call
            )

            if payload:
                visited_assign.value.args = [new_rhs]
                new_assign = visited_assign
            else:
                new_assign = ast.Assign(targets=[lhs_visited], value=new_rhs)

            ast.fix_missing_locations(new_assign)

            if vectorization_context is not None:
                vectorization_context['metadata'] = old_metadata

            return [new_assign]

        except Exception as e:
            self.logger.exception('Exception in _transform_accumulation:', e)
            raise

    def _substitute_loop_index_with_offset(self, node: ast.AST, loop_index: str) -> ast.AST:
        """
        Replace a bare ``loop_index - <anything>`` minuend with
        ``layers - depth_idx_start`` broadcast appropriately, to encode
        the relative offset correctly when the loop start is dynamic.

        Example::

            (jjj - locflag[jj, 0]) * snowrho[ji, jjj] * snowdz_old[ji, jjj]

        becomes, after visiting (``jjj`` → layers slice)::

            (layers[None,:] - depth_idx_start[None,:]) * snowrho[:,layers] * ...

        Only the standalone ``loop_index`` appearing as the left
        operand of a ``Sub`` is rewritten; other occurrences are left
        to the normal visit pipeline.

        Parameters
        ----------
        node : ast.AST
            The reduction term to rewrite.
        loop_index : str
            The dynamic loop's index variable name.

        Returns
        -------
        ast.AST
            A deep-copied, rewritten version of *node*.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            class OffsetSubstitutor(ast.NodeTransformer):
                def visit_BinOp(self, n):
                    self.generic_visit(n)
                    if (
                        isinstance(n.op, ast.Sub)
                        and isinstance(n.left, ast.Name)
                        and n.left.id == loop_index
                    ):
                        n.left = ast.Subscript(
                            value=ast.Name(id='layers', ctx=ast.Load()),
                            slice=ast.Tuple(elts=[ast.Constant(value=None), ast.Slice()], ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                        n.right = ast.Subscript(
                            value=ast.Name(id='depth_idx_start', ctx=ast.Load()),
                            slice=ast.Tuple(elts=[ast.Slice(), ast.Constant(value=None)], ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                    return n

            return OffsetSubstitutor().visit(copy.deepcopy(node))
        except Exception as e:
            self.logger.exception('Exception in _substitute_loop_index_with_offset:', e)
            raise

    def _extract_reduction_term(self, rhs_node: ast.AST, lhs_node: ast.AST, loop_index: str) -> ast.AST:
        """
        Strip the accumulation base from an RHS expression, returning
        only the per-iteration term.

        Given ``rhs = lhs + f(jj)``, returns ``f(jj)``. More generally,
        strips any sub-expression of a top-level ``Add`` that is
        structurally identical (via ``ast.unparse`` comparison) to
        *lhs_node*. Falls back to the whole RHS if the pattern is not
        matched.

        Parameters
        ----------
        rhs_node : ast.AST
            The full accumulation payload.
        lhs_node : ast.AST
            The accumulation base to strip out.
        loop_index : str
            The dynamic loop's index variable name (unused directly
            here but kept for call-site symmetry).

        Returns
        -------
        ast.AST
            The isolated reduction term, or *rhs_node* unchanged if no
            base could be stripped.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            lhs_src = ast.unparse(ast.fix_missing_locations(copy.deepcopy(lhs_node)))

            def _strip(node):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    left_src = ast.unparse(ast.fix_missing_locations(copy.deepcopy(node.left)))
                    right_src = ast.unparse(ast.fix_missing_locations(copy.deepcopy(node.right)))
                    if left_src == lhs_src:
                        return node.right
                    if right_src == lhs_src:
                        return node.left
                return node

            return _strip(rhs_node)
        except Exception as e:
            self.logger.exception('Exception in _extract_reduction_term:', e)
            raise

    def _extract_accum_base_from_payload(self, payload: ast.AST, lhs_visited: ast.AST) -> ast.AST:
        """
        Identify the accumulation-base sub-expression within *payload*.

        Given a payload like ``zsnowrhon[:, jj] + (layers...) *
        snowrho * ...``, returns the operand whose array name matches
        *lhs_visited*'s array name.

        Parameters
        ----------
        payload : ast.AST
            The full ``.set()``/RHS payload expression.
        lhs_visited : ast.AST
            The (already-visited) assignment target, used as the
            fallback and for name matching.

        Returns
        -------
        ast.AST
            The matching operand if *payload* is a top-level ``Add``
            and one side's array name equals *lhs_visited*'s, otherwise
            *lhs_visited* itself.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(payload, ast.BinOp) or not isinstance(payload.op, ast.Add):
                return lhs_visited

            lhs_arr = get_name(lhs_visited)
            for candidate in (payload.left, payload.right):
                candidate_arr = get_name(candidate)
                if candidate_arr and candidate_arr == lhs_arr:
                    return candidate

            return lhs_visited
        except Exception as e:
            self.logger.exception('Exception in _extract_accum_base_from_payload:', e)
            raise

    def _rhs_is_accumulation(self, rhs_node: ast.AST, lhs_node: ast.AST) -> bool:
        """
        Return ``True`` if *rhs_node* has the shape ``lhs + expr`` or
        ``expr + lhs``.

        Parameters
        ----------
        rhs_node : ast.AST
            The expression to check.
        lhs_node : ast.AST
            The accumulation base to compare against, via
            ``ast.unparse`` structural equality.

        Returns
        -------
        bool
            ``True`` if *rhs_node* is a top-level ``Add`` with one
            operand textually identical to *lhs_node*.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(rhs_node, ast.BinOp):
                return False
            if not isinstance(rhs_node.op, ast.Add):
                return False

            lhs_src = ast.unparse(ast.fix_missing_locations(copy.deepcopy(lhs_node)))
            left_src = ast.unparse(ast.fix_missing_locations(copy.deepcopy(rhs_node.left)))
            right_src = ast.unparse(ast.fix_missing_locations(copy.deepcopy(rhs_node.right)))
            return lhs_src in (left_src, right_src)
        except Exception as e:
            self.logger.exception('Exception in _rhs_is_accumulation:', e)
            raise