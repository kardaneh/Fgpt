import ast
from typing import Dict, List, Set, Tuple, Optional
from jax_utils import get_name

class _Masking:
    """
    Builds and applies boolean masks for conditional and bounded-slice
    array updates.

    Composes onto ``JaxConverter`` to implement every operation that
    turns a Python-level masked write into a JAX-compatible
    ``jnp.where`` / ``.at[...].set()`` expression. Three related
    concerns live here:

    - **WHERE/ELSEWHERE and ``arr[mask] = ...`` lowering** —
      :meth:`normalize_masked_node` extracts a boolean mask from a
      subscript chain (stripping it and any redundant full slices);
      :meth:`strip_any` unwraps a trailing ``.any()`` call so the
      underlying comparison can be reused directly as a mask;
      :meth:`handle_masked_where` lowers a Fortran-style WHERE block
      (recursing into nested ELSEWHERE-as-WHERE chains) by pairing
      branch assignments and calling :meth:`build_masked_update`;
      :meth:`handle_masked_arrays` and :meth:`_flatten_subscript`
      handle the ``arr[mask] = value`` and ``arr[:][:]`` cleanup forms
      directly from ``visit_Assign``.
    - **Per-statement mask application under vectorisation** —
      :meth:`_mask_plain_assign` and :meth:`_mask_vector_assign` are
      the workhorses called from every conditional-lowering branch in
      ``visit_If`` (``'vector'``, ``'index_loop'``,
      ``_emit_convergence_break``) to rewrite a single assignment into
      its masked equivalent, distinguishing plain-name targets from
      ``.at[...].<op>(...)`` array updates and detecting whether the
      target is already an in-place JAX op via :meth:`is_arr_at_op_call`
      (defined elsewhere) and :meth:`_is_boolean_mask`.
      :meth:`_boolean_mask` performs the final ``jnp.logical_*``
      rewrite of a mask's RHS expression before it is stored.
    - **Dynamic (runtime-bound) slice masking** — for slices whose
      bounds are not statically known (``arr[i:n]`` where ``n`` depends
      on a runtime attribute), :meth:`get_dynamic_slices` and
      :meth:`_is_dynamic_slice` detect the pattern,
      :meth:`check_for_dynamic_slice` is the boolean entry point used
      by ``visit_Assign``, :meth:`create_mask` builds an
      index-comparison mask sized to the slice's declared dimension
      (with :meth:`expand_to_full_shape` broadcasting lower/upper
      bounds to match), :meth:`remove_dynamic_slice` clears the
      now-redundant bounds back to a full slice, and
      :meth:`wrap_with_slice_specific_masks` /
      :meth:`handle_dynamic_slice` orchestrate building one mask per
      distinct slice group and wrapping the RHS expression with
      ``jnp.where`` accordingly.
    """

    def normalize_masked_node(
        self,
        node: ast.AST,
        mask_names: Set[str] = None,
    ) -> Tuple[ast.AST, Optional[ast.AST]]:
        """
        Extract a boolean mask from a subscript chain and clean the remainder.

        Recursively walks *node* looking for a ``Subscript`` whose slice
        contains a boolean-mask expression (either a name in *mask_names*
        or a structurally boolean sub-expression detected via
        :meth:`_is_boolean_mask`). The mask is removed from the subscript
        (along with any full ``[:]`` slices), and returned separately so
        the caller can apply it via ``jnp.where`` instead of fancy indexing.

        For non-subscript nodes (``BinOp``, ``Call``, ``UnaryOp``,
        ``Compare``, ``BoolOp``), recurses into children to strip masks
        from nested subscripts without extracting a top-level mask.

        Parameters
        ----------
        node : ast.AST
            Expression node to normalise — typically a WHERE-block target
            or RHS expression.
        mask_names : Set[str], optional
            Names known to represent boolean masks (e.g. from a prior
            ``mask = cond`` assignment in the same WHERE block).

        Returns
        -------
        Tuple[ast.AST, Optional[ast.AST]]
            ``(cleaned_node, extracted_mask)``. *extracted_mask* is
            ``None`` when no mask was found at this level.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Subscript):
                base, slices = self._flatten_subscript(node)

                extracted_mask = None
                new_slices = []

                for s in slices:
                    is_named_mask = (
                        isinstance(s, ast.Name)
                        and mask_names is not None
                        and s.id in mask_names
                    )
                    is_expr_mask = self._is_boolean_mask(s)

                    if is_named_mask or is_expr_mask:
                        if extracted_mask is None:
                            extracted_mask = s
                    elif not self.is_full_slice(s):
                        new_slices.append(s)

                if not new_slices:
                    return base, extracted_mask

                new_node = ast.Subscript(
                    value=base,
                    slice=(
                        new_slices[0]
                        if len(new_slices) == 1
                        else ast.Tuple(elts=new_slices, ctx=ast.Load())
                    ),
                    ctx=node.ctx,
                )
                return new_node, extracted_mask

            elif isinstance(node, ast.BinOp):
                node.left, _ = self.normalize_masked_node(node.left, mask_names)
                node.right, _ = self.normalize_masked_node(node.right, mask_names)
                return node, None

            elif isinstance(node, ast.Call):
                node.args = [
                    self.normalize_masked_node(arg, mask_names)[0]
                    for arg in node.args
                ]
                node.keywords = [
                    ast.keyword(
                        arg=kw.arg,
                        value=self.normalize_masked_node(kw.value, mask_names)[0],
                    )
                    for kw in node.keywords
                ]
                return node, None

            elif isinstance(node, ast.UnaryOp):
                node.operand, _ = self.normalize_masked_node(node.operand, mask_names)
                return node, None

            elif isinstance(node, ast.Compare):
                node.left, _ = self.normalize_masked_node(node.left, mask_names)
                node.comparators = [
                    self.normalize_masked_node(c, mask_names)[0]
                    for c in node.comparators
                ]
                return node, None

            elif isinstance(node, ast.BoolOp):
                node.values = [
                    self.normalize_masked_node(v, mask_names)[0]
                    for v in node.values
                ]
                return node, None

            return node, None

        except Exception as e:
            self.logger.exception('Exception in normalize_masked_node:', e)
            raise
    
    def strip_any(self, expr: ast.AST) -> ast.AST:
        """
        Unwrap a redundant ``.any()`` call around a boolean expression.

        Converts ``(a < b).any()`` back to ``a < b`` when the receiver is
        already a ``Compare``, ``BinOp``, or ``BoolOp`` — such ``.any()``
        calls appear in source Fortran-derived code as a scalar-reduction
        idiom that is unnecessary once the expression is used directly as
        a ``jnp.where`` mask.

        Parameters
        ----------
        expr : ast.AST
            Expression to inspect.

        Returns
        -------
        ast.AST
            The unwrapped receiver expression, or *expr* unchanged if the
            ``.any()`` pattern is not matched.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == 'any'
                and isinstance(expr.func.value, (ast.Compare, ast.BinOp, ast.BoolOp))
            ):
                return expr.func.value
            return expr
        except Exception as e:
            self.logger.exception('Exception in strip_any:', e)
            raise
    
    def handle_masked_where(self, node: ast.If) -> List[ast.Assign]:
        """
        Lower a Fortran-style masked WHERE/ELSEWHERE block to ``jnp.where``.

        Treats *node* as a WHERE construct: ``node.test`` is the mask
        expression, ``node.body`` holds the WHERE assignments, and
        ``node.orelse`` holds the ELSEWHERE assignments (which may itself
        be a nested WHERE, handled recursively).

        For each target assigned in both branches, builds a masked update
        via :meth:`build_masked_update`. Plain-name masks (``mask = cond``)
        used inside nested subscripts are detected and threaded through via
        :meth:`normalize_masked_node`.

        Parameters
        ----------
        node : ast.If
            The WHERE block, structured as an ``ast.If`` with the mask
            condition in ``test``.

        Returns
        -------
        List[ast.Assign]
            One ``jnp.where``-based assignment per target found in the
            WHERE branch.

        Raises
        ------
        NotImplementedError
            If a branch contains a statement structure that is not a
            recognised WHERE pattern (single nested ``If`` or a list of
            plain ``Assign`` statements).
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            mask_expr = self.visit(node.test)

            def build_block(block):
                found_masked_write = False
                assignments = []
                mask_names = set()

                if len(block) == 1 and isinstance(block[0], ast.If):
                    return self.handle_masked_where(block[0])

                if len(block) > 1:
                    for stmt in block:
                        if not isinstance(stmt, ast.Assign):
                            continue

                        target = stmt.targets[0]

                        if isinstance(target, ast.Name):
                            mask_names.add(target.id)

                        elif isinstance(target, ast.Subscript):
                            if isinstance(target.slice, (ast.Compare, ast.Subscript, ast.BoolOp)):
                                found_masked_write = True
                            elif isinstance(target.slice, ast.Name):
                                if target.slice.id in mask_names:
                                    found_masked_write = True

                        if found_masked_write:
                            if isinstance(target, ast.Subscript):
                                target, _ = self.normalize_masked_node(target, mask_names)

                            value, _ = self.normalize_masked_node(stmt.value, mask_names)
                            value = self.visit(value)
                            assignments.append((target, value))

                    return assignments

                raise NotImplementedError('Unsupported WHERE block structure')

            true_branch = build_block(node.body)
            false_branch = build_block(node.orelse) if node.orelse else []
            result = []

            for i, (target, true_expr) in enumerate(true_branch):
                false_expr = false_branch[i][1] if false_branch else None
                assign = self.build_masked_update(
                    target=target,
                    mask=mask_expr,
                    true_expr=true_expr,
                    false_expr=false_expr,
                )
                result.append(assign)

            return result

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception('Exception in handle_masked_where:', e)
            raise

    def _mask_plain_assign(
        self,
        assign: ast.Assign,
        mask_name: str,
    ) -> List[ast.Assign]:
        """
        Apply a vectorised mask to a plain (non-subscript) assignment.

        Rewrites ``var = value`` into ``var = jnp.where(mask, value,
        old_val)``, where *old_val* is either the variable's previous
        value (if already defined in an outer scope) or a freshly created
        default (``jnp.ones_like(1)``) registered in :attr:`_local_defaults`.

        If *var_name* has been identified as ``'stateful'`` in
        :attr:`var_state` and has a known shape, the result is further
        wrapped in ``var.at[:].set(masked_val)`` to preserve array identity
        across iterations.

        Parameters
        ----------
        assign : ast.Assign
            The original plain assignment, with a single ``ast.Name``
            target.
        mask_name : str
            Name of the boolean mask variable to apply.

        Returns
        -------
        List[ast.Assign]
            A single-element list containing the masked assignment, or
            ``[assign]`` unchanged if the target is not a plain name.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            target = assign.targets[0]
            value = self.visit(assign.value)
            if not isinstance(target, ast.Name):
                return [assign]

            var_name = target.id
            mask_expr = ast.Name(id=mask_name, ctx=ast.Load())

            if self.is_arr_at_op_call(assign.value, var_name):
                value = assign.value.func.args[0]

            if var_name in self._local_defaults or (
                self._local_defined_stack
                and len(self._local_defined_stack) > 1
                and var_name in self._local_defined_stack[-2]
            ):
                old_val = ast.Name(id=var_name, ctx=ast.Load())
            else:
                default_val = ast.Call(
                    func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='ones_like', ctx=ast.Load()),
                    args=[ast.Constant(value=1)],
                    keywords=[],
                )
                self._local_defaults[var_name] = default_val
                old_val = default_val

            masked_val = ast.Call(
                func=ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr='where', ctx=ast.Load()),
                args=[mask_expr, value, old_val],
                keywords=[],
            )

            shape = None
            if (
                self._local_defined_stack
                and len(self._local_defined_stack) > 1
                and var_name in self._local_defined_stack[-2]
            ):
                shape = self._local_defined_stack[-2][var_name]

            if var_name in self.var_state and shape:
                state, _ = self.var_state.get(var_name, ())
                if state == 'stateful':
                    masked_val = ast.Call(
                        func=ast.Attribute(
                            value=ast.Subscript(
                                value=ast.Attribute(value=ast.Name(id=var_name, ctx=ast.Load()), attr='at', ctx=ast.Load()),
                                slice=ast.Slice() if len(shape) == 1 else [ast.Slice()] * len(shape),
                                ctx=ast.Load(),
                            ),
                            attr='set',
                            ctx=ast.Load(),
                        ),
                        args=[masked_val],
                        keywords=[],
                    )

            new_assign = ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=masked_val,
            )
            return [ast.fix_missing_locations(new_assign)]

        except Exception as e:
            self.logger.exception('Exception in _mask_plain_assign:', e)
            raise
    
    def _mask_vector_assign(
        self,
        assign: ast.Assign,
        mask_name: str,
        assigned: List,
        used_after: List,
    ) -> List[ast.AST]:
        """
        Apply conditional masking to a vectorised assignment.

        Transforms a vectorised assignment into a masked update that preserves
        JAX functional semantics. Assignments represented as
        ``arr = arr.at[idx].<op>(value)`` are rewritten so that updates are
        applied only where the generated mask evaluates to ``True``.

        For overwrite operations (``set``), the original array value is
        preserved wherever the mask evaluates to ``False`` by constructing an
        equivalent ``jnp.where`` expression. For arithmetic updates such as
        ``add``, ``subtract``, ``multiply``, ``divide`` and ``power``, a masked
        delta is constructed and applied through the corresponding ``.at``
        operator.

        Mask broadcasting and rank alignment are performed automatically using
        :meth:`maybe_add_index` and :meth:`expand_to_full_shape` when
        vectorisation introduces additional dimensions.

        Parameters
        ----------
        assign : ast.Assign
            Assignment previously transformed into a JAX-style functional
            update.
        mask_name : str
            Name of the boolean mask variable controlling the update.
        assigned : list
            Variables assigned within the enclosing control-flow region.
        used_after : list
            Variables referenced after the control-flow region.

        Returns
        -------
        List[ast.stmt]
            One or more transformed statements implementing the masked update.
            If the assignment does not match a supported ``.at`` update pattern,
            ``[assign]`` is returned unchanged.

        Notes
        -----
        This method assumes assignments have already been converted into
        functional update form before masking is applied.

        Mask-aware assignments may rely on metadata stored in
        :attr:`_control_stack`, :attr:`var_state`, and
        :attr:`dynamic_variable_lift`.

        Related functionality is provided by
        :meth:`build_masked_update`,
        :meth:`_mask_plain_assign`, and
        :meth:`handle_dynamic_slice`.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during the
            transformation process.
        """
        try:
            # Only handle simple assignments
            if not isinstance(assign, ast.Assign):
                return [assign]

            target = assign.targets[0]
            value = assign.value
            target_name = get_name(target)
            # Only handle arr = arr.at[...].<op>(...) pattern
            if not self.is_arr_at_op_call(value, target_name):
                if isinstance(assign, ast.Assign) and self._control_stack: 
                    if not self.check_if_array(assign) or (target_name in self.var_state): 
                        if self._is_control_temporary(target_name, assigned, used_after):
                            # Do NOT mask control temporaries
                            return [assign]
                        
                        if target_name in assigned:
                            return self._mask_plain_assign(assign, mask_name)
                        else:
                            return [assign]
                return [assign]
            
            at_sub = value.func.value   # arr.at[idx]
            op_name = value.func.attr   # set / add / multiply / etc...
            args = value.args
            
            arr_name = at_sub.value.value.id if isinstance(at_sub.value.value, ast.Name) else at_sub.value.value.attr
            
            if arr_name is None:
                return [assign]

            # Determine old value expression
            if (self.counter == 0 and self.for_counter == 0) and self.cls_info[self.cls_name].get('attributes').get(arr_name):
                target_value = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=arr_name, ctx=ast.Load())
                if isinstance(at_sub.value.value, ast.Name):
                    at_sub.value.value =  ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),attr=arr_name, ctx=ast.Load())
            else:
                target_value = ast.Name(id=arr_name, ctx=ast.Load())
            
            if isinstance(at_sub.slice, ast.Tuple):
                if all(isinstance(s, ast.Slice) for s in at_sub.slice.elts):
                    old_val = target_value
                else:
                    old_val = ast.Subscript(value=target_value, slice=at_sub.slice, ctx=ast.Load())
            else:
                old_val = ast.Subscript(value=target_value, slice=at_sub.slice, ctx=ast.Load())
            
            # Determine rhs expression
            rhs_expr = args[0] if op_name == "set" else ast.BinOp(left=old_val, op={
                "add": ast.Add(),
                "multiply": ast.Mult(),
                "subtract": ast.Sub(),
                "divide": ast.Div(),
                "power": ast.Pow(),
            }[op_name], right=args[0])

            vect_context = None
            if self._control_stack:
                vect_context = self._control_stack[-1].to_dict()

            mask_expr = ast.Name(id=mask_name, ctx=ast.Load())
            target_rank = self._target_rank(at_sub.slice)

            # Align the RHS rank with the target rank before constructing the mask.
            # Vectorization may introduce additional dimensions that require
            # explicit indexing or broadcasting.
            rhs_expr, elts_list = self.maybe_add_index(rhs_expr, target_rank, vect_context)
            # Adjust the mask shape when nested vectorized loops introduce
            # additional dimensions. The mask must broadcast to the same rank
            # as the assignment target.
            if elts_list:
                if (vect_context and vect_context['metadata'].get('current_mask_rank', 0) == 0):
                    pass 
                else:
                    mask_expr = ast.Subscript(
                        value=mask_expr,
                        slice=ast.Tuple(elts=elts_list, ctx=ast.Load()),
                        ctx=ast.Load()
                    )
            else:
                if target_rank > 1 and vect_context:
                    current_mask_rank = vect_context.get("metadata", {}).get("current_mask_rank", 0)
                    if current_mask_rank == 0:
                        pass 
                    elif current_mask_rank < target_rank:
                        dimensions = self.get_active_dims(target)
                        loop_info = vect_context.get('loop_info', {})
                        vectorization_axis = vect_context.get('vectorization_axis', {})

                        elts = [ast.Constant(value=None)] * target_rank
                        for loop_dim, loop_var in loop_info.items():
                            if loop_dim not in dimensions:
                                continue
                            dim_index = dimensions.index(loop_dim)
                            vect_axis = vectorization_axis.get(loop_var, [])
                            if dim_index not in vect_axis:
                                continue
                            elts[dim_index] = ast.Slice()

                        if any(isinstance(e, ast.Slice) for e in elts):
                            mask_expr = ast.Subscript(
                                value=mask_expr,
                                slice=ast.Tuple(elts=elts, ctx=ast.Load()),
                                ctx=ast.Load()
                            )

            if op_name != "set" and isinstance(rhs_expr, ast.BinOp):
                # Select the appropriate identity element for the update dtype.
                attributes = self.cls_info[self.cls_name].get('attributes')
                methods = self.cls_info[self.cls_name]['methods'].get(self.func_name)
                local_arrays = methods.get('local_arr')
                dtype = 'float64'
                if attributes and target_name in attributes:
                    attr_data = attributes.get(target_name)
                    if attr_data:
                        dtype = attr_data.get('dtype', 'float64')
                elif local_arrays and target_name in local_arrays:
                        arr_data = local_arrays.get(target_name)
                        if arr_data:
                            dtype = arr_data.get('dtype', 'float64')
                
                is_float = dtype == "float64"
                constant_value = 0.0 if is_float else 0
                multi_constant_value = 1.0 if is_float else 1

                if isinstance(rhs_expr.op, ast.Add):
                    op_name = 'add'
                    identity = ast.Constant(value=constant_value)
                    delta_expr = rhs_expr.right

                elif isinstance(rhs_expr.op, ast.Sub):
                    op_name = 'subtract'
                    identity = ast.Constant(value=constant_value)
                    delta_expr = rhs_expr.right

                elif isinstance(rhs_expr.op, ast.Mult):
                    op_name = 'multiply'
                    identity = ast.Constant(value=multi_constant_value)
                    delta_expr = rhs_expr.right

                elif isinstance(rhs_expr.op, ast.Div):
                    op_name = 'divide'
                    identity = ast.Constant(value=multi_constant_value)
                    delta_expr = rhs_expr.right

                elif isinstance(rhs_expr.op, ast.Pow):
                    op_name = 'power'
                    identity = ast.Constant(value=multi_constant_value)
                    delta_expr = rhs_expr.right

                masked_delta = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='jnp', ctx=ast.Load()),
                        attr='where',
                        ctx=ast.Load(),
                    ),
                    args=[mask_expr, delta_expr, identity],
                    keywords=[],
                )
                delta_name = f'delta{mask_name}'
                delta_value = ast.Assign(
                    targets=[ast.Name(id=delta_name, ctx=ast.Store())],
                    value=masked_delta
                )
                new_assign = ast.Assign(
                    targets=[ast.Name(id=arr_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=at_sub, attr=op_name, ctx=ast.Load()),
                        args=[ast.Name(id=delta_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
                return [ast.fix_missing_locations(delta_value), ast.fix_missing_locations(new_assign)]
            else:
                # Overwrite / set pattern
                masked_val = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="where", ctx=ast.Load()),
                    args=[mask_expr, rhs_expr, old_val],
                    keywords=[]
                )

                new_assign = ast.Assign(
                    targets=[ast.Name(id=arr_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=at_sub, attr='set', ctx=ast.Load()),
                        args=[masked_val],
                        keywords=[]
                    )
                )
                
                return [ast.fix_missing_locations(new_assign)]
        except Exception as e:
            self.logger.exception('Exception in _mask_vector_assign:', e)
            raise

    def _boolean_mask(self, node: ast.Assign) -> ast.Assign:
        """
        Rewrite boolean expressions into JAX-compatible operations.

        Transforms boolean and comparison expressions appearing on the
        right-hand side of an assignment into their equivalent ``jnp``
        representations. This ensures that logical operations participate
        correctly in vectorised and JAX-transformed execution.

        Supported transformations include:

        * ``and`` → ``jnp.logical_and``
        * ``or`` → ``jnp.logical_or``
        * ``not`` → ``jnp.logical_not``
        * Comparison operators rewritten through
        :meth:`_transform_if_test`

        Parameters
        ----------
        node : ast.Assign
            Assignment whose value expression should be analysed and
            transformed.

        Returns
        -------
        ast.Assign
            Assignment containing the transformed boolean expression.

        Notes
        -----
        The transformation is applied only to the assignment value and does
        not modify assignment targets.

        Boolean expression rewriting shares the same comparison handling
        logic used by :meth:`_transform_if_test`.

        This method is typically invoked before vectorisation-specific
        transformations are applied.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during boolean
            expression transformation.
        """
        try:
            value = node.value
            # Only transform if it’s a boolean expression
            if isinstance(value, (ast.BoolOp, ast.Compare, ast.UnaryOp, ast.BinOp)):
                node.value = self._transform_if_test(value)
            return node
        except Exception as e:
            self.logger.exception('Exception in _boolean_mask:', e)
            raise
    
    def _is_boolean_mask(self, node: ast.AST) -> bool:
        """
        Determine whether a mask is boolean or not 
        based on the given node(ast.Subscript) slice.

        Parameters
        ----------
        node : ast.AST
            AST node to inspect.

        Returns
        -------
        bool
            ``True`` if the node is boolean otherwise ``False``.

        Notes
        -----
        This method serves as a lightweight predicate to check if
        we don't have a case of masking process inside a subscript
        slice, used by :meth:`handle_masked_arrays`
        """
        return isinstance(node, (ast.Compare, ast.UnaryOp, ast.BoolOp))
    
    def build_masked_update(
        self,
        target: ast.AST,
        mask: ast.AST,
        true_expr: ast.AST,
        false_expr: Optional[ast.AST] = None,
    ) -> ast.Assign:
        """
        Construct a masked assignment using ``jnp.where``.

        Builds the canonical masked-update representation used throughout the
        transformation pipeline. The generated expression evaluates::

            jnp.where(mask, true_expr, false_expr)

        and rewrites the result according to the type of assignment target.

        For array targets, updates are emitted using JAX functional update
        semantics::

            arr[idx] = value

        becomes::

            arr = arr.at[idx].set(
                jnp.where(mask, value, old_value)
            )

        For variable and attribute targets, the masked value is assigned
        directly::

            var = jnp.where(mask, true_expr, false_expr)

        If *false_expr* is omitted, the current value of *target* is used as
        the fallback branch.

        Parameters
        ----------
        target : ast.AST
            Assignment target. Supported target types are
            :class:`ast.Subscript`, :class:`ast.Name`, and
            :class:`ast.Attribute`.
        mask : ast.AST
            Boolean mask controlling the update.
        true_expr : ast.AST
            Expression evaluated when the mask is ``True``.
        false_expr : ast.AST, optional
            Expression evaluated when the mask is ``False``. If omitted,
            the current value of *target* is used.

        Returns
        -------
        ast.Assign
            Assignment implementing the masked update.

            For subscript targets, the result is emitted as a JAX
            ``.at[...,].set(...)`` update. For variable and attribute targets,
            a direct masked assignment is returned.

        Notes
        -----
        Nested subscript expressions are normalised using
        :meth:`_flatten_subscript`.

        Boolean masks are typically generated by
        :meth:`create_mask`,
        :meth:`handle_dynamic_slice`, or control-flow masking
        transformations such as :meth:`_mask_vector_assign`.

        This method serves as the canonical masked-update constructor used
        throughout the AST transformation pipeline.

        Raises
        ------
        NotImplementedError
            If *target* is not a supported assignment target type.

        Exception
            Re-raises any unexpected error encountered during update
            construction.
        """

        mask = self.strip_any(mask)

        # If false_expr not provided → use current value of target
        if false_expr is None:
            false_expr = self.visit(target)

        where_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="where",
                ctx=ast.Load()
            ),
            args=[mask, true_expr, false_expr],
            keywords=[]
        )

        # Case 1: array update (A[...] = ...)
        if isinstance(target, ast.Subscript):
            base, slices = self._flatten_subscript(target)

            filtered = [s for s in slices if not self.is_full_slice(s)]
            if not filtered:
                filtered = [slices[-1]]

            slice_value = (
                filtered[0]
                if len(filtered) == 1
                else ast.Tuple(elts=filtered, ctx=ast.Load())
            )

            at_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Attribute(value=base, attr='at', ctx=ast.Load()),
                        slice=slice_value,
                        ctx=ast.Load()
                    ),
                    attr='set',
                    ctx=ast.Load()
                ),
                args=[where_call],
                keywords=[]
            )

            return ast.Assign(
                targets=[base],
                value=at_call
            )

        # Case 2: simple variable (A = ...)
        elif isinstance(target, (ast.Name, ast.Attribute)):
            lhs = self.visit(target)
            lhs.ctx = ast.Store()

            return ast.Assign(
                targets=[lhs],
                value=where_call
            )

        else:
            raise NotImplementedError(f"Unsupported target type: {type(target)}")
        
    def handle_masked_arrays(
        self,
        node: ast.Assign,
        sub: ast.Subscript,
        base: ast.AST,
        slices: List[ast.AST],
    ) -> ast.Assign:
        """
        Transform assignments involving boolean-masked array indexing.

        This method handles assignments whose target contains nested
        subscript expressions that cannot be processed through the standard
        ``.at[...]`` assignment path.

        Two cases are supported:

        * Boolean-mask assignments such as ``arr[mask] = value``.
        * Nested slice expressions such as ``arr[:][:] = value``.

        For boolean-mask assignments, the target and mask are first
        normalised using :meth:`normalize_masked_node`. The mask expression
        is then transformed through :meth:`_transform_if_test`, and a masked
        update assignment is generated via :meth:`build_masked_update`.

        For nested slice expressions, redundant full slices are removed and
        the target is rewritten into an equivalent flattened subscript
        expression before returning the modified assignment node.

        Parameters
        ----------
        node : ast.Assign
            Assignment currently being transformed.
        sub : ast.Subscript
            Target subscript expression extracted from the assignment.
        base : ast.AST
            Base array expression returned by
            :meth:`_flatten_subscript`.
        slices : List[ast.AST]
            Sequence of subscript components returned by
            :meth:`_flatten_subscript`.

        Returns
        -------
        ast.Assign
            Either a masked update assignment generated by
            :meth:`build_masked_update` or a rewritten assignment with
            simplified indexing.

        Notes
        -----
        Boolean-mask detection is delegated to
        :meth:`_is_boolean_mask`.

        Slice simplification removes redundant full slices identified by
        :meth:`is_full_slice` while preserving the effective indexing
        semantics.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during the
            transformation process.
        """
        try:
            # Case: arr[mask] = value
            if self._is_boolean_mask(sub.slice):

                target, mask = self.normalize_masked_node(sub)
                return self.build_masked_update(
                    target=target,
                    mask=self._transform_if_test(mask),
                    true_expr=self.visit(node.value),
                    false_expr=None
                )

            # Case: cleanup arr[:][:] → arr[:]
            else:
                filtered = [s for s in slices if not self.is_full_slice(s)]

                if not filtered:
                    filtered = [slices[-1]]

                new_slice = (
                    filtered[0]
                    if len(filtered) == 1
                    else ast.Tuple(elts=filtered, ctx=ast.Load())
                )

                node.targets[0] = ast.Subscript(
                    value=base,
                    slice=new_slice,
                    ctx=ast.Store()
                )

                return node
        except Exception as e:
            self.logger.exception('Exception in handle_masked_arrays:', e)
            raise
    
    def _flatten_subscript(
        self, 
        sub: ast.Subscript
    ) -> Tuple[ast.AST, List[ast.AST]]:
        """
        Flatten a nested subscript expression into its base value and indices.

        Traverses a chain of nested `ast.Subscript` nodes and extracts
        all indexing operations into a single ordered list. The returned slice
        sequence preserves the original indexing order from outermost to
        innermost access.

        For example::

            arr[:, i][mask]

        is flattened into::

            base   = arr
            slices = [(:, i), mask]

        This representation is used by transformation routines such as
        :meth:`handle_masked_arrays` and :meth:`build_masked_update` when
        rewriting nested indexing expressions into JAX-compatible update
        operations.

        Parameters
        ----------
        sub : ast.Subscript
            Subscript expression to flatten.

        Returns
        -------
        Tuple[ast.AST, List[ast.AST]]
            A tuple containing:

            * The base expression being indexed.
            * A list of slice expressions in evaluation order.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during the
            flattening process.
        """
        try:
            slices = []
            value = sub

            while isinstance(value, ast.Subscript):
                slices.append(value.slice)
                value = value.value

            slices.reverse()
            return value, slices
        except Exception as e:
            self.logger.exception('Exception in _flatten_subscript:', e)
            raise
    
    def create_mask(self, axis: int, sub: ast.Subscript, mask_name: str):
        """
        Construct a boolean mask corresponding to a dynamic slice expression.

        Generates AST statements that build a boolean mask representing the
        bounds of a slice along a specified array dimension. The resulting mask
        is equivalent to evaluating::

            lower <= idx < upper

        for every index along the selected axis.

        The generated mask is stored in *mask_name* and is intended for use in
        subsequent masked assignment transformations, including
        :meth:`build_masked_update` and :meth:`handle_masked_arrays`.

        When slice bounds depend on vectorized dimensions, the bounds are
        broadcast to the target rank using :meth:`expand_to_full_shape` to
        ensure shape compatibility.

        Parameters
        ----------
        axis : int
            Axis corresponding to the slice dimension being transformed.
        sub : ast.AST
            Subscript node containing the dynamic slice expression.
        mask_name : str
            Name of the variable that will receive the generated mask.

        Returns
        -------
        List[ast.Assign]
            Sequence of assignment statements that:

            * Construct an index array using ``jnp.arange``.
            * Generate any required broadcasted bound expressions.
            * Create and assign the final boolean mask.

        Notes
        -----
        Array dimension metadata is retrieved through
        :meth:`get_declared_dims`.

        Class attributes used as dimensions are automatically converted into
        ``self.<attr>`` references when constructing the index range.

        Broadcasting of dynamic bounds is handled through
        :meth:`expand_to_full_shape`.

        Raises
        ------
        ValueError
            When the given argument is not that of ast.Subscript
        Exception
            Re-raises any unexpected error encountered during mask
            construction.
        """
        try:
            if not isinstance(sub, ast.Subscript):
                raise ValueError(f'The given argument is not that of ast.Subscript \
                                , {type(sub)}')
            new_stmts = []
            if isinstance(sub.slice, ast.Slice):
                slice_node = sub.slice
            else:
                slice_node = sub.slice.elts[axis]

            lower = slice_node.lower
            upper = slice_node.upper
            # build idx = jnp.arange(dim)
            dim_name = self.get_declared_dims(sub)[axis]
            if dim_name in self.cls_info[self.cls_name].get('attributes'):
                dim_ast = ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=dim_name,
                    ctx=ast.Load()
                )
            else:
                dim_ast = ast.Name(id=dim_name, ctx=ast.Load())
            idx_assign = ast.Assign(
                targets=[ast.Name(id='idx', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='jnp', ctx=ast.Load()),
                        attr='arange',
                        ctx=ast.Load()
                    ),
                    args=[dim_ast],
                    keywords=[]
                )
            )
            new_stmts.append(idx_assign)
            node_dim = self.get_declared_dims(sub)
            # This defines the broadcasting issues that might linked to the fact 
            # that the layers idx dimension and the upper dimenion 
            # might mismatch and the fact that both needs to have an end dimension 
            # corresponding to the subscript containing the dynamic slice
            idx = ast.Name(id='idx', ctx=ast.Load())
            if len(node_dim) != len([dim_name]):
                idx_expr = self.expand_to_full_shape(idx, [dim_name], node_dim)
            else:
                idx_expr = idx 
                
            upper_dims = self.get_declared_dims(upper) if upper is not None else None
            if upper is not None:
                if upper_dims:
                    upper_expr = self.expand_to_full_shape(upper, upper_dims, node_dim)
                else:
                    upper_expr = upper  # scalar bound, no broadcasting needed
            else:
                upper_expr = None

            lower_dims = self.get_declared_dims(lower) if lower is not None else None
            if lower is not None:
                if lower_dims:
                    lower_expr = self.expand_to_full_shape(lower, lower_dims, node_dim)
                else:
                    lower_expr = lower  # scalar bound, no broadcasting needed
            else:
                lower_expr = None
            
            if lower_expr is not None:
                lower_cmp = ast.Compare(
                    left=idx_expr,
                    ops=[ast.GtE()],
                    comparators=[lower_expr]
                )
            else:
                lower_cmp = None

            if upper_expr is not None:
                upper_cmp = ast.Compare(
                    left=idx_expr,
                    ops=[ast.Lt()],
                    comparators=[upper_expr]
                )
            else:
                upper_cmp = None
            
            if lower_cmp and upper_cmp:
                mask_expr = ast.BinOp(
                    left=lower_cmp,
                    op=ast.BitAnd(),
                    right=upper_cmp
                )
            elif upper_cmp:
                mask_expr = upper_cmp
            else:
                mask_expr = lower_cmp
            
            mask_assign = ast.Assign(
                targets=[ast.Name(id=mask_name, ctx=ast.Store())],
                value=mask_expr
            )
            new_stmts.append(mask_assign)
            
            return new_stmts
        except Exception as e:
            self.logger.exception('Exception in create_mask:', e)
            raise
    
    def remove_dynamic_slice(
        self, 
        axis: int, 
        subs: List[Tuple[ast.Subscript, ast.Slice]]
    ) -> None:
        """
        Remove dynamic slice bounds from transformed subscript expressions.

        Normalises dynamic slices after mask generation by replacing any
        computed lower and upper bounds with full slices. This converts
        expressions such as::

            arr[:n]
            arr[i, start:end]

        into::

            arr[:]
            arr[i, :]

        while preserving the original array rank and indexing structure.

        This method is typically invoked by :meth:`handle_dynamic_slice`
        after the corresponding slice bounds have been encoded into explicit
        boolean masks.

        Parameters
        ----------
        axis : int
            Axis containing the dynamic slice to be removed.
        subs : List[Tuple[ast.Subscript, ast.Slice]]
            Collection of subscript expressions and their associated dynamic
            slice nodes.

        Notes
        -----
        The generated masks created by :meth:`create_mask` preserve the
        original slice semantics, allowing the explicit slice bounds to be
        removed safely.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during slice
            normalisation.
        """
        try:
            for sub in subs:
                sub_slice = sub[0].slice
                if isinstance(sub_slice, ast.Slice):
                    # single slice a[:n] -> a[:]
                    sub_slice.lower = None
                    sub_slice.upper = None
                else:
                    # tuple slice a[i, :n] -> a[i, :]
                    elt = sub_slice.elts[axis]
                    if isinstance(elt, ast.Slice):
                        elt.lower = None
                        elt.upper = None
        except Exception as e:
            self.logger.exception('Exception in remove_dynamic_slice:', e)
            raise
    
    def wrap_with_slice_specific_masks(
        self, 
        expr: ast.AST, 
        mask_map: Dict[int, str]
    ) -> ast.AST:
        """
        Apply slice-specific masks to dynamic slice accesses.

        Recursively traverses an expression tree and replaces selected
        subscript expressions with masked equivalents using ``jnp.where``.
        Each subscript identified in *mask_map* is rewritten as::

            jnp.where(mask, expr, 0)

        where the mask is associated with the corresponding subscript node.

        The transformation is applied recursively to supported expression
        types, including binary operations and function calls.

        Parameters
        ----------
        expr : ast.AST
            Expression to transform.
        mask_map : Dict[int, str]
            Mapping from subscript node identifiers to the names of mask
            variables generated by :meth:`create_mask`.

        Returns
        -------
        ast.AST
            Expression with all matching dynamic slice accesses wrapped in
            mask-aware ``jnp.where`` expressions.

        Notes
        -----
        Mask variables are typically generated by
        :meth:`handle_dynamic_slice`.

        Only subscript expressions explicitly registered in *mask_map* are
        modified.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during expression
            transformation.
        """
        try:
            if isinstance(expr, ast.Subscript):
                if id(expr) in mask_map:
                    mask_name = mask_map[id(expr)]

                    return ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="where",
                            ctx=ast.Load()
                        ),
                        args=[
                            ast.Name(id=mask_name, ctx=ast.Load()),
                            expr,
                            ast.Constant(value=0)
                        ],
                        keywords=[]
                    )

            elif isinstance(expr, ast.BinOp):
                expr.left = self.wrap_with_slice_specific_masks(expr.left, mask_map)
                expr.right = self.wrap_with_slice_specific_masks(expr.right, mask_map)
                return expr

            elif isinstance(expr, ast.Call):
                expr.args = [self.wrap_with_slice_specific_masks(arg, mask_map) for arg in expr.args]
                return expr

            return expr
        except Exception as e:
            self.logger.exception('Exception in wrap_with_slice_specific_masks:', e)
            raise

    def handle_dynamic_slice(
        self, 
        node: ast.Assign
    ) -> Tuple[List[ast.Assign], ast.Assign] | ast.Assign:
        """
        Transform dynamic slice expressions into explicit mask operations.

        Identifies dynamic slice accesses occurring within reduction
        expressions and rewrites them into mask-based equivalents suitable
        for vectorised execution.

        The transformation proceeds in several stages:

        #. Detect dynamic slices using :meth:`get_dynamic_slices`.
        #. Group compatible slices by axis and slice structure.
        #. Generate boolean masks through :meth:`create_mask`.
        #. Replace dynamic slice accesses with masked expressions via
        :meth:`wrap_with_slice_specific_masks`.
        #. Remove the original dynamic slice bounds using
        :meth:`remove_dynamic_slice`.

        Currently only a single dynamic slice axis is supported.

        Parameters
        ----------
        node : ast.Assign
            Assignment node containing one or more dynamic slice accesses.

        Returns
        -------
        Tuple[List[ast.Assign], ast.Assign] | ast.Assign
            If dynamic slices are found, returns a tuple containing:

            * Mask construction statements.
            * The transformed assignment.

            Otherwise, returns *node* unchanged.

        Notes
        -----
        Mask generation is performed independently for each distinct dynamic
        slice pattern to avoid constructing redundant masks.

        Dynamic slice detection relies on :meth:`_is_dynamic_slice`.

        Raises
        ------
        ValueError
            If dynamic slices are detected on multiple axes.

        Exception
            Re-raises any unexpected error encountered during
            transformation.
        """
        try:
            results = self.get_dynamic_slices(node)

            # Step 1: group by axis ONLY
            axis_map = {}
            for sub in results:
                slice_node = sub.slice

                if isinstance(slice_node, ast.Slice):
                    if self._is_dynamic_slice(slice_node):
                        axis_map.setdefault(0, []).append((sub, slice_node))

                elif isinstance(slice_node, ast.Tuple):
                    for idx, elt in enumerate(slice_node.elts):
                        if isinstance(elt, ast.Slice) and self._is_dynamic_slice(elt):
                            axis_map.setdefault(idx, []).append((sub, elt))

            if not axis_map:
                return node

            # NOTE: right now, the dynamic slice can handle 
            # one axis with slice of the same nature and not 
            # Only support single axis for now
            if len(axis_map) > 1:
                raise ValueError("Not implemented for multi axial slices")

            new_stmts = []

            # Step 2: group by slice
            slice_groups = {}
            for axis, subs in axis_map.items():
                for sub, slice_node in subs:
                    key = ast.dump(slice_node)
                    slice_groups.setdefault((axis, key), []).append((sub, slice_node))

            # Step 3: create mask per slice group
            subscript_to_mask = {}

            for (axis, key), subs in slice_groups.items():
                ref_sub, _ = subs[0]

                # mask_name = f"mask_axis_{axis}_{len(new_stmts)}"
                mask_name = f"mask_axis_{self.mask_axis_counter}"
                self.mask_axis_counter += 1
                stmts = self.create_mask(axis, ref_sub, mask_name=mask_name)
                new_stmts.extend(stmts)

                for sub, _ in subs:
                    subscript_to_mask[id(sub)] = mask_name

            # Step 4: wrap RHS 
            node.value = self.wrap_with_slice_specific_masks(node.value, subscript_to_mask)

            # Step 5: remove slices AFTER wrapping
            for axis, subs in axis_map.items():
                self.remove_dynamic_slice(axis, subs)

            return new_stmts, node
        
        except Exception as e:
            self.logger.exception('Exception in handle_dynamic_slice:', e)
            raise
    
    def get_dynamic_slices(
        self, 
        node: ast.AST, 
        inside_reduction: bool = False
    ) -> List[ast.Subscript]:
        """
        Collect dynamic slice expressions occurring within reductions.

        Recursively traverses an AST and identifies subscript expressions
        containing dynamic slice bounds that appear inside supported
        reduction operations.

        A slice is considered dynamic when determined by
        :meth:`_is_dynamic_slice`.

        Supported reductions include:

        * ``sum``
        * ``mean``
        * ``max``
        * ``min``
        * ``prod``
        * ``nansum``
        * ``nanmean``

        Parameters
        ----------
        node : ast.AST
            AST node to inspect.
        inside_reduction : bool, optional
            Indicates whether the current traversal location is already
            nested inside a reduction call.

        Returns
        -------
        List[ast.Subscript]
            Dynamic slice expressions discovered within reduction contexts.

        Notes
        -----
        Reduction context is propagated recursively through the AST so that
        dynamic slices are only collected when they contribute directly to a
        reduction operation.

        The resulting subscript nodes are typically processed by
        :meth:`handle_dynamic_slice`.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during traversal.
        """
        try:
            results = []
            
            # Check if current node is a reduction call
            is_reduction = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {'sum', 'mean', 'max', 'min', 'prod', 'nansum', 'nanmean'}
            )
            
            if isinstance(node, ast.Subscript):
                slice_node = node.slice
                if inside_reduction:
                    # Case 1: single slice a[1:n]
                    if isinstance(slice_node, ast.Slice):
                        if self._is_dynamic_slice(slice_node):
                            results.append(node)
                    # Case 2: multidimensional a[i, 1:n]
                    elif isinstance(slice_node, ast.Tuple):
                        for elt in slice_node.elts:
                            if isinstance(elt, ast.Slice) and self._is_dynamic_slice(elt):
                                results.append(node)
                                break
            
            # Recurse into children, propagating reduction context
            for child in ast.iter_child_nodes(node):
                results.extend(
                    self.get_dynamic_slices(child, inside_reduction=inside_reduction or is_reduction)
                )
            
            return results
        except Exception as e:
            self.logger.exception('Exception in get_dynamic_slice:', e)
            raise

    def _is_dynamic_slice(self, slice_node: ast.Slice) -> bool:
        """
        Determine whether a slice contains dynamic bounds.

        A slice is considered dynamic if one or more of its bounds depend on
        runtime values, vectorised dimensions, model attributes, local array
        dimensions, or previously tracked dependencies.

        The following constructs are treated as dynamic:

        * Subscript expressions used as slice bounds.
        * ``self.<attr>`` references.
        * Expressions containing ``self.<attr>`` references.
        * Variables recorded in :attr:`var_deps`.
        * Variables associated with known array dimensions.
        * Function input dimensions recorded in
        :attr:`func_input_dim`.

        Parameters
        ----------
        slice_node : ast.Slice
            Slice node to analyse.

        Returns
        -------
        bool
            ``True`` if the slice contains dynamic bounds, otherwise
            ``False``.

        Notes
        -----
        Dimension metadata is obtained from :attr:`cls_info`,
        :attr:`var_deps`, and :attr:`func_input_dim`.

        This method is used by :meth:`get_dynamic_slices` and
        :meth:`handle_dynamic_slice`.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during analysis.
        """
        try:
            parts = (slice_node.lower, slice_node.upper, slice_node.step)
            attributes = self.cls_info[self.cls_name].get('attributes')
            methods = self.cls_info[self.cls_name].get('methods')
            local_arr = methods.get(self.func_name)
            
            for part in parts:
                if part is None:
                    continue
                # Case 1: definitely dynamic 
                if isinstance(part, ast.Subscript):
                    return True
                # Case 2a: self.X — always dynamic as a slice bound
                if (
                    isinstance(part, ast.Attribute)
                    and isinstance(part.value, ast.Name)
                    and part.value.id == "self"
                ):
                    return True
                # Case 2b: BinOp like self.nslm + 1 — if contains self.X, dynamic
                if isinstance(part, ast.BinOp):
                    for subnode in ast.walk(part):
                        if (
                            isinstance(subnode, ast.Attribute)
                            and isinstance(subnode.value, ast.Name)
                            and subnode.value.id == "self"
                        ):
                            return True
                    continue
                # Case 2c: plain variable -> check dependency list
                if isinstance(part, (ast.Name, ast.Attribute)):
                    name = get_name(part)
                    if name in self.var_deps:
                        return True
                    else:
                        if name in attributes:
                            if attributes.get(name).get('dimensions'):
                                return True
                        elif name in local_arr:
                            if local_arr.get(name).get('dimensions'):
                                return True
                        elif name in self.func_input_dim:
                            return True
            return False
        except Exception as e:
            self.logger.exception('Exception in _is_dynamic_slice:', e)
            raise
    
    def check_for_dynamic_slice(self, node: ast.AST) -> bool:
        """
        Determine whether an AST contains dynamic slice expressions.

        Checks whether any dynamic slices are present within the supplied
        AST node by delegating detection to
        :meth:`get_dynamic_slices`.

        Parameters
        ----------
        node : ast.AST
            AST node to inspect.

        Returns
        -------
        bool
            ``True`` if at least one dynamic slice is present,
            otherwise ``False``.

        Notes
        -----
        This method serves as a lightweight predicate used before invoking
        the more expensive transformation logic implemented by
        :meth:`handle_dynamic_slice`.
        """
        return len(self.get_dynamic_slices(node)) > 0
    
    def expand_to_full_shape(self, expr: ast.AST, expr_dims, target_dims):
        """
        Broadcast an expression to match a target dimensional structure.

        Constructs an indexing expression that expands *expr* to the rank
        defined by *target_dims*. Dimensions present in *expr_dims* are
        preserved using full slices, while missing dimensions are inserted
        using ``None`` indexing.

        For example::

            expr_dims   = ["layers"]
            target_dims = ["batch", "layers"]

        produces an expression equivalent to::

            expr[None, :]

        This transformation is primarily used when constructing masks and
        broadcasting dynamic slice bounds.

        Parameters
        ----------
        expr : ast.AST
            Expression to broadcast.
        expr_dims : Sequence[str]
            Dimensions currently represented by *expr*.
        target_dims : Sequence[str]
            Desired dimensional structure.

        Returns
        -------
        ast.Subscript
            Broadcasted indexing expression matching the target rank.

        Notes
        -----
        This method is used by :meth:`create_mask` to align dynamic slice
        bounds with the dimensionality of the target array.

        Inserted ``None`` indices correspond to NumPy/JAX broadcasting
        semantics.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during shape
            expansion.
        """
        try:
            elts = []
            for dim in target_dims:
                if dim in expr_dims:
                    elts.append(ast.Slice())
                else:
                    elts.append(ast.Constant(value=None))
            return ast.Subscript(
                value=expr,
                slice=ast.Tuple(elts=elts, ctx=ast.Load()),
                ctx=ast.Load()
            )
        except Exception as e:
            self.logger.exception('Exception in expand_to_full_shape:', e)
            raise