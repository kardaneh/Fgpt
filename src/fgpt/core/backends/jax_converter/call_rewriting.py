# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import ast

from fgpt.core.backends.utils import collect_reads_before_def, contains_name, get_name


class _CallRewriting:
    """
    Rewrites calls between transformed methods and finalises
    helper-function bodies and return statements.

    Composes onto ``JaxConverter`` to handle everything that happens at
    a method-call boundary once individual statements have already
    been lowered by the other mixins:

    - :meth:`visit_Call` intercepts calls to sibling methods that have
      already been (or will be) vectorised, wrapping them in
      ``jax.vmap(..., in_axes=...)`` when the call-site argument ranks
      exceed what the callee's own input-dimension metadata expects;
      it also finalises any pending ``dynamic_variable_lift`` subscript
      corrections that were registered before the lift was known about
      at the call site, and delegates reduction-call rewriting
      (``np.sum``/``mean``/etc.) to :attr:`reduction`.
    - :meth:`visit_Expr` handles *bare* (non-assigned) method calls —
      ``self.other_method(...)`` as a statement — by consulting
      :attr:`ret_per_func` to learn which arguments/attributes the
      callee modifies and returns, remapping declared parameter names
      back to the actual call-site argument names, then emitting the
      unpacking assignment plus, where needed, an ``eqx.tree_at(...)``
      update of ``self`` for any modified attributes (with
      :meth:`_is_vectorized_child_call` and
      :meth:`_emit_scan_wrapped_call` handling the case where the
      callee must be wrapped in a ``lax.scan`` rather than called
      directly because it sits inside a vectorised loop).
    - :meth:`visit_Attribute` rewrites ``self.attr`` references to bare
      ``attr`` names inside synthetic helper functions, when *attr* is
      part of the helper's declared input/output set.
    - :meth:`process_helpers` drains :attr:`_pending_helpers`,
      recursively visiting each synthetic helper body (pushing a fresh
      :attr:`_context_stack` / :attr:`_func_arg_stack` entry per helper
      so nested helper generation resolves scoping correctly) until no
      pending helpers remain, appending each finished helper to
      :attr:`helpers`.
    - :meth:`add_return_stmt` and :meth:`analyze_function_statefulness`
      close out a single function's transformation: the former builds
      the function's final ``return`` (a raw tuple for
      callee-of-another-method functions, or an ``eqx.tree_at(...)``
      pytree update for top-level entry points), and the latter runs a
      use-before-def analysis over the function body up front so that
      :meth:`visit_Assign` and the vectorisation mixin can correctly
      classify variables as ``'stateful'`` versus ``'temporary'``.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Transform function-call expressions and apply vectorization-aware
        rewrites.

        This visitor performs several call-site transformations during AST
        lowering. Child-function invocations recorded in :attr:`call_edge`
        are analysed to determine whether additional vectorization is
        required. When a callee receives inputs with higher-rank shapes than
        expected, the call is automatically wrapped in :func:`jax.vmap`
        using axes inferred from the active control-flow context.

        The visitor also updates dynamically lifted array indexing metadata
        recorded in :attr:`dynamic_variable_lift`, ensuring that lifted
        dimensions are reinserted into generated slice expressions before
        reduction processing.

        After vectorization and lifting rewrites have been applied, the call
        is delegated to :attr:`reduction` for reduction-specific
        transformations.

        Parameters
        ----------
        node : ast.Call
            Call expression node to transform.

        Returns
        -------
        ast.AST
            The transformed call node. This may be the original call,
            a :func:`jax.vmap` wrapped invocation, or a reduction-transformed
            call returned by :meth:`reduction.process_call`.

        Notes
        -----
        Vectorization decisions are based on call-graph metadata stored in
        :attr:`call_edge`, argument shape information recorded in
        ``edge.arg_shapes``, and active loop metadata obtained from
        :attr:`_control_stack`.

        When automatic vectorization is required, the generated call has the
        form::

            jax.vmap(callee, in_axes=...)(...)

        Dynamic variable lifting uses metadata from
        :attr:`dynamic_variable_lift` to reconstruct omitted dimensions in
        lifted indexing expressions.

        Prior to invoking :meth:`reduction.process_call`, the current
        transformation state is synchronized with :attr:`reduction`,
        including:

        * :attr:`func_name`
        * :attr:`func_input_dim`
        * active vectorization axes
        * active loop metadata

        This method serves as the primary integration point between
        call-graph analysis, vectorization lowering, dynamic lifting, and
        reduction processing.

        See Also
        --------
        :meth:`get_active_dims`
            Determines active vectorized dimensions for call arguments.

        :meth:`visit_Expr`
            Rewrites child-function invocations and mutation propagation.

        :attr:`call_edge`
            Stores caller-callee relationships and argument metadata.

        :attr:`dynamic_variable_lift`
            Records lifted array dimensions requiring reconstruction.

        :meth:`reduction.process_call`
            Performs reduction-specific call transformations.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during call
            transformation.
        """
        try:
            node = self.generic_visit(node)

            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            # This is being done due to the fact that the functions(callees)
            # might already be transformed and/or
            # used by another caller functions thus we use vmap
            callees = {edge.callee for edge in self.call_edge[self.func_name]}
            needs_vmap = False
            in_axes = []
            if (
                vectorization_context
                and vectorization_context.get("vectorization_axis")
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in callees
            ):
                loop_info = (
                    vectorization_context.get("loop_info", {})
                    if vectorization_context
                    else {}
                )

                for edge in self.call_edge[self.func_name]:
                    # Skip edges that don't match the current call
                    if edge.callee != node.func.attr:
                        continue

                    needs_vmap = False  # reset per edge
                    in_axes = []
                    fn_input_dim = {}

                    call_node = edge.call_node
                    for i, arg in enumerate(call_node.args):
                        active_dims = self.get_active_dims(arg)
                        if active_dims:
                            fn_input_dim[edge.func_args[i]] = active_dims

                    if edge.arg_shapes:
                        for arg_name, actual_shape in edge.arg_shapes.items():
                            expected_shape = fn_input_dim.get(arg_name, [])
                            if len(actual_shape) > len(expected_shape):
                                needs_vmap = True
                                for i, sh in enumerate(actual_shape):
                                    if sh in loop_info and len(in_axes) < len(
                                        node.args
                                    ):
                                        in_axes.append(i)

                    if needs_vmap:
                        vmap_call = ast.Call(
                            func=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="jax", ctx=ast.Load()),
                                    attr="vmap",
                                    ctx=ast.Load(),
                                ),
                                args=[node.func],
                                keywords=[
                                    ast.keyword(
                                        arg="in_axes",
                                        value=ast.Tuple(
                                            elts=[
                                                ast.Constant(value=el) for el in in_axes
                                            ],
                                            ctx=ast.Load(),
                                        ),
                                    )
                                ],
                            ),
                            args=node.args,
                            keywords=[],
                        )
                        node = vmap_call
                    break

            # This is usually done inside visit_subscript
            # but in some case the lifted variables might be
            # intialized prior to the lifting thus we loop back for the lifting
            if getattr(self, "dynamic_variable_lift", None):
                for key, values in self.dynamic_variable_lift.items():
                    is_present = (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "set"
                        and isinstance(node.func.value, ast.Subscript)
                    ) and contains_name(node.func.value, key)
                    if is_present:
                        batched_axes = values["batched_axis"]
                        original_shape = values["original_shape"]
                        sub = node.func.value.slice
                        vect_loop = values["vectorized_loop"]
                        if isinstance(sub, ast.Slice):
                            sub_elts = [sub]
                        elif isinstance(sub, ast.Tuple):
                            sub_elts = list(sub.elts)
                        else:
                            sub_elts = [sub]

                        for ax in sorted(batched_axes):
                            vect_value = (
                                vect_loop[ax].id
                                if isinstance(vect_loop[ax], ast.Name)
                                else vect_loop[ax].attr
                            )
                            if original_shape and vect_value not in original_shape:
                                sub_elts.insert(ax, ast.Slice())
                            else:
                                if (
                                    original_shape
                                    and vect_value in original_shape
                                    and len(original_shape) != len(sub_elts)
                                ):
                                    sub_elts.insert(ax, ast.Slice())

                        if len(sub_elts) == 1:
                            node.func.value.slice = sub_elts[0]
                        else:
                            node.func.value.slice = ast.Tuple(
                                elts=sub_elts, ctx=ast.Load()
                            )

            self.reduction.dynamic_variable_lift = self.dynamic_variable_lift

            self.reduction.func_name = (
                self.func_name if hasattr(self, "func_name") else None
            )
            self.reduction.vectorization_axis = (
                vectorization_context["vectorization_axis"]
                if vectorization_context
                else {}
            )
            self.reduction.loop_info = (
                vectorization_context.get("loop_info", {})
                if vectorization_context
                else {}
            )
            self.reduction.func_input_dim = self.func_input_dim
            new_node = self.reduction.process_call(node)

            if new_node:
                node = ast.copy_location(new_node, node)

            return node
        except Exception as e:
            self.logger.exception("Exception in visit_Call:", e)
            raise

    def _is_vectorized_child_call(self, node: ast.Expr) -> bool:
        """
        Determine whether a function call should be lowered into a scan-based
        vectorized invocation.

        A call is considered a vectorized child call when all of the
        following conditions hold:

        * The expression consists of a method call of the form
        ``self.<method>(...)``.
        * The called method is a known child of the current function in
        :attr:`call_edge`.
        * The current control-flow context contains an active vectorization
        axis recorded in :attr:`_control_stack`.
        * At least one argument of the call corresponds to a loop variable
        associated with the active vectorization axis.

        Parameters
        ----------
        node : ast.Expr
            Expression node to analyse.

        Returns
        -------
        bool
            ``True`` if the call should be transformed into a
            :func:`lax.scan`-based vectorized invocation, otherwise
            ``False``.

        Notes
        -----
        Vectorization metadata is obtained from the active control-flow
        context stored in :attr:`_control_stack`.

        The method compares loop variables recorded in ``loop_info`` against
        entries in ``vectorization_axis`` and verifies that the vectorized
        loop variable is passed as an argument to the child function.

        This method is used by :meth:`visit_Expr` to determine whether a
        child-function call should be lowered through
        :meth:`_emit_scan_wrapped_call`.

        See Also
        --------
        :meth:`visit_Expr`
            Rewrites child-function calls and mutation propagation.

        :meth:`_emit_scan_wrapped_call`
            Generates the scan-based implementation for vectorized child
            calls.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during analysis.
        """
        try:
            # 1. RHS must be a Call
            rhs = node.value
            if not isinstance(rhs, ast.Call):
                return False

            # Must be self.method(...) form
            if not (
                isinstance(rhs.func, ast.Attribute)
                and isinstance(rhs.func.value, ast.Name)
                and rhs.func.value.id == "self"
            ):
                return False

            callee_name = rhs.func.attr

            # 2. Callee must be a known child of current function in call_edge
            current_edges = self.call_edge.get(self.func_name, [])
            callee_names = {edge.callee for edge in current_edges}
            if callee_name not in callee_names:
                return False

            # 3. Control stack must have an active vectorization axis
            if not self._control_stack:
                return False

            ctx = self._control_stack[-1].to_dict()

            loop_info = ctx.get("loop_info", {})
            vectorization_axis = ctx.get("vectorization_axis", {})

            # Check that at least one loop_info entry maps to a vectorized axis and is
            # present inside the arguments of the function
            arg_names = [get_name(n) for n in rhs.args]
            for dim, loop_var in loop_info.items():
                if loop_var in vectorization_axis and loop_var in arg_names:
                    return True

            return False
        except Exception as e:
            self.logger.exception("Exception in _is_vectorized_child_call:", e)
            raise

    def _emit_scan_wrapped_call(
        self,
        new_assign: ast.Assign,
        vectorization_context: dict,
        callee_name: str,
        new_call: ast.Call,
        modified_vars: list[str],
        func_args: list[str],
        actual_args: list[str],
    ) -> ast.Assign | list[ast.AST]:
        """
        Lower a vectorized child-function call into a :func:`lax.scan`
        transformation.

        This method generates an inner scan-body function that executes the
        original child call for a single vectorized iteration and propagates
        mutated state through the scan carry. The resulting helper function
        is then wrapped in a :func:`lax.scan` invocation over the active
        vectorization dimension.

        Modified variables are treated as scan carry values and are threaded
        through each iteration of the generated scan body. Upon completion,
        the final carry values are unpacked back into the corresponding
        variables in the enclosing scope.

        Parameters
        ----------
        new_assign : ast.Assign
            Assignment generated for the transformed child-function call.
        vectorization_context : dict
            Active vectorization metadata extracted from
            :attr:`_control_stack`.
        callee_name : str
            Name of the child function being transformed.
        new_call : ast.Call
            Rewritten function-call node.
        modified_vars : list[str]
            Variables returned by the child function that must be propagated
            through scan carry state.
        func_args : list[str]
            Formal parameter names of the child function.
        actual_args : list[str]
            Argument names supplied at the call site.

        Returns
        -------
        ast.Assign or list[ast.AST]
            The original assignment when no active vectorization dimension
            is found, otherwise a sequence of AST nodes consisting of a
            generated scan-body function and the corresponding
            :func:`lax.scan` invocation.

        Notes
        -----
        The vectorized dimension is obtained from the ``loop_info`` and
        ``vectorization_axis`` entries in the supplied
        ``vectorization_context``.

        The generated scan body has the form::

            def _scan_body_<callee>(carry, loop_var):
                ...
                return carry_out, None

        Iteration indices are generated using ``jnp.arange`` over the
        vectorized model dimension stored on ``self``.

        State propagation is implemented using scan carry variables derived
        from ``modified_vars``.

        This method is used by :meth:`visit_Expr` when
        :meth:`_is_vectorized_child_call` identifies a vectorized child
        invocation.

        See Also
        --------
        :meth:`_is_vectorized_child_call`
            Detects calls that require scan-based lowering.

        :meth:`visit_Expr`
            Entry point for call transformation.

        :attr:`_control_stack`
            Stores active control-flow and vectorization metadata.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during AST
            generation.
        """
        try:
            loop_info = (vectorization_context or {}).get("loop_info", {})

            vect_dim = None
            loop_var = None
            for dim, lvar in loop_info.items():
                vect_axis = (vectorization_context or {}).get("vectorization_axis", {})
                if lvar in vect_axis and vect_axis[lvar]:
                    vect_dim = dim
                    loop_var = lvar
                    break

            if vect_dim is None:
                return new_assign

            inner_func_name = f"_scan_body_{callee_name}"

            carry_in_elts = [ast.Name(id=v, ctx=ast.Load()) for v in modified_vars]
            carry_out_elts = [
                ast.Name(id=f"{v}_out", ctx=ast.Store()) for v in modified_vars
            ]
            carry_ret_elts = [
                ast.Name(id=f"{v}_out", ctx=ast.Load()) for v in modified_vars
            ]

            new_call_args = []
            for arg in new_call.args:
                arg_name = (
                    arg.id
                    if isinstance(arg, ast.Name)
                    else (arg.attr if isinstance(arg, ast.Attribute) else None)
                )
                if arg_name == loop_var:
                    new_call_args.append(ast.Name(id=loop_var, ctx=ast.Load()))
                else:
                    new_call_args.append(arg)

            inner_call = ast.Call(
                func=new_call.func, args=new_call_args, keywords=new_call.keywords
            )

            inner_assign = ast.Assign(
                targets=[ast.Tuple(elts=carry_out_elts, ctx=ast.Store())],
                value=inner_call,
            )

            inner_return = ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Tuple(elts=carry_ret_elts, ctx=ast.Load()),
                        ast.Constant(value=None),
                    ],
                    ctx=ast.Load(),
                )
            )

            carry_unpack = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=v, ctx=ast.Store()) for v in modified_vars],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Name(id="carry", ctx=ast.Load()),
            )

            inner_func = ast.FunctionDef(
                name=inner_func_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="carry"), ast.arg(arg=loop_var)],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[carry_unpack, inner_assign, inner_return],
                decorator_list=[],
                returns=None,
            )

            scan_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="lax", ctx=ast.Load()),
                    attr="scan",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id=inner_func_name, ctx=ast.Load()),
                    ast.Tuple(elts=carry_in_elts, ctx=ast.Load()),
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="arange",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=vect_dim,
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )

            outer_assign = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=v, ctx=ast.Store())
                                    for v in modified_vars
                                ],
                                ctx=ast.Store(),
                            ),
                            ast.Name(id="_", ctx=ast.Store()),
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=scan_call,
            )

            return [inner_func, outer_assign]
        except Exception as e:
            self.logger.exception("Exception in _emit_scan_wrapped_call:", e)
            raise

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        """
        Transform expression statements containing calls to tracked helper
        functions.

        This visitor rewrites function-call expressions whose callees appear
        in the call graph recorded in :attr:`call_edge`. When a callee
        returns modified arguments or attributes, the call is converted into
        an assignment that captures the returned values and propagates
        mutations through the transformed program state.

        Modified object attributes are synchronized back onto ``self`` using
        :func:`eqx.tree_at`. When the call occurs inside a vectorized control
        context, the transformation may additionally emit scan- or
        vectorization-specific statements through
        :meth:`_emit_scan_wrapped_call`.

        The visitor also tracks returned mutations through
        :attr:`_modified_ret_stack`, updates mutation metadata in
        :attr:`_mutated_attrs` and :attr:`_var_modif`, and records carry
        dependencies for active scan contexts.

        Parameters
        ----------
        node : ast.Expr
            Expression node to transform.

        Returns
        -------
        ast.AST or list[ast.AST]
            The original node if no transformation is required, a rewritten
            assignment node, or a sequence of statements when additional
            synchronization or vectorization logic must be emitted.

        Notes
        -----
        Return metadata is obtained from :attr:`ret_per_func` and is used to
        determine which variables must be captured from the callee.

        Actual call arguments are mapped back to formal parameters using
        method metadata stored in :attr:`cls_info`.

        Attribute synchronization is performed using :func:`eqx.tree_at`
        whenever modified attributes must be propagated back onto ``self``.

        This method cooperates closely with :meth:`add_return_stmt` and
        :meth:`visit_Attribute` during helper-function lowering.

        See Also
        --------
        :meth:`add_return_stmt`
            Generates return statements for transformed helper functions.

        :meth:`visit_Attribute`
            Rewrites accesses to mutated attributes.

        :meth:`_emit_scan_wrapped_call`
            Generates scan-aware call transformations.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during AST
            transformation.
        """
        try:
            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            # Fast exit if nothing to do
            if not self.call_edge:
                return node

            methods = self.cls_info[self.cls_name].get("methods")
            attributes = self.cls_info[self.cls_name].get("attributes")
            value = node.value
            if not isinstance(value, ast.Call):
                return node

            func = value.func
            if not isinstance(func, ast.Attribute):
                return node

            callee_name = func.attr
            callees = {edge.callee for edge in self.call_edge[self.func_name]}

            if callee_name not in callees:
                return node

            ret_info = self.ret_per_func.get(callee_name)
            if not ret_info:
                return node

            # Function arguments
            func_args = methods.get(callee_name).get("args")
            # Actual arguments list
            actual_args = [
                arg.id if isinstance(arg, ast.Name) else arg.attr for arg in value.args
            ]
            # NOTE:
            # The arguments passed at the call site may differ from the function’s
            # formal parameters. In such cases, we use both the actual arguments and
            # the function signature arguments to construct a mapping. This allows
            # `ret_info["var_modif_args"]` (which is expressed in terms of formal
            # parameters) to be translated back to the corresponding actual arguments
            # at the call site, thereby identifying the true variables that were
            # modified and returned.
            mapped_args = []

            for arg in ret_info["var_modif_args"]:
                if arg in func_args:
                    index = func_args.index(arg)
                    if index < len(actual_args):
                        mapped_args.append(actual_args[index])
                    else:
                        mapped_args.append(arg)
                else:
                    mapped_args.append(arg)

            if mapped_args:
                ret_info["var_modif_args"] = mapped_args

            # Build tuple assignment for modified variables
            modified_vars = list(ret_info["var_modif_args"]) + list(
                ret_info["var_modif_attr"]
            )
            # NOTE:
            # These variables must be propagated because `var_modif_args` only
            # captures arguments modified within the function, but those arguments
            # may correspond to attributes originally passed into the function.
            # Additionally, other attributes may have been modified by earlier
            # function calls, and in such cases the actual arguments and formal
            # parameters may differ.

            # This also accounts for cases where the data originates from local
            # arrays rather than instance attributes, requiring explicit tracking
            # to ensure all modified values are correctly propagated.
            # current_modified = [vars for vars in modified_vars if vars in attributes]
            if self._modified_ret_stack:
                self._modified_ret_stack[-1].extend(modified_vars)
            else:
                self._modified_ret_stack.append(modified_vars)

            tuple_targets = [ast.Name(id=var, ctx=ast.Store()) for var in modified_vars]

            # Update self using eqx.tree_at for modified attributes
            # BUT also in somes cases the args that we sent are modified and returned but these args might be attributes as well
            # thus we need to ensure that these are also taken into account
            var_modif_attr = list(ret_info["var_modif_attr"]) + [
                vars for vars in ret_info["var_modif_args"] if vars in attributes
            ]

            # NOTE:
            # We need to adjust this step because arguments passed to a function may
            # have been modified as attributes in a previous function. Therefore, we
            # must retrieve and reconcile values using the current function’s variable
            # names to determine whether the arguments being passed have already been
            # updated as attributes.
            previously_modified_vars = set()
            for callee, ret_info in self.ret_per_func.items():
                # Only consider callees that appear before or are relevant to current function
                previously_modified_vars.update(ret_info.get("var_modif_attr", []))

            new_args = []
            for arg in value.args:  # iterate over the AST args of the call
                if (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "self"
                ):
                    attr_name = arg.attr
                    if attr_name in previously_modified_vars:
                        # Previously modified, pass as Name instead of self.attr
                        new_args.append(ast.Name(id=attr_name, ctx=ast.Load()))
                    else:
                        # keep as self.attr
                        new_args.append(arg)
                else:  # Rest of the args
                    new_args.append(arg)

            new_call = ast.Call(func=value.func, args=new_args, keywords=value.keywords)
            new_assign = ast.Assign(
                targets=[ast.Tuple(elts=tuple_targets, ctx=ast.Store())], value=new_call
            )

            if vectorization_context and self._is_vectorized_child_call(node):
                vmap_stmts = self._emit_scan_wrapped_call(
                    new_assign,
                    vectorization_context,
                    callee_name=callee_name,
                    new_call=new_call,
                    modified_vars=modified_vars,
                    func_args=func_args,
                    actual_args=actual_args,
                )
                if not var_modif_attr:
                    return vmap_stmts

                new_assign = vmap_stmts

            if not var_modif_attr:
                return new_assign

            lambda_body_elts = [
                ast.Attribute(
                    value=ast.Name(id="m", ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                )
                for attr in var_modif_attr
            ]

            tree_at_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="eqx", ctx=ast.Load()),
                    attr="tree_at",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg="m")],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=ast.Tuple(elts=lambda_body_elts, ctx=ast.Load()),
                    ),
                    ast.Name(id="self", ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Name(id=a, ctx=ast.Load()) for a in var_modif_attr],
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            )

            self_assign = ast.Assign(
                targets=[ast.Name(id="self", ctx=ast.Store())],
                value=tree_at_call,
            )
            if self._modified_ret_stack:
                self._modified_ret_stack[-1].append("self")

            fn_node = self.fn_index[callee_name]
            prev_attr_used = set(
                [self.to_arg(value) for value in self._first_reads(fn_node.body)]
            )
            # Attributes that are read before being modified inside the function.
            # These are used early in the function body and are not directly modified at that point.

            # NOTE:
            # In some cases, attributes may already be used before the function call,
            # and then used again inside the function where they are modified and returned.
            # Since we know these attributes will be modified and returned, we can safely
            # ensure they are updated on `self` before the function call.
            if (self._mutated_attrs | self._var_modif["attr"]) & (
                prev_attr_used | set(var_modif_attr)
            ) - {"self"}:
                pre_self_attributes = (
                    self._mutated_attrs | self._var_modif["attr"]
                ) & (prev_attr_used | set(var_modif_attr)) - {"self"}
                # These attributes must be updated on `self` before calling the function.
                # This includes:
                # - Attributes already known to be mutated, and attributes that are read early in the function but were modified previously.
                # Not all attributes in `var_modif_attr` are required, only those that intersect
                # with already mutated attributes or previously-used attributes.
                lambda_body_elts = [
                    ast.Attribute(
                        value=ast.Name(id="m", ctx=ast.Load()),
                        attr=attr,
                        ctx=ast.Load(),
                    )
                    for attr in pre_self_attributes
                ]

                tree_at_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="eqx", ctx=ast.Load()),
                        attr="tree_at",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[ast.arg(arg="m")],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                            ),
                            body=ast.Tuple(elts=lambda_body_elts, ctx=ast.Load()),
                        ),
                        ast.Name(id="self", ctx=ast.Load()),
                        ast.Tuple(
                            elts=[
                                ast.Name(id=a, ctx=ast.Load())
                                for a in pre_self_attributes
                            ],
                            ctx=ast.Load(),
                        ),
                    ],
                    keywords=[],
                )

                pre_self_assign = ast.Assign(
                    targets=[ast.Name(id="self", ctx=ast.Store())],
                    value=tree_at_call,
                )
                return_set = [pre_self_assign, new_assign, self_assign]
            else:
                return_set = [new_assign, self_assign]

            current_modified = (
                set(self._modified_ret_stack[-1]) if self._modified_ret_stack else set()
            )
            if self._scan_stack:
                ctx = self._scan_stack[-1]
                ctx["mutated"].update(current_modified)
                for var in list(current_modified | set(modified_vars)):
                    if var not in ctx["carry"]:
                        ctx["introduced"].add(var)

            elif not self._modified_ret_stack or len(self._modified_ret_stack) == 1:
                self._mutated_attrs |= current_modified
                if current_modified:
                    current_modified.remove("self")
                current_modified = [
                    vars for vars in current_modified if vars in attributes
                ]
                self._var_modif["attr"].update(set(current_modified))

            return return_set

        except Exception as e:
            self.logger.exception("Exception in visit_Expr:", e)
            raise

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """
        Rewrite accesses to mutable object attributes.

        References of the form ``self.<attr>`` are replaced by local variable
        accesses when the attribute is known to participate in mutation
        propagation. This allows transformed helper functions to operate on
        explicit state variables rather than directly accessing object
        attributes.

        Attributes are rewritten when they are tracked as mutated in
        :attr:`_mutated_attrs`, recorded in :attr:`_var_modif`, introduced
        through helper-function arguments, or propagated through
        :attr:`_modified_ret_stack`.

        Parameters
        ----------
        node : ast.Attribute
            Attribute access node to analyse.

        Returns
        -------
        ast.AST
            A rewritten :class:`ast.Name` node when the attribute should be
            treated as local state, otherwise the original attribute access
            after recursive visitation.

        Notes
        -----
        This transformation is primarily used for synthetic helper
        functions generated during control-flow lowering.

        Rewritten attributes are expected to be supplied explicitly as
        helper-function inputs and outputs.

        This method relies on mutation metadata maintained in
        :attr:`_mutated_attrs`, :attr:`_modified_ret_stack`, and
        :attr:`_func_arg_stack`.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during visitation.
        """
        try:
            # Only rewrite inside helper functions
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                attr = node.attr
                # If this attribute is mutated anywhere in the function, ALWAYS rewrite
                # `self.attr` as `attr` inside helper functions.
                # These helper functions receive the variable as an explicit input, so
                # instance access is no longer required.
                # This applies to variables introduced within the scope of loops or
                # conditionals that originate from modified return values of call
                # functions (tracked in `self.modified_ret_stack`).
                if (
                    attr in self._mutated_attrs
                    and attr in self._var_modif["attr"]
                    or (self._func_arg_stack[-1] and attr in self._func_arg_stack[-1])
                    or (
                        self._modified_ret_stack
                        and attr in self._modified_ret_stack[-1]
                    )
                ):
                    return ast.copy_location(ast.Name(id=attr, ctx=node.ctx), node)

            return self.generic_visit(node)
        except Exception as e:
            self.logger.exception("Exception in visit_Attribute:", e)
            raise

    def process_helpers(self) -> None:
        """
        Process deferred helper functions until all nested transformations
        have been resolved.

        Synthetic helper functions generated during control-flow lowering are
        revisited under a fresh scope and transformation context. Each
        helper body is recursively transformed, allowing nested helpers,
        mutation propagation, attribute rewriting, and call-graph-based
        rewrites to be applied until no pending helpers remain.

        For each helper, a context describing the current nesting level,
        helper arguments, and previously known mutations is pushed onto
        :attr:`_context_stack`. Scope information is maintained through
        :meth:`_push_scope` and :meth:`_pop_scope`.

        Notes
        -----
        Each helper context contains:

        * ``level`` – current helper nesting depth.
        * ``helper_args`` – argument names available within the helper.
        * ``mutated_before`` – snapshot of
        :attr:`_mutated_attrs` before processing.

        Helper argument information is recorded in
        :attr:`_func_arg_stack` and used by :meth:`visit_Attribute`
        when rewriting mutable state references.

        Processed helpers are normalized with
        :func:`ast.fix_missing_locations` before being appended to
        :attr:`helpers`.

        This method repeatedly consumes helper functions from
        :attr:`_pending_helpers` until a fixed point is reached.

        See Also
        --------
        :meth:`visit_Expr`
            Rewrites helper-function calls and mutation propagation.

        :meth:`visit_Attribute`
            Rewrites mutable attribute accesses within helper bodies.

        :meth:`_push_scope`
            Creates a new transformation scope.

        :meth:`_pop_scope`
            Restores the previous transformation scope.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during helper
            processing.
        """
        try:
            while self._pending_helpers:
                helper = self._pending_helpers.pop(0)
                helper_arg_names = {a.arg for a in helper.args.args}
                self._push_scope()
                if "arg" in helper_arg_names:
                    arg_unpack = helper.body[0]
                    helper_arg_names = set()
                    for node in arg_unpack.targets[0].elts:
                        helper_arg_names.add(node.id)

                mutated_before = set(self._mutated_attrs)
                level = len(self._func_arg_stack)
                ctx = {
                    "level": level,
                    "helper_args": helper_arg_names,
                    "mutated_before": mutated_before,
                }
                self._context_stack.append(ctx)
                self._func_arg_stack.append(helper_arg_names)

                new_body = []
                for stmt in helper.body:
                    transformed = self.visit(stmt)
                    if isinstance(transformed, list):
                        new_body.extend(transformed)
                    elif transformed is None:
                        continue
                    else:
                        new_body.append(transformed)

                helper.body = new_body

                self._func_arg_stack.pop()
                self._context_stack.pop()
                self._pop_scope()  # This belongs the reduction parent stack
                self.helpers.append(ast.fix_missing_locations(helper))

        except Exception as e:
            self.logger.exception("Exception in process_helpers:", e)
            raise

    def add_return_stmt(self, var_modif_args: set, var_modif_attr: set) -> ast.Return:
        """
        Construct the return statement for the current function.

        The generated return value depends on the function's position in the
        call graph. Functions that are invoked by another function return a
        raw tuple containing modified arguments and attributes so that the
        caller can propagate state changes. Top-level functions instead
        return updated model attributes through :func:`eqx.tree_at` together
        with any modified arguments.

        The sets of modified arguments and attributes are also recorded in
        :attr:`ret_per_func` for later analysis.

        Parameters
        ----------
        var_modif_args : set
            Names of function arguments modified within the current function.
        var_modif_attr : set
            Names of object attributes modified within the current function.

        Returns
        -------
        ast.Return
            Return statement node representing the transformed function
            output.

        Notes
        -----
        Call graph information is obtained from :attr:`call_edge` to
        determine whether the current function has a parent caller.

        When returning model attributes from a top-level function, the
        returned value is constructed using :func:`eqx.tree_at` to produce
        an updated instance of ``self``.

        This method updates :attr:`ret_per_func` and is typically used
        during function transformation after variable modification analysis.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during AST
            construction.
        """
        try:
            self.ret_per_func[self.func_name] = {
                "var_modif_args": var_modif_args,
                "var_modif_attr": var_modif_attr,
            }

            edges = self.call_edge
            # has_children = bool(edges.get(self.func_name, []))
            has_parent = any(
                edge.callee == self.func_name
                for edge_list in edges.values()
                for edge in edge_list
            )
            # no children AND parent -> raw tuple
            # (calling children funciton inside parent function)
            if has_parent:
                elts = [
                    ast.Name(id=v, ctx=ast.Load())
                    for v in list(var_modif_args) + list(var_modif_attr)
                ]
                return ast.Return(value=ast.Tuple(elts=elts, ctx=ast.Load()))

            lambda_elts = [
                ast.Attribute(
                    value=ast.Name(id="m", ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                )
                for attr in var_modif_attr
            ]

            tree_at_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="eqx", ctx=ast.Load()),
                    attr="tree_at",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg="m")],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=ast.Tuple(elts=lambda_elts, ctx=ast.Load()),
                    ),
                    ast.Name(id="self", ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Name(id=a, ctx=ast.Load()) for a in var_modif_attr],
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            )

            arg_elts = [ast.Name(id=arg, ctx=ast.Load()) for arg in var_modif_args]

            if arg_elts:
                return ast.Return(
                    value=ast.Tuple(elts=arg_elts + [tree_at_call], ctx=ast.Load())
                )

            return ast.Return(value=tree_at_call)

        except Exception as e:
            self.logger.exception("Exception in add_return_stmt:", e)
            raise

    def analyze_function_statefulness(self, func_node: ast.FunctionDef) -> None:
        """
        Analyse variable statefulness within a function body.

        Performs a use-before-definition analysis over the supplied function
        body to identify variables whose values depend on previous
        iterations or external state. The resulting set of variables is
        pushed onto :attr:`_stateful_vars_stack` for later processing during
        transformation.

        Variables considered already defined include:

        * Loop index variables recorded in :attr:`index_variables`.
        * Variables from enclosing scopes recorded in
        :attr:`outer_scope_vars`.
        * Function parameters.
        * The ``self`` reference.

        Parameters
        ----------
        func_node : ast.FunctionDef
            Function definition node to analyse.

        Notes
        -----
        The initial definition set is constructed from
        :attr:`index_variables`, :attr:`outer_scope_vars`, function
        arguments, and ``self``.

        The analysis is delegated to :func:`collect_reads_before_def`, and
        the resulting stateful variable set is stored in
        :attr:`_stateful_vars_stack`.

        This method should be called when entering a function scope before
        visiting assignment statements.

        See Also
        --------
        :func:`collect_reads_before_def`
            Performs the underlying use-before-definition analysis.

        :attr:`_stateful_vars_stack`
            Stack used to track stateful variables across nested scopes.

        Raises
        ------
        Exception
            Re-raises any unexpected error encountered during analysis.
        """
        try:
            # Parameters and outer-scope vars are "already defined"
            already_defined = set(self.index_variables) | set(
                getattr(self, "outer_scope_vars", set())
            )
            for arg in func_node.args.args:
                already_defined.add(arg.arg)
            # Also mark any class attributes / self.x as defined
            already_defined.add("self")

            self._stateful_vars_stack.append(
                collect_reads_before_def(func_node.body, already_defined)
            )
        except Exception as e:
            self.logger.exception("Exception in analyze_function_statefulness:", e)
            raise
