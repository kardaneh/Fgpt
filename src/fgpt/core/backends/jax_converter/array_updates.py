# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import ast
import copy

from fgpt.core.backends.utils import contains_name, get_name


class _ArrayUpdate:
    """
    Rewrites in-place array mutations into JAX's immutable
    ``.at[...].<op>(...)`` update pattern.

    Composes onto ``JaxConverter`` to implement ``visit_Assign``, the
    single largest entry point in the converter: every subscript
    assignment (``arr[i] = v``, ``self.attr[i] += v``, …) and every
    plain-name/attribute assignment passes through here before any
    other mixin's logic runs, since :meth:`visit_Assign` is responsible
    for routing to dynamic-slice masking, vectorisation-axis
    broadcasting, dynamic-variable-lift registration, and final
    ``.at[]`` rewriting in the correct order.

    Three supporting concerns live alongside it:

    - :meth:`check_in_place_modif` classifies the assignment's RHS as
      one of ``set``/``add``/``subtract``/``multiply``/``divide``/
      ``power`` by detecting the in-place pattern ``a[i] = a[i] <op>
      expr`` (including one level of nested ``BinOp``), so
      :meth:`visit_Assign` can choose the matching ``.at[...].<op>(...)``
      call instead of always falling back to ``.set``.
    - :meth:`visit_Subscript` performs the companion read-side
      rewriting: inserting vectorisation-axis slices, pairwise
      ``arange`` indices, and dynamic-lift batch axes into subscript
      expressions so that reads of an array agree with the shape
      produced by the writes :meth:`visit_Assign` emits.

    :meth:`update_cls_dict` is the shared mutation point both
    :meth:`visit_Assign` and :meth:`visit_Subscript` call into when a
    local array's declared dimensions or dtype change as a result of
    dynamic lifting, keeping :attr:`cls_info` in sync with the
    transformations applied.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
        """
        Transform assignment statements during vectorization and JAX conversion.

        This visitor rewrites assignment expressions according to the current
        vectorization context and mutation-tracking rules. In particular, it
        converts indexed assignments into functional JAX-style updates using
        ``.at[...]`` operations and performs dependency analysis for variables
        created or modified inside vectorized control flow.

        The transformation supports:

        * Conversion of indexed assignments into ``.at`` update expressions.
        * Handling of in-place operators (e.g. ``+=``, ``*=``).
        * Pairwise indexing for vectorized assignments.
        * Broadcasting-aware index rewriting.
        * Dynamic slice and mask-based assignment transformations.
        * Tracking of stateful variables created within vectorized loops.
        * Registration of dynamically lifted variables and arrays.
        * Dependency analysis for local variables and attributes.

        Parameters
        ----------
        node : ast.Assign
            Assignment node to transform.

        Returns
        -------
        ast.AST | list[ast.AST]
            The transformed assignment node.

            Depending on the transformation, the return value may be:

            * A rewritten `ast.Assign`.
            * A list of statements when dynamic masking introduces
            prerequisite operations.
            * The original assignment node if no transformation is required.

        Notes
        -----
        Subscript assignments are converted from an imperative form::

            arr[i] = value

        into the equivalent functional update form::

            arr = arr.at[i].set(value)

        or the corresponding JAX update operator when an in-place
        modification is detected.

        During vectorization, assignments are analyzed to determine whether
        variables should be treated as temporary values or promoted to
        stateful variables. Stateful variables may later be dynamically
        lifted into batched arrays when required by vectorized execution.

        Dynamic slicing and masked assignments may introduce additional
        statements before the rewritten assignment.

        Side Effects
        ------------
        * Updates variable dependency information in :attr:`var_deps`.
        * Updates variable state information in :attr:`var_state`.
        * Registers modified arguments in :attr:`_var_modif`.
        * Registers modified attributes in :attr:`_var_modif`.
        * Updates :attr:`dynamic_variable_lift` for promoted variables.
        * Updates :attr:`_lifted_vars` for dynamically lifted arrays.
        * May update class metadata through :meth:`update_cls_dict`.
        * May add local variable definitions through :meth:`_add_local`.
        """
        if (
            node.targets
            and isinstance(node.targets[0], ast.Name)
            and not self.check_if_array(node)
        ):
            self._seen_assigned_names.add(node.targets[0].id)
        assign_vect = None
        if self._control_stack:
            assign_vect = self._control_stack[-1].to_dict()
            self._analyze_local_assignment(node, assign_vect)

        # NOTE:
        # The parent NodeTransformer.generic_visit dispatches to the parent
        # visit_Call implementation rather than the specialized transformer
        # defined in this class. Calls are therefore explicitly revisited here.
        if isinstance(node.value, ast.Call):
            node.value = self.visit_Call(node.value)
        elif isinstance(node.value, ast.BoolOp | ast.Compare | ast.UnaryOp):
            node.value = self._transform_if_test(node.value)

        if node.targets and isinstance(node.targets[0], ast.Subscript):
            sub = node.targets[0]
            operator, rhs = self.check_in_place_modif(node)
            node.value = rhs
            # This is to just remove the self in the case we have an instances of a[i] = a[i] + ...
            node.value = self.visit(node.value)
            dyn_slice = self.check_for_dynamic_slice(node.value)

            if dyn_slice:
                mask_stmts, node = self.handle_dynamic_slice(node)
            else:
                mask_stmts = []

            pairwise = False
            axis = None
            loop_info = None

            if assign_vect and assign_vect.get("vectorization_axis"):
                axis = list(assign_vect["vectorization_axis"].keys())
                loop_info = assign_vect["loop_info"]
                pairwise = self._rhs_is_pairwise(node.value)
                if assign_vect and pairwise:
                    if isinstance(sub, ast.Subscript):
                        self._force_subscript_pairwise(sub, axis, loop_info)

                elif assign_vect:
                    _ = self._apply_broadcasting_slice(sub, assign_vect)

            sub_modified = self.visit_Subscript(copy.deepcopy(sub))
            slice_value = sub.slice
            if isinstance(sub_modified, ast.Subscript):
                slice_value = sub_modified.slice
            else:
                vect = self._control_stack[-1].to_dict()
                vectorization_axis = vect.get("vectorization_axis")
                metadata = vect.get("metadata", {})
                loop_index = metadata.get("loop_index", None)

                if not vectorization_axis:
                    slice_value = sub.slice
                else:
                    axis = list(vectorization_axis.keys()) + (
                        [loop_index] if loop_index else []
                    )

                    # is_axis = lambda idx: isinstance(idx, ast.Name) and idx.id in axis
                    def is_axis(idx):
                        return isinstance(idx, ast.Name) and idx.id in axis

                    if isinstance(sub.slice, ast.Slice):
                        slice_value = sub.slice
                    elif isinstance(sub.slice, ast.Tuple):
                        sub.slice.elts = [
                            ast.Slice() if is_axis(node) else node
                            for node in sub.slice.elts
                        ]
                    elif isinstance(sub.slice, ast.Name) and is_axis(sub.slice):
                        sub.slice = ast.Slice()
                        slice_value = sub.slice

            new_assign = None
            if isinstance(sub.value, ast.Name):
                arr_name = sub.value.id
                if self._outer_func_args and arr_name in self._outer_func_args:
                    self._var_modif["args"].add(arr_name)

                at_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Subscript(
                            value=ast.Attribute(
                                value=ast.Name(id=arr_name, ctx=ast.Load()),
                                attr="at",
                                ctx=ast.Load(),
                            ),
                            slice=slice_value,
                            ctx=ast.Load(),
                        ),
                        attr=operator,
                        ctx=ast.Load(),
                    ),
                    args=[node.value],
                    keywords=[],
                )
                new_assign = ast.Assign(
                    targets=[ast.Name(id=arr_name, ctx=ast.Store())], value=at_call
                )

            if (
                isinstance(sub.value, ast.Attribute)
                and isinstance(sub.value.value, ast.Name)
                and sub.value.value.id == "self"
            ):
                arr_name = sub.value.attr
                if sub.value.attr in self._mutated_attrs:
                    at_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Subscript(
                                value=ast.Attribute(
                                    value=ast.Name(id=sub.value.attr, ctx=ast.Load()),
                                    attr="at",
                                    ctx=ast.Load(),
                                ),
                                slice=slice_value,
                                ctx=ast.Load(),
                            ),
                            attr=operator,
                            ctx=ast.Load(),
                        ),
                        args=[node.value],
                        keywords=[],
                    )
                else:
                    at_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Subscript(
                                value=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id="self", ctx=ast.Load()),
                                        attr=sub.value.attr,
                                        ctx=ast.Load(),
                                    ),
                                    attr="at",
                                    ctx=ast.Load(),
                                ),
                                slice=slice_value,
                                ctx=ast.Load(),
                            ),
                            attr=operator,
                            ctx=ast.Load(),
                        ),
                        args=[node.value],
                        keywords=[],
                    )
                self._var_modif["attr"].add(sub.value.attr)
                new_assign = ast.Assign(
                    targets=[ast.Name(id=sub.value.attr, ctx=ast.Store())],
                    value=at_call,
                )

            if mask_stmts:
                return mask_stmts + [ast.copy_location(new_assign, node)]
            elif new_assign:
                return ast.copy_location(new_assign, node)
            else:  # This is to handle these cases like : arr[i][mask] = ...
                if isinstance(sub, ast.Subscript):
                    base, slices = self._flatten_subscript(sub)
                    if len(slices) > 1:
                        return self.handle_masked_arrays(node, sub, base, slices)

        elif isinstance(node.targets[0], ast.Attribute | ast.Name):
            vectorization_context = None
            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()

            dyn_slice = self.check_for_dynamic_slice(node.value)
            if dyn_slice:
                mask_stmts, node = self.handle_dynamic_slice(node)
            else:
                mask_stmts = []

            if vectorization_context:
                deps_rhs = self._expr_depends_on_axes(
                    node.value, vectorization_context["vectorization_axis"]
                )

                target = node.targets[0]

                def _collect_names(expr):
                    names = set()
                    for subnode in ast.walk(expr):
                        if isinstance(subnode, ast.Name):
                            names.add(subnode.id)
                    return names

                if isinstance(target, ast.Name):
                    if not self.check_if_array(node):
                        var_name = target.id
                        rhs_names = _collect_names(node.value)
                        is_self_dependent = var_name in rhs_names
                        is_index_var = var_name in self.index_variables

                        # TODO: Need to handle the case where the deep_rhs is empty tuple
                        # due to teh fact the vectorization pushed it be empty thus
                        # the loop(vectorized) dependencies is not directly present
                        self.var_deps[target.id] = deps_rhs

                        is_control_flow_dependant = var_name in self._stateful_vars
                        if not is_index_var:
                            if is_self_dependent or is_control_flow_dependant:
                                if var_name in self.var_state:
                                    if self.var_state[var_name][0] == "temporary":
                                        self.var_state[var_name] = (
                                            "stateful",
                                            self.var_state[var_name][1],
                                        )
                                else:
                                    self.var_state[var_name] = ("stateful", node)
                                    # Here stateful especially in the context of vectorization
                                    # requires the promotion of that scalar to vector(array like)
                            else:
                                self.var_state.setdefault(var_name, ("temporary", node))

                        is_array = self.check_if_array(node)
                        shape = ()
                        if not is_array:
                            shape = self._infer_scalar_shape(assign_vect)

                        state = self.var_state.get(var_name)

                        is_stateful = state is not None and state[0] == "stateful"
                        origin_name = self._resolve_dynamic_origin(var_name)

                        is_dynamic_created = (
                            origin_name in self.var_state
                            and self.var_state.get(origin_name)[0] == "stateful"
                        )

                        if not self._is_local(target.id):
                            if shape:
                                self._add_local(target.id, shape)
                        else:
                            if is_stateful or is_dynamic_created:
                                if shape:
                                    self._local_defined_stack[-1][origin_name] = shape
                                attributes = self.cls_info[self.cls_name].get(
                                    "attributes"
                                )

                                # 2. register for dynamic lifting (if not already)
                                if origin_name not in self.dynamic_variable_lift:
                                    # get original shape if exists, otherwise scalar
                                    original_shape = []
                                    # current vectorization axis
                                    vectorization_axis = (
                                        vectorization_context or {}
                                    ).get("vectorization_axis")
                                    loop_info = (vectorization_context or {}).get(
                                        "loop_info"
                                    )
                                    batched_axes = {
                                        i
                                        for i, dim in enumerate(shape)
                                        if dim in loop_info
                                    }
                                    vect_loop = []
                                    for dim in shape:
                                        if dim in attributes:
                                            vect_loop.append(
                                                ast.Attribute(
                                                    value=ast.Name(
                                                        id="self", ctx=ast.Load()
                                                    ),
                                                    attr=dim,
                                                    ctx=ast.Load(),
                                                )
                                            )
                                        else:
                                            vect_loop.append(
                                                ast.Name(id=dim, ctx=ast.Load())
                                            )

                                    # register lift
                                    dtype = self.infer_dtype(
                                        self.var_state[origin_name][1].value
                                    )

                                    if dtype == "unknown":
                                        dtype = "float64"

                                    if origin_name not in self.dynamic_variable_lift:
                                        self.dynamic_variable_lift[origin_name] = {
                                            "original_shape": original_shape,
                                            "axis_map": 0,  # Since this adds a new dimension to scalar
                                            "batched_axis": batched_axes,
                                            "vectorized_loop": vect_loop,
                                            "dtype": dtype,
                                        }

                                        if (
                                            self.func_name,
                                            origin_name,
                                        ) not in self._lifted_vars:
                                            self._lifted_vars.add(origin_name)
                                            func_args = {
                                                "func_name": self.func_name,
                                                "local_arr": {
                                                    "var": origin_name,
                                                    "update_value": {
                                                        "dimensions": list(shape),
                                                        "dtype": dtype,
                                                        "type": "jnp.ndarray",
                                                    },
                                                },
                                            }
                                            self.update_cls_dict(
                                                method=True, func_args=func_args
                                            )

                elif isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        self.var_deps[target.attr] = deps_rhs

            node.value = self.generic_visit(node.value)
            if mask_stmts:
                return mask_stmts + [node]

            return node

        return node

    def check_in_place_modif(self, node: ast.Assign) -> tuple[str, ast.AST]:
        """
        Detect in-place array element updates that can be translated to JAX
        ``Array.at`` operations.

        This method identifies assignment patterns such as::

            a[i] = a[i] + x
            a[i] = a[i] - x
            a[i] = a[i] * x
            a[i] = a[i] / x
            a[i] = (a[i] + b) * c

        and converts them into an operation name and update expression suitable
        for translation into JAX's ``array.at[index].<op>()`` API. Assignments
        that do not match a supported in-place update pattern are treated as
        regular ``set`` operations:
        https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html

        Parameters
        ----------
        node : ast.Assign
            Assignment node to analyze.

        Returns
        -------
        tuple[str, ast.AST]
            A tuple ``(operation, value)`` where:

            - ``operation`` is one of ``{"add", "subtract", "multiply",
            "divide", "power", "set"}``.
            - ``value`` is the update expression associated with the operation.

            For unsupported patterns, returns ``("set", node.value)``.

        Notes
        -----
        Supported transformations include:

        - ``a[idx] = a[idx] + x`` -> ``("add", x)``
        - ``a[idx] = a[idx] - x`` -> ``("subtract", x)``
        - ``a[idx] = a[idx] * x`` -> ``("multiply", x)``
        - ``a[idx] = a[idx] / x`` -> ``("divide", x)``
        - ``a[idx] = (a[idx] + b) * c`` ->
        ``("add", b * c)``

        The method verifies that both sides reference the same indexed array
        location before classifying an assignment as an in-place update.
        """

        ops = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.Div: "divide",
            ast.Pow: "power",
        }

        try:
            sub = node.targets[0]
            if isinstance(sub.value, ast.Name):
                lhs_name = sub.value.id
            elif isinstance(sub.value, ast.Attribute):
                lhs_name = sub.value.attr
            else:
                return "set", node.value
            lhs_name = get_name(sub.value)

            # Only handle simple binary assignments: a[:] = a <op> expr
            if not isinstance(node.value, ast.BinOp):
                return "set", node.value

            rhs = node.value.left
            # in-place case: a[:] = a + expr
            same_index = (
                isinstance(sub.value, ast.Subscript)
                and isinstance(rhs, ast.Subscript)
                and get_name(sub.value) == get_name(rhs)
                and ast.dump(sub.value.slice) == ast.dump(rhs.slice)
            )
            if same_index:
                operator = ops.get(type(node.value.op))
                if operator:
                    return operator, node.value.right

            # nested binop case: a[:] = (a <op> b) <op> c
            if isinstance(rhs, ast.BinOp):
                left_inner = rhs.left
                inner_name = None

                if isinstance(left_inner, ast.Name):
                    inner_name = left_inner.id
                elif isinstance(left_inner, ast.Attribute):
                    inner_name = left_inner.attr
                elif isinstance(left_inner, ast.Subscript):
                    inner_name = (
                        left_inner.value.id
                        if isinstance(left_inner.value, ast.Name)
                        else left_inner.value.attr
                    )

                if inner_name == lhs_name:
                    operator = ops.get(type(rhs.op))
                    if operator:
                        new_binop = ast.BinOp(
                            left=rhs.right, op=node.value.op, right=node.value.right
                        )
                        return operator, new_binop

            return "set", node.value
        except Exception as e:
            self.logger.exception("Exception in update_cls_dict", e)
            raise

    def update_cls_dict(
        self,
        method: bool = False,
        func_args: dict = None,
        attribute: bool = False,
        attribute_args: dict = None,
    ) -> None:
        """
        Update stored class metadata for methods and attributes.

        This method modifies entries in ``self.cls_info`` associated with the
        current class. It supports updating method metadata (such as argument
        information and local array definitions) as well as attribute metadata.

        Parameters
        ----------
        method : bool, optional
            Whether to update method-related metadata. If True, ``func_args``
            must be provided. Default is False.
        func_args : dict, optional
            Method update information. Expected keys include:

            - ``func_name`` : str
                Name of the method to update.
            - ``args`` : dict, optional
                Updated method argument information.
            - ``local_arr`` : dict, optional
                Local array metadata containing:

                - ``var`` : str
                    Name of the local array variable.
                - ``update_value`` : dict
                    Replacement metadata for the variable.

        attribute : bool, optional
            Whether to update attribute-related metadata. If True,
            ``attribute_args`` must be provided. Default is False.
        attribute_args : dict, optional
            Attribute update information. Expected keys include:

            - ``var`` : str
                Name of the attribute to update.
            - ``update_value`` : dict
                Metadata values to merge into the existing attribute entry.

        Raises
        ------
        ValueError
            If required update arguments are not provided or if the requested
            method does not exist in the stored class metadata.
        """

        try:
            cls_data = self.cls_info[self.cls_name]

            if method:
                if not func_args:
                    raise ValueError("If method=True, func_args must be provided")

                func_name = func_args.get("func_name")
                methods = cls_data.get("methods", {})
                data = methods.get(func_name)

                if not data:
                    raise ValueError(f"{func_name} doesn't have any data in cls_info")

                # Update args
                if func_args.get("args"):
                    data["args"] = func_args["args"]

                # Update local arrays
                local_arr = func_args.get("local_arr")
                if local_arr:
                    var = local_arr.get("var")
                    update_value = local_arr.get("update_value", {})
                    # data["local_arr"][var].update(update_value)
                    data["local_arr"][var] = update_value

            if attribute:
                if not attribute_args:
                    raise ValueError(
                        "If attribute=True, attribute_args must be provided"
                    )

                attributes = cls_data.get("attributes", {})
                var = attribute_args["var"]
                update_values = attribute_args["update_value"]

                attributes[var].update(update_values)

        except Exception as e:
            self.logger.exception("Exception in update_cls_dict", e)
            raise

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """
        Transform subscript expressions during vectorization.

        This visitor rewrites `ast.Subscript` nodes according to the
        active vectorization context stored on the control stack. Depending on
        the current vectorization state, the transformation may:

        * Replace vectorized index variables with full slices.
        * Generate pairwise indexing expressions using vectorized axes.
        * Convert loop-index-dependent accesses into slices or indexed ranges.
        * Lift dynamically created arrays by inserting batch dimensions.
        * Simplify fully sliced array accesses when possible.

        When no vectorization context is active, the subscript expression is
        returned unchanged.

        Parameters
        ----------
        node : ast.Subscript
            Subscript node to transform.

        Returns
        -------
        ast.AST
            The transformed AST node. This may be:

            * The original subscript with rewritten indices.
            * A simplified array expression when all indices become full slices.
            * The original node when no transformation is required.

        Notes
        -----
        The transformation depends on metadata stored in the current
        vectorization context, including:

        * ``vectorization_axis`` for identifying vectorized dimensions.
        * ``loop_info`` for pairwise indexing generation.
        * ``loop_index`` for loop-dependent index rewriting.
        * ``n_layers`` and ``use_layers_index`` for dynamic-loop handling.

        Dynamic array lifting may update internal bookkeeping structures and
        class metadata through :meth:`update_cls_dict`.

        Side Effects
        ------------
        * Updates :attr:`index_variables` with discovered index variables.
        * Updates :attr:`_lifted_vars` for dynamically lifted arrays.
        * May modify class metadata through :meth:`update_cls_dict`.
        * May record lifted array dimensions and vectorization information.
        """
        node = self.generic_visit(node)
        # No vectorization context
        if not self._control_stack:
            return node

        vect = self._control_stack[-1].to_dict()
        vectorization_axis = vect.get("vectorization_axis", {}) if vect else {}

        if vectorization_axis is None:
            return node

        metadata = vect.get("metadata", {})

        loop_index = metadata.get("loop_index")
        n_layers = metadata.get("n_layers", None)
        use_layers_index = metadata.get("use_layers_index", None)
        if not vectorization_axis:
            return node

        axis = list(vectorization_axis.keys())

        # Normalize indices
        if isinstance(node.slice, ast.Tuple):
            indices = list(node.slice.elts)
            for elt in node.slice.elts:
                if isinstance(elt, ast.Name):
                    self.index_variables.add(elt.id)
        else:
            indices = [node.slice]
            if isinstance(node.slice, ast.Name):
                self.index_variables.add(node.slice.id)

        # Array lifting
        var_name = get_name(node)
        lift_info = self.dynamic_variable_lift.get(var_name)
        if lift_info:
            batched_axes = lift_info["batched_axis"]
            original_shape = lift_info["original_shape"]
            vect_loop = lift_info["vectorized_loop"]
            local_arrays = (
                self.cls_info[self.cls_name]
                .get("methods", {})
                .get(self.func_name, {})
                .get("local_arr", {})
            )
            # Insert batch axes FIRST
            indices = list(indices)
            for ax in sorted(batched_axes):
                vect_value = (
                    vect_loop[ax].id
                    if isinstance(vect_loop[ax], ast.Name)
                    else vect_loop[ax].attr
                )
                if vect_value not in original_shape:
                    indices.insert(ax, ast.Slice())
                else:
                    if vect_value in original_shape and len(original_shape) != len(
                        indices
                    ):
                        indices.insert(ax, ast.Slice())

            if (self.func_name, var_name) not in self._lifted_vars:
                new_dim = original_shape
                for idx in sorted(batched_axes):
                    new_dim.insert(
                        idx,
                        vect_loop[idx].id
                        if isinstance(vect_loop[idx], ast.Name)
                        else vect_loop[idx].attr,
                    )
                self._lifted_vars.add((self.func_name, var_name))
                func_args = {
                    "func_name": self.func_name,
                    "local_arr": {
                        "var": var_name,
                        "update_value": {
                            "dimensions": new_dim,
                            "dtype": local_arrays.get(var_name).get("dtype"),
                            "type": "jnp.ndarray",
                        },
                    },
                }

                self.update_cls_dict(method=True, func_args=func_args)

        needs_pairwise = False
        for idx in indices:
            deps = self.expr_deps.get(idx, ())
            if deps:
                if set(axis).issubset(list(deps)):
                    if not (isinstance(idx, ast.Name) and idx.id in axis):
                        needs_pairwise = True
                        break
            else:
                if any(
                    (isinstance(n, ast.Name) and n.id in self.dynamic_variable_lift)
                    or (
                        isinstance(n, ast.Name)
                        and (state := self.var_state.get(n.id))
                        and state[0] == "stateful"
                    )
                    for n in ast.walk(node)
                ):
                    needs_pairwise = True
                    break

        new_indices = []
        loop_info = vect.get("loop_info", {}) if needs_pairwise else None

        # is_axis = lambda idx: isinstance(idx, ast.Name) and idx.id in axis
        def is_axis(idx):
            return isinstance(idx, ast.Name) and idx.id in axis

        for idx in indices:
            if needs_pairwise and is_axis(idx):
                new_indices.append(self._make_arange(axis, loop_info))

            elif not needs_pairwise and is_axis(idx):
                new_indices.append(ast.Slice())

            elif loop_index and contains_name(idx, loop_index):
                if n_layers:  # This is from the loop to mask for dynamic loops
                    offset = self._extract_offset(idx, loop_index)
                    lower = (
                        ast.BinOp(
                            left=n_layers[0],
                            op=ast.Add(),
                            right=ast.Constant(value=offset),
                        )
                        if offset
                        else n_layers[0]
                    )
                    upper = (
                        ast.BinOp(
                            left=n_layers[1],
                            op=ast.Add(),
                            right=ast.Constant(value=offset),
                        )
                        if offset
                        else n_layers[1]
                    )
                    new_indices.append(ast.Slice(lower=lower, upper=upper, step=None))
                elif use_layers_index:
                    # conditional search: index with the layers arange directly
                    offset = self._extract_offset(idx, loop_index)
                    layers_node = use_layers_index
                    if offset:
                        new_indices.append(
                            ast.BinOp(
                                left=layers_node,
                                op=ast.Add(),
                                right=ast.Constant(value=offset),
                            )
                        )
                    else:
                        new_indices.append(layers_node)
                else:
                    new_indices.append(ast.Slice())
            else:
                new_indices.append(idx)

        # Tranform arr[:, :] -> arr not lift_info and
        if not lift_info and all(self.is_full_slice(i) for i in new_indices):
            if (
                isinstance(node.value, ast.Attribute)
                and getattr(node.value, "attr", None) == "at"
            ):
                if len(new_indices) == 1:
                    node.slice = new_indices[0]
                else:
                    node.slice = ast.Tuple(elts=new_indices, ctx=ast.Load())
                return node
            else:
                return node.value  # flatten completely

        if len(new_indices) == 1:
            node.slice = new_indices[0]
        else:
            node.slice = ast.Tuple(elts=new_indices, ctx=ast.Load())

        return node
