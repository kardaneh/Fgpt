import ast

from fgpt.jax_utils import Control, MaybeAddIndexTransformer, contains_name, get_name


class _Vectorization:
    """
    Vectorisation analysis and rank/shape inference for ``JaxConverter``.

    Composes onto ``JaxConverter`` to answer the questions that drive
    every other lowering pass: which loop index maps to which axis of
    a vectorised array, what rank a given expression produces once
    subscripts and reductions are accounted for, whether a name is a
    declared array versus a scalar, and how to broadcast or index an
    expression so its shape matches a target rank.

    Three groups of responsibility live here:

    - **Axis discovery** — :meth:`compute_vectorization_axes`,
      :meth:`collect_subscripts`, :meth:`_extract_indices`,
      :meth:`get_vectorized_arrays`, :meth:`check_condition` determine,
      from the raw subscript usage in a loop body, which axis each
      loop index occupies in each array it indexes.
    - **Shape/rank inference** — :meth:`get_declared_dims`,
      :meth:`get_active_dims`, :meth:`_target_rank`,
      :meth:`subscript_ranks`, :meth:`_is_arange_over_dim`,
      :meth:`is_class_attribute`, :meth:`_base_symbol`,
      :meth:`infer_dtype` resolve the dimensionality and dtype of an
      arbitrary expression by consulting :attr:`cls_info`,
      :attr:`func_input_dim`, and :attr:`_inferred_ranks`.
    - **Broadcasting and dynamic lifting** — :meth:`maybe_add_index`,
      :meth:`_apply_broadcasting_slice`, :meth:`_rhs_is_pairwise`,
      :meth:`_force_subscript_pairwise`, :meth:`_make_arange`,
      :meth:`broadcast_scalar`, :meth:`_expr_depends_on_axes`,
      :meth:`_infer_scalar_shape`, :attr:`_inside_axis_condition`,
      :meth:`_analyze_local_assignment`, :meth:`_extract_target`,
      :meth:`_process_lift`, :meth:`_compute_new_axis`,
      :meth:`_build_vectorized_loop`, :meth:`_resolve_dynamic_origin`
      implement the promotion of scalar variables to vectorised arrays
      (:attr:`dynamic_variable_lift`) when a vectorisation axis forces
      it, and the index-rewriting needed to keep broadcast shapes
      consistent afterward.

    Convergence-loop support (:meth:`_emit_convergence_break`,
    :meth:`_get_or_create_converged_var`) also lives here since it
    depends heavily on rank inference and mask construction shared with
    the rest of the mixing.
    """

    def compute_vectorization_axes(
        self,
        body_ast: list[ast.AST],
        loop_index: str,
    ) -> dict[str, set[int]]:
        """
        Determine which subscript axis *loop_index* occupies across all
        array accesses in *body_ast*.

        Collects every subscript in the body via
        :meth:`collect_subscripts`, restricts attention to
        *loop_index* if it is a known loop variable (per
        ``self.analyzer.loop_stack``), then records, for each array
        access where *loop_index* appears, which positional axis it
        occupied.

        Parameters
        ----------
        body_ast : list[ast.AST]
            Statement list to scan — typically a loop or branch body.
        loop_index : str
            The loop variable name to compute axes for.

        Returns
        -------
        dict[str, set[int]]
            Mapping ``{loop_index: {axis, ...}}`` — usually a single
            key, since this is computed per loop index, but the axis
            set can contain multiple values if the same index appears
            at different positions across different arrays.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            array_indices = self.collect_subscripts(body_ast)
            loop_vars = [lv for group in self.analyzer.loop_stack for lv in group]
            loop_vars = {loop_index} & set(loop_vars)
            axis_map = {lv: set() for lv in loop_vars}

            for arr, accesses in array_indices.items():
                for idx_tuple in accesses:
                    for axis, idx in enumerate(idx_tuple):
                        if idx in axis_map:
                            axis_map[idx].add(axis)

            return axis_map
        except Exception as e:
            self.logger.exception("Exception in compute_vectorization_axes:", e)
            raise

    def collect_subscripts(self, node: ast.AST) -> dict[str, list[tuple]]:
        """
        Walk an AST subtree and collect every subscript access.

        Parameters
        ----------
        node : ast.AST
            Subtree to walk — typically a loop or branch body.

        Returns
        -------
        dict[str, list[tuple]]
            Mapping ``{array_name: [index_tuple, ...]}`` — one entry
            per ``ast.Subscript`` found, where each index tuple is
            produced by :meth:`_extract_indices`.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            results: dict[str, list[tuple]] = {}
            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript):
                    arr = get_name(sub.value)
                    if arr is None:
                        continue
                    idx = self._extract_indices(sub.slice)
                    if idx:
                        results.setdefault(arr, []).append(idx)
            return results
        except Exception as e:
            self.logger.exception("Exception in collect_subscripts:", e)
            raise

    def _extract_indices(self, node: ast.AST) -> tuple:
        """
        Convert a subscript slice node into a tuple of string indices.

        Each index element is rendered to a string representation
        suitable for comparison against loop-variable names: plain
        names and constants render as themselves, full slices render
        as ``":"``, and anything else falls back to ``ast.unparse``
        (or ``None`` if unparsing fails).

        Parameters
        ----------
        node : ast.AST
            The subscript's ``slice`` — either a single index or an
            ``ast.Tuple`` of indices.

        Returns
        -------
        tuple
            One string (or ``None``) per index dimension.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:

            def extract(n):
                if isinstance(n, ast.Name):
                    return n.id
                if isinstance(n, ast.Attribute):
                    return ast.unparse(n)
                if isinstance(n, ast.Constant):
                    return str(n.value)
                if isinstance(n, ast.Slice):
                    return ":"
                try:
                    return ast.unparse(n)
                except Exception:
                    return None

            if isinstance(node, ast.Tuple):
                return tuple(extract(e) for e in node.elts)
            return (extract(node),)
        except Exception as e:
            self.logger.exception("Exception in _extract_indices:", e)
            raise

    def get_vectorized_arrays(
        self,
        node: ast.AST,
        vectorization_context: dict,
    ) -> set[str]:
        """
        Identify which arrays referenced in *node* are vectorised under
        the current axis mapping.

        An array counts as vectorised if either:

        - it is subscripted with a full slice (``:``) or a plain name
          at a position matching an active vectorisation axis, or
        - one of its declared dimensions (from :attr:`cls_info`
          attributes or the current method's ``local_arr``) matches a
          loop dimension present in ``vectorization_context['loop_info']``.

        Parameters
        ----------
        node : ast.AST
            Subtree to scan.
        vectorization_context : dict
            The active :class:`Control` context as a dict, with
            ``vectorization_axis`` and ``loop_info`` keys.

        Returns
        -------
        set[str]
            Names of arrays found to be vectorised.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            vector_arrays = set()
            vectorization_axis = vectorization_context.get("vectorization_axis")

            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript):
                    indices = (
                        sub.slice.elts
                        if isinstance(sub.slice, ast.Tuple)
                        else [sub.slice]
                    )
                    sub_name = get_name(sub)
                    for i, idx in enumerate(indices):
                        if isinstance(idx, ast.Slice):
                            if i in list(vectorization_axis.values()):
                                vector_arrays.add(sub_name)
                        elif isinstance(idx, ast.Name):
                            if idx.id in list(vectorization_axis.keys()):
                                vector_arrays.add(sub_name)

                if isinstance(sub, ast.Name | ast.Attribute):
                    cls_data = self.cls_info[self.cls_name]
                    attributes = cls_data.get("attributes", {})
                    methods = cls_data.get("methods", {})
                    func_data = methods.get(self.func_name, {})
                    local_arrays = func_data.get("local_arr", {}) if func_data else {}

                    loop_info = vectorization_context.get("loop_info")
                    sub_name = get_name(sub)
                    if sub_name in attributes:
                        dimension = attributes.get(sub_name).get("dimensions")
                        for idx in dimension:
                            if idx in loop_info:
                                vector_arrays.add(sub_name)
                    elif sub_name in local_arrays:
                        dimension = local_arrays.get(sub_name).get("dimensions")
                        for idx in dimension:
                            if idx in loop_info:
                                vector_arrays.add(sub_name)

            return vector_arrays
        except Exception as e:
            self.logger.exception("Exception in get_vectorized_arrays:", e)
            raise

    def check_condition(
        self,
        node: ast.AST,
        vectorization_context: dict | None,
    ) -> tuple:
        """
        Check whether *node* references any vectorised array.

        Thin wrapper around :meth:`get_vectorized_arrays` that also
        handles the no-context case.

        Parameters
        ----------
        node : ast.AST
            Subtree to check — typically a ``while`` condition.
        vectorization_context : Optional[dict]
            The active :class:`Control` context as a dict, or ``None``.

        Returns
        -------
        tuple
            ``(is_vectorized, vector_arrays)`` where *is_vectorized* is
            ``True`` if at least one vectorised array was found.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if vectorization_context is None:
                return False, set()
            vector_arrays = self.get_vectorized_arrays(node, vectorization_context)
            return len(vector_arrays) > 0, vector_arrays
        except Exception as e:
            self.logger.exception("Exception in check_condition:", e)
            raise

    def get_declared_dims(self, node: ast.AST) -> list | None:
        """
        Resolve the declared dimension names for *node*.

        Looks up *node*'s name in, in order: the current method's
        ``local_arr``, the class's ``attributes``, then
        :attr:`func_input_dim`. For an ``ast.Subscript``, recurses into
        the base value, since indexing does not change the declared
        (full) dimension list — only :meth:`get_active_dims` accounts
        for dimensions removed by indexing.

        Parameters
        ----------
        node : ast.AST
            ``ast.Name``, ``ast.Attribute``, or ``ast.Subscript`` to
            resolve.

        Returns
        -------
        Optional[list]
            The declared dimension names, or ``None`` if *node* is not
            a known array (scalar or unrecognised name).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            cls = self.cls_name if isinstance(self.cls_name, str) else self.cls_name[0]
            func = self.func_name

            local_arr = self.cls_info[cls]["methods"][func].get("local_arr", {})
            attributes = self.cls_info[cls]["attributes"]

            if isinstance(node, ast.Name):
                if node.id in local_arr:
                    return local_arr[node.id].get("dimensions")
                elif node.id in attributes:
                    return attributes[node.id].get("dimensions")
                elif hasattr(self, "func_input_dim") and node.id in list(
                    self.func_input_dim.keys()
                ):
                    return self.func_input_dim.get(node.id)

            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    attr = node.attr
                    if attr in attributes:
                        return attributes[attr].get("dimensions")
                    elif hasattr(self, "func_input_dim") and attr in list(
                        self.func_input_dim.keys()
                    ):
                        return self.func_input_dim.get(attr)

            if isinstance(node, ast.Subscript):
                base_dims = self.get_declared_dims(node.value)
                return base_dims if base_dims is not None else None

            return None
        except Exception as e:
            self.logger.exception("Exception in get_declared_dims:", e)
            raise

    def get_active_dims(self, node: ast.AST) -> list:
        """
        Resolve the dimensions of *node* that remain active after
        indexing.

        Unlike :meth:`get_declared_dims` (the full declared shape),
        this drops any dimension consumed by a non-slice index (a
        ``Name``, ``Attribute``, ``BinOp``, or ``Constant`` index
        removes that dimension; a ``Slice`` keeps it).

        Parameters
        ----------
        node : ast.AST
            Node to resolve — typically a subscript expression.

        Returns
        -------
        list
            Dimension names still present after indexing, or the full
            declared list if *node* is not a subscript.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            declared = self.get_declared_dims(node)
            if not declared:
                return []

            if isinstance(node, ast.Subscript):
                sl = node.slice
                active = []
                if isinstance(sl, ast.Tuple):
                    for dim, idx in zip(declared, sl.elts):
                        if isinstance(
                            idx, ast.Name | ast.Attribute | ast.BinOp | ast.Constant
                        ):
                            continue
                        active.append(dim)
                else:
                    active = declared[1:]
                return active

            return declared
        except Exception as e:
            self.logger.exception("Exception in get_active_dims:", e)
            raise

    def _target_rank(self, slice_node: ast.AST) -> int:
        """
        Compute the rank a target subscript will have after
        vectorisation.

        Counts the number of elements in *slice_node* that are either
        a full slice or a ``Name`` matching an active vectorisation
        axis. Falls back to ``1`` when no control context is active
        (a bare scalar target outside vectorisation still occupies one
        rank slot for downstream broadcasting logic).

        Parameters
        ----------
        slice_node : ast.AST
            The subscript's ``slice`` — a single index or an
            ``ast.Tuple`` of indices.

        Returns
        -------
        int
            The inferred target rank.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            vec_context = None
            if self._control_stack:
                vec_context = (
                    self._control_stack[-1].to_dict().get("vectorization_axis")
                )
            else:
                return 1

            if isinstance(slice_node, ast.Tuple):
                return sum(
                    isinstance(e, ast.Name)
                    and e.id in vec_context
                    or isinstance(e, ast.Slice)
                    for e in slice_node.elts
                )
            return 1
        except Exception as e:
            self.logger.exception("Exception in _target_rank:", e)
            raise

    def _base_symbol(self, node: ast.AST) -> ast.AST | None:
        """
        Strip nested subscripts to find the base ``Name``/``Attribute``.

        Parameters
        ----------
        node : ast.AST
            Node to unwrap — typically a (possibly multiply) subscripted
            expression.

        Returns
        -------
        Optional[ast.AST]
            The innermost ``ast.Name`` or ``ast.Attribute``, or ``None``
            if the base is some other node type.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            while isinstance(node, ast.Subscript):
                node = node.value
            if isinstance(node, ast.Name | ast.Attribute):
                return node
            return None
        except Exception as e:
            self.logger.exception("Exception in _base_symbol:", e)
            raise

    def is_class_attribute(self, arg: ast.AST) -> bool:
        """
        Return ``True`` if *arg* refers to (or contains a reference to)
        a declared class attribute.

        Recurses into ``ast.BinOp`` operands so that expressions like
        ``self.kjpindex + 1`` are still recognised as
        attribute-dependent.

        Parameters
        ----------
        arg : ast.AST
            Expression to check — ``ast.Attribute``, ``ast.Name``, or
            ``ast.BinOp``.

        Returns
        -------
        bool
            ``True`` if *arg* (or a sub-expression) names a known class
            attribute.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(arg, ast.Attribute):
                return arg.attr in self.cls_info[self.cls_name]["attributes"]
            if isinstance(arg, ast.Name):
                return arg.id in self.cls_info[self.cls_name]["attributes"]
            if isinstance(arg, ast.BinOp):
                return self.is_class_attribute(arg.left) or self.is_class_attribute(
                    arg.right
                )
            return False
        except Exception as e:
            self.logger.exception("Exception in is_class_attribute:", e)
            raise

    def _is_arange_over_dim(self, node: ast.AST) -> bool:
        """
        Return ``True`` if *node* is ``jnp.arange``/``np.arange`` (or
        bare ``arange``) called over an expression that references a
        known dimension variable.

        Recurses into ``ast.BinOp`` so that ``arange(n) + 1`` is still
        recognised.

        Parameters
        ----------
        node : ast.AST
            Node to check.

        Returns
        -------
        bool
            ``True`` if *node* is an ``arange`` call with at least one
            argument resolving to a class attribute via
            :meth:`is_class_attribute`.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.BinOp):
                return self._is_arange_over_dim(node.left) or self._is_arange_over_dim(
                    node.right
                )

            if not isinstance(node, ast.Call):
                return False

            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr != "arange":
                    return False
            elif isinstance(func, ast.Name):
                if func.id != "arange":
                    return False
            else:
                return False

            return any(self.is_class_attribute(arg) for arg in node.args)
        except Exception as e:
            self.logger.exception("Exception in _is_arange_over_dim:", e)
            raise

    def subscript_ranks(
        self,
        node: ast.AST,
        ranks: dict | None = None,
        seen: set | None = None,
    ) -> dict[ast.AST, int]:
        """
        Recursively infer the array rank produced by every
        sub-expression of *node*.

        Handles, with rank computed accordingly:

        - ``ast.Subscript`` — counts full slices, ``None`` constants
          (newaxis), and dimension-``arange`` indices as
          rank-preserving; integer/variable indices drop a dimension.
        - ``ast.Name`` / ``ast.Attribute`` — looked up via
          :meth:`get_declared_dims`, falling back to
          :attr:`_inferred_ranks` for compiler-created names.
        - ``ast.Call`` — reduction functions (``sum``, ``mean``,
          ``prod``) reduce the input rank by the number of reduced
          axes, honouring ``keepdims=True``.
        - ``ast.BinOp`` / ``ast.BoolOp`` / ``ast.Compare`` — rank is the
          max of the operand ranks, with comparisons additionally
          checking :attr:`dynamic_variable_lift` / :attr:`var_state`
          for scalars that have been promoted to rank-1.
        - any other node — recurses into children without assigning a
          rank to the node itself.

        Parameters
        ----------
        node : ast.AST
            Expression to analyse.
        ranks : Optional[dict]
            Accumulator dict, mutated in place across the recursion;
            created fresh if ``None``.
        seen : Optional[set]
            set of base symbols already covered by an enclosing
            subscript, to avoid double-counting; created fresh if
            ``None``.

        Returns
        -------
        dict[ast.AST, int]
            Mapping from AST node to inferred rank, covering *node* and
            every sub-expression visited.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if ranks is None:
                ranks = {}
            if seen is None:
                seen = set()

            if isinstance(node, ast.Subscript):
                sl = node.slice

                if isinstance(sl, ast.Tuple):
                    rank = 0
                    for elt in sl.elts:
                        if isinstance(elt, ast.Slice) and self.is_full_slice(elt):
                            rank += 1
                        elif isinstance(elt, ast.Constant) and elt.value is None:
                            rank += 1
                        elif self._is_arange_over_dim(elt):
                            rank += 1
                else:
                    rank = 1 if isinstance(sl, ast.Slice) else 0

                ranks[node] = rank

                base = self._base_symbol(node)
                if base is not None:
                    seen.add(base)

                self.subscript_ranks(node.value, ranks, seen)
                if isinstance(sl, ast.Tuple):
                    for elt in sl.elts:
                        self.subscript_ranks(elt, ranks, seen)
                else:
                    self.subscript_ranks(sl, ranks, seen)

                return ranks

            if isinstance(node, ast.Name | ast.Attribute):
                if node in seen:
                    return ranks

                rank = (
                    len(self.get_declared_dims(node))
                    if self.get_declared_dims(node)
                    else 0
                )
                if rank == 0 and isinstance(node, ast.Name):
                    rank = self._inferred_ranks.get(node.id, 0)

                if rank > 0:
                    ranks[node] = rank
                    seen.add(node)

                return ranks

            if isinstance(node, ast.Call):
                func_name = get_name(node.func)

                if func_name in {"sum", "mean", "prod"}:
                    arg = node.args[0]
                    inner_ranks = self.subscript_ranks(arg, {}, set())
                    ranks.update(inner_ranks)
                    input_rank = max(inner_ranks.values(), default=0)

                    axis = None
                    for kw in node.keywords:
                        if kw.arg == "axis" and isinstance(kw.value, ast.Constant):
                            axis = kw.value.value

                    if axis is None:
                        rank = 0
                    elif isinstance(axis, tuple):
                        rank = input_rank - len(axis)
                    else:
                        rank = input_rank - 1

                    for kw in node.keywords:
                        if (
                            kw.arg == "keepdims"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value
                        ):
                            rank = input_rank

                    ranks[node] = rank
                    return ranks

            if isinstance(node, ast.BinOp):
                left_ranks = self.subscript_ranks(node.left, {}, set())
                right_ranks = self.subscript_ranks(node.right, {}, set())
                left_rank = left_ranks.get(node.left, 0)
                right_rank = right_ranks.get(node.right, 0)
                rank = max(left_rank, right_rank)

                ranks.update(left_ranks)
                ranks.update(right_ranks)
                ranks[node] = rank
                return ranks

            if isinstance(node, ast.BoolOp):
                max_rank = 0
                for val in node.values:
                    child_ranks = self.subscript_ranks(val, {}, set())
                    ranks.update(child_ranks)
                    max_rank = max(max_rank, child_ranks.get(val, 0))
                ranks[node] = max_rank
                return ranks

            if isinstance(node, ast.Compare):
                left_ranks = self.subscript_ranks(node.left, {}, set())
                comp_ranks = {}
                for comp in node.comparators:
                    comp_ranks.update(self.subscript_ranks(comp, {}, set()))

                left_rank = left_ranks.get(node.left, 0)
                right_rank = max(
                    (comp_ranks.get(comp, 0) for comp in node.comparators), default=0
                )

                if left_rank == 0 and isinstance(node.left, ast.Name):
                    dims = self.get_declared_dims(node.left)
                    left_rank = len(dims) if dims else 0
                    if left_rank == 0 and (
                        node.left.id in self.dynamic_variable_lift
                        or (state := self.var_state.get(node.left.id))
                        and state[0] == "stateful"
                    ):
                        left_rank = 1

                if right_rank == 0:
                    for comp in node.comparators:
                        if isinstance(comp, ast.Name):
                            dims = self.get_declared_dims(comp)
                            comp_rank = len(dims) if dims else 0
                            if comp_rank == 0 and (
                                comp.id in self.dynamic_variable_lift
                                or (state := self.var_state.get(comp.id))
                                and state[0] == "stateful"
                            ):
                                comp_rank = 1
                            right_rank = max(right_rank, comp_rank)

                rank = max(left_rank, right_rank)
                ranks.update(left_ranks)
                ranks.update(comp_ranks)
                ranks[node] = rank
                return ranks

            for child in ast.iter_child_nodes(node):
                self.subscript_ranks(child, ranks, seen)

            return ranks

        except Exception as e:
            self.logger.exception("Exception in subscript_ranks:", e)
            raise

    def maybe_add_index(
        self,
        node: ast.AST,
        target_rank: int,
        vect_context: dict,
    ) -> tuple:
        """
        Insert broadcasting indices into *node* so its rank matches
        *target_rank*.

        Computes per-node ranks via :meth:`subscript_ranks`, then
        delegates the actual index-insertion logic to
        ``MaybeAddIndexTransformer``, which has access to class
        metadata, local scope, and dynamic-lift state needed to decide
        where ``None`` (newaxis) or ``Slice`` indices belong.

        Parameters
        ----------
        node : ast.AST
            Expression to rewrite.
        target_rank : int
            The rank *node* must end up matching.
        vect_context : dict
            The active :class:`Control` context as a dict.

        Returns
        -------
        tuple
            ``(rewritten_node, elts_list)`` where *elts_list* is the
            broadcast index list used (if any), passed back to callers
            that need to align a companion mask to the same shape.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            ranks = self.subscript_ranks(node)
            transformer = MaybeAddIndexTransformer(
                cls_info=self.cls_info,
                cls_name=self.cls_name,
                func_name=self.func_name,
                ranks=ranks,
                target_rank=target_rank,
                vect_context=vect_context,
                local_defined_variables=self._local_defined_stack[-1]
                if self._local_defined_stack
                else [],
                dynamic_variable_lift=self.dynamic_variable_lift
                if self.dynamic_variable_lift
                else {},
                inferred_ranks=self._inferred_ranks,
                func_input_dim=self.func_input_dim
                if hasattr(self, "func_input_dim")
                else None,
            )
            return transformer.visit(node), transformer.elts_list
        except Exception as e:
            self.logger.exception("Exception in maybe_add_index:", e)
            raise

    def _apply_broadcasting_slice(self, sub: ast.Subscript, assign_vect: dict):
        """
        Replace vectorisation-axis names in *sub*'s slice with full
        slices, in place.

        Parameters
        ----------
        sub : ast.Subscript
            The subscript node to mutate.
        assign_vect : dict
            The active :class:`Control` context as a dict, with a
            ``vectorization_axis`` key.

        Returns
        -------
        ast.AST or None
            The new slice expression (``ast.Tuple`` or ``ast.Slice``),
            or ``None`` if no axis name was matched.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(sub.slice, tuple):
                new_slices = []
                for s in sub.slice:
                    if (
                        isinstance(s, ast.Name)
                        and s.id in assign_vect["vectorization_axis"]
                    ):
                        new_slices.append(ast.Slice())
                    else:
                        new_slices.append(s)
                sub.slice = tuple(new_slices)
                return ast.Tuple(new_slices, ctx=ast.Load())
            else:
                if (
                    isinstance(sub.slice, ast.Name)
                    and sub.slice.id in assign_vect["vectorization_axis"]
                ):
                    sub.slice = ast.Slice()
                    return ast.Slice()
        except Exception as e:
            self.logger.exception("Exception in _apply_broadcasting_slice:", e)
            raise

    def _rhs_is_pairwise(self, node: ast.AST) -> bool:
        """
        Return ``True`` if any subscript inside *node* has been flagged
        ``_pairwise`` by ``MaybeAddIndexTransformer``.

        Parameters
        ----------
        node : ast.AST
            Expression to scan — typically the RHS of an assignment.

        Returns
        -------
        bool
            ``True`` if a pairwise-indexed subscript is present.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            found = False
            for child in ast.walk(node):
                if isinstance(child, ast.Subscript) and getattr(
                    child, "_pairwise", False
                ):
                    found = True
                    break
            return found
        except Exception as e:
            self.logger.exception("Exception in _rhs_is_pairwise:", e)
            raise

    def _force_subscript_pairwise(
        self,
        node: ast.Subscript,
        axis: list[str],
        loop_info: dict,
    ) -> None:
        """
        Rewrite every full-slice or axis-name index in *node* into an
        explicit ``jnp.arange(...)`` call, in place.

        Used when a subscript needs pairwise (gather-style) indexing
        rather than broadcasting — converting ``arr[:, j]`` style
        slices into ``arr[jnp.arange(dim), j]`` so JAX performs
        element-wise pairing instead of an outer-product broadcast.

        Parameters
        ----------
        node : ast.Subscript
            The subscript whose slice is rewritten in place.
        axis : list[str]
            Names of the active vectorisation axes.
        loop_info : dict
            Mapping from loop dimension name to loop variable name,
            used by :meth:`_make_arange` to resolve the arange bound.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            indices = (
                list(node.slice.elts)
                if isinstance(node.slice, ast.Tuple)
                else [node.slice]
            )
            new_indices = []

            for idx in indices:
                if isinstance(idx, ast.Slice):
                    new_indices.append(self._make_arange(axis, loop_info))
                elif isinstance(idx, ast.Name) and idx.id in axis:
                    new_indices.append(self._make_arange(axis, loop_info))
                else:
                    new_indices.append(idx)

            if len(new_indices) == 1:
                node.slice = new_indices[0]
            else:
                node.slice = ast.Tuple(elts=new_indices, ctx=ast.Load())
        except Exception as e:
            self.logger.exception("Exception in _force_subscript_pairwise:", e)
            raise

    def _make_arange(self, axis: list[str], loop_info: dict) -> ast.Call:
        """
        Build a ``jnp.arange(<dim>)`` call for the loop dimension
        backing *axis*.

        Parameters
        ----------
        axis : list[str]
            Names of the active vectorisation axes.
        loop_info : dict
            Mapping from loop dimension name to loop variable name.

        Returns
        -------
        ast.Call
            ``jnp.arange(self.<dim>)`` if the dimension is a class
            attribute, otherwise ``jnp.arange(<dim>)``.

        Raises
        ------
        ValueError
            If no entry in *loop_info* maps to a value in *axis*.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            attributes = self.cls_info[self.cls_name]["attributes"]
            loop_var = next(
                (key for key, val in loop_info.items() if val in axis), None
            )

            if loop_var is None:
                raise ValueError(f"No loop_info entry found for index '{axis}'")

            if loop_var in attributes:
                val = ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=loop_var,
                    ctx=ast.Load(),
                )
            else:
                val = ast.Name(id=loop_var, ctx=ast.Load())

            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="arange",
                    ctx=ast.Load(),
                ),
                args=[val],
                keywords=[],
            )
        except ValueError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _make_arange:", e)
            raise

    def _expr_depends_on_axes(self, node: ast.AST, vectorization_axis: dict) -> tuple:
        """
        Determine which vectorisation axes *node* transitively depends
        on.

        Walks *node*, following ``Name`` references into
        :attr:`var_deps` when the name itself is not a raw axis name,
        and recording any direct axis-name hit. The result is cached
        in :attr:`expr_deps` for ``Name`` and ``BinOp`` nodes so
        repeated lookups for the same expression are cheap.

        Parameters
        ----------
        node : ast.AST
            Expression to analyse.
        vectorization_axis : dict
            Mapping of active axis names to their axis indices.

        Returns
        -------
        tuple
            Axis names *node* depends on, ordered to match
            *vectorization_axis*'s iteration order.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            deps = set()

            def visit(n):
                if n is None:
                    return
                if isinstance(n, ast.Name):
                    if n.id in vectorization_axis:
                        deps.add(n.id)
                    elif n.id in self.var_deps:
                        deps.update(self.var_deps.get(n.id, set()))
                    return
                if isinstance(n, ast.Attribute):
                    visit(n.value)
                    return
                if isinstance(n, ast.Subscript):
                    visit(n.value)
                    visit(n.slice)
                    return
                if isinstance(n, ast.Call):
                    visit(n.func)
                    for a in n.args:
                        visit(a)
                    for kw in n.keywords:
                        visit(kw.value)
                    return
                if isinstance(n, ast.Tuple):
                    for e in n.elts:
                        visit(e)
                    return
                if isinstance(n, ast.BinOp | ast.BoolOp | ast.Compare):
                    for child in ast.iter_child_nodes(n):
                        visit(child)
                    return
                if isinstance(n, ast.UnaryOp | ast.IfExp):
                    for child in ast.iter_child_nodes(n):
                        visit(child)
                    return
                if isinstance(n, ast.Constant):
                    return
                for child in ast.iter_child_nodes(n):
                    visit(child)

            visit(node)
            ordered = tuple(ax for ax in vectorization_axis if ax in deps)

            if isinstance(node, ast.Name) and node.id not in vectorization_axis:
                self.expr_deps[node] = ordered
            if isinstance(node, ast.BinOp):
                self.expr_deps[node] = ordered

            return ordered
        except Exception as e:
            self.logger.exception("Exception in _expr_depends_on_axes:", e)
            raise

    def _infer_scalar_shape(self, vectorization_context: dict) -> tuple:
        """
        Resolve the dimension names backing the active vectorisation
        axes.

        Parameters
        ----------
        vectorization_context : dict
            The active :class:`Control` context as a dict, with
            ``vectorization_axis`` and ``loop_info`` keys.

        Returns
        -------
        tuple
            One dimension name per active axis, in axis-iteration
            order.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            axes = vectorization_context.get("vectorization_axis", {})
            loop_info = vectorization_context.get("loop_info", {})
            return tuple(
                list(loop_info.keys())[list(loop_info.values()).index(axis)]
                for axis in axes
            )
        except Exception as e:
            self.logger.exception("Exception in _infer_scalar_shape:", e)
            raise

    @property
    def _inside_axis_condition(self) -> bool:
        """
        ``True`` if any enclosing :class:`Control` on the stack is an
        ``if``-block.

        Indicates that the current assignment is guarded by a
        condition that was classified as ``'vector'`` or
        ``'index_loop'`` (i.e. depends on the vectorisation axis).

        Returns
        -------
        bool
            ``True`` if an ``if``-kind :class:`Control` is present
            anywhere on :attr:`_control_stack`.
        """
        for ctrl in reversed(self._control_stack):
            if ctrl.kind == "if":
                return True
        return False

    def _analyze_local_assignment(
        self,
        node: ast.AST,
        vectorization_context: dict,
    ) -> None:
        """
        Decide whether a local array assignment must be promoted
        (dynamically lifted) due to vectorisation.

        Operates purely via side effects on :attr:`dynamic_variable_lift`
        through :meth:`_process_lift`. A local array is lifted when one
        of three conditions holds and the LHS does **not** already
        subscript using a loop variable (i.e. it is not already
        correctly indexed):

        1. The RHS directly uses a loop variable in a subscript.
        2. The RHS depends on a name already registered in
           :attr:`dynamic_variable_lift`.
        3. The assignment sits inside an axis-dependent ``if`` block
           (:attr:`_inside_axis_condition`) and is not itself already
           shaped to the loop dimensions.

        Parameters
        ----------
        node : ast.AST
            The assignment to analyse — expected to be an
            ``ast.Assign`` whose value is checked via
            :meth:`check_if_array`.
        vectorization_context : dict
            The active :class:`Control` context as a dict, or falsy to
            skip analysis entirely.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not vectorization_context:
                return
            if not self.check_if_array(node):
                return

            axes = vectorization_context.get("vectorization_axis", {})
            loop_info = vectorization_context.get("loop_info", {})
            if not axes:
                return

            target_node = self._extract_target(node)
            array_name = get_name(target_node)

            local_arrays = (
                self.cls_info[self.cls_name]
                .get("methods", {})
                .get(self.func_name, {})
                .get("local_arr", {})
            )
            if array_name not in local_arrays:
                return

            lhs_uses_loop = self._subscript_uses_loop_vars(
                target_node, list(axes.keys())
            )
            rhs_uses_loop = self._subscript_uses_loop_vars(
                node.value, list(axes.keys())
            )

            if not lhs_uses_loop:
                if rhs_uses_loop:
                    self._process_lift(array_name, axes, loop_info, local_arrays)
                    return

                for name in list(self.dynamic_variable_lift.keys()):
                    if contains_name(node.value, name):
                        self._process_lift(array_name, axes, loop_info, local_arrays)
                        return

                if self._inside_axis_condition and not self.check_if_array(
                    node, required_dims=list(loop_info.keys())
                ):
                    self._process_lift(array_name, axes, loop_info, local_arrays)
                    return

        except Exception as e:
            self.logger.exception("Exception in _analyze_local_assignment:", e)
            raise

    def _extract_target(self, node: ast.AST) -> ast.AST:
        """
        Return the LHS target of an assignment, or *node* unchanged.

        Parameters
        ----------
        node : ast.AST
            Node to extract from.

        Returns
        -------
        ast.AST
            ``node.targets[0]`` if *node* is an ``ast.Assign``,
            otherwise *node* itself.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Assign):
                return node.targets[0]
            return node
        except Exception as e:
            self.logger.exception("Exception in _extract_target:", e)
            raise

    def _process_lift(
        self,
        array_name: str,
        axes: dict,
        loop_info: dict,
        local_arrays: dict,
    ) -> None:
        """
        Register *array_name* in :attr:`dynamic_variable_lift` with its
        new batched shape.

        No-ops if the array has no declared dimensions or if
        :meth:`_compute_new_axis` cannot determine an insertion point.
        Idempotent — does nothing if *array_name* is already
        registered.

        Parameters
        ----------
        array_name : str
            Name of the local array to lift.
        axes : dict
            Active vectorisation axes, ``{loop_dim: {axis, ...}}``.
        loop_info : dict
            Mapping from loop dimension name to loop variable name.
        local_arrays : dict
            The current method's ``local_arr`` metadata dict.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            dimension = local_arrays[array_name].get("dimensions")
            dtype = local_arrays[array_name].get("dtype")

            if not dimension:
                return

            new_axis = self._compute_new_axis(dimension, axes)
            if new_axis is None:
                return

            vect_loop = self._build_vectorized_loop(loop_info, axes)
            batch_axis_key = list(axes.keys())[-1]

            if array_name not in self.dynamic_variable_lift:
                self.dynamic_variable_lift[array_name] = {
                    "original_shape": dimension,
                    "axis_map": new_axis,
                    "batched_axis": axes[batch_axis_key],
                    "vectorized_loop": vect_loop,
                    "dtype": dtype,
                }
        except Exception as e:
            self.logger.exception("Exception in _process_lift:", e)
            raise

    def _compute_new_axis(self, dimension: list, axes: dict) -> int:
        """
        Compute the shifted axis index after batch-dimension insertion.

        Parameters
        ----------
        dimension : list
            The array's original declared dimensions.
        axes : dict
            Active vectorisation axes, ``{loop_dim: {axis, ...}}``.

        Returns
        -------
        int
            The axis index ``0`` shifted forward by the number of
            batch axes inserted at or before it.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            axis_index = 0
            batch_axes = set()
            for ax in axes.values():
                batch_axes |= ax
            shift = sum(1 for b in batch_axes if b <= axis_index)
            return axis_index + shift
        except Exception as e:
            self.logger.exception("Exception in _compute_new_axis:", e)
            raise

    def _build_vectorized_loop(self, loop_info: dict, axes: dict) -> list[ast.AST]:
        """
        Build the AST node list representing the loop bound(s) used to
        size a lifted array's new batch dimension.

        Parameters
        ----------
        loop_info : dict
            Mapping from loop dimension name to loop variable name.
        axes : dict
            Active vectorisation axes, ``{loop_dim: {axis, ...}}``.

        Returns
        -------
        list[ast.AST]
            Single-element list containing ``self.<dim>`` (if the
            backing dimension is a class attribute) or a bare
            ``ast.Name`` otherwise.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            batch_axis_key = list(axes.keys())[-1]
            loop_var = next(k for k, v in loop_info.items() if v == batch_axis_key)
            attributes = self.cls_info[self.cls_name].get("attributes", {})

            if loop_var in attributes:
                return [
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=loop_var,
                        ctx=ast.Load(),
                    )
                ]
            return [ast.Name(id=loop_var, ctx=ast.Load())]
        except Exception as e:
            self.logger.exception("Exception in _build_vectorized_loop:", e)
            raise

    def _resolve_dynamic_origin(self, var_name: str) -> str:
        """
        Resolve compiler-created aliases back to their original source
        variable.

        Example::

            depth_idx -> locjj
            tmp2 -> depth_idx -> locjj
            returns: locjj

        Walks :attr:`dynamic_created_variables` following the alias
        chain, guarding against cycles via a ``seen`` set.

        Parameters
        ----------
        var_name : str
            The (possibly aliased) variable name to resolve.

        Returns
        -------
        str
            The original source variable name, or *var_name* itself if
            it is not a registered alias.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            seen = set()
            current = var_name

            while current in self.dynamic_created_variables:
                if current in seen:
                    break
                seen.add(current)
                parent = self.dynamic_created_variables[current]
                if not isinstance(parent, str):
                    break
                current = parent

            return current
        except Exception as e:
            self.logger.exception("Exception in _resolve_dynamic_origin:", e)
            raise

    def infer_dtype(self, node: ast.AST) -> str:
        """
        Infer the JAX/numpy dtype string produced by evaluating *node*.

        Resolution order by node type:

        - ``ast.Constant`` — Python type maps directly
          (``bool``/``int`` → ``'int64'``, ``float`` → ``'float64'``,
          ``complex`` → ``'complex128'``).
        - ``ast.Name`` — checked against :attr:`symbol_table`, then the
          current method's ``local_arr``, then class ``attributes``.
        - ``ast.Attribute`` on ``self`` — checked against class
          ``attributes``.
        - ``ast.Call`` — array constructors (``zeros``/``ones``/``full``)
          honour an explicit ``dtype`` keyword, defaulting to
          ``'float64'``; ``jnp.<dtype>(...)`` casts are mapped directly;
          otherwise inferred recursively from the first argument.
        - ``ast.BinOp`` — both operand dtypes are inferred and combined
          via a numpy-style promotion table.
        - ``ast.UnaryOp`` — propagated from the operand.

        Parameters
        ----------
        node : ast.AST
            Expression to infer a dtype for.

        Returns
        -------
        str
            A dtype string (e.g. ``'float64'``, ``'int32'``), or
            ``'unknown'`` if no rule matches.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool):
                    return "int64"
                if isinstance(val, int):
                    return "int64"
                if isinstance(val, float):
                    return "float64"
                if isinstance(val, complex):
                    return "complex128"

            if isinstance(node, ast.Name):
                name = node.id
                if name in self.symbol_table:
                    return self.symbol_table[name]
                local_arr = (
                    self.cls_info.get(self.cls_name, {})
                    .get("methods", {})
                    .get(self.func_name, {})
                    .get("local_arr", {})
                )
                if name in local_arr:
                    return local_arr[name].get("dtype", "unknown")
                attributes = self.cls_info.get(self.cls_name, {}).get("attributes", {})
                if name in attributes:
                    return attributes[name].get("dtype", "unknown")
                return "unknown"

            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    attributes = self.cls_info.get(self.cls_name, {}).get(
                        "attributes", {}
                    )
                    if node.attr in attributes:
                        return attributes[node.attr].get("dtype", "unknown")
                return "unknown"

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["zeros", "ones", "full"]:
                        for kw in node.keywords:
                            if kw.arg == "dtype" and isinstance(kw.value, ast.Constant):
                                return str(kw.value.value)
                        return "float64"
                    attr = node.func.attr
                    dtype_map = {
                        "int32": "int32",
                        "int64": "int64",
                        "float32": "float32",
                        "float64": "float64",
                        "bool_": "bool",
                    }
                    if attr in dtype_map:
                        return dtype_map[attr]
                if node.args:
                    return self.infer_dtype(node.args[0])
                return "unknown"

            if isinstance(node, ast.BinOp):
                left_dtype = self.infer_dtype(node.left)
                right_dtype = self.infer_dtype(node.right)

                promotion = {
                    frozenset(["complex128"]): "complex128",
                    frozenset(["float64", "complex128"]): "complex128",
                    frozenset(["float64"]): "float64",
                    frozenset(["float32", "float64"]): "float64",
                    frozenset(["float32"]): "float32",
                    frozenset(["int64", "float64"]): "float64",
                    frozenset(["int64", "float32"]): "float64",
                    frozenset(["int32", "float64"]): "float64",
                    frozenset(["int32", "float32"]): "float32",
                    frozenset(["int64"]): "int64",
                    frozenset(["int32", "int64"]): "int64",
                    frozenset(["int32"]): "int32",
                }
                key = frozenset([left_dtype, right_dtype])
                if key in promotion:
                    return promotion[key]
                if left_dtype != "unknown":
                    return left_dtype
                if right_dtype != "unknown":
                    return right_dtype
                return "unknown"

            if isinstance(node, ast.UnaryOp):
                return self.infer_dtype(node.operand)

            return "unknown"
        except Exception as e:
            self.logger.exception("Exception in infer_dtype:", e)
            raise

    def broadcast_scalar(
        self,
        expr: ast.AST,
        expr_axes: set,
        target_axes: list,
    ) -> ast.AST:
        """
        Insert ``None`` (newaxis) and ``Slice`` indices so *expr*'s
        shape matches *target_axes*.

        No-ops if *expr* has no axes at all, or if its axes already
        equal *target_axes* exactly.

        Parameters
        ----------
        expr : ast.AST
            Expression to broadcast.
        expr_axes : set
            Axes already present on *expr*.
        target_axes : list
            Axes *expr* must be broadcast to match, in order.

        Returns
        -------
        ast.AST
            *expr* unchanged, or wrapped in an ``ast.Subscript`` that
            inserts ``Slice()`` for axes already present and
            ``Constant(None)`` for new axes.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not expr_axes:
                return expr
            if expr_axes == target_axes:
                return expr

            index = []
            expr_axis_set = set(expr_axes)
            for ax in target_axes:
                if ax in expr_axis_set:
                    index.append(ast.Slice())
                else:
                    index.append(ast.Constant(value=None))

            return ast.Subscript(
                value=expr,
                slice=ast.Tuple(elts=index, ctx=ast.Load()),
                ctx=ast.Load(),
            )
        except Exception as e:
            self.logger.exception("Exception in broadcast_scalar:", e)
            raise

    def _emit_convergence_break(
        self,
        node: ast.If,
        vectorization_context: dict | None,
        assigned: list[str],
        used_after: set[str],
    ) -> list[ast.AST]:
        """
        Lower an ``index_loop`` conditional that contains a ``break``
        into a convergence-mask update sequence.

        Rather than literally breaking out of a ``lax.scan`` body
        (which JAX does not support), this builds an accumulating
        ``converged`` boolean array: once an element's condition fires
        the break, it is marked converged and excluded from all future
        updates to that element, even though the scan keeps iterating
        for every element together.

        Mask construction depends on which branch holds the ``break``:

        - **break in body** — elements where the mask is ``True`` have
          converged; updates (in ``orelse``) apply where the mask is
          ``False``.
        - **break in orelse** — elements where the mask is ``False``
          have converged; updates (in ``body``) apply where the mask is
          ``True``.

        The active update mask additionally excludes already-converged
        elements (``jnp.logical_and(jnp.logical_not(converged),
        update_mask)``), then a new ``if``-scoped :class:`Control` is
        pushed using that active mask before visiting the update
        statements. ``break``/``continue`` statements are dropped from
        the visited output. Finally ``converged`` is updated via
        ``jnp.logical_or``.

        Parameters
        ----------
        node : ast.If
            The conditional containing a ``break`` in ``body`` or
            ``orelse``.
        vectorization_context : Optional[dict]
            The active :class:`Control` context as a dict, or ``None``.
        assigned : list[str]
            Names assigned anywhere in the branches.
        used_after : set[str]
            Names assigned and later read, passed through to
            :meth:`_mask_vector_assign`.

        Returns
        -------
        list[ast.AST]
            The mask, active-mask, masked-update, and converged-update
            statements, in emission order.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            new_stmts = []

            node.test = self.visit(node.test)
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
            new_stmts.append(mask_assign)

            break_in_body = any(isinstance(s, ast.Break) for s in node.body)
            update_stmts = [
                s
                for s in (node.orelse if break_in_body else node.body)
                if not isinstance(s, ast.Break)
            ]

            if break_in_body:
                break_mask = mask_name
                update_mask = f"_mask_{self._mask_counter}"
                self._mask_counter += 1
                inv_assign = ast.Assign(
                    targets=[ast.Name(id=update_mask, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="logical_not",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=mask_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
                new_stmts.append(inv_assign)
            else:
                break_mask = f"_mask_{self._mask_counter}"
                self._mask_counter += 1
                inv_assign = ast.Assign(
                    targets=[ast.Name(id=break_mask, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="logical_not",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=mask_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
                new_stmts.append(inv_assign)
                update_mask = mask_name

            converged_name = self._get_or_create_converged_var(vectorization_context)

            active_mask_name = f"_mask_{self._mask_counter}"
            self._mask_counter += 1
            active_mask_assign = ast.Assign(
                targets=[ast.Name(id=active_mask_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_and",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()),
                                attr="logical_not",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id=converged_name, ctx=ast.Load())],
                            keywords=[],
                        ),
                        ast.Name(id=update_mask, ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
            new_stmts.append(active_mask_assign)

            if vectorization_context:
                ranks = self.subscript_ranks(node.test)
                mask_rank = ranks.get(node.test, 0)
                self._control_stack.append(
                    Control(
                        kind="if",
                        loop_info=vectorization_context["loop_info"],
                        transform_type="index_loop",
                        vectorization_axis=vectorization_context["vectorization_axis"],
                        metadata={
                            "current_mask_assign": active_mask_assign,
                            "current_mask_rank": mask_rank,
                        },
                    )
                )

            self._push_scope()
            for stmt in update_stmts:
                visited = self.visit(stmt)
                if isinstance(visited, list):
                    for v in visited:
                        if isinstance(v, ast.Break | ast.Continue):
                            continue
                        if isinstance(v, ast.Assign):
                            new_stmts.extend(
                                self._mask_vector_assign(
                                    v, active_mask_name, assigned, used_after
                                )
                            )
                        else:
                            new_stmts.append(v)
                elif isinstance(visited, ast.Break | ast.Continue):
                    pass
                elif isinstance(visited, ast.Assign):
                    new_stmts.extend(
                        self._mask_vector_assign(
                            visited, active_mask_name, assigned, used_after
                        )
                    )
                elif visited is not None:
                    new_stmts.append(visited)
            self._pop_scope()

            converged_update = ast.Assign(
                targets=[ast.Name(id=converged_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="logical_or",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id=converged_name, ctx=ast.Load()),
                        ast.Name(id=break_mask, ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
            new_stmts.append(ast.fix_missing_locations(converged_update))

            if self._control_stack and self._control_stack[-1].kind == "if":
                self._control_stack.pop()

            return new_stmts

        except Exception as e:
            self.logger.exception("Exception in _emit_convergence_break:", e)
            raise

    def _get_or_create_converged_var(self, vectorization_context: dict | None) -> str:
        """
        Return the name of the ``converged`` boolean array for the
        current scan scope, creating and registering it if absent.

        Falls back to a fixed name ``'_converged'`` when no scan is
        active (e.g. a convergence-break pattern outside any
        ``lax.scan``). Otherwise registers a fresh
        ``_converged_N = jnp.zeros(self.<dim>, dtype=jnp.bool_)``
        initialiser into the current scan context's ``carry``/
        ``introduced``/``mutated`` sets and stashes the init statement
        under ``ctx['converged_init']`` for :meth:`handle_scan` to
        prepend.

        Parameters
        ----------
        vectorization_context : Optional[dict]
            The active :class:`Control` context as a dict, used to
            determine the dimension to size the converged array with.

        Returns
        -------
        str
            The converged-variable name for this scope.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not self._scan_stack:
                return "_converged"

            ctx = self._scan_stack[-1]
            if "converged_var" in ctx:
                return ctx["converged_var"]

            name = f"_converged_{self._mask_counter}"
            self._mask_counter += 1
            ctx["converged_var"] = name

            ctx["carry"].add(name)
            ctx["introduced"].add(name)
            ctx["mutated"].add(name)

            vect_dim = (
                list(vectorization_context["loop_info"].keys())[0]
                if vectorization_context
                else "kjpindex"
            )
            init_stmt = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="zeros",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id="self", ctx=ast.Load()),
                            attr=vect_dim,
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[
                        ast.keyword(
                            arg="dtype",
                            value=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()),
                                attr="bool_",
                                ctx=ast.Load(),
                            ),
                        )
                    ],
                ),
            )
            ctx["converged_init"] = ast.fix_missing_locations(init_stmt)

            return name
        except Exception as e:
            self.logger.exception("Exception in _get_or_create_converged_var:", e)
            raise
