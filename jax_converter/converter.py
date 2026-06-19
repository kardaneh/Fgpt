import ast
from typing import Dict, List, Set, Union

from .analysis import _BranchAnalysis
from .array_updates import _ArrayUpdate
from .call_rewriting import _CallRewriting
from .conditionals import _ConditionalLowering
from .dynamic_loops import _DynamicLoop
from .loops import _LoopLowering
from .masking import _Masking
from .scope_utils import _Scope
from .vectorization import _Vectorization

from logger import Logger
from jax_utils import get_name, ReductionHandler, Control, VectorizationAnalyzer,\
CallEdge, WhileVectorToScalar
 
class JaxConverter(
    _BranchAnalysis,
    _ArrayUpdate,
    _CallRewriting,
    _ConditionalLowering,
    _DynamicLoop,
    _LoopLowering,
    _Masking,
    _Scope,
    _Vectorization,
    ast.NodeTransformer
    ):
    """
    Transform Python control flow into JAX/XLA-compatible functional primitives.

    Operates as an ``ast.NodeTransformer`` that rewrites a single class method
    in place, covering three categories of transformation:

    - **Conditionals** — ``if`` / ``elif`` / ``else`` blocks are classified
      and rewritten into one of:

      - ``jnp.where`` for pure value-select patterns (no side effects),
      - ``lax.cond`` with synthetic ``_if_true_N`` / ``_if_false_N`` helper
        functions for index-dependent or stateful branches,
      - vectorised mask assignments (``jnp.logical_and``, ``jnp.where``) for
        branches that operate element-wise over a vectorised axis.

    - **Loops** — ``for`` loops are classified and rewritten into one of:

      - ``lax.scan`` for sequential index loops that carry state,
      - vectorised body expansion (loop removed, body kept) for loops over
        a batch axis,
      - ``jax.vmap`` wrapping an ``eqx.internal.while_loop`` for loops that
        contain a ``while`` whose condition depends on the vectorised axis.

    - **Array updates** — in-place subscript assignments (``a[i] = v``) are
      rewritten to ``a = a.at[i].set(v)`` (or ``.add``, ``.multiply``, etc.)
      to satisfy JAX's immutability requirement.

    Synthetic helper functions generated during transformation are collected in
    :attr:`_pending_helpers` and finalised by :meth:`process_helpers` after the
    main visit pass completes.

    Parameters
    ----------
    cls_info : Dict
        Class metadata produced by ``get_class_info_from_ast``, containing
        attribute types, dimensions, dtypes, and per-method local arrays.
    logger : Logger, optional
        Logger instance for structured output.  A default :class:`Logger` is
        created if ``None`` is passed.
    mode : str, optional
        Differentiation mode.  ``'jax'`` (default) and ``'fwd'`` use
        ``lax``-based while loops; ``'bwd'`` uses ``checkpointed`` while loops
        to reduce memory during reverse-mode AD.

    Attributes
    ----------
    counter : int
        Monotonic counter for generating unique ``_if_true_N`` /
        ``_if_false_N`` helper names.
    for_counter : int
        Monotonic counter for generating unique ``_scan_body_N`` names.
    helpers : List[ast.FunctionDef]
        Finalised helper functions ready to be prepended to the module.
    _pending_helpers : List[ast.FunctionDef]
        Helper functions whose bodies still need a recursive visit pass
        (processed by :meth:`process_helpers`).
    _func_arg_stack : List[Set[str]]
        Stack of argument-name sets for the current lexical scope, used to
        decide whether a name should be passed by value or read from ``self``.
    _always_exclude : Set[str]
        Names never included as helper inputs regardless of usage
        (``self``, ``np``, ``jnp``, ``lax``, ``logging``, ``math``,
        ``range``).
    _context_stack : List[Dict]
        Scoped context pushed/popped around each helper-body visit, carrying
        ``level``, ``helper_args``, and ``mutated_before``.
    _control_stack : List[Control]
        Stack of :class:`Control` objects representing active loop /
        if-block scopes; drives vectorisation-axis and mask propagation.
    _mutated_attrs : Set[str]
        Class attributes written inside the current function body — drives
        ``self.attr`` → ``attr`` rewriting in helpers and return-statement
        construction.
    _var_modif : Dict[str, Set]
        Tracks which class attributes (``'attr'``) and function arguments
        (``'args'``) are modified and must appear in the return tuple.
    _outer_func_args : List[str] or None
        Argument list of the outermost function being transformed; set once
        on first :meth:`visit_FunctionDef` entry and cleared by
        :meth:`reset_all`.
    cls_info : Dict
        Reference to the full class-info dict passed at construction.
    cls_name : str or List[str]
        Name(s) of the class(es) found in *cls_info*.
    analyzer : VectorizationAnalyzer
        Classifies ``for`` and ``if`` nodes as ``index_loop``, ``vector``,
        ``masked``, etc.
    reduction : ReductionHandler
        Handles ``np.sum`` / ``np.mean`` / … rewrites under vectorisation.
    while_transformer : WhileVectorToScalar
        Rewrites ``while``-loop bodies into state-tuple form for
        ``eqx.internal.while_loop``.
    expr_deps : Dict[ast.AST, tuple]
        Maps AST expression nodes to the vectorisation-axis names they depend
        on; used for broadcast-shape inference.
    var_deps : Dict[str, tuple]
        Maps variable names to the vectorisation axes their assigned value
        depends on.
    call_edge : Union[CallEdge, List]
        Per-function call-dependency edges, set by :meth:`set_working_function`
        before each function is visited.
    ret_per_func : Dict[str, Dict]
        Stores ``var_modif_args`` and ``var_modif_attr`` for each completed
        function, consumed by :meth:`visit_Expr` when rewriting callee sites.
    var_state : Dict[str, Tuple[str, ast.AST]]
        Maps scalar variable names to ``('temporary' | 'stateful', node)``
        used to decide whether the variable must be lifted to an array under
        vectorisation.
    dynamic_variable_lift : Dict[str, Dict]
        Registry of scalar variables that have been promoted to arrays by the
        vectorisation pass, carrying shape, dtype, and axis metadata.
    symbol_table : Dict[str, str]
        Maps loop-index and scalar variable names to their inferred dtype
        strings (e.g. ``'int64'``, ``'float64'``).
    mode : str
        Active differentiation mode (``'jax'``, ``'fwd'``, or ``'bwd'``).
    fn_index : Dict[str, ast.FunctionDef]
        Index of function nodes whose bodies modified ``_var_modif``;
        used by :meth:`visit_Expr` for pre-call ``self`` update decisions.
    """

    def __init__(
        self,
        cls_info: Dict,
        logger: Logger = None,
        mode: str = 'jax',
    ) -> None:
        # Counters for unique helper-function name generation
        self.counter = 0
        self.for_counter = 0
        self._mask_counter = 0

        # Helper pipeline: pending = awaiting recursive visit; helpers = finalised
        self.helpers: List[ast.FunctionDef] = []
        self._pending_helpers: List[ast.FunctionDef] = []

        # Lexical-scope argument stack — outermost args set once on first FunctionDef
        self._func_arg_stack: List[Set[str]] = []
        self._always_exclude: Set[str] = {
            'self', 'np', 'jnp', 'lax', 'logging', 'math', 'range'
        }
        self._outer_func_args = None

        # Context stacks for nested helper / loop / conditional scopes
        self._context_stack: List[Dict] = []
        self._control_stack: List[Control] = []

        # Mutation tracking across the current function body
        self._mutated_attrs: Set[str] = set()
        self._var_modif: Dict[str, Set] = {'attr': set(), 'args': set()}

        # Class metadata
        self.cls_info = cls_info
        self.cls_name = (
            list(cls_info.keys())[0]
            if len(cls_info) == 1
            else list(cls_info.keys())
        )

        # Sub-transformers
        self.analyzer = VectorizationAnalyzer()
        self.reduction = ReductionHandler(cls_info=cls_info, cls_name=self.cls_name)
        self.while_transformer = WhileVectorToScalar()

        # Vectorisation dependency tracking
        # expr_deps: AST node → tuple of axis names it depends on
        # var_deps:  variable name → tuple of axis names
        self.expr_deps: Dict = {}
        self.var_deps: Dict = {}

        # Call-graph and return-value bookkeeping
        self.call_edge: Union[CallEdge, List] = []
        self.ret_per_func: Dict = {}
        self.fn_index: Dict = {}

        # Local-scope definition tracking (scope stack mirrors lexical blocks)
        self._local_defaults: Dict = {}
        self._local_defined_stack: List[Dict] = [{}]

        # Stateful-variable analysis (filled once before each function transform)
        self.var_state: Dict = {}
        self._stateful_vars_stack: List[Set] = []

        # Scan-specific stacks
        # _scan_stack: per-scan-body carry / mutated / introduced sets
        # _modified_ret_stack: attributes returned from nested calls, stacked by region
        self._scan_stack: List[Dict] = []
        self._modified_ret_stack: List[List[str]] = []

        # Dynamic variable lifting due to vectorisation promotion
        self.dynamic_variable_lift: Dict = {}
        self._lifted_vars: Set = set()
        self.dynamic_created_variables: Dict = {}

        # Index variables introduced by for-loop targets
        self.index_variables: Set = set()
        self.mask_axis_counter: int = 0

        # Scalar type table: variable name → dtype string
        self.symbol_table: Dict = {}

        # Rank inference cache (populated during subscript visits)
        self._inferred_ranks: Dict[str, int] = {}

        # Differentiation mode controls while-loop kind ('lax' vs 'checkpointed')
        self.mode: str = mode

        if logger is None:
            self.logger = Logger()
        else:
            self.logger = logger

        self.logger.show_header('JaxConverter')

    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Enter a function definition and apply pre-visit structural rewrites.

        Performs two operations before delegating to ``generic_visit``:

        1. Pushes the function's argument names onto :attr:`_func_arg_stack`
        so that inner helper generation can distinguish local arguments
        from class attributes.
        2. Calls :meth:`_lift_if_return` to hoist any ``if`` block that
        contains a ``return`` into a form the conditional lowering pass
        can handle uniformly.

        After visiting, records the function node in :attr:`fn_index` if any
        variable modifications were detected (used later by
        :meth:`visit_Expr`).

        Parameters
        ----------
        node : ast.FunctionDef
            The function node being entered.

        Returns
        -------
        ast.FunctionDef
            The transformed function node.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            args = [a.arg for a in node.args.args]
            self._func_arg_stack.append(args)
            if not self._outer_func_args:
                self._outer_func_args = args

            node = self._lift_if_return(node)
            new_node = self.generic_visit(node)

            if any(self._var_modif.values()):
                self.fn_index[node.name] = node

            return new_node

        except Exception as e:
            self.logger.exception('Exception in visit_FunctionDef:', e)
            raise

    def visit(self, node: ast.AST) -> ast.AST:
        """
        Visit a node in the AST while maintaining reduction context.

        This method overrides the base visitor to manage a reduction context stack.
        It ensures that reduction operations can correctly infer axis corrections,
        particularly in scalar contexts. The current AST node is pushed onto the
        reduction stack before visiting children and popped afterward.

        Parameters
        ----------
        node : ast.AST
            The AST node to visit.

        Returns
        -------
        ast.AST 
            The transformed node returned by the superclass visitor.
        """
        self.reduction.push_parent(node) # <--- this ensure that the reduction methods 
                                         # infer the proper axis correction in most case
                                         # when we require a scalar context 
        new_node = super().visit(node)
        self.reduction.pop_parent()
        return new_node
    
    def generic_visit(self, node: ast.AST) -> ast.AST:
        """
        Generic visitor for AST nodes with additional control-flow analysis.

        After visiting child nodes, this method optionally records dependencies
        on vectorization axes when inside a control-flow context. It enriches
        traversal by tracking how expressions depend on loop/vectorization axes.

        Parameters
        ----------
        node : ast.AST
            The AST node being visited.

        Returns
        -------
        ast.AST
            The result of the superclass generic visit.
        
        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            result = super().generic_visit(node)

            if self._control_stack:
                vectorization_context = self._control_stack[-1].to_dict()
                self._expr_depends_on_axes(
                    node,
                    vectorization_context["vectorization_axis"]
                )

            return result
        except Exception as e:
            self.logger.exception('Exception in generic_visit:', e)
            raise
    
    def is_full_slice(self, s: ast.Slice) -> bool:
        """
        Check whether a slice represents a full dimension slice.

        A full slice is defined as `[:]`, meaning no lower bound, upper bound,
        or step is specified.

        Parameters
        ----------
        s : ast.slice
            The slice node to evaluate.

        Returns
        -------
        bool
            True if the slice is equivalent to `[:]`, False otherwise.
        """
        return isinstance(s, ast.Slice) and s.lower is None and s.upper is None and s.step is None

    
    def _extract_offset(self, node: ast.AST, loop_index: str):
        """
        Extract a constant offset relative to a loop index expression.

        This method detects simple affine index patterns of the form:
        - i
        - i + c
        - i - c

        where `i` is a loop index and `c` is an integer constant.

        Parameters
        ----------
        node : ast.AST
            The AST node representing an index expression.
        loop_index : str
            Name of the loop index variable.

        Returns
        -------
        int or None
            The extracted integer offset if the pattern matches:
            - 0 for `i`
            - +c for `i + c`
            - -c for `i - c`
            Otherwise, returns None.
        
        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Name) and node.id == loop_index:
                return 0

            if isinstance(node, ast.BinOp):
                if isinstance(node.left, ast.Name) and node.left.id == loop_index:
                    if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                        if isinstance(node.op, ast.Add):
                            return node.right.value
                        if isinstance(node.op, ast.Sub):
                            return -node.right.value

            return None
        
        except Exception as e:
            self.logger.exception('Exception in _extract_offset:', e)
            raise
    
    def _replace_subscript_with_name(self, node: ast.AST) -> ast.AST:
        """
        Convert ``BoolOp`` (``and``/``or``) into chained bitwise ``BinOp``.

        Despite its name (retained for compatibility with existing call
        sites), the current implementation only rewrites ``ast.BoolOp``
        nodes — ``and`` becomes a chain of ``ast.BitAnd`` and ``or`` becomes
        a chain of ``ast.BitOr`` — so that the expression can be evaluated
        element-wise once converted to ``jnp`` calls downstream.

        Parameters
        ----------
        node : ast.AST
            Expression to rewrite — typically an ``if`` test copy.

        Returns
        -------
        ast.AST
            The rewritten node, or *node* unchanged if no ``BoolOp`` is
            present at the top level (children are still visited).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            class Replacer(ast.NodeTransformer):
                def visit_BoolOp(self, node):
                    node = self.generic_visit(node)
                    op = node.op
                    values = node.values

                    if isinstance(op, ast.And):
                        new_op = ast.BitAnd()
                    elif isinstance(op, ast.Or):
                        new_op = ast.BitOr()
                    else:
                        return node

                    left = values[0]
                    for right in values[1:]:
                        left = ast.BinOp(left=left, op=new_op, right=right)
                    return ast.copy_location(left, node)

            return Replacer().visit(node)

        except Exception as e:
            self.logger.exception('Exception in _replace_subscript_with_name:', e)
            raise
    
    def check_if_array(
        self,
        node: ast.AST,
        required_dims: List[str] = None,
    ) -> bool:
        """
        Return ``True`` if *node* represents a variable that is an array.

        Resolves the variable name from *node* (an ``Assign`` target, a
        ``Name``, or an ``Attribute``) and checks three sources in order:

        1. the current function's input dimensions (:attr:`func_input_dim`),
        2. the class's declared attributes (:attr:`cls_info`), and
        3. local arrays declared in the current method's metadata.

        Parameters
        ----------
        node : ast.AST
            Node to classify — ``ast.Assign``, ``ast.Name``, or
            ``ast.Attribute``. Any other type returns ``False``.
        required_dims : List[str], optional
            If given, the array must declare at least one dimension name
            in this list to count as a match; otherwise any array
            qualifies.

        Returns
        -------
        bool
            ``True`` if a matching array declaration is found in any of
            the three sources.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Assign):
                target_node = node.targets[0]
            elif isinstance(node, (ast.Name, ast.Attribute)):
                target_node = node
            else:
                return False

            var_name = get_name(target_node)
            if not var_name:
                return False

            def _has_required_dims(dimensions):
                if not required_dims:
                    return True
                return bool(set(required_dims) & set(dimensions))

            if self.func_input_dim and var_name in self.func_input_dim:
                if _has_required_dims(self.func_input_dim[var_name]):
                    return True

            cls_attrs = self.cls_info[self.cls_name]['attributes']
            if var_name in cls_attrs and 'dimensions' in cls_attrs[var_name]:
                if _has_required_dims(cls_attrs[var_name]['dimensions']):
                    return True

            func_locals = self.cls_info[self.cls_name]['methods'][self.func_name].get('local_arr', {})
            if var_name in func_locals and 'dimensions' in func_locals[var_name]:
                if _has_required_dims(func_locals[var_name]['dimensions']):
                    return True

            return False

        except Exception as e:
            self.logger.exception('Exception in check_if_array:', e)
            raise
    
    @property
    def _stateful_vars(self) -> Set:
        """
        Get the set of stateful variables for the current analysis scope.

        This property returns the stateful-variable set associated with the
        innermost active scope on the stateful-variable stack. If no scope
        is active, an empty set is returned.

        Returns
        -------
        set
            The set of stateful variable names in the current scope. Returns
            an empty set when no stateful-variable context is active.
        """
        return self._stateful_vars_stack[-1] if self._stateful_vars_stack else set()
    
    def is_arr_at_op_call(self, node: ast.AST, arr_name: str) -> bool:
        """
        Return ``True`` if *node* matches the pattern
        ``arr_name.at[...].<op>(...)`` where ``arr_name`` is a bare local
        variable.

        Walks down the attribute chain from *node* until reaching a
        ``Subscript`` (the ``.at[...]`` part), then confirms the subscript
        base is a plain ``ast.Name`` equal to *arr_name* — this does
        **not** match ``self.arr_name.at[...]``, since the base in that
        case is an ``ast.Attribute`` rather than an ``ast.Name``.

        Parameters
        ----------
        node : ast.AST
            Node to check — only ``ast.Call`` nodes can match.
        arr_name : str
            Expected array name at the base of the ``.at[...]`` chain;
            must refer to a plain local variable, not a class attribute.

        Returns
        -------
        bool
            ``True`` if *node* is a JAX ``.at[...].<op>(...)`` call rooted
            at a local ``Name`` matching *arr_name*.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(node, ast.Call):
                return False

            func = node.func
            if not isinstance(func, ast.Attribute):
                return False

            attr_chain = func
            while isinstance(attr_chain, ast.Attribute):
                attr_chain = attr_chain.value

            if not isinstance(attr_chain, ast.Subscript):
                return False
            if not isinstance(attr_chain.value, ast.Attribute):
                return False
            if not isinstance(attr_chain.value.value, ast.Name):
                return False
            if attr_chain.value.value.id != arr_name:
                return False

            return True

        except Exception as e:
            self.logger.exception('Exception in is_arr_at_op_call:', e)
            raise
    
    def _is_control_temporary(self, var_name: str, assigned, used_after) -> bool:
        """
        Determine whether a variable is a temporary value created within a control-flow block.

        A variable is considered temporary if it is assigned inside a control-flow
        construct, not part of a scan carry, not an array, and used later after assignment.

        Parameters
        ----------
        var_name : str
            Name of the variable being analyzed.
        assigned : set
            Set of variables assigned within the current control block.
        used_after : set
            Set of variables that are used after the current point in execution.

        Returns
        -------
        bool
            True if the variable is considered a temporary control-flow variable,
            False otherwise.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            # 1. Must be inside control flow
            if not self._control_stack:
                return False

            # 2. Must NOT be an array
            if self.check_if_array(ast.Name(id=var_name, ctx=ast.Load())):
                return False

            # 3. Must NOT be scan carry
            scan_carry_vars = []
            if self._scan_stack:
                scan_stack = self._scan_stack[-1]
                scan_carry_vars = scan_stack['carry']
            if var_name in scan_carry_vars: # self._scan_carry_vars
                return False

            # 4. Must be assigned in this control block
            if var_name not in assigned:
                return False

            # 5. Must be used later (RHS usage anywhere after)
            return var_name in used_after
        except Exception as e:
            self.logger.exception('Exception in _is_control_temporary:', e)
            raise
    
    def set_working_function(
        self,
        func_name: str,
        func_input_dim: Dict,
        call_edge: Union[CallEdge, List]
    ) -> None:
        """
        Set metadata for the currently analyzed working function.

        This method updates internal state with information about the function
        being processed, including its name, input dimensions, and call graph edge(s).

        Parameters
        ----------
        func_name : str
            Name of the function being analyzed.
        func_input_dim : Dict
            Mapping describing input tensor dimensions or shapes for the function.
        call_edge : Union[CallEdge, List]
            Representation of the call relationship(s) associated with the function.
            Can be a single CallEdge or a list of CallEdge objects.
        """
        self.func_name = func_name
        self.func_input_dim = func_input_dim
        self.call_edge = call_edge

    def reset_all(self) -> None:
        """
        Reset all internal analysis state to initial defaults.

        This method clears counters, variable tracking structures, mutation records,
        lifted variable sets, and inferred rank information. It is typically called
        before starting analysis of a new function or computation graph.
        """
        self.for_counter = 0
        self.counter = 0
        self._mask_counter = 0
        self._outer_func_args = None
        self._var_modif = {
            'attr': set(),
            'args': set()
        }
        self.var_state = {}
        self.dynamic_variable_lift = {}
        self._lifted_vars = set()
        self.index_variables = set()
        self._mutated_attrs = set()
        self._modified_ret_stack = []
        self._inferred_ranks = {}