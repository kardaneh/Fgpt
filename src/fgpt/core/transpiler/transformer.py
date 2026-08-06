# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

from __future__ import annotations

import ast
import copy
import os
import stat
import subprocess
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk

if TYPE_CHECKING:
    from fgpt.core.frontend.extractor import Extractor
    from fgpt.isolator import Isolator

from fgpt.core.common.logger import Logger
from fgpt.core.common.utils import (
    AdjustIndices,
    ReplaceGlobals,
    ast_walk,
    attach_instance,
    find_folder,
    find_used_globals,
    get_instance_name,
    identify_replace_all,
    load_code_templates,
    order_assignments,
    python_parser,
    safe_eval_expr,
    search_convar_dependencies,
    update_methods,
)
from fgpt.core.transpiler.f2np import F2NP
from fgpt.core.transpiler.intrinsic import build_fortran_reshape_helper

DEFAULT_TEMPLATE = Path(__file__).parent / "templates" / "default.yaml"


class Transformer:
    """
    Convert Fortran source code into Python code via an AST-based pipeline.

    The ``Transformer`` parses Fortran declarations and subroutines (with the
    help of :class:`~isolator.Isolator` and :class:`~extractor.Extractor`),
    builds an intermediate Python :class:`ast.Module` representation, and
    emits either a class-based "global module" file or an executable "main"
    script. It is the central orchestrator of the Fortran-to-Python
    translation pipeline: declaration conversion, class/instance metadata
    extraction, dependency resolution, function-call rewriting, array index
    adjustment, and binary I/O code generation all happen here.

    Parameters
    ----------
    benchmark_dir : str
        Directory used to store per-subroutine benchmark artifacts (timing
        files, binary dumps, etc.). If ``None``, a ``benchmark`` directory is
        created under the current working directory.
    isolator : Isolator
        Instance providing access to isolated subroutine ASTs
        (``working_subroutines``) and input ordering metadata
        (``input_dict``), used e.g. in :meth:`retreive_variable_order`.
    extractor : Extractor
        Instance providing extracted Fortran metadata (array info, dummy
        argument lists, call graphs, variable modification info, etc.)
        consumed throughout the class, notably in :meth:`correct_function`
        and :meth:`update_global_python`.
    ignore_case : list of str
        Names of variables or functions to be excluded from processing.
    config_path : str
        Path to the ``template.yaml`` configuration file containing the
        Python code templates used by :func:`load_code_templates`.
    logger : Logger, optional
        Logger instance used for structured logging and exception
        reporting. If ``None``, a default :class:`Logger` is created.

    Attributes
    ----------
    benchmark_dir : str
        Resolved benchmark directory path.
    ignore_case : list of str
        Names ignored during transformation.
    isolator : Isolator
        Reference to the isolator instance.
    extractor : Extractor
        Reference to the extractor instance.
    cls_mode : bool
        Whether the current output target is class-based.
    config_path : str
        Path to the template configuration file.
    global_state : bool
        Whether the current declaration set being processed is global
        (module-level) rather than local to a subroutine.
    logger : Logger
        Logger instance used across the class.
    f2np : F2NP
        Helper responsible for translating individual Fortran expressions
        and subroutines into Python AST nodes.
    pre_init : list of str
        Variable names already declared/initialized in the active code
        template; populated by :meth:`pre_init_variables`.
    dependant_variables : dict
        Mapping of array variable names to the (uninitialized) variables
        their shape bounds depend on; populated by
        :meth:`search_dependant_variables`.
    scalar : list of str
        Scalar/logical variable names for the current subroutine context;
        populated by :meth:`separate_scalar`.
    variable_order : list of str
        Order in which variables are read from the binary input file;
        populated by :meth:`retreive_variable_order`.

    Notes
    -----
    The typical pipeline driven by this class is:

    1. :meth:`update_global_python` builds the class-based global module
       AST for a subroutine, including nested child subroutines collected
       via :meth:`collect_descendants_dfs`.
    2. :meth:`update_main_python` builds the executable ``main()`` script
       that instantiates the global module, reads binary inputs, calls the
       translated subroutine, and optionally runs a generated comparison
       test (see :meth:`create_test_function`).
    3. :meth:`transfer_to_pyfile` writes the finalized AST to disk.
    4. :meth:`run_python_scripts` executes and validates the generated
       scripts against expected binary outputs.

    ``update_global_python``, ``update_main_python``, and
    ``run_python_scripts`` are wrapped with :meth:`Logger.log_event` at
    construction time so their execution is automatically logged.
    """

    def __init__(
        self,
        benchmark_dir: str,
        isolator: Isolator,
        extractor: Extractor,
        ignore_case: list[str],
        config_path: str,
        logger: Logger = None,
    ):
        if benchmark_dir is None:  # The benchmark directory
            current_dir = os.getcwd()
            self.benchmark_dir = os.path.join(current_dir, "benchmark")
            os.makedirs(self.benchmark_dir, exist_ok=True)
        else:
            self.benchmark_dir = benchmark_dir

        self.ignore_case = ignore_case  # List of string of variables or functions names that are to be ignored
        self.isolator = isolator  # An instance of isolator class just used to retrieve variable order
        self.extractor = extractor  # An instance of extractor class
        self.cls_mode = (
            False  # Defines if we should create a class global module or not
        )
        if config_path is None:
            self.config_path = DEFAULT_TEMPLATE
        else:
            self.config_path = Path(config_path).resolve()
        self.global_state = (
            False  # Allows to define if the given code template is for global or not
        )

        if logger is None:
            self.logger = Logger()
        else:
            self.logger = logger
        self.logger.show_header("Transformer")

        self.f2np = F2NP(
            extractor
        )  # Class in charge of transforming a subroutine from fortran to python

        # This meant to be done to ensure that we get the log events happening
        # inside the function thus wraps the method itself upon the wrapper function
        self.update_global_python = self.logger.log_event("Update_global_python")(
            self.update_global_python
        )
        self.update_main_python = self.logger.log_event("Update_main_python")(
            self.update_main_python
        )
        self.run_python_scripts = self.logger.log_event("Run python scripts")(
            self.run_python_scripts
        )

    def get_imports_from_specs(
        self,
        import_specs: list[tuple[str, list[str]]],
    ) -> list[ast.ImportFrom] | None:
        """
        Build ``ast.ImportFrom`` nodes from a list of ``(module, names)``
        specifications.

        Used by :meth:`create_cls_info` to generate the imports needed for
        each discovered class (the convention being that a class
        ``Foo`` lives in a module named ``foo``).

        Parameters
        ----------
        import_specs : list[tuple[str, list[str]]]
            Each tuple is ``(module_name, names_to_import)``. Specs with an
            empty *names* list are skipped.

        Returns
        -------
        list[ast.ImportFrom] or None
            One ``ImportFrom`` node per non-empty spec, or ``None`` if
            *import_specs* is empty or an error occurs.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """

        if not import_specs:
            return None

        import_nodes = []

        try:
            for module_name, names in import_specs:
                if not names:
                    continue

                aliases = [ast.alias(name=name) for name in names]
                import_node = ast.ImportFrom(module=module_name, names=aliases, level=0)
                import_nodes.append(import_node)

            return import_nodes

        except Exception as e:
            self.logger.exception("Exception in get_imports_from_specs", e)
            return None

    def create_instances(
        self,
        nodes: list,
        self_mode: bool | None = False,
    ) -> list[ast.Assign] | None:
        """
        Build ``ast.Assign`` nodes instantiating each given class.

        Generates ``var = ClassName()`` (or ``self.var = ClassName()`` when
        *self_mode* is ``True``), using :func:`get_instance_name` to derive
        the variable name from the class name. Used by
        :meth:`create_cls_info` to build the instance nodes paired with
        each discovered ``ast.ClassDef``.

        Parameters
        ----------
        nodes : list
            ``ast.ClassDef`` nodes to instantiate.
        self_mode : bool, optional
            If ``True``, targets are ``self.<name>`` attributes; otherwise
            plain local names.

        Returns
        -------
        list[ast.Assign] or None
            One instantiation assignment per class, or ``None`` if an
            error occurs.

        Raises
        ------
        ValueError
            If any element of *nodes* is not an ``ast.ClassDef``.
        Exception
            Re-raises any unexpected error after logging.
        """
        instance_nodes = []
        try:
            for node in nodes:
                if not isinstance(node, ast.ClassDef):
                    raise ValueError(f"Node is not an ast.ClassDef: {node}")

                instance_name = get_instance_name(node.name)
                # Create Class() constructor call
                constructor_call = ast.Call(
                    func=ast.Name(id=node.name, ctx=ast.Load()), args=[], keywords=[]
                )

                # Choose target based on self_mode
                if self_mode:
                    target = ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=instance_name,
                        ctx=ast.Store(),
                    )
                else:
                    target = ast.Name(id=instance_name, ctx=ast.Store())

                assign_node = ast.Assign(targets=[target], value=constructor_call)

                instance_nodes.append(assign_node)

            return instance_nodes

        except Exception:
            self.logger.exception(
                "Failed to create instance nodes in create_instances."
            )
            return None

    def create_cls_info(
        self,
        out_module: ast.Module,
        subroutine_key: str,
        instance_node: list | None = None,
        self_mode: bool | None = False,
    ) -> tuple[dict, list, list]:
        """
        Build the nested class/instance metadata structure consumed
        throughout the transformation pipeline.

        Walks *out_module* for ``ast.ClassDef`` nodes, extracts each
        class's attributes/methods/instances via
        :meth:`_extract_class_members`, generates the corresponding import
        nodes via :meth:`get_imports_from_specs` and instance nodes via
        :meth:`create_instances`, then assembles everything into the final
        ``cls_info`` dict via :meth:`_assemble_cls_info`. ``cls_info`` is
        the structure consumed by :meth:`correct_function`,
        :meth:`add_instance`, :meth:`update_global_python`, and
        :meth:`update_main_python`.

        Parameters
        ----------
        out_module : ast.Module
            Module AST to scan for class definitions.
        subroutine_key : str
            Identifier for the current subroutine, passed through to
            :meth:`_extract_class_members` for array-info lookups.
        instance_node : list, optional
            Pre-existing instance creation nodes, used when classes
            instantiate other classes internally.
        self_mode : bool, optional
            Whether the top-level instance should be bound to ``self``
            rather than a plain local name.

        Returns
        -------
        tuple[dict, list, list]
            ``(cls_info, import_nodes, instance_nodes)``.

        Raises
        ------
        ValueError
            if no class definitions are found, or instance/target nodes
            don't match expectations.
        Exception
            Re-raises any unexpected error after logging.

        Notes
        -----
        This function assumes standard class structure and may not detect dynamically defined
        attributes or methods.
        """
        try:
            class_defs = list(ast_walk(out_module, ast.ClassDef))
            class_members = {
                cls.name: self._extract_class_members(
                    cls, instance_node, subroutine_key
                )
                for cls in class_defs
            }

            specs = [(class_name.lower(), [class_name]) for class_name in class_members]
            import_nodes = self.get_imports_from_specs(specs)
            instance_nodes = self.create_instances(class_defs)

            cls_info = self._assemble_cls_info(
                class_defs, class_members, instance_nodes, instance_node, self_mode
            )

            return cls_info, import_nodes, instance_nodes

        except ValueError as e:
            self.logger.error("ValueError in create_cls_info.", e)
            raise
        except Exception:
            self.logger.exception("Failed to create class dict.")
            raise

    def _extract_class_members(
        self,
        class_def: list,
        instance_node: list,
        subroutine_key: str,
    ) -> dict:
        """
        Extract attributes, methods, and (optionally) instances from a
        single class definition.

        Every ``ast.Assign`` in *class_def* is classified via
        :meth:`_handle_assignment`, mutating the *attributes*/*instances*
        dicts in place; every ``ast.FunctionDef`` directly in the class
        body is recorded as a method. Used by :meth:`create_cls_info`.

        Parameters
        ----------
        class_def : list
            The ``ast.ClassDef`` node to inspect.
        instance_node : list
            Known instance creation nodes; when non-empty, an
            ``'instances'`` key is included in the result.
        subroutine_key : str
            Current subroutine identifier, forwarded to
            :meth:`_handle_assignment` for array-info resolution.

        Returns
        -------
        dict
            ``{'attributes': ..., 'methods': ..., 'instances': ...}``
            (the last key present only when *instance_node* is non-empty).
        """

        attributes = {}
        methods = {}
        instances = {}

        for assign in ast_walk(class_def, ast.Assign):
            self._handle_assignment(
                assign, attributes, instances, instance_node, subroutine_key
            )

        for node in class_def.body:
            if isinstance(node, ast.FunctionDef):
                methods[node.name] = node

        result = {
            "attributes": attributes,
            "methods": methods,
        }

        if instance_node:
            result["instances"] = instances

        return result

    def _handle_assignment(
        self,
        assign: ast.Assign,
        attributes: dict,
        instances: dict,
        instance_node: list,
        subroutine_key: str,
    ) -> None:
        """
        Classify a single ``self.<attr> = ...`` assignment and record it
        into *attributes* or *instances*.

        Dispatches by RHS shape: a plain class-constructor call →
        :meth:`_build_instance_entry`; ``np.int32``/``np.float64`` scalar →
        :meth:`_get_scalar_info`; ``np.bool`` → recorded directly;
        ``np.zeros``/``np.array`` → :meth:`_handle_array`; a bare
        ``self.other`` reference → :meth:`_handle_attribute_copy`. Mutates
        *attributes* and *instances* in place. Used by
        :meth:`_extract_class_members`.

        Parameters
        ----------
        assign : ast.Assign
            A class-body assignment node.
        attributes : dict
            Attribute metadata dict, mutated in place.
        instances : dict
            Instance metadata dict, mutated in place.
        instance_node : list
            Known instance creation nodes, forwarded to
            :meth:`_build_instance_entry`.
        subroutine_key : str
            Current subroutine identifier, forwarded to
            :meth:`_handle_array`.

        Notes
        -----
        - Detects and handles:
            - instance creation
            - numpy scalar attributes
            - numpy boolean attributes
            - numpy array attributes
            - attribute copying (self.x = self.y)
        - Mutates `attributes` and `instances` in-place.
        """
        for target in assign.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue

            name = target.attr

            # Instance creation
            if (
                instance_node is not None
                and isinstance(assign.value, ast.Call)
                and isinstance(assign.value.func, ast.Name)
            ):
                instances[name] = self._build_instance_entry(assign, instance_node)
                return

            # Scalar / dtype
            if (
                isinstance(assign.value, ast.Call)
                and isinstance(assign.value.func, ast.Attribute)
                and assign.value.func.attr in ["int32", "float64"]
            ):
                attributes[name] = self._get_scalar_info(assign, attributes)
                return

            # Boolean
            if (
                isinstance(assign.value, ast.Call)
                and isinstance(assign.value.func, ast.Attribute)
                and assign.value.func.attr in ["bool"]
            ):
                attributes[name] = [assign.value.args[0].value, "bool"]
                return

            # Array
            if (
                isinstance(assign.value, ast.Call)
                and isinstance(assign.value.func, ast.Attribute)
                and assign.value.func.attr in ["zeros", "array"]
            ):
                attributes[name] = self._handle_array(assign, name, subroutine_key)
                return

            # Attribute copy self.x = ... self.y = self.x
            if isinstance(assign.value, ast.Attribute):
                self._handle_attribute_copy(assign, attributes)

    def _build_instance_entry(self, assign: ast.AST, instance_node: list) -> dict:
        """
        Build the metadata entry for a ``self.<attr> = SomeClass()``
        instance-creation assignment.

        Matches *assign*'s constructor call against *instance_node* by
        class name to confirm the instance is one of the already-known
        instantiations. Used by :meth:`_handle_assignment`.

        Parameters
        ----------
        assign : ast.AST
            The instance-creation assignment.
        instance_node : list
            Known instance creation nodes to match against.

        Returns
        -------
        dict
            ``{'class_name': <target node>, 'attributes': {}, 'methods': {}}``.

        Raises
        ------
        ValueError
            If no entry in *instance_node* matches *assign*'s class name.
        """
        class_name = assign.value.func.id  # Global_module_xxx

        for instance in instance_node:
            if (
                isinstance(instance, ast.Assign)
                and isinstance(instance.value, ast.Call)
                and isinstance(instance.value.func, ast.Name)
                and instance.value.func.id == class_name
            ):
                return {
                    "class_name": instance.targets[0],
                    "attributes": {},
                    "methods": {},
                }

        raise ValueError("Instance node and class name differ")

    def _handle_attribute_copy(self, assign: ast.AST, attributes: dict) -> None:
        """
        Propagate dtype metadata for a ``self.x = self.y`` attribute-copy
        assignment.

        No-ops unless both target and value are ``self.attr`` references
        and the source attribute already has recorded dtype metadata in
        *attributes* (relying on declaration order: the source is assumed
        to have been processed earlier). Used by :meth:`_handle_assignment`.

        Parameters
        ----------
        assign : ast.AST
            The attribute-copy assignment.
        attributes : dict
            Attribute metadata dict, mutated in place.

        Raises
        ------
        KeyError
            if *attributes* access patterns change to
            require the source key to exist rather than tolerating its
            absence.
        Exception
            Re-raises any unexpected error after logging.

        Notes
        -----
        - Handles simple attribute copies only.
        - Inherits dtype metadata from source attribute.
        - Does nothing if source attribute metadata is missing.
        """
        try:
            target = assign.targets[0]

            if not isinstance(target, ast.Attribute):
                return

            value = assign.value

            # Only handle simple attribute access like: self.x = self.y
            if not isinstance(value, ast.Attribute):
                return

            source_name = value.attr
            target_name = target.attr

            if not isinstance(value.value, ast.Name):
                return

            # Lookup dtype from existing attributes
            dtype_info = attributes.get(source_name)

            if dtype_info:
                attributes[target_name] = [
                    source_name,
                    dtype_info[
                        1
                    ],  # inherit dtype This is because we the self.y is usually present
                    # before the self.x thus the type of self.x is that of self.y
                ]
        except KeyError as e:
            self.logger.error("KeyError in _handle_attribute_copy.", e)
            raise
        except Exception:
            self.logger.exception("Exception in _handle_attribute_copy.")
            raise

    def _get_scalar_info(self, assign: ast.AST, attributes: dict) -> list:
        """
        Extract ``[value, dtype]`` metadata from a NumPy scalar assignment.

        Constant or simple ``BinOp`` arguments are evaluated via
        :func:`safe_eval_expr` against *attributes* (so expressions
        depending on already-declared attributes can resolve); other
        argument shapes are kept symbolic. Used by
        :meth:`_handle_assignment`.

        Parameters
        ----------
        assign : ast.AST
            The scalar-constructor assignment (``np.int32(...)`` /
            ``np.float64(...)``).
        attributes : dict
            Existing attribute metadata, used for expression evaluation.

        Returns
        -------
        list
            ``[value, dtype]`` where *value* is the evaluated constant or
            the raw AST argument.

        Raises
        ------
        Exception
            Propagated from :func:`safe_eval_expr` if
            expression evaluation fails on a malformed ``BinOp``.
        """
        try:
            arg = assign.value.args[0]
            dtype = assign.value.func.attr

            if isinstance(arg, ast.BinOp | ast.Constant):
                value = safe_eval_expr(arg, attributes)
                return [value, dtype]

            return [arg, dtype]
        except Exception as e:
            self.logger.exception("Exception in _get_scalar_info:", e)
            raise

    def _handle_array(self, assign: ast.AST, name: str, subroutine_key: str) -> list:
        """
        Extract ``[array_info, dtype]`` metadata from a NumPy array
        assignment.

        Array shape/type metadata is resolved via
        :meth:`_resolve_array_info`; *dtype* is read from the assignment's
        ``dtype`` keyword if present. Used by :meth:`_handle_assignment`.

        Parameters
        ----------
        assign : ast.AST
            The array-constructor assignment (``np.zeros(...)`` /
            ``np.array(...)``).
        name : str
            Name of the array attribute.
        subroutine_key : str
            Current subroutine identifier, forwarded to
            :meth:`_resolve_array_info`.

        Returns
        -------
        list
            ``[array_info, dtype]``; *dtype* may be ``None`` if no
            ``dtype`` keyword was present.

        Raises
        ------
        ValueError
            if array metadata cannot be found in any context.
        """
        try:
            array_info = self._resolve_array_info(name, subroutine_key)

            dtype = None
            for kw in assign.value.keywords:
                if kw.arg == "dtype":
                    dtype = getattr(kw.value, "attr", getattr(kw.value, "id", None))

            return [array_info, dtype]
        except ValueError as e:
            self.logger.error("ValueError in _handle_array:", e)
            raise

    def _resolve_array_info(self, name: str, subroutine_key: str) -> dict:
        """
        Look up array shape/type metadata for *name*, searching the
        current subroutine first and falling back to other isolated
        subroutines.

        Case-insensitive lookup against :attr:`extractor`'s
        ``all_array_info``. Used by :meth:`_handle_array`.

        Parameters
        ----------
        name : str
            Array variable name to resolve.
        subroutine_key : str
            Primary subroutine context to search first.

        Returns
        -------
        dict
            The array's metadata dictionary.

        Raises
        ------
        ValueError
            If *name* is not found in the current subroutine's array info
            nor in any of :attr:`isolator`'s ``working_subroutines``.
        """

        def lookup(arrays):
            return {k.casefold(): v for k, v in arrays.items()}.get(name.casefold())

        # current subroutine
        arrays = self.extractor.all_array_info[subroutine_key]
        result = lookup(arrays)

        if result:
            return result

        # fallback search
        for key in self.isolator.working_subroutines:
            arrays = self.extractor.all_array_info[key]
            result = lookup(arrays)
            if result:
                self.logger.info(f"Found array_info for {name} in {key}")
                return result

        raise ValueError(f"Array info not found for {name}")

    def _assemble_cls_info(
        self,
        class_defs: list,
        class_members: dict,
        instance_nodes: list,
        instance_node: ast.AST,
        self_mode: bool,
    ) -> dict:
        """
        Combine per-class member dicts and instance nodes into the final
        ``class_name -> instance_name -> metadata`` structure.

        Used by :meth:`create_cls_info` as the final assembly step.

        Parameters
        ----------
        class_defs : list
            ``ast.ClassDef`` nodes, in the same order as *instance_nodes*.
        class_members : dict
            Per-class ``{'attributes': ..., 'methods': ..., 'instances': ...}``
            from :meth:`_extract_class_members`, keyed by class name.
        instance_nodes : list
            Instance creation nodes, in the same order as *class_defs*.
        instance_node : ast.AST
            Truthy when nested instance metadata should be included in the
            assembled entry.
        self_mode : bool
            Whether the instance name should be forced to ``"self"``.

        Returns
        -------
        dict
            ``{class_name: {instance_name: {'attributes': ..., 'methods': ..., ['instances': ...]}}}``.

        Raises
        ------
        TypeError
            If a corresponding instance node is not an ``ast.Assign``, or
            its target is not an ``ast.Name``.
        """
        cls_info = {}

        for cls, inst in zip(class_defs, instance_nodes):
            class_name = cls.name

            if not isinstance(inst, ast.Assign):
                raise TypeError("Invalid instance node")

            target = inst.targets[0]
            if not isinstance(target, ast.Name):
                raise TypeError("Invalid instance target")

            instance_name = target.id if not self_mode else "self"

            entry = {
                "attributes": class_members[class_name]["attributes"],
                "methods": class_members[class_name]["methods"],
            }

            if instance_node:
                entry["instances"] = class_members[class_name]["instances"]

            cls_info[class_name] = {instance_name: entry}

        return cls_info

    def add_instance(
        self,
        idx: int,
        instance_node: ast.AST,
        cls_info: dict,
        functions_def: ast.FunctionDef,
        method_names: list[str],
    ) -> None:
        """
        Insert an instance-creation node, plus optional follow-up method
        calls, into a function body.

        The insertion index is first resolved via
        :meth:`_resolve_insert_index` and adjusted to respect existing
        dependency ordering via :meth:`_adjust_for_dependencies`. After
        inserting the instance, requested method calls are built via
        :meth:`_build_method_calls` and inserted immediately after.

        Parameters
        ----------
        idx : int
            Desired insertion index (passed through to
            :meth:`_resolve_insert_index`).
        instance_node : ast.AST
            The instance-creation ``ast.Assign`` to insert.
        cls_info : dict
            Class metadata used to resolve method ASTs in
            :meth:`_build_method_calls`.
        functions_def : ast.FunctionDef
            Function whose body is mutated in place.
        method_names : list[str]
            Names of methods to call on the new instance, in order.

        Raises
        ------
        NotImplementedError
            If *functions_def* is falsy (insertion outside a function body
            is unsupported).
        """

        if not functions_def:
            raise NotImplementedError(
                "add_instance only supports insertion inside functions"
            )

        try:
            insert_idx = self._resolve_insert_index(idx, functions_def)
            insert_idx = self._adjust_for_dependencies(insert_idx, functions_def)

            # Insert instance
            functions_def.body.insert(insert_idx, instance_node)
            insert_idx += 1

            # Add method calls
            calls = self._build_method_calls(instance_node, cls_info, method_names)

            for call in calls:
                functions_def.body.insert(insert_idx, call)
                insert_idx += 1

        except Exception:
            self.logger.exception("Exception in add_instance")
            raise

    def _resolve_insert_index(self, idx: int, func_def: ast.FunctionDef) -> int:
        """
        Resolve a clamped insertion index, defaulting to just after the
        last ``ast.Assign`` in the function body when *idx* is ``None``.

        Used by :meth:`add_instance`.

        Parameters
        ----------
        idx : int or None
            Caller-supplied index, or ``None`` for the default heuristic.
        func_def : ast.FunctionDef
            Function whose body length bounds the result.

        Returns
        -------
        int
            A valid index in ``[0, len(func_def.body)]``.

        Notes
        -----
        - If `idx` is provided, it is clamped to valid bounds.
        - If `idx` is None, the index is set after the last assignment.
        - Falls back to appending at the end if no assignments are found.
        """
        if idx is not None:
            return max(0, min(idx, len(func_def.body)))

        # default: after last assignment
        for i in reversed(range(len(func_def.body))):
            if isinstance(func_def.body[i], ast.Assign):
                return i + 1

        return len(func_def.body)

    def _adjust_for_dependencies(self, idx: int, func_def: ast.FunctionDef) -> int:
        """
        Push an insertion index forward past the first "complex"
        assignment (one whose RHS is a call, attribute, or subscript) if
        *idx* would otherwise precede it.

        Used by :meth:`add_instance` to avoid inserting an instance
        creation before a statement that might already depend on
        constructs the instance provides.

        Parameters
        ----------
        idx : int
            Initial insertion index.
        func_def : ast.FunctionDef
            Function whose body is scanned.

        Returns
        -------
        int
            The original *idx*, or the index of the first complex
            assignment if that comes later than *idx*.

        Notes
        -----
        - Moves the insertion point forward if it precedes a "complex" assignment.
        - Complex assignments include calls, attributes, or subscript expressions.
        - Ensures dependent computations are not bypassed.
        """
        for i, stmt in enumerate(func_def.body):
            if isinstance(stmt, ast.Assign):
                # heuristic: first "complex" assignment
                if isinstance(stmt.value, ast.Call | ast.Attribute | ast.Subscript):
                    if idx < i:
                        self.logger.info(f"Adjusting insert index from {idx} -> {i}")
                        return i
                    break
        return idx

    def _get_instance_info(
        self,
        instance_node: ast.AST,
    ) -> tuple[str, ast.Name] | tuple[str, ast.Attribute]:
        """
        Extract an instance's name and a load-context reference node from
        its creation assignment.

        Used by :meth:`_build_method_calls`.

        Parameters
        ----------
        instance_node : ast.AST
            The instance-creation ``ast.Assign``.

        Returns
        -------
        Union[tuple[str, ast.Name], tuple[str, ast.Attribute]]
            ``(name, reference_node)``, where *reference_node* is a fresh
            ``ast.Name`` (for plain-name targets) or the original
            ``ast.Attribute`` target (for ``self.attr`` targets).

        Raises
        ------
        TypeError
            If the assignment's target is neither an ``ast.Name`` nor an
            ``ast.Attribute``

        Notes
        -----
        - Supports simple names and attribute-based targets.
        """
        target = instance_node.targets[0]

        if isinstance(target, ast.Name):
            return target.id, ast.Name(id=target.id, ctx=ast.Load())

        if isinstance(target, ast.Attribute):
            return target.attr, target

        raise TypeError("Unsupported instance target")

    def _build_method_calls(
        self,
        instance_node: ast.AST,
        cls_info: dict,
        method_names: list,
    ) -> list:
        """
        Build call statements for each requested method on a newly created
        instance.

        Resolves each method's AST via *cls_info*, builds the call via
        :meth:`create_call_statements`. Methods not found are skipped with
        a warning rather than raising, since some requested method names
        may legitimately not exist for a given class. Used by
        :meth:`add_instance`.

        Parameters
        ----------
        instance_node : ast.AST
            The instance-creation assignment.
        cls_info : dict
            Class metadata used to resolve method definitions.
        method_names : list
            Method names to build calls for, in order.

        Returns
        -------
        list
            Call statement AST nodes (``ast.Expr``/``ast.Assign``), one per
            successfully resolved method.

        Notes
        -----
        - Only processes instances created via direct class calls.
        - Skips methods not found in the class metadata.
        - Returns an empty list if no valid methods are available.
        """
        if not method_names:
            return []

        if not isinstance(instance_node.value, ast.Call):
            return []

        if not isinstance(instance_node.value.func, ast.Name):
            return []

        class_name = instance_node.value.func.id
        instance_name, instance_ref = self._get_instance_info(instance_node)

        methods = cls_info.get(class_name, {}).get(instance_name, {}).get("methods")

        if not methods:
            self.logger.info(f"Class {class_name} has no methods")
            return []

        calls = []

        for method in method_names:
            method_ast = methods.get(method)

            if not method_ast:
                self.logger.warning(
                    f"Method '{method}' not found in class '{class_name}' skipping..."
                )
                continue

            calls.append(self.create_call_statements(method_ast, instance_ref))

        return calls

    def create_call_statements(
        self,
        function_ast: ast.FunctionDef,
        instance: str | ast.AST | None = None,
    ) -> ast.Expr | ast.Assign:
        """
        Build a call statement for *function_ast*, choosing between a bare
        expression and an assignment based on whether the function returns
        a value.

        Detects method-vs-function calling convention from whether the
        first argument is named ``self``. Arguments are built via
        :meth:`_build_args`. If the function contains a non-``None``
        ``return``, the call is wrapped in an assignment via
        :meth:`_build_assignment`; otherwise a bare ``ast.Expr`` is
        returned. Used throughout the pipeline (:meth:`_build_method_calls`,
        :meth:`update_main_python`) to synthesise call sites for already-
        translated functions/methods.

        Parameters
        ----------
        function_ast : ast.FunctionDef
            The function/method definition to call.
        instance : str or ast.AST, optional
            The instance to call the method on (a name string, an AST
            reference node, or ``None`` to default to ``self``). Ignored
            for non-method functions.

        Returns
        -------
        Union[ast.Expr, ast.Assign]
            The constructed call statement.

        Raises
        ------
        ValueError
            If *instance* is given but the function is not a method (its
            first argument is not ``self``).
        TypeError
            If *instance* is neither ``None``, a ``str``, nor an
            ``ast.AST``.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            function_def = function_ast
            function_name = function_def.name

            is_method = (
                function_def.args.args and function_def.args.args[0].arg == "self"
            )

            if instance and not is_method:
                raise ValueError(
                    "If instance node given and is_method=False `self` argument not present inside the method"
                )

            args = self._build_args(function_def=function_def)
            # Build call target
            if is_method:
                if instance is None:
                    value = ast.Name(id="self", ctx=ast.Load())
                elif isinstance(instance, str):
                    value = ast.Name(id=instance, ctx=ast.Load())
                elif isinstance(instance, ast.AST):
                    value = instance
                else:
                    raise TypeError(f"Unexpected type for instance: {type(instance)}")

                func = ast.Attribute(value=value, attr=function_name, ctx=ast.Load())
            else:
                func = ast.Name(id=function_name, ctx=ast.Load())

            call_expr = ast.Call(func=func, args=args, keywords=[])

            # Detect return
            return_stmt = next(
                (
                    n
                    for n in ast.walk(function_def)
                    if isinstance(n, ast.Return) and n.value is not None
                ),
                None,
            )

            if return_stmt:
                return self._build_assignment(return_stmt, call_expr)

            return ast.Expr(value=call_expr)

        except Exception as e:
            self.logger.exception("Exception in create_call_statements", e)
            raise

    def _build_assignment(
        self, return_node: ast.AST, call_expr: ast.Call
    ) -> ast.Assign | None:
        """
        Build an assignment mapping a function's return targets to a call
        expression.

        Handles single-name returns (``a = func(...)``) and tuple returns
        (``(a, b) = func(...)``, with non-``Name`` tuple elements silently
        dropped). Used by :meth:`create_call_statements`.

        Parameters
        ----------
        return_node : ast.AST
            The function's ``ast.Return`` node.
        call_expr : ast.Call
            The call expression to assign.

        Returns
        -------
        Optional[ast.Assign]
            The constructed assignment, or ``None`` if a tuple return
            contained no ``ast.Name`` elements.

        Raises
        ------
        AttributeError
            If the return value is neither an ``ast.Tuple`` nor an
            ``ast.Name``.

        Notes
        -----
        - Supports single variable returns and tuple unpacking.
        - Ignores non-name elements in tuple returns.
        - Produces assignments of the form:
            - `a = func(...)`
            - `(a, b) = func(...)`
        """
        value = return_node.value

        if isinstance(value, ast.Tuple):
            targets = [
                ast.Name(id=elt.id, ctx=ast.Store())
                for elt in value.elts
                if isinstance(elt, ast.Name)
            ]
            if targets:
                return ast.Assign(
                    targets=[ast.Tuple(elts=targets, ctx=ast.Store())], value=call_expr
                )

        elif isinstance(value, ast.Name):
            return ast.Assign(
                targets=[ast.Name(id=value.id, ctx=ast.Store())], value=call_expr
            )

        else:
            raise AttributeError(f"Unexcpected return_node value type:{type(value)}")

    def _build_args(self, function_def: ast.FunctionDef) -> list:
        """
        Build the positional/keyword/starred argument list for a call to
        *function_def*, excluding ``self``.

        Used by :meth:`create_call_statements`.

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function whose signature determines the call arguments.

        Returns
        -------
        list
            Argument expression nodes (``ast.Name``/``ast.Starred``) in
            signature order: positional-only and regular args, then
            ``*args``, then keyword-only args.

        Notes
        -----
        - Excludes the 'self' argument.
        - Includes positional, keyword-only, and variadic (*args) arguments.
        """
        args = []

        for arg in function_def.args.posonlyargs + function_def.args.args:
            if arg.arg != "self":
                args.append(ast.Name(id=arg.arg, ctx=ast.Load()))

        if function_def.args.vararg:
            args.append(
                ast.Starred(
                    value=ast.Name(id=function_def.args.vararg.arg, ctx=ast.Load()),
                    ctx=ast.Load(),
                )
            )

        for arg in function_def.args.kwonlyargs:
            args.append(ast.Name(id=arg.arg, ctx=ast.Load()))

        return args

    def get_timer(self, subroutine_key: str) -> ast.FunctionDef | None:
        """
        Build the ``@timer`` decorator function from the configured
        template.

        Substitutes *subroutine_key* into the benchmark output path before
        parsing. Used by :meth:`update_global_python` to attach timing to
        the outermost translated subroutine.

        Parameters
        ----------
        subroutine_key : str
            Current subroutine identifier, used to build the benchmark
            output path.

        Returns
        -------
        Optional[ast.FunctionDef]
            The parsed timer function, or ``None`` if templates fail to
            load or parsing fails.

        Raises
        ------
        ValueError
            if templates could not be loaded.
        SyntaxError
            if the rendered template is not valid Python.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            templates = load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")

            code_template = templates["Python_templates"]["Python_timer_template"][
                "template"
            ]
            path = f"{self.benchmark_dir}/{subroutine_key}/time.txt"
            code_template = Template(code_template).substitute(path=f'"{path}"')
            try:
                parsed_ast = ast.parse(code_template).body[0]
                return parsed_ast
            except SyntaxError as e:
                self.logger.error(f"Syntax error while parsing the timer template: {e}")
                raise
        except Exception:
            self.logger.exception("Exception occurred in get_timer")
            return None

    def correct_function(
        self,
        function_def: ast.FunctionDef,
        cls_info: dict,
        subroutine_key: str | None = None,
        main_file_attributes: list | None = None,
    ) -> None:
        """
        Normalise a freshly translated function: fix its argument list,
        rewrite nested calls, compute and insert return values, and adjust
        array indices.

        Five-step pipeline, each delegated to a dedicated helper:

        1. :meth:`_get_primary_instance` resolves the function's owning
        module/instance.
        2. :meth:`_fix_function_arguments` strips global/instance-bound
        arguments and ensures ``self``/instance is present when needed.
        3. :meth:`_fix_function_calls` rewrites nested calls to known
        internal subroutines with correctly mapped arguments.
        4. :meth:`_compute_return_variables` determines which scalar
        OUT/INOUT variables must be returned.
        5. :meth:`_adjust_function_indices` applies Fortran→Python index
        correction via ``AdjustIndices``.

        A return statement is appended via :meth:`_insert_return` if
        step 4 produced any return variables. Used by
        :meth:`_process_procedures` for every translated subroutine.

        Parameters
        ----------
        function_def : ast.FunctionDef
            The translated function to normalise, mutated in place.
        cls_info : dict
            Class metadata for the owning module/instance.
        subroutine_key : str, optional
            Current subroutine identifier.
        main_file_attributes : list, optional
            Attribute names defined at the main-file level, excluded from
            scalar return consideration.
        """

        try:
            module_name, instance_name, instance_data = self._get_primary_instance(
                cls_info
            )

            global_attr = instance_data["attributes"]

            # 1. Fix arguments
            self._fix_function_arguments(
                function_def,
                instance_name,
                global_attr,
                instance_data,
            )

            # 2. Fix nested function calls
            self._fix_function_calls(
                function_def,
                cls_info,
                subroutine_key,
                module_name,
                instance_name,
            )

            # 3. Compute return variables
            return_list, scalar_variables = self._compute_return_variables(
                function_def,
                cls_info,
                subroutine_key,
                main_file_attributes,
                module_name,
                instance_name,
            )

            # 4. Adjust indices (Fortran -> Python)
            self._adjust_function_indices(
                function_def,
                cls_info,
                subroutine_key,
                scalar_variables,
                module_name,
                instance_name,
            )

            # 5. Insert return
            if return_list:
                self._insert_return(function_def, return_list)

        except Exception:
            self.logger.exception("Exception in correct_function")
            raise

    def _get_primary_instance(self, cls_info: dict) -> tuple:
        """
        Extract the first ``(module_name, instance_name, instance_data)``
        triple from a ``cls_info`` structure.

        Used by :meth:`correct_function`.

        Parameters
        ----------
        cls_info : dict
            Nested ``module -> instance -> metadata`` structure.

        Returns
        -------
        tuple
            ``(module_name, instance_name, instance_data)``.

        Raises
        ------
        StopIteration
            if *cls_info* is empty.
        """
        try:
            module_name = next(iter(cls_info))
            instance_name = next(iter(cls_info[module_name]))
            instance_data = cls_info[module_name][instance_name]
            return module_name, instance_name, instance_data
        except StopIteration:
            raise

    def _fix_function_arguments(
        self,
        function_def: ast.FunctionDef,
        instance_name: str,
        global_attr: dict,
        instance_data: dict,
    ) -> None:
        """
        Remove arguments that duplicate global or instance attributes, and
        ensure the function still has a way to reach them (via ``self`` or
        the instance name).

        Used by :meth:`correct_function` (step 1).

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function whose argument list is mutated in place.
        instance_name : str
            Name to (re-)insert as an argument if global/instance
            references remain inside the body — ``"self"`` for class
            methods, otherwise the bare instance variable name.
        global_attr : dict
            Global attribute names to strip from the argument list.
        instance_data : dict
            Metadata for nested instances, whose attributes are also
            stripped from the argument list.
        """

        args = function_def.args.args
        arg_names = [a.arg for a in args]

        global_args = set(arg_names) & set(global_attr)

        other_args = set()
        for inst in instance_data.get("instances", {}).values():
            other_args |= set(arg_names) & set(inst.get("attributes", []))

        to_remove = global_args | other_args

        function_def.args.args = [
            a for a in args if a.arg not in to_remove and a.arg != "self"
        ]

        used_globals = find_used_globals(function_def, global_attr)

        if to_remove or used_globals:
            if instance_name == "self":
                if not any(a.arg == "self" for a in function_def.args.args):
                    function_def.args.args.insert(0, ast.arg(arg="self"))
            else:
                if not any(a.arg == instance_name for a in function_def.args.args):
                    function_def.args.args.append(ast.arg(arg=instance_name))

    def _fix_function_calls(
        self,
        function_def: ast.FunctionDef,
        cls_info: dict,
        subroutine_key: str,
        module_name: str,
        instance_name: str,
    ) -> None:
        """
        Rewrite arguments of nested calls to other internal subroutines so
        they match the callee's resolved dummy-argument mapping.

        Only calls whose target name appears in :attr:`extractor`'s
        ``call_within_sub`` for *subroutine_key* are touched; arguments are
        resolved per call via :meth:`_resolve_call_arguments`. Used by
        :meth:`correct_function` (step 2).

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function whose nested calls are rewritten in place.
        cls_info : dict
            Class metadata used to resolve the callee's method AST.
        subroutine_key : str
            Current subroutine identifier.
        module_name : str
            Owning module name, used to index into *cls_info*.
        instance_name : str
            Owning instance name, used to index into *cls_info*.
        """
        call_indices = defaultdict(int)

        for node in ast_walk(function_def, ast.Call):
            if not isinstance(node.func, ast.Name):
                continue

            func_name = node.func.id

            if func_name not in self.extractor.call_within_sub[subroutine_key]:
                continue

            method = cls_info[module_name][instance_name]["methods"].get(func_name)

            new_args = self._resolve_call_arguments(func_name, call_indices, method)
            if new_args:
                node.args = new_args

    def _resolve_call_arguments(
        self,
        func_name: str,
        call_indices: dict,
        method: ast.FunctionDef,
    ) -> list:
        """
        Resolve the actual argument expressions for one call to
        *func_name*, mapping the callee's dummy-argument positions back to
        the call site recorded by :attr:`extractor`.

        Tracks per-function call occurrence counts in *call_indices* (since
        a routine may be called multiple times and each call's actual
        arguments must be matched independently). Falls back from
        subroutine-style ``Call_Stmt`` argument lists to function-style
        ``Part_Ref``/``Assignment_Stmt`` parsing when no ``Call_Stmt`` is
        found. Used by :meth:`_fix_function_calls`.

        Parameters
        ----------
        func_name : str
            Name of the called function/subroutine.
        call_indices : dict
            Mutable per-function call-occurrence counter.
        method : ast.FunctionDef
            The callee's already-translated signature, used to determine
            which dummy-argument positions are needed.

        Returns
        -------
        list
            Resolved argument expression nodes, or an empty list if
            resolution fails.

        Raises
        ------
        AssertionError
            if the actual-argument count for a
            function-style call doesn't match the expected dummy-argument
            count.
        IndexError
        KeyError
        """
        # This retrieves all the call entries of the called function
        i = call_indices[func_name]
        call_indices[func_name] += 1
        # Clamp i so it doesn’t exceed available subroutine entries
        i = min(i, len(self.extractor.call_subroutines.get(func_name, [])) - 1)
        indexes = []
        dummy_args_list = self.extractor.dummy_arg_list[func_name]
        folded_args = [a.casefold() for a in dummy_args_list]
        for arg in method.args.args:
            key = arg.arg.casefold()
            if key in folded_args:
                indexes.append(folded_args.index(key))

        try:
            actual_args_list = None
            part_ref = None
            if walk(self.extractor.call_subroutines[func_name], F23.Call_Stmt):
                # Subroutines
                actual_args_list = walk(
                    self.extractor.call_subroutines[func_name], F23.Actual_Arg_Spec_List
                )
            if actual_args_list:
                args = [
                    self.f2np.handle_expr(actual_args_list[i].children[idx])
                    for idx in indexes
                ]
                return args
            else:
                # Fall back to another parsing route which usually means
                # we are in the case of Functions
                # Functions in most cases, seems to appear as either
                # Part ref or just as an assign statement like that of def in Python
                if isinstance(
                    self.extractor.call_subroutines[func_name][i], F23.Part_Ref
                ):
                    part_ref = walk(
                        self.extractor.call_subroutines[func_name][i], F23.Part_Ref
                    )
                else:
                    if isinstance(
                        self.extractor.call_subroutines[func_name][i],
                        F23.Assignment_Stmt,
                    ):
                        _, _, func = self.extractor.call_subroutines[func_name][
                            i
                        ].children
                        part_ref = walk(func, F23.Part_Ref)

                if part_ref:
                    actual_args_list = walk(part_ref, F23.Section_Subscript_List)[0]
                assert len(actual_args_list.children) == len(indexes), (
                    f"The actual arguments for the funciton and \
                    that of the dummy arg list of function: {func_name} should match"
                )
                args = []
                for idx in indexes:
                    if idx < len(actual_args_list.children):
                        args.append(
                            self.f2np.handle_expr(actual_args_list.children[idx])
                        )
                    else:
                        self.logger.warning(
                            f"Skipping missing actual argument at index {idx} for {func_name}: "
                            f"{len(actual_args_list.children)} actual args found."
                        )
                return args
        except (IndexError, KeyError) as e:
            self.logger.error(
                f"Error mapping arguments for expr to '{func_name}' at index {i}:", e
            )
            return []
        except AssertionError as e:
            self.logger.error(
                "if the actual-argument count for a \
            function-style call doesn't match the expected dummy-argument \
            count.",
                e,
            )

    def _compute_return_variables(
        self,
        function_def: ast.FunctionDef,
        cls_info: dict,
        subroutine_key: str,
        main_file_attributes: list,
        module_name: str,
        instance_name: str,
    ) -> tuple[list, set]:
        """
        Determine which scalar variables must be returned from a
        translated function.

        A variable qualifies when it is declared ``OUT``/``INOUT``, is
        recorded as modified by :attr:`extractor`, is not a global
        attribute, and is not an array (arrays are passed/modified by
        reference and don't need explicit returning). Existing
        ``ast.Return`` statements suppress this computation entirely. Used
        by :meth:`correct_function` (step 3).

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function being analysed.
        cls_info : dict
            Class metadata for the owning instance.
        subroutine_key : str
            Current subroutine identifier.
        main_file_attributes : list
            Attribute names defined at main-file level, excluded from
            the scalar-variable set.
        module_name : str
            Owning module name.
        instance_name : str
            Owning instance name.

        Returns
        -------
        tuple[list, set]
            ``(return_list, scalar_variables)`` — names to return, and the
            set of scalar dummy-argument names (used later for index
            adjustment exclusion).
        """
        return_list = []
        scalar_variables = set()

        instance_data = cls_info[module_name][instance_name]
        global_attr = instance_data["attributes"]

        func_name = function_def.name
        method = instance_data["methods"].get(func_name)
        if not method:
            return return_list, scalar_variables

        scalars = [s.string for s in self.extractor.scalar_variables[func_name]]

        for arg in method.args.args:
            if arg.arg in scalars and arg.arg not in main_file_attributes:
                scalar_variables.add(arg.arg)

        variables_output = []
        for variables in self.extractor.var_dummy[subroutine_key]:
            if any(
                [
                    var
                    for var in walk(variables, F23.Intent_Spec)
                    if var.tostr() in ["OUT", "INOUT"]
                ]
            ):
                name = walk(walk(variables, F23.Entity_Decl), F23.Name)[0].string
                variables_output.append(name)

        # Retreive the output variables that might be modified
        var_modified = set(variables_output) & set(
            self.extractor.var_modif[subroutine_key]
        )

        non_global = var_modified - set(global_attr)  # Filter out global variables

        # Filter out arrays (those with 'DIMENSION' in their type info) which allows us to return only the scalars
        # since arrays are sent as reference arguments and doesn't need to be returned to be updated
        scalars_to_return = {
            var
            for var in non_global
            if "DIMENSION"
            not in self.extractor.var_modif_info[subroutine_key].get(var, [])
        }

        if scalars_to_return and not any(ast_walk(function_def, ast.Return)):
            return_list.extend(scalars_to_return)

        return return_list, scalar_variables

    def _adjust_function_indices(
        self,
        function_def: ast.FunctionDef,
        cls_info: dict,
        subroutine_key: str,
        scalar_variables: list,
        module_name: str,
        instance_name: str,
    ) -> None:
        """
        Apply Fortran-to-Python array index correction to a function body
        via ``AdjustIndices``.

        Loop variables are collected via :meth:`_collect_loop_variables`;
        additional adjusted-variable dependencies are resolved via
        :func:`search_convar_dependencies`. Used by :meth:`correct_function`
        (step 4).

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function whose body is visited and mutated in place by
            ``AdjustIndices``.
        cls_info : dict
            Class metadata for the owning instance.
        subroutine_key : str
            Current subroutine identifier.
        scalar_variables : list
            Scalar variable names excluded from index adjustment.
        module_name : str
            Owning module name.
        instance_name : str
            Owning instance name.
        """
        cons_var = self._collect_loop_variables(function_def, subroutine_key)

        kwargs = {"exclude_index": scalar_variables} if scalar_variables else {}

        adjusted = search_convar_dependencies(cons_var, function_def)
        if adjusted:
            kwargs["adjusted_vars"] = adjusted

        adjuster = AdjustIndices(
            cons_var,
            self.extractor.all_array_info[subroutine_key],
            cls_info[module_name][instance_name],
            **kwargs,
        )

        for stmt in function_def.body:
            adjuster.visit(stmt)

    def _collect_loop_variables(
        self,
        function_def: ast.FunctionDef,
        subroutine_key: str,
    ) -> set:
        """
        Collect the set of variable names used as loop-control indices,
        from both :attr:`extractor`'s ``loop_dict`` and the function body's
        actual ``ast.For`` targets.

        Used by :meth:`_adjust_function_indices`.

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function whose ``ast.For`` nodes are scanned.
        subroutine_key : str
            Current subroutine identifier, used to look up
            :attr:`extractor`'s ``loop_dict``.

        Returns
        -------
        set
            Loop-index variable names.
        """
        cons_var = set()
        loop_dict_values = self.extractor.loop_dict.get(subroutine_key, {})

        for values in loop_dict_values.values():
            if not values:
                continue

            for v in values:
                cons_var.add(v)

        for loop in ast_walk(function_def, ast.For):
            cons_var.update(self._extract_loop_targets(loop.target))

        return cons_var

    def _extract_loop_targets(self, target: ast.AST) -> set:
        """
        Recursively extract variable names from a ``for`` loop's target
        expression, handling tuple/list unpacking.

        Used by :meth:`_collect_loop_variables`.

        Parameters
        ----------
        target : ast.AST
            A loop's ``target`` node.

        Returns
        -------
        set
            Variable names found in *target*.
        """
        names = set()

        if isinstance(target, ast.Name):
            names.add(target.id)

        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                names.update(self._extract_loop_targets(elt))

        return names

    def _insert_return(self, function_def: ast.FunctionDef, return_list: list) -> None:
        """
        Append a ``return`` statement to a function body, returning a
        single value or a tuple depending on *return_list*'s length.

        Used by :meth:`correct_function` (step 5).

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function whose body is mutated in place.
        return_list : list
            Variable names to return.
        """
        values = [ast.Name(id=v, ctx=ast.Load()) for v in return_list]

        return_node = ast.Return(
            value=values[0]
            if len(values) == 1
            else ast.Tuple(elts=values, ctx=ast.Load())
        )

        function_def.body.append(return_node)

    def out_module_python(self) -> ast.Module | None:
        """
        Load and parse the configured global-module Python code template.

        Used as the starting point for both
        :meth:`transform_to_class` (class-based global module) consumers.

        Returns
        -------
        Optional[ast.Module]
            The parsed template AST, or ``None`` if loading or parsing
            fails.

        Raises
        ------
        ValueError
            if templates fail to
            load, the global-class template entry is missing, or parsing
            returns ``None``.
        """
        try:
            templates = load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")

            code = templates["Python_templates"]["Python_global_class_template"][
                "template"
            ]

            if code is None:
                raise ValueError(
                    "The code template for out_module_python wasn't retreived"
                )

            parsed_ast = python_parser(code)
            if parsed_ast is None:
                raise ValueError("Parsed AST is None due to prior error")

            return parsed_ast

        except Exception:
            self.logger.exception("Exception occurred while loading out_module_python")
            return None

    def out_main_python(self) -> ast.Module | None:
        """
        Load and parse the configured main-script Python code template.

        Used as the starting point for :meth:`update_main_python`.

        Returns
        -------
        Optional[ast.Module]
            The parsed template AST, or ``None`` if loading or parsing
            fails.

        Raises
        ------
        ValueError
            if templates fail to
            load, the main template entry is missing, or parsing returns
            ``None``.
        """
        try:
            templates = load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")

            code = templates["Python_templates"]["Python_main_template"]["template"]
            if code is None:
                raise ValueError(
                    "The code template for out_main_python wasn't retreived"
                )

            parsed_ast = python_parser(code)
            if parsed_ast is None:
                raise ValueError("Parsed AST is None due to prior error")

            return parsed_ast

        except Exception:
            self.logger.exception("Exception occurred while loading out_main_python")
            return None

    def retreive_variable_order(self) -> None:
        """
        Determine the order in which dummy-argument variables are laid out
        in the binary input file, populating :attr:`variable_order`.

        Reads from :attr:`isolator`'s ``input_dict`` (combining
        ``reads_non_allocatables`` and ``reads_allocatables``). Used by
        :meth:`update_global_python`, :meth:`update_main_python`, and
        :meth:`prepare_read_code_for_main_template`, since the binary
        layout determines both read order and which variables can be
        grouped into a single read loop.

        Raises
        ------
        KeyError
            if :attr:`isolator`'s ``input_dict`` is missing
            the expected keys.
        """
        self.variable_order = []
        try:
            combined = (
                self.isolator.input_dict["reads_non_allocatables"]
                + self.isolator.input_dict["reads_allocatables"]
            )
            for read_dec in combined:
                read_stmt = walk(read_dec, F23.Input_Item_List)
                for item in read_stmt:
                    self.variable_order.append(item.children[0].string)
        except Exception:
            self.logger.exception("Exception in retrieve_variable_order")
            raise
        except KeyError as e:
            self.logger.error("Expected key missing in input_dict", e)
            raise

    def convert_SPECIFICATION_PART(
        self,
        declaration_stmts: list,
        fix_loc: bool = False,
        cls_mode: bool = False,
    ) -> list:
        """
        Convert a list of Fortran declaration blocks into Python AST
        assignment/import nodes.

        Each block is first checked via :meth:`_contains_function` (blocks
        containing nested function subprograms are skipped entirely), then
        normalised via :meth:`_preprocess_declarations`. ``Use_Stmt`` nodes
        are routed to :meth:`_handle_use_stmt`; ``Type_Declaration_Stmt``
        nodes are routed to :meth:`_handle_type_decl`. Used by
        :meth:`update_global_python` and :meth:`update_main_python` as the
        entry point for translating a routine's declaration section.

        Parameters
        ----------
        declaration_stmts : list
            Fortran declaration block lists to convert.
        fix_loc : bool, optional
            If ``True``, calls ``ast.fix_missing_locations`` on each
            resulting node — needed when the nodes will be unparsed
            independently of a parent tree that already has locations.
        cls_mode : bool, optional
            Whether the resulting assignments target ``self.<attr>``
            (class context) or plain module-level names.

        Returns
        -------
        list
            The converted AST nodes (assignments and/or imports).
        """

        ast_nodes = []
        try:
            for declarations in declaration_stmts:
                if self._contains_function(declarations):
                    continue

                declarations = self._preprocess_declarations(declarations)

                for node in walk(
                    declarations, (F23.Type_Declaration_Stmt, F23.Use_Stmt)
                ):
                    if isinstance(node, F23.Use_Stmt):
                        ast_nodes.extend(self._handle_use_stmt(node))

                    elif isinstance(node, F23.Type_Declaration_Stmt):
                        ast_nodes.extend(self._handle_type_decl(node, cls_mode))

            if fix_loc:
                ast_nodes = [ast.fix_missing_locations(n) for n in ast_nodes]

            return ast_nodes

        except Exception:
            self.logger.exception("Error in convert_SPECIFICATION_PART")
            raise

    def _contains_function(self, declarations: list) -> bool:
        """
        Return ``True`` if any declaration block contains a nested
        ``F23.Function_Subprogram``.

        Used by :meth:`convert_SPECIFICATION_PART` to skip blocks that
        define internal functions, which are not handled by this
        declaration-conversion path.

        Parameters
        ----------
        declarations : list
            Fortran declaration block to inspect.

        Returns
        -------
        bool
            ``True`` if a nested function subprogram is present.
        """
        return any(walk(decl, F23.Function_Subprogram) for decl in declarations)

    def _preprocess_declarations(self, declarations: list) -> Any:
        """
        Normalise a declaration block before AST conversion, delegating to
        :attr:`isolator`'s processor.

        Two-element blocks are merged via ``combine_allocate_declaration``
        (handling split ``ALLOCATABLE`` + dimension declarations); other
        blocks have ``INTENT``/``SAVE`` attributes stripped via
        ``remove_intent_and_save``. Used by
        :meth:`convert_SPECIFICATION_PART` and
        :meth:`search_dependant_variables`.

        Parameters
        ----------
        declarations : list
            Raw Fortran declaration block.

        Returns
        -------
        Any
            The preprocessed declaration node.
        """
        if len(declarations) == 2:
            return self.isolator.processor.combine_allocate_declaration(declarations)
        return self.isolator.processor.remove_intent_and_save(declarations)

    def _is_array_declaration(self, decl: ast.AST | list) -> bool:
        """
        Return ``True`` if *decl* declares an array (has a dimension
        attribute or an explicit shape specification).

        Used by :meth:`_handle_type_decl` and
        :meth:`search_dependant_variables`.

        Parameters
        ----------
        decl : Union[ast.AST, list]
            A declaration statement or list of statements.

        Returns
        -------
        bool
            ``True`` if a dimension/shape specifier is present.
        """
        return any(walk(decl, F23.Dimension_Attr_Spec)) or any(
            walk(decl, F23.Explicit_Shape_Spec)
        )

    def _handle_use_stmt(self, node: Any) -> list:
        """
        Convert a Fortran ``USE`` statement into one or more
        ``ast.ImportFrom`` nodes.

        Selective imports (``USE mod, ONLY: a, b``) are resolved via
        :meth:`_extract_use_names`; unrestricted ``USE`` statements are
        converted to a wildcard import. Used by
        :meth:`convert_SPECIFICATION_PART`.

        Parameters
        ----------
        node : Any
            Fortran ``F23.Use_Stmt`` node.

        Returns
        -------
        list
            Zero or one ``ast.ImportFrom`` node, depending on whether
            :meth:`_extract_use_names` yields any names.
        """
        _, _, module_name, _, only_stmt = node.children

        if only_stmt:
            names = self._extract_use_names(only_stmt)
            if not names:
                return []
            return [ast.ImportFrom(module=module_name.string, names=names, level=0)]

        return [
            ast.ImportFrom(
                module=module_name.string, names=[ast.alias(name="*")], level=0
            )
        ]

    def _extract_use_names(self, only_stmt: Any) -> list:
        """
        Extract import aliases from a Fortran ``USE ... ONLY:`` clause.

        Names matching :attr:`extractor`'s ``allowed_external_subroutines``
        are skipped (these are handled separately as logging calls rather
        than real imports — see :meth:`F2NP._build_call_logging`). Used by
        :meth:`_handle_use_stmt`.

        Parameters
        ----------
        only_stmt : Any
            Fortran ``ONLY:`` clause node.

        Returns
        -------
        list
            ``ast.alias`` nodes for each retained imported name, with
            renames (``USE mod, ONLY: a => b``) preserved via the
            ``asname`` field.
        """
        names = []

        for el in only_stmt.children:
            if isinstance(el, F23.Name):
                if el.string in self.extractor.allowed_external_subroutines:
                    continue
                names.append(ast.alias(name=el.string))

            elif isinstance(el, F23.Rename):
                _, name, asname = el.children
                if name.string in self.extractor.allowed_external_subroutines:
                    continue
                names.append(ast.alias(name=name.string, asname=asname.string))

        return names

    def _handle_type_decl(self, node: Any, cls_mode: bool) -> list:
        """
        Lower a Fortran type declaration statement into one or more Python
        assignment nodes.

        Dispatches per declared entity based on the declaration's
        attributes: ``PARAMETER`` declarations go through
        :meth:`_handle_parameter`; array declarations (per
        :meth:`_is_array_declaration`) go through :meth:`_handle_dimension`;
        everything else goes through :meth:`_handle_scalar`. Used by
        :meth:`convert_SPECIFICATION_PART`.

        Parameters
        ----------
        node : Any
            Fortran ``F23.Type_Declaration_Stmt`` node.
        cls_mode : bool
            Whether targets should be ``self.<attr>`` or plain names.

        Returns
        -------
        list
            One assignment node per declared entity in *node*.
        """
        intrinsic_type_spec, _, entity_decl_list = node.children

        dtype = self._get_dtype(intrinsic_type_spec)
        attr_specs = [p.string for p in walk(node, F23.Attr_Spec)]
        has_dimension = self._is_array_declaration(node)
        has_kind = any(walk(node, F23.Kind_Selector))

        ast_nodes = []

        for entity in entity_decl_list.children:
            target = self._make_target(entity, cls_mode)
            _, _, _, init = entity.children
            value = init.children[1] if init else None

            if "PARAMETER" in attr_specs:
                ast_nodes.append(
                    self._handle_parameter(
                        target, value, dtype, has_kind, has_dimension, cls_mode
                    )
                )

            elif has_dimension:
                ast_nodes.append(
                    self._handle_dimension(node, target, value, dtype, cls_mode)
                )

            else:
                ast_nodes.append(
                    self._handle_scalar(
                        target, value, dtype, intrinsic_type_spec, has_kind, cls_mode
                    )
                )

        return ast_nodes

    def _make_target(self, entity: Any, cls_mode: bool) -> ast.Attribute | ast.Name:
        """
        Build the assignment target for a single declared entity.

        In class mode, uppercase Fortran names are lowercased for the
        Python attribute name (matching the convention used elsewhere in
        the pipeline). Used by :meth:`_handle_type_decl`.

        Parameters
        ----------
        entity : Any
            Fortran ``F23.Entity_Decl`` node.
        cls_mode : bool
            Whether to build ``self.<attr>`` or a plain ``ast.Name``.

        Returns
        -------
        ast.Attribute or ast.Name
            The store-context target node.
        """
        var_name = entity.children[0].string

        if cls_mode:
            return ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=var_name.lower() if var_name.isupper() else var_name,
                ctx=ast.Store(),
            )

        return ast.Name(id=var_name, ctx=ast.Store())

    def _get_dtype(self, intrinsic: Any) -> tuple[str, str]:
        """
        Map a Fortran intrinsic type to its ``(module, attribute)`` NumPy
        dtype pair, defaulting to ``('np', 'float64')`` for unrecognised
        types.

        Used by :meth:`_handle_type_decl`.

        Parameters
        ----------
        intrinsic : Any
            Fortran intrinsic type specification node.

        Returns
        -------
        tuple[str, str]
            ``(module, attribute)``, e.g. ``('np', 'int32')``.
        """
        kind_map = {
            "REAL": ("np", "float64"),
            "INTEGER": ("np", "int32"),
            "LOGICAL": ("np", "bool"),
        }
        return kind_map.get(intrinsic.children[0], ("np", "float64"))

    def _dtype_call(self, dtype: tuple[str, str], value: ast.AST) -> ast.Call:
        """
        Wrap *value* in a NumPy dtype constructor call, e.g.
        ``np.float64(value)``.

        Used by :meth:`_handle_parameter` and :meth:`_handle_scalar`.

        Parameters
        ----------
        dtype : tuple[str, str]
            ``(module, attribute)`` dtype pair from :meth:`_get_dtype`.
        value : ast.AST
            The value expression to cast.

        Returns
        -------
        ast.Call
            The constructor call.
        """
        return ast.Call(func=self._dtype_attr(dtype=dtype), args=[value], keywords=[])

    def _assign(self, target: ast.AST, value: ast.AST) -> ast.Assign:
        """
        Build a single-target ``ast.Assign`` node.

        Parameters
        ----------
        target : ast.AST
            The assignment target.
        value : ast.AST
            The value expression.

        Returns
        -------
        ast.Assign
            ``target = value``.
        """
        return ast.Assign(targets=[target], value=value)

    def _maybe_attach(self, node: ast.AST, cls_mode: bool) -> ast.AST:
        """
        Conditionally qualify *node* with instance context via
        :func:`attach_instance`.

        No-op when *cls_mode* is ``False``. Used throughout
        :meth:`_handle_parameter`, :meth:`_handle_scalar`, and
        :meth:`_extract_shape` to ensure bare-name references inside
        initial-value expressions resolve correctly once embedded in a
        class.

        Parameters
        ----------
        node : ast.AST
            Expression to possibly qualify.
        cls_mode : bool
            Whether class-context qualification should be applied.

        Returns
        -------
        ast.AST
            *node*, possibly wrapped via :func:`attach_instance`.
        """
        return attach_instance(node) if cls_mode else node

    def _handle_parameter(
        self,
        target: ast.AST,
        value: ast.AST,
        dtype: tuple[str, str],
        has_kind: bool,
        has_dimension: bool,
        cls_mode: bool,
    ) -> ast.Assign:
        """
        Convert a Fortran ``PARAMETER`` declaration into an initialising
        assignment.

        Handles four shapes: no initial value (defaults to dtype-cast
        zero); an array constructor (built via :meth:`_build_array`); a
        bare name/attribute reference (assigned directly, untyped); and a
        general expression (wrapped in ``np.zeros``-style array
        construction if *has_dimension*, otherwise dtype-cast). Used by
        :meth:`_handle_type_decl`.

        Parameters
        ----------
        target : ast.AST
            The assignment target.
        value : ast.AST
            The initial-value expression node, or ``None``.
        dtype : tuple[str, str]
            Dtype pair from :meth:`_get_dtype`.
        has_kind : bool
            Whether the declaration specifies a kind (currently unused in
            the body but kept for signature symmetry with
            :meth:`_handle_scalar`).
        has_dimension : bool
            Whether the declaration is array-shaped.
        cls_mode : bool
            Whether to attach instance context to resolved expressions.

        Returns
        -------
        ast.Assign
            The constructed assignment.
        """
        if value is None:
            return self._assign(target, self._dtype_call(dtype, ast.Constant(0)))

        if walk(value, F23.Array_Constructor):
            elements = self._extract_array_elements(value)
            return self._assign(
                target, self._build_array(elements, dtype, raw_list=True)
            )

        expr = self.f2np.handle_expr(value)
        expr = self._maybe_attach(expr, cls_mode)

        if isinstance(expr, ast.Name) or isinstance(expr, ast.Attribute):
            return self._assign(target, expr)

        if has_dimension:
            return self._assign(target, self._build_array(expr, dtype))

        return self._assign(target, self._dtype_call(dtype, expr))

    def _handle_dimension(
        self,
        node: ast.AST,
        target: ast.AST,
        value: ast.AST,
        dtype: tuple[str, str],
        cls_mode: bool,
    ) -> ast.Assign:
        """
        Convert a Fortran array declaration (with explicit dimensions)
        into an initialising assignment.

        An explicit array constructor takes priority (built via
        :meth:`_build_array`); otherwise a ``np.zeros((...), dtype=...)``
        call is built with the shape resolved via :meth:`_extract_shape`.
        Used by :meth:`_handle_type_decl`.

        Parameters
        ----------
        node : ast.AST
            The Fortran declaration node, used to extract shape
            specifications.
        target : ast.AST
            The assignment target.
        value : ast.AST
            The initial-value expression node, or ``None``.
        dtype : tuple[str, str]
            Dtype pair from :meth:`_get_dtype`.
        cls_mode : bool
            Whether to attach instance context to shape expressions.

        Returns
        -------
        ast.Assign
            The constructed assignment.
        """
        if value and walk(value, F23.Array_Constructor):
            elements = self._extract_array_elements(value)
            return self._assign(
                target, self._build_array(elements, dtype, raw_list=True)
            )

        shape = self._extract_shape(node, cls_mode)

        return self._assign(
            target,
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="zeros",
                    ctx=ast.Load(),
                ),
                args=[ast.Tuple(elts=shape, ctx=ast.Load())],
                keywords=[ast.keyword(arg="dtype", value=self._dtype_attr(dtype))],
            ),
        )

    def _handle_scalar(
        self,
        target: ast.AST,
        value: ast.AST,
        dtype: tuple[str, str],
        intrinsic: Any,
        has_kind: bool,
        cls_mode: bool,
    ) -> ast.Assign:
        """
        Convert a non-parameter, non-array Fortran scalar declaration into
        an initialising assignment.

        ``LOGICAL`` declarations without a kind are special-cased into
        ``np.bool(True/False)``; declarations with no initial value
        default to a dtype-cast zero; bare name/attribute initial values
        are assigned directly (untyped); other expressions are dtype-cast.
        Used by :meth:`_handle_type_decl`.

        Parameters
        ----------
        target : ast.AST
            The assignment target.
        value : ast.AST
            The initial-value expression node, or ``None``.
        dtype : tuple[str, str]
            Dtype pair from :meth:`_get_dtype`.
        intrinsic : Any
            Fortran intrinsic type node, inspected for the ``LOGICAL``
            special case.
        has_kind : bool
            Whether a kind is specified (suppresses the ``LOGICAL``
            special case if ``True``).
        cls_mode : bool
            Whether to attach instance context to resolved expressions.

        Returns
        -------
        ast.Assign
            The constructed assignment.
        """
        if intrinsic.children[0] == "LOGICAL" and not has_kind:
            val = False
            if value and value.string.upper() == ".TRUE.":
                val = True
            return self._assign(target, self._dtype_call(dtype, ast.Constant(val)))

        if value is None:
            return self._assign(target, self._dtype_call(dtype, ast.Constant(0)))

        expr = self.f2np.handle_expr(value)
        expr = self._maybe_attach(expr, cls_mode)

        if isinstance(expr, ast.Name) or isinstance(expr, ast.Attribute):
            return self._assign(target, expr)

        return self._assign(target, self._dtype_call(dtype, expr))

    def _build_array(
        self, elements: list, dtype: tuple[str, str], raw_list: bool = False
    ) -> ast.Call:
        """
        Build a ``np.array(elements, dtype=...)`` call.

        Used by :meth:`_handle_parameter` and :meth:`_handle_dimension`.

        Parameters
        ----------
        elements : list
            Element expressions, or a single expression when *raw_list* is
            ``False``.
        dtype : tuple[str, str]
            Dtype pair from :meth:`_get_dtype`.
        raw_list : bool, optional
            If ``True``, *elements* is wrapped in an ``ast.List`` before
            being passed as the array constructor's argument; if ``False``,
            *elements* is passed through directly (already an expression).

        Returns
        -------
        ast.Call
            The ``np.array(...)`` call.
        """
        if raw_list:
            elements = ast.List(elts=elements, ctx=ast.Load())

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()
            ),
            args=[elements],
            keywords=[ast.keyword(arg="dtype", value=self._dtype_attr(dtype))],
        )

    def _dtype_attr(self, dtype: tuple[str, str]) -> ast.Attribute:
        """
        Build the AST node for a NumPy dtype reference, e.g. ``np.float64``.

        Used throughout this class and :class:`F2NP` wherever a ``dtype=``
        keyword value is needed.

        Parameters
        ----------
        dtype : tuple[str, str]
            ``(module, attribute)`` pair.

        Returns
        -------
        ast.Attribute
            The dtype reference expression.
        """
        mod, attr = dtype
        return ast.Attribute(
            value=ast.Name(id=mod, ctx=ast.Load()), attr=attr, ctx=ast.Load()
        )

    def _extract_array_elements(self, value: Any) -> list:
        """
        Extract element expressions from a Fortran array constructor via
        :attr:`f2np`'s expression handler.

        Used by :meth:`_handle_parameter` and :meth:`_handle_dimension`.

        Parameters
        ----------
        value : Any
            Fortran node containing an ``F23.Array_Constructor``.

        Returns
        -------
        list
            Converted element expressions.
        """
        elements = []
        array_list = walk(walk(value, F23.Array_Constructor), F23.Ac_Value_List)[0]
        for val in array_list.children:
            elements.append(self.f2np.handle_expr(val))
        return elements

    def _extract_shape(self, node: Any, cls_mode: bool) -> list:
        """
        Compute per-dimension size expressions from an array declaration's
        explicit shape specification.

        Mirrors :meth:`F2NP._extract_shapes` but additionally applies
        :meth:`_maybe_attach` to each bound for class-context resolution.
        Used by :meth:`_handle_dimension`.

        Parameters
        ----------
        node : Any
            Fortran node containing explicit shape specifications.
        cls_mode : bool
            Whether to attach instance context to bound expressions.

        Returns
        -------
        list
            One size expression per dimension.

        Raises
        ------
        ValueError
            If a dimension has neither a lower nor an upper bound.
        """
        shape = []

        for dim in walk(node, F23.Explicit_Shape_Spec):
            lb, ub = dim.children

            if lb and ub:
                lower = self._maybe_attach(self.f2np.handle_expr(lb), cls_mode)
                upper = self._maybe_attach(self.f2np.handle_expr(ub), cls_mode)

                shape.append(
                    ast.BinOp(
                        left=ast.BinOp(left=upper, op=ast.Sub(), right=lower),
                        op=ast.Add(),
                        right=ast.Constant(1),
                    )
                )

            elif lb:
                shape.append(self._maybe_attach(self.f2np.handle_expr(lb), cls_mode))

            elif ub:
                shape.append(self._maybe_attach(self.f2np.handle_expr(ub), cls_mode))

            else:
                raise ValueError("Invalid dimension spec")

        return shape

    def _collect_target_names(self, target: ast.AST) -> None:
        """
        Recursively collect assignment target names into :attr:`pre_init`.

        Handles plain names, ``self.attr`` attributes, and tuple/list
        unpacking. Used by :meth:`pre_init_variables`.

        Parameters
        ----------
        target : ast.AST
            An assignment target (LHS) node.
        """

        if isinstance(target, ast.Name):
            self.pre_init.add(target.id)

        elif isinstance(target, ast.Attribute):
            self.pre_init.add(target.attr)

        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._collect_target_names(elt)

    def pre_init_variables(self, code_template: ast.Module) -> None:
        """
        Populate :attr:`pre_init` with every variable name already
        assigned somewhere in *code_template*.

        Walks all ``ast.Assign``/``ast.AnnAssign`` nodes, collecting
        targets via :meth:`_collect_target_names`. Used by
        :meth:`update_global_python` before declaration conversion, so that
        :meth:`_resolve_dependencies` can distinguish variables that are
        already initialised by the template itself from ones that still
        need a dependency-ordered read/assignment.

        Parameters
        ----------
        code_template : ast.Module
            The template AST to scan.
        """
        self.pre_init = set()
        try:
            for node in ast.walk(code_template):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self._collect_target_names(target)

                elif isinstance(node, ast.AnnAssign):
                    self._collect_target_names(node.target)

            self.pre_init = list(self.pre_init)

        except Exception:
            self.logger.exception("Exception in pre_init_variables")
            raise

    def search_dependant_variables(self, declaration_stmts: list) -> None:
        """
        Populate :attr:`dependant_variables` by analysing array-dimension
        bounds for references to other not-yet-initialised variables.

        Builds a name → initialisation-status table via
        :meth:`_build_symbol_table`, then for each array declaration
        extracts its bound expressions (:meth:`_extract_bounds`) and
        resolves which referenced names are themselves uninitialised
        dependencies (:meth:`_resolve_dependencies`). Used by
        :meth:`update_global_python` to drive
        :meth:`init_dependant_variables`'s insertion ordering.

        Parameters
        ----------
        declaration_stmts : list
            Declaration blocks to analyse.
        """
        self.dependant_variables = {}
        try:
            symbol_table = self._build_symbol_table(declaration_stmts)

            for decl in declaration_stmts:
                decl = self._preprocess_declarations(decl)

                if not self._is_array_declaration(decl):
                    continue

                var_name = self._get_decl_name(decl)
                bounds = self._extract_bounds(decl)

                deps = self._resolve_dependencies(bounds, symbol_table)

                if deps:
                    self.dependant_variables[var_name] = deps

        except Exception:
            self.logger.exception("Exception in search_dependant_variables")
            raise

    def _build_symbol_table(self, declaration_stmts: list) -> dict:
        """
        Build a ``name -> {'initialized': bool}`` table from declared
        entities.

        Used by :meth:`search_dependant_variables`.

        Parameters
        ----------
        declaration_stmts : list
            Declaration blocks to scan.

        Returns
        -------
        dict
            Per-variable initialisation status.
        """
        table = {}

        for decl in declaration_stmts:
            decl = self._preprocess_declarations(decl)

            for entity in walk(decl, F23.Entity_Decl):
                name = entity.children[0].string
                _, _, _, init = entity.children

                table[name] = {"initialized": init is not None}

        return table

    def _get_decl_name(self, decl: Any) -> Any:
        """
        Extract the declared entity's name from a declaration node.

        Used by :meth:`search_dependant_variables`.

        Parameters
        ----------
        decl : Any
            A declaration node.

        Returns
        -------
        Any
            The declared name (string).
        """
        entity = walk(decl, F23.Entity_Decl)[0]
        return entity.children[0].string

    def _extract_bounds(self, decl: Any) -> list:
        """
        Collect lower/upper bound expressions from a declaration's
        explicit shape specifications.

        Used by :meth:`search_dependant_variables`.

        Parameters
        ----------
        decl : Any
            A declaration node.

        Returns
        -------
        list
            Bound expression nodes.
        """
        bounds = []

        for dim in walk(decl, F23.Explicit_Shape_Spec):
            lb, ub = dim.children

            if lb:
                bounds.append(lb)
            if ub:
                bounds.append(ub)

        return bounds

    def _resolve_dependencies(self, bounds: list, symbol_table: dict) -> list:
        """
        Identify which names referenced in *bounds* are dependencies
        requiring prior initialisation.

        A name qualifies as a dependency when it is not already in
        :attr:`pre_init` and the symbol table shows it as declared but not
        yet initialised. Used by :meth:`search_dependant_variables`.

        Parameters
        ----------
        bounds : list
            Bound expressions, potentially containing variable references.
        symbol_table : dict
            Table from :meth:`_build_symbol_table`.

        Returns
        -------
        list
            Names of uninitialised dependencies.
        """
        deps = set()

        for b in bounds:
            names = walk(b, F23.Name)

            for name_node in names:
                name = name_node.string

                if name in self.pre_init:
                    continue

                info = symbol_table.get(name)

                if info and not info["initialized"]:
                    deps.add(name)

        return list(deps)

    def _is_class_instantiation(self, stmt: ast.AST, class_name: str) -> bool:
        """
        Determine whether a statement instantiates a specific class.

        A statement is considered a class instantiation when it matches the
        pattern ``var = ClassName(...)`` where the called object is an
        :class:`ast.Name` whose identifier equals *class_name*.

        Parameters
        ----------
        stmt : ast.AST
            Statement node to inspect.
        class_name : str
            Name of the class expected to be instantiated.

        Returns
        -------
        bool
            ``True`` if *stmt* is an :class:`ast.Assign` whose value is a
            call to *class_name*; otherwise ``False``.

        Notes
        -----
        Only direct instantiations of the form ``ClassName(...)`` are
        recognized. Calls through attributes such as
        ``module.ClassName(...)`` are ignored.
        """
        return (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == class_name
        )

    def _is_method_call(self, stmt: ast.AST, var_name: str, method_name: str) -> bool:
        """
        Determine whether a statement is a method call on a target variable.

        Supported call patterns include:

        - ``var.method(...)``
        - ``self.var.method(...)``

        Parameters
        ----------
        stmt : ast.AST
            Statement node to inspect.
        var_name : str
            Name of the variable expected to own the method.
        method_name : str
            Name of the method expected to be called.

        Returns
        -------
        bool
            ``True`` if *stmt* matches a supported method-call pattern for
            *var_name* and *method_name*; otherwise ``False``.

        Notes
        -----
        The statement must be represented as an expression containing an
        :class:`ast.Call` whose callable is an :class:`ast.Attribute`.
        Nested ownership beyond ``self.var`` is not supported.
        """
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
        ):
            return False

        func = stmt.value.func

        # var.method()
        if isinstance(func.value, ast.Name):
            return func.value.id == var_name and func.attr == method_name

        # self.var.method()
        if isinstance(func.value, ast.Attribute):
            return func.value.attr == var_name and func.attr == method_name

        return False

    def _find_init_pattern(
        self, function: ast.FunctionDef, class_name: str, method_name: str
    ) -> int | None:
        """
        Locate the insertion point following a class initialization pattern.

        The function body is scanned for a consecutive pair of statements
        matching:

        1. An instantiation of *class_name*, as determined by
        :meth:`_is_class_instantiation`.
        2. A call to *method_name* on the created instance, as determined by
        :meth:`_is_method_call`.

        If the pattern is found, the index immediately following the method
        call is returned. If any assignment statement appears after the
        matched method call, the pattern is considered unsafe and
        ``None`` is returned.

        Parameters
        ----------
        function : ast.FunctionDef
            Function whose body is searched for the initialization pattern.
        class_name : str
            Name of the class expected to be instantiated.
        method_name : str
            Name of the method expected to be invoked on the instance.

        Returns
        -------
        int or None
            Index immediately after the matched method call, suitable for
            inserting additional statements. Returns ``None`` if no valid
            pattern is found or if subsequent assignments make insertion
            unsafe.
        """
        body = function.body

        for i in range(len(body) - 1):
            stmt1, stmt2 = body[i], body[i + 1]

            if not self._is_class_instantiation(stmt1, class_name):
                continue

            var_name = self._get_assigned_name(stmt1, require_self=True)

            if not var_name:
                continue

            if not self._is_method_call(stmt2, var_name, method_name):
                continue

            # Check for unsafe assignments after
            if any(isinstance(stmt, ast.Assign) for stmt in body[i + 2 :]):
                return None

            return i + 2

        return None

    def _check_position_within_function(
        self, function: ast.FunctionDef, class_name: str, method_name: str
    ) -> tuple[int, int | None]:
        """
        Determine the most appropriate insertion position within a function body.

        The insertion position is first resolved by searching for a recognized
        initialization pattern via :meth:`_find_init_pattern`. If no suitable
        pattern is found, the insertion point falls back to the position
        immediately following the last assignment statement in the function.

        If the function contains one or more return statements, the final
        insertion position is adjusted to ensure insertion occurs before the
        last return statement.

        Parameters
        ----------
        function : ast.FunctionDef
            Function whose body is analyzed.
        class_name : str
            Name of the class whose instantiation should be detected.
        method_name : str
            Name of the method expected to be called on the instantiated
            object.

        Returns
        -------
        tuple[int, int | None]
            Two-element tuple containing:

            - ``insert_pos``: Index at which new statements should be inserted.
            - ``return_pos``: Index of the last :class:`ast.Return`
            statement, or ``None`` if no return statement exists.

        See Also
        --------
        :meth:`_find_init_pattern`
            Resolves insertion positions based on initialization patterns.

        Notes
        -----
        The returned insertion position is always guaranteed to be less than
        or equal to the position of the last return statement, when one is
        present.
        """
        body = function.body

        # Try pattern-based insertion
        insert_pos = self._find_init_pattern(function, class_name, method_name)

        # Fallback to place after last assignment
        if insert_pos is None:
            assign_positions = [
                i for i, stmt in enumerate(body) if isinstance(stmt, ast.Assign)
            ]
            insert_pos = assign_positions[-1] + 1 if assign_positions else len(body)

        # Check for return statements
        return_positions = [
            i for i, stmt in enumerate(body) if isinstance(stmt, ast.Return)
        ]

        return_pos = return_positions[-1] if return_positions else None

        if return_pos is not None:
            insert_pos = min(insert_pos, return_pos)

        return insert_pos, return_pos

    def insert_at(
        self,
        idx: int,
        ast_node: ast.AST,
        python_template: ast.AST,
        method_name: str = None,
        **kwargs,
    ) -> None:
        """
        Insert an AST node into a Python AST at a context-appropriate location.

        The insertion strategy is delegated to a handler selected by
        :meth:`_get_insert_handler`. The chosen handler is responsible for
        determining the exact insertion location and mutating the target AST.

        Supported insertion targets include imports, functions, classes,
        assignments, and expression statements.

        Parameters
        ----------
        idx : int
            Desired insertion index within the target scope.
        ast_node : ast.AST
            AST node to insert.
        python_template : ast.Module
            Target module whose AST is modified in place.
        method_name : str, optional
            Name of the function or method used for context-sensitive
            insertion. Required by some insertion handlers.

        Other Parameters
        ----------------
        **kwargs
            Additional arguments forwarded directly to the selected insertion
            handler.

        Raises
        ------
        TypeError
            If *ast_node* is not supported by
            :meth:`_get_insert_handler`.
        Exception
            Re-raises any exception raised by the selected insertion handler
            after logging the failure.

        See Also
        --------
        :meth:`_get_insert_handler`
            Resolves the insertion handler for a given AST node type.
        """
        try:
            class_exists = any(
                isinstance(n, ast.ClassDef) for n in python_template.body
            )

            handler = self._get_insert_handler(ast_node, class_exists)
            handler(idx, ast_node, python_template, method_name, **kwargs)

        except Exception as e:
            self.logger.exception("Exception in insert_at", e)
            raise
        except TypeError as e:
            self.logger.error(
                "If the node is not supported by teh _get_insert_handler method", e
            )
            raise

    def _get_insert_handler(self, node: ast.AST, class_exists: bool) -> Callable:
        """
        Resolve the insertion handler for a given AST node.

        The returned handler is responsible for inserting the supplied node
        type into the target AST. Function definitions are dispatched
        differently depending on whether a class definition already exists in
        the target module.

        Parameters
        ----------
        node : ast.AST
            AST node for which an insertion handler should be selected.
        class_exists : bool
            Whether the target module already contains a
            :class:`ast.ClassDef`.

        Returns
        -------
        callable
            Bound method that performs insertion for the supplied node type.

        Raises
        ------
        TypeError
            If *node* is not one of the supported AST node types.

        Notes
        -----
        Currently supported node types include:

        - :class:`ast.Import`
        - :class:`ast.ImportFrom`
        - :class:`ast.FunctionDef`
        - :class:`ast.ClassDef`
        - :class:`ast.Assign`
        - :class:`ast.Expr`
        """
        if isinstance(node, ast.Import | ast.ImportFrom):
            return self._insert_import

        if isinstance(node, ast.FunctionDef):
            return (
                self._insert_function_in_class
                if class_exists
                else self._insert_function_module
            )

        if isinstance(node, ast.ClassDef):
            return self._insert_class

        if isinstance(node, ast.Assign):
            return self._insert_assign

        if isinstance(node, ast.Expr):
            return self._insert_expr

        raise TypeError(f"Unsupported node type: {type(node)}")

    def _insert_import(self, idx, node, module, *_) -> None:
        """
        Insert an import statement into a module.

        The import is inserted immediately after the last existing
        :class:`ast.Import` or :class:`ast.ImportFrom` statement. If the
        module contains no imports, the node is inserted at the beginning of
        the module body.

        Parameters
        ----------
        idx : int or None
            Ignored for import insertion.
        node : ast.AST
            Import node to insert.
        module : ast.Module
            Module whose body is modified in place.
        *_ : Any
            Additional positional arguments accepted for interface
            compatibility and ignored.
        """
        positions = self._find_positions(module.body, (ast.Import, ast.ImportFrom))
        insert_pos = positions[-1] + 1 if positions else 0
        module.body.insert(insert_pos, node)

    def _insert_class(self, idx, node, module, *_) -> None:
        """
        Insert a class definition into a module.

        If *idx* is provided, insertion is attempted at that position.
        However, class definitions are never inserted within the import
        section; when necessary, the insertion point is shifted to the first
        position after the final import statement.

        When *idx* is not provided, the insertion position is determined
        automatically using the following precedence:

        1. Before the last function definition.
        2. Before the ``if __name__ == "__main__"`` guard.
        3. After the final import statement.
        4. At the end of the module.

        Parameters
        ----------
        idx : int or None
            Desired insertion index.
        node : ast.ClassDef
            Class definition to insert.
        module : ast.Module
            Module whose body is modified in place.
        *_ : Any
            Additional positional arguments accepted for interface
            compatibility and ignored.
        """
        import_positions = self._find_positions(
            module.body, (ast.Import, ast.ImportFrom)
        )
        function_positions = self._find_positions(module.body, ast.FunctionDef)
        name_guard_positions = self._find_name_guard(module.body)

        if idx is not None:
            if import_positions:
                last_import = import_positions[-1] + 1

                if idx <= last_import:
                    self.logger.info(
                        f"idx {idx} inside imports -> shifting to {last_import}"
                    )
                    module.body.insert(last_import, node)
                    return

            module.body.insert(idx, node)
            return

        # No idx -> smart placement
        if function_positions:
            module.body.insert(function_positions[-1], node)

        elif name_guard_positions:
            module.body.insert(name_guard_positions[0], node)

        elif import_positions:
            module.body.insert(import_positions[-1] + 1, node)

        else:
            module.body.append(node)

    def _insert_function_in_class(self, idx, node, module, *_) -> None:
        """
        Insert a method into the first class definition in a module.

        If *idx* is provided and falls within the class body, the method is
        inserted at that position. Otherwise, the method is inserted
        immediately after the last existing method in the class. If the class
        contains no methods, the new method is appended to the class body.

        Parameters
        ----------
        idx : int or None
            Desired insertion index within the class body.
        node : ast.FunctionDef
            Method definition to insert.
        module : ast.Module
            Module containing the target class definition.
        *_ : Any
            Additional positional arguments accepted for interface
            compatibility and ignored.

        Raises
        ------
        ValueError
            If no :class:`ast.ClassDef` exists in the target module.
        """
        class_node = next((n for n in module.body if isinstance(n, ast.ClassDef)), None)

        if not class_node:
            raise ValueError("No class found for method insertion")

        function_positions = self._find_positions(class_node.body, ast.FunctionDef)

        if idx is not None and idx <= len(class_node.body):
            class_node.body.insert(idx, node)
            return

        # Default: after last method
        if function_positions:
            insert_pos = function_positions[-1] + 1
        else:
            insert_pos = len(class_node.body)

        class_node.body.insert(insert_pos, node)

    def _insert_function_module(self, idx, node, module, *_) -> None:
        """
        Insert a function definition into a module.

        If *idx* is provided, insertion is attempted at that position.
        However, function definitions are never inserted within the import
        section; when necessary, the insertion point is shifted to the first
        position after the final import statement.

        When *idx* is not provided, the function is inserted immediately
        before the last existing function definition. If no functions are
        present, it is appended to the module body.

        Parameters
        ----------
        idx : int or None
            Desired insertion index.
        node : ast.FunctionDef
            Function definition to insert.
        module : ast.Module
            Module whose body is modified in place.
        *_ : Any
            Additional positional arguments accepted for interface
            compatibility and ignored.
        """
        import_positions = self._find_positions(
            module.body, (ast.Import, ast.ImportFrom)
        )
        function_positions = self._find_positions(module.body, ast.FunctionDef)

        if idx is not None:
            if import_positions:
                last_import = import_positions[-1] + 1

                if idx <= last_import:
                    self.logger.info(
                        f"idx {idx} inside imports -> shifting to {last_import}"
                    )
                    module.body.insert(last_import, node)
                    return

            module.body.insert(idx, node)
            return

        # Default placement
        if function_positions:
            module.body.insert(function_positions[-1], node)
        else:
            module.body.append(node)

    def _insert_assign(self, idx, node, module, method_name, **kwargs) -> None:
        """
        Insert an assignment statement into a function or module.

        The target function is first resolved via
        :meth:`_resolve_target_function`. If a matching function is found,
        the insertion position is computed via
        :meth:`_compute_safe_insert_position` and the assignment is inserted
        within that function body.

        If no target function can be resolved, the assignment is inserted at
        module scope after the last existing assignment statement.

        Parameters
        ----------
        idx : int or None
            Desired insertion index within the target scope.
        node : ast.Assign
            Assignment statement to insert.
        module : ast.Module
            Module whose AST is modified in place.
        method_name : str
            Name of the target function or method.
        **kwargs
            Additional context forwarded to
            :meth:`_compute_safe_insert_position`.
        """
        target_func = self._resolve_target_function(module, method_name)

        if target_func:
            insert_pos = self._compute_safe_insert_position(target_func, idx, **kwargs)
            target_func.body.insert(insert_pos, node)
        else:
            insert_pos = self._last_position(module.body, ast.Assign)
            module.body.insert(insert_pos, node)

    def _insert_expr(self, idx, node, module, method_name, **kwargs) -> None:
        """
        Insert an expression statement into a function or module.

        The target function is first resolved via
        :meth:`_resolve_target_function`. If no matching function is found,
        the expression is appended at module scope.

        Special handling is applied to calls of the form
        ``read_dummy(...)``. Such expressions use
        :meth:`_handle_read_dummy` to determine a safe insertion location.
        All other expressions use
        :meth:`_compute_safe_insert_position`.

        Parameters
        ----------
        idx : int or None
            Desired insertion index within the target scope.
        node : ast.Expr
            Expression statement to insert.
        module : ast.Module
            Module whose AST is modified in place.
        method_name : str
            Name of the target function or method.
        **kwargs
            Additional context forwarded to insertion-position helpers.
        """
        target_func = self._resolve_target_function(module, method_name)

        if not target_func:
            module.body.append(node)
            return

        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "read_dummy"
        ):
            insert_pos = self._handle_read_dummy(target_func, idx, **kwargs)
        else:
            insert_pos = self._compute_safe_insert_position(target_func, idx, **kwargs)

        target_func.body.insert(insert_pos, node)

    def _find_positions(self, body: list, node_type: ast.AST) -> list[int]:
        """
        Locate all occurrences of a given AST node type within a node list.

        Parameters
        ----------
        body : list of ast.AST
            Sequence of AST nodes to inspect.
        node_type : type or tuple[type, ...]
            AST node type or types to match.

        Returns
        -------
        list[int]
            Indices of all nodes whose type matches *node_type*.
        """
        return [i for i, stmt in enumerate(body) if isinstance(stmt, node_type)]

    def _find_name_guard(self, body: list[ast.AST]) -> list[int]:
        """
        Locate conditional guard statements within a module body.

        This helper identifies :class:`ast.If` statements whose test
        expression is represented by an :class:`ast.Compare`. It is
        primarily intended to detect module-level guard blocks such as
        ``if __name__ == "__main__":``.

        Parameters
        ----------
        body : list[ast.AST]
            Module body to inspect.

        Returns
        -------
        list[int]
            Indices of matching guard statements.
        """
        positions = []

        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare):
                positions.append(i)

        return positions

    def _resolve_target_function(
        self, module: ast.AST, method_name: str
    ) -> ast.FunctionDef | None:
        """
        Resolve the target function for AST insertion.

        If *module* is itself a :class:`ast.FunctionDef`, it is returned
        directly. Otherwise, the AST is searched for a function definition
        whose name matches *method_name*.

        Parameters
        ----------
        module : ast.AST
            AST subtree to search.
        method_name : str
            Name of the function to locate.

        Returns
        -------
        ast.FunctionDef or None
            Matching function definition, or ``None`` if no matching
            function is found.

        Notes
        -----
        If multiple matching functions exist, only the first match returned
        by :func:`ast.walk` is used.
        """
        if isinstance(module, ast.FunctionDef):
            return module

        matches = [
            f
            for f in ast.walk(module)
            if isinstance(f, ast.FunctionDef) and f.name == method_name
        ]

        return matches[0] if matches else None

    def _compute_safe_insert_position(
        self, func: ast.FunctionDef, idx: int, **kwargs
    ) -> int:
        """
        Compute a valid insertion position within a function body.

        The default insertion position is determined via
        :meth:`_check_position_within_function`, which accounts for
        recognized initialization patterns and return statements.

        If *idx* is provided and falls within the valid insertion region,
        it is used instead of the computed position.

        Parameters
        ----------
        func : ast.FunctionDef
            Function whose body will receive the new statement.
        idx : int or None
            Desired insertion index.
        **kwargs
            Additional context used for insertion analysis. Expected keys
            include ``class_name`` and ``method``.

        Returns
        -------
        int
            Safe insertion index within *func*.
        """
        class_name = kwargs.get("class_name")
        method = kwargs.get("method")

        insert_pos, return_pos = self._check_position_within_function(
            func, class_name, method
        )

        if (
            idx is not None
            and (return_pos is None or idx < return_pos)
            and idx > insert_pos
        ):
            return idx

        return insert_pos

    def _last_position(self, body: list, node_type: ast.AST) -> int:
        """
        Return the insertion position following the last matching node.

        Parameters
        ----------
        body : list of ast.AST
            Sequence of AST nodes to inspect.
        node_type : type or tuple[type, ...]
            AST node type or types to match.

        Returns
        -------
        int
            Index immediately after the last matching node. If no matching
            node exists, returns ``len(body)``.
        """
        positions = self._find_positions(body, node_type)
        return positions[-1] + 1 if positions else len(body)

    def _handle_read_dummy(self, func: ast.FunctionDef, idx: int, **kwargs) -> int:
        """
        Compute a safe insertion position for ``read_dummy`` calls.

        The insertion location is determined using
        :meth:`_check_position_within_function` and adjusted according to
        the presence of existing assignment statements and function
        boundary constraints.

        Parameters
        ----------
        func : ast.FunctionDef
            Function whose body will receive the expression.
        idx : int or None
            Desired insertion index.
        **kwargs
            Additional context used for insertion analysis. Expected keys
            include ``class_name`` and ``method``.

        Returns
        -------
        int
            Safe insertion index for the ``read_dummy`` expression.
        """
        assign_names = {
            stmt.targets[0].id
            for stmt in func.body
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
        }

        insert_pos, return_pos = self._check_position_within_function(
            func, kwargs.get("class_name"), kwargs.get("method")
        )

        if assign_names and (
            idx is not None
            and (return_pos is None or idx < return_pos)
            and idx > insert_pos
        ):
            return idx

        return insert_pos

    def _is_scalar_var(self, dec_statement: Any) -> str | None:
        """
        Determine whether a declaration represents a scalar variable.

        A declaration is considered scalar when it has no initialization,
        dimension attribute, or explicit shape specification.

        Parameters
        ----------
        dec_statement : Any
            Declaration statement to inspect. The object is expected to be
            compatible with :func:`walk` and the relevant Fortran parser
            node types.

        Returns
        -------
        str or None
            Name of the declared variable if it is identified as a scalar
            variable; otherwise ``None``.

        Notes
        -----
        Declarations containing initialization expressions, dimension
        attributes, or explicit shape specifications are excluded.
        """
        var = walk(dec_statement, F23.Entity_Decl)[0].string
        init_spec = any(walk(dec_statement, F23.Initialization))
        alloc_spec = any(walk(dec_statement, F23.Dimension_Attr_Spec))
        explicit_shape = any(walk(dec_statement, F23.Explicit_Shape_Spec))

        if not init_spec and not alloc_spec and not explicit_shape:
            return var
        return None

    def separate_scalar(self, subroutine_key: str, dec_stmts: list = None) -> None:
        """
        Populate :attr:`scalar` with scalar or logical variable names.

        The source of declarations depends on the supplied arguments and the
        current value of :attr:`global_state`:

        1. If *dec_stmts* is provided, only the supplied declaration
        statements are processed.
        2. If *dec_stmts* is ``None`` and :attr:`global_state` is ``True``,
        declarations are obtained from
        ``self.extractor.dec_global[subroutine_key]``.
        3. If *dec_stmts* is ``None`` and :attr:`global_state` is ``False``,
        declarations are obtained from
        ``self.extractor.var_dummy[subroutine_key]`` and filtered by
        intent attributes.

        Variables identified as scalar or logical via
        :meth:`_is_scalar_var` are added to :attr:`scalar`.

        Parameters
        ----------
        subroutine_key : str
            Identifier of the subroutine whose declarations should be
            analyzed.
        dec_stmts : list, optional
            Declaration statements to process. If omitted, declarations are
            obtained from internal extractor state.

        Raises
        ------
        Exception
            Re-raises any exception encountered during declaration analysis
            after logging the error.
        """
        try:
            self.scalar = []

            # Step 1: determine source of declaration statements
            if dec_stmts is not None:
                statements = dec_stmts

            elif self.global_state:
                statements = [
                    self.extractor.dec_global[subroutine_key][var]
                    for var in self.variable_order
                ]

            else:
                statements = self.extractor.var_dummy[subroutine_key]

            # Step 2: process statements
            for dec_statement in statements:
                # filter by intent only for dummy args
                if dec_stmts is None and not self.global_state:
                    has_valid_intent = any(
                        i.tostr() in ["IN", "INOUT", "OUT"]
                        for i in walk(dec_statement, F23.Intent_Spec)
                    )
                    if not has_valid_intent:
                        continue

                if not dec_statement:
                    continue

                varname = self._is_scalar_var(dec_statement)
                if varname:
                    self.scalar.append(varname)

        except Exception as e:
            self.logger.exception("Exception in separate_scalar", e)
            raise

    def read_file_ast(self, assign_nodes: list[ast.Assign]) -> list[ast.Assign]:
        """
        Generate AST assignments for reading variables from a binary file.

        Each assignment node is converted into an equivalent file-read
        operation. Scalar variables are handled via
        :meth:`_build_scalar_read`, while array variables are handled via
        :meth:`_build_array_read`.

        Parameters
        ----------
        assign_nodes : list[ast.Assign]
            Assignment nodes describing variables that should be read from
            the binary file.

        Returns
        -------
        list[ast.Assign]
            AST assignment nodes that perform the corresponding file-read
            operations.
        """
        var_list = []

        for node in assign_nodes:
            if not isinstance(node, ast.Assign):
                continue

            target = self._make_read_target(node)
            value = node.value

            if isinstance(value, ast.Call) and len(value.keywords) > 0:
                new_node = self._build_array_read(target, value)
            else:
                new_node = self._build_scalar_read(target, value)

            var_list.append(new_node)

        return var_list

    def _make_read_target(self, node: ast.Assign) -> ast.Attribute | ast.Name:
        """
        Construct a writable AST target from an assignment target.

        The target is converted into a node with a
        :class:`ast.Store` context suitable for use on the left-hand side of
        a generated assignment statement.

        Parameters
        ----------
        node : ast.Assign
            Assignment node whose target is to be transformed.

        Returns
        -------
        ast.Name or ast.Attribute
            Writable AST target corresponding to the original assignment
            target.

        Raises
        ------
        TypeError
            If the assignment target type is unsupported.
        """
        target = node.targets[0]

        if isinstance(target, ast.Name):
            return ast.Name(id=target.id, ctx=ast.Store())

        if isinstance(target, ast.Attribute):
            return ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=target.attr,
                ctx=ast.Store(),
            )

        raise TypeError(f"Unsupported target type: {type(target)}")

    def _build_scalar_read(self, target: ast.AST, value: ast.AST) -> ast.Assign:
        """
        Build an AST assignment that reads a scalar value from a file.

        The generated assignment reads a single value from ``ffile`` using
        the appropriate reader function and assigns the first element of the
        returned array-like object to *target*.

        Logical values receive special handling and are converted using
        ``np.bool`` to preserve Fortran logical semantics.

        Parameters
        ----------
        target : ast.AST
            Assignment target.
        value : ast.AST
            AST node describing the scalar datatype to read.

        Returns
        -------
        ast.Assign
            Assignment statement that performs the scalar read operation.
        """
        attr_type = value.func.attr
        read_type = self._get_read_func(attr_type)

        read_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ffile", ctx=ast.Load()),
                attr=read_type,
                ctx=ast.Load(),
            ),
            args=[self._dtype_attr(("np", attr_type))],
            keywords=[],
        )

        # NOTE: Special case: LOGICAL -> np.bool(...)
        # Logical values representation: .TRUE. is mostly respresented with -1 because all bits are set to 1
        # .FALSE. is represented by 0
        # https://stackoverflow.com/a/39454385
        if attr_type == "bool":
            return ast.Assign(
                targets=[target],
                value=ast.Call(
                    func=self._dtype_attr(("np", "bool")),
                    args=[ast.Subscript(read_call, ast.Constant(0), ctx=ast.Load())],
                    keywords=[],
                ),
            )

        return ast.Assign(
            targets=[target],
            value=ast.Subscript(read_call, ast.Constant(0), ctx=ast.Load()),
        )

    def _build_array_read(self, target: ast.AST, value: ast.AST) -> ast.Assign:
        """
        Build an AST assignment that reads and reshapes array data.

        The generated assignment reads array contents from ``ffile`` using
        the appropriate reader function and reshapes the resulting data
        according to the metadata contained in *value*. Reshaping is
        performed using Fortran ordering (``order="F"``).

        Parameters
        ----------
        target : ast.AST
            Assignment target representing the destination array.
        value : ast.AST
            AST node containing datatype and shape information.

        Returns
        -------
        ast.Assign
            Assignment statement that performs the array read and reshape
            operation.
        """
        attr_type = value.keywords[0].value.attr
        read_type = self._get_read_func(attr_type)

        read_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ffile", ctx=ast.Load()),
                attr=read_type,
                ctx=ast.Load(),
            ),
            args=[self._dtype_attr(("np", attr_type))],
            keywords=[],
        )

        reshape_call = ast.Call(
            func=ast.Attribute(value=read_call, attr="reshape", ctx=ast.Load()),
            args=[value.args[0]],
            keywords=[ast.keyword(arg="order", value=ast.Constant("F"))],
        )

        return ast.Assign(
            targets=[ast.Subscript(target, ast.Slice(), ctx=ast.Store())],
            value=reshape_call,
        )

    def _get_read_func(self, dtype: str) -> str:
        """
        Map data type to corresponding binary read function.

        Parameters
        ----------
        dtype : str
            Data type string (e.g., 'float64', 'int32').

        Returns
        -------
        str
            Name of the reader function ('read_reals' or 'read_ints').
        """
        return "read_reals" if dtype == "float64" else "read_ints"

    def init_dependant_variables(
        self, read_ast: ast.Module, assign_nodes: list
    ) -> list:
        """
        Insert dependency-managed variable initializations into a read AST.

        The insertion order is determined by :attr:`dependant_variables`,
        which maps variables to the variables on which they depend. For each
        managed variable, the latest assignment position among its dependees
        is identified and the variable's initialization statement is inserted
        immediately afterward.

        Insertions are planned before modification of the AST and then
        applied in sorted order to avoid index-shifting issues during
        insertion.

        Parameters
        ----------
        read_ast : ast.Module
            Module AST whose body will receive dependency-managed
            initialization statements.
        assign_nodes : list[ast.Assign]
            Assignment nodes representing variables requiring deferred
            initialization.

        Returns
        -------
        list[ast.AST]
            Updated module body containing the inserted initialization
            statements.

        Raises
        ------
        Exception
            Re-raises any exception encountered during dependency analysis
            or AST modification after logging the error.
        """
        try:
            # Build index of existing assignments
            assign_positions = {}  # var_name -> position

            for i, stmt in enumerate(read_ast.body):
                if isinstance(stmt, ast.Assign):
                    name = self._get_assigned_name(stmt)
                    if name:
                        assign_positions[name] = i

            # Map new assignments by variable name
            new_assign_map = {
                self._get_assigned_name(node): node
                for node in assign_nodes
                if self._get_assigned_name(node)
            }

            # Compute insertion plan
            insert_plan = []  # list of (position, node)

            for var, dependees in self.dependant_variables.items():
                # Skip if no assignment exists for this variable
                assign_node = new_assign_map.get(var)
                if not assign_node:
                    continue

                # Find latest dependency position
                max_pos = max(
                    (assign_positions.get(dep, -1) for dep in dependees), default=-1
                )

                insert_pos = max_pos + 1
                insert_plan.append((insert_pos, assign_node))

            # Sort insertions to avoid shifting issues
            insert_plan.sort(key=lambda x: x[0])

            # Apply insertions with offset correction
            offset = 0
            for pos, node in insert_plan:
                read_ast.body.insert(pos + offset, node)
                offset += 1

            return read_ast.body

        except Exception:
            self.logger.exception("Exception in init_dependant_variables")
            raise

    def _get_assigned_name(
        self, node: ast.Assign, require_self: bool = False
    ) -> str | None:
        """
        Extract the assigned variable name from an assignment statement.

        Supports both direct assignments and attribute assignments. When
        *require_self* is ``True``, only assignments to ``self`` attributes
        are considered valid.

        Examples of supported assignments include:

        - ``x = value``
        - ``self.x = value``

        Parameters
        ----------
        node : ast.Assign
            Assignment node to inspect.
        require_self : bool, optional
            If ``True``, only assignments targeting ``self`` attributes are
            accepted.

        Returns
        -------
        str or None
            Extracted variable name, or ``None`` if no supported assignment
            target is found.
        """
        if not node.targets:
            return None

        target = node.targets[0]

        if isinstance(target, ast.Name):
            return target.id

        if isinstance(target, ast.Attribute):
            if require_self:
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    return target.attr
                return None

            return target.attr

        return None

    def transfer_to_pyfile(
        self,
        tree: ast.Module,
        subroutine_key: str,
        folder_name: str | None = "hydrol",
        python_file_type: Literal["global_module", "main"] = "global_module",
    ) -> None:
        """
        Write a generated Python AST to a Python source file.

        The output file is written beneath *folder_name* using a subdirectory
        named after *subroutine_key*. The generated source is obtained via
        :func:`ast.unparse` and written with executable user permissions.

        The output filename follows the convention::

            {python_file_type}_{subroutine_key}.py

        Parameters
        ----------
        tree : ast.Module
            Python AST to serialize and write.
        subroutine_key : str
            Name of the target subroutine used to construct the output path.
        folder_name : str, optional
            Root directory containing generated Python files.
        python_file_type : {"global_module", "main"}, optional
            Type of Python file being generated.

        Raises
        ------
        ValueError
            If *folder_name* cannot be located by
            :func:`find_folder`.
        OSError
            If directory creation, file writing, or permission updates
            fail.
        """
        try:
            current_dir = os.getcwd()

            path_to_folder = find_folder(current_dir, target_folder=folder_name)

            if path_to_folder is None:
                raise ValueError(
                    f"For the given folder, it couldn't be found: {path_to_folder}"
                )

            subroutine_path = os.path.join(path_to_folder, subroutine_key)
            file_path = os.path.join(
                subroutine_path, f"{python_file_type}_{subroutine_key}.py"
            )

            # First create python benchmark directory which will contain the directories of each subroutines
            # dir within which contains the output of the subroutines test
            self.logger.info("Creating benchmark directory...")
            os.makedirs(path_to_folder, exist_ok=True)

            # Then the subroutine directory within the benchmark
            self.logger.info("Creating subroutine directory...")
            os.makedirs(subroutine_path, exist_ok=True)

            self.logger.info(f"Writing Python file: {file_path}")
            with open(file_path, "w") as f:
                f.write("#!/usr/bin/env python3\n")
                f.write(ast.unparse(tree))

            rights = stat.S_IRWXU
            os.chmod(file_path, rights)

            self.logger.info("File successfully written.")

        except Exception as e:
            self.logger.exception("Exception in transfer_to_pyfile", e)
        except OSError as e:
            self.logger.error(
                "If directory creation, file writing, or permission updates \
            fail.",
                e,
            )

    def insert_all_assign_nodes(
        self, assign_nodes: list, code_tree: ast.Module, method_name: str, **kwargs
    ) -> None:
        """
        Insert multiple assignment statements into a target AST.

        Assignments are first mapped by variable name and then inserted in an
        order determined by the current processing mode.

        When :attr:`global_state` is ``True``, variables declared in
        :attr:`variable_order` are inserted according to declaration order,
        while dependency-managed variables are excluded and handled
        separately. Newly discovered variables are inserted first using
        dependency-aware ordering.

        When :attr:`global_state` is ``False``, all assignments are ordered
        via :func:`order_assignments` and inserted into the target method.

        Parameters
        ----------
        assign_nodes : list[ast.Assign]
            Assignment nodes to insert.
        code_tree : ast.Module
            Module AST that will be modified in place.
        method_name : str
            Name of the target function or method.
        **kwargs
            Additional context forwarded to :meth:`insert_at`.
        Raises
        ------
        ValueError
            If an assignment node does not contain a valid assignment target.
        Exception
            Re-raises any exception encountered during insertion after
            logging the error.
        """

        try:
            name_to_node = {}
            # Create node map
            for node in assign_nodes:
                name = self._get_assigned_name(node)
                if not name:
                    raise ValueError(
                        f"Invalid assignment node: {ast.unparse(ast.fix_missing_locations(node))}"
                    )
                name_to_node[name] = node

            all_names = set(name_to_node.keys())

            if self.global_state:
                declared = set(self.variable_order)
                dependant = set(self.dependant_variables.keys())
                new_vars = all_names - declared

                # Insert new variables first (dependency ordered)
                if new_vars:
                    ordered_new = order_assignments(assign_nodes, new_vars)

                    for var in ordered_new:
                        node = name_to_node.get(var)
                        if node:
                            self.insert_at(
                                None, node, code_tree, method_name=method_name
                            )

                # Insert declared variables
                for var in self.variable_order:
                    if var in dependant:
                        continue  # skip dependency-managed vars

                    node = name_to_node.get(var)
                    if node:
                        self.insert_at(None, node, code_tree, method_name=method_name)

            else:
                ordered_vars = order_assignments(assign_nodes, None)

                for var in ordered_vars:
                    node = name_to_node.get(var)
                    if node:
                        self.insert_at(
                            None, node, code_tree, method_name=method_name, **kwargs
                        )

        except Exception:
            self.logger.exception("Exception in insert_all_assign_nodes")
            raise

    def create_test_function(
        self, cls_info: dict, subroutine_key: str
    ) -> ast.FunctionDef:
        """
        Create a test function to compare the output of the
        Python code with the FORTRAN output saved in `output.bin`.

        Parameters
        ----------
        cls_info : dict
            Dictionary containing all the information of
            classes that some variable might depend on.

        Returns
        -------
        ast.FunctionDef
            Function AST to test the output of Python with that of FORTRAN.
        """
        try:
            # 1. Extract instance + attributes
            instance_name, attributes = self._extract_instance_attributes(cls_info)

            # 2. Build function arguments
            args = self._build_test_args(instance_name, attributes, subroutine_key)

            # 3. Create function skeleton
            function_def = ast.FunctionDef(
                name=f"test_{subroutine_key}",
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

            # 4. Add setup code (file reading)
            function_def.body.extend(self._build_test_setup_block(subroutine_key))

            # For arrays, we will use the allclose to see if two arrays has the same shape and values:
            # https://numpy.org/doc/2.3/reference/generated/numpy.allclose.html
            # For scalars, we will use isclose which is helpful when comparing floating points precision :
            # https://numpy.org/devdocs/reference/generated/numpy.isclose.html
            # 5. Build comparison loop
            comp_loop = self._build_test_loop(cls_info, subroutine_key)

            function_def.body.extend(comp_loop)

            return function_def

        except Exception:
            self.logger.exception("Exception in create_test_function")
            return None

    def _extract_instance_attributes(self, cls_info: dict) -> tuple:
        """
        Retrieves the first instance in `cls_info` and merges its attributes
        with any nested instance attributes.

        Parameters
        ----------
        cls_info : dict
            Nested class/instance metadata structure.

        Returns
        -------
        tuple
            (instance_name, attributes)
            - instance_name : str
            - attributes : dict
                Merged attribute dictionary for the instance.

        Raises
        ------
        ValueError
            If no instance is found in `cls_info`.
        """
        for class_data in cls_info.values():
            instance_name = next(iter(class_data))
            instance_data = class_data[instance_name]

            attributes = copy.deepcopy(instance_data.get("attributes", {}))

            # Merge nested instance attributes
            for inst in instance_data.get("instances", {}).values():
                attributes.update(inst.get("attributes", {}))

            return instance_name, attributes

        raise ValueError("No instance found in cls_info")

    def _build_test_args(
        self, instance_name: str, attributes: dict, subroutine_key: str
    ) -> list:
        """
        Build argument list for generated test function.

        Parameters
        ----------
        instance_name : str
            Name of the class instance.
        attributes : dict
            Instance attribute dictionary.
        subroutine_key : str
            Subroutine identifier used to fetch modified variables.

        Returns
        -------
        list
            List of ast.arg nodes representing function arguments.
        """
        args = []

        modif_vars = set(self.extractor.var_modif[subroutine_key])

        # If any modified variable is a class attribute -> pass instance
        if attributes.keys() & modif_vars:
            args.append(ast.arg(arg=instance_name))

        # Add non-attribute variables
        for var in modif_vars - attributes.keys():
            args.append(ast.arg(arg=var))

        return args

    def _build_test_setup_block(self, subroutine_key) -> list:
        """
        Generate AST nodes for test function setup section.

        Parameters
        ----------
        subroutine_key : str
            Identifier for selecting benchmark/output directory.

        Returns
        -------
        list
            List of AST nodes representing setup statements.
        """

        code = """
print('--- inside the test function for {subroutine_name} ---')
path = f'{benchmark_dir}/{subroutine_name}/output.bin'
ffile = FortranFile(path, 'r')
        """
        code = code.format(
            benchmark_dir=self.benchmark_dir, subroutine_name=subroutine_key
        )
        return ast.parse(code).body

    def _build_test_loop(self, cls_info: dict, subroutine_key: str):
        """
        Construct an AST-based test loop for evaluating modified variables
        against reference outputs using a generated comparison template.

        This method builds two AST nodes:

        1. A list assignment containing modified variable names extracted from
        `self.extractor.var_modif_info[subroutine_key]`.
        2. A `for` loop that iterates over zipped modified variables and their
        corresponding values, then injects a comparison logic template into
        the loop body after applying global replacements.

        The loop body is populated using a Python code template loaded from
        configuration via :func:`load_code_templates`. Global replacements
        are applied using :class:`ReplaceGlobals`.

        Parameters
        ----------
        cls_info : dict
            Metadata describing the class or instance context used for resolving
            global references and applying transformations inside the AST.
        subroutine_key : str
            Identifier used to select the appropriate set of modified variables
            and associated template logic from internal extractor storage.

        Returns
        -------
        list of ast.AST
            A list containing:
            - An `ast.Assign` node defining `modif_var`
            - An `ast.For` node representing the constructed test loop

        Raises
        ------
        ValueError
            If the loaded comparison template is `None`.
        SyntaxError
            If the injected template code cannot be parsed into valid Python AST.
        Exception
            If global replacement via :class:`ReplaceGlobals` fails.
        """
        variables = list(self.extractor.var_modif_info[subroutine_key].keys())

        # modif_var = [...]
        modif_var_assign = ast.Assign(
            targets=[ast.Name(id="modif_var", ctx=ast.Store())],
            value=ast.List(elts=[ast.Constant(v) for v in variables], ctx=ast.Load()),
        )

        # for variable, value in zip(...)
        loop = ast.For(
            target=ast.Tuple(
                elts=[
                    ast.Name(id="variable", ctx=ast.Store()),
                    ast.Name(id="value", ctx=ast.Store()),
                ],
                ctx=ast.Store(),
            ),
            iter=ast.Call(
                func=ast.Name(id="zip", ctx=ast.Load()),
                args=[
                    ast.Name(id="modif_var", ctx=ast.Load()),
                    ast.List(
                        elts=[ast.Name(id=v, ctx=ast.Load()) for v in variables],
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            ),
            body=[],
            orelse=[],
        )

        try:
            loop = ReplaceGlobals(cls_info).visit_For(loop)
        except Exception:
            self.logger.exception("ReplaceGlobals failed")
            raise

        # Load comparison template
        templates = load_code_templates(self.config_path)
        code = templates["Python_templates"]["Python_test_output_template"]["template"]

        if code is None:
            raise ValueError("Test output template is None")

        try:
            loop.body = ast.parse(code).body
        except SyntaxError as e:
            self.logger.error("Syntax error while parsing the code template:", e)
            raise

        return [modif_var_assign, loop]

    def prepare_read_code_for_global_template(
        self, assign_nodes: list, subroutine_key: str
    ) -> ast.Module:
        """
        This method assumes that the necessary `for` loops for reading array data are already
        present in the `read_code_template`. It focuses on scalar variables, which are read
        individually and inserted into their appropriate positions within the code template.
        During this process, the names of the scalar variables are also added to the list
        of variables handled inside the `for` loop.

        Parameters
        ----------
        assign_nodes : list
            list of `ast.Assign` nodes for each variable.

        Returns
        -------
        ast.Module
            Python AST of the read template with the newly added elements.
        """

        try:
            templates = load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")

            for_template_str = templates["Python_templates"][
                "Python_read_for_loop_class_template"
            ]["template"]

            read_code_template = for_template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=subroutine_key,
            )

            read_ast = python_parser(read_code_template)
            if read_ast is None:
                raise ValueError(
                    "read_ast for the python read code for global template is None due to prior error"
                )

            if assign_nodes:
                var_list = self.read_file_ast(assign_nodes)
                for_node = self._find_first_node(read_ast, ast.For)

                insert_pos = read_ast.body.index(for_node)
                for var in var_list:
                    read_ast.body.insert(insert_pos, var)
                    insert_pos += 1

            self._populate_for_loop_iterable(read_ast)

            return read_ast

        except Exception:
            self.logger.exception(
                "Error from prepare_read_code_for_global_template method"
            )
            return None

    def _populate_for_loop_iterable(self, read_ast: ast.Module) -> None:
        """
        Populate empty for-loop iterable with ordered non-scalar variables.

        If the template contains a for-loop with an empty list iterator,
        this fills it using variables from `self.variable_order`, excluding
        scalars.

        Parameters
        ----------
        read_ast : ast.Module
            AST of the read function containing the target for-loop.
        """
        for_node = self._find_first_node(read_ast, ast.For)

        if not for_node:
            raise ValueError("No FOR loop found in template")

        if isinstance(for_node.iter, ast.List) and not for_node.iter.elts:
            variables = [v for v in self.variable_order if v not in self.scalar]

            for_node.iter.elts = [ast.Constant(v) for v in variables]

    def _find_first_node(
        self, tree: ast.AST, node_type: type[ast.AST]
    ) -> ast.AST | None:
        """
        Return the first AST node matching a given type.

        Performs a depth-first traversal and returns the first occurrence
        of the specified AST node type.

        Parameters
        ----------
        tree : ast.AST
            AST to search.
        node_type : type[ast.AST]
            AST node class to locate (e.g., ast.For, ast.FunctionDef).

        Returns
        -------
        ast.AST or None
            First matching node if found, otherwise None.
        """
        return next((n for n in ast.walk(tree) if isinstance(n, node_type)), None)

    def transform_to_class(self, ast_nodes: list, subroutine_key: str) -> ast.Module:
        """
        This builds a global module class by separating imports and assignments,
        injecting initialization logic, and wiring declaration routines.

        Parameters
        ----------
        ast_nodes : list
            AST nodes including ast.Assign, ast.Import, and ast.ImportFrom.
        subroutine_key : str
            Identifier used to specialize class and variable handling.

        Returns
        -------
        ast.Module
            Final Python class-based AST module.
        """

        try:
            class_tree = self.out_module_python()
            if class_tree is None:
                raise ValueError("Class_tree is None")

            class_defs = ast_walk(class_tree, ast.ClassDef)

            for class_def in class_defs:
                if class_def.name == "Global_module":
                    class_def.name = "_".join(["Global_module", subroutine_key])

            assign_nodes = []
            procedure_nodes = []

            for node in ast_nodes:
                if isinstance(node, ast.Import | ast.ImportFrom):
                    procedure_nodes.append(node)
                elif isinstance(node, ast.Assign):
                    assign_nodes.append(node)

            # If the procedures are present then that we add them to the template
            if procedure_nodes:
                for procedure_node in procedure_nodes:
                    self.insert_at(None, procedure_node, class_tree)

            self.separate_scalar(subroutine_key=subroutine_key)

            for func in ast_walk(class_tree, ast.FunctionDef):
                if func.name == "__init__":
                    self.insert_all_assign_nodes(
                        assign_nodes, class_tree, method_name=func.name
                    )

                elif func.name == "declaration_initialization":
                    self._handle_declaration_init(
                        func, assign_nodes, class_tree, subroutine_key
                    )

            # What this does that it fixes the missing location(lineno,end_lineno,col_offset,end_col_offset)
            # based on the parent node
            # https://docs.python.org/3/library/ast.html#ast.fix_missing_locations
            class_tree = ast.fix_missing_locations(class_tree)

            return class_tree
        except Exception:
            self.logger.exception("Exception error in transform_to_class")
            raise

    def _handle_declaration_init(
        self,
        func: ast.FunctionDef,
        assign_nodes: list,
        class_tree: ast.AST,
        subroutine_key: str,
    ) -> None:
        """
        Populate declaration_initialization method with read and setup logic.

        Builds scalar reads and dependent variable initialization, then injects
        them into the function body while cleaning up placeholders.

        Parameters
        ----------
        func : ast.FunctionDef
            Function inside which to populate.
        assign_nodes : list
            Assignement nodes to check for scalar variables.
        class_tree : ast.AST
            Module-level AST containing class definitions.
        subroutine_key : str
        """
        # Remove placeholder pass
        if func.body and isinstance(func.body[0], ast.Pass):
            func.body.pop(0)

        scalar_nodes = self._get_scalar_nodes(assign_nodes)

        read_ast = self.prepare_read_code_for_global_template(
            scalar_nodes, subroutine_key=subroutine_key
        )

        if read_ast is None:
            raise ValueError("read_ast is None")

        read_body = self.init_dependant_variables(read_ast, assign_nodes)

        func.body.extend(read_body)

        self._cleanup_declaration_function(func, class_tree, read_body)

    def _get_scalar_nodes(self, assign_nodes: list) -> list:
        """
        Extract assignment nodes corresponding to scalar variables.

        Parameters
        ----------
        assign_nodes : list
            Assignement nodes to check for scalar variables

        Returns
        -------
        list
            Filtered list of scalar ast.Assign nodes.
        """
        if not self.scalar:
            return []

        name_map = {self._get_assigned_name(node): node for node in assign_nodes}

        return [name_map[name] for name in self.scalar if name in name_map]

    def _cleanup_declaration_function(
        self, func: ast.FunctionDef, class_tree: ast.AST, read_body: list
    ) -> None:
        """
        Handles cleanup cases such as:
        - Empty for-loops with no iterations
        - Missing file read initialization (ffile-only case)
        - Removal of unnecessary declaration functions

        Parameters
        ----------
        func : ast.FunctionDef
            Function inside which to apply cleanup
        class_tree : ast.AST
            Module-level AST containing class definitions.
        read_body : list
            Read elements
        """
        if not read_body:
            return

        last_node = read_body[-1]

        # Case 1: Empty for loop
        if (
            isinstance(last_node, ast.For)
            and isinstance(last_node.iter, ast.List)
            and not last_node.iter.elts
        ):
            if not self.scalar:
                # No scalars and empty for loop -> remove entier function
                self._remove_function(class_tree, "declaration_initialization")
            else:
                # Only scalars and empty for loop -> remove for loop
                func.body = [stmt for stmt in func.body if stmt is not last_node]

        # Case 2: No file read
        elif (
            isinstance(last_node, ast.Assign)
            and isinstance(last_node.targets[0], ast.Name)
            and last_node.targets[0].id == "ffile"
        ):
            self._remove_function(class_tree, "declaration_initialization")

    def _remove_function(self, class_tree: ast.AST, name: str) -> None:
        """
        Remove a method from all classes in the AST module.

        Parameters
        ----------
        class_tree : ast.AST
            Module-level AST containing class definitions.
        name : str
            Function name to remove.
        """
        for cls in ast_walk(class_tree, ast.ClassDef):
            cls.body = [
                item
                for item in cls.body
                if not (isinstance(item, ast.FunctionDef) and item.name == name)
            ]

    def _maybe_insert_fortran_reshape_helper(self, tree: ast.Module) -> None:
        """
        Splice the ``fortran_reshape`` runtime helper into *tree*'s top level
        if the translation that produced *tree* emitted a call to it (i.e.
        :attr:`f2np`'s ``needs_fortran_reshape_helper`` flag is set).

        No-ops if the flag is unset, or if a function named
        ``fortran_reshape`` is already present in *tree* (defensive guard
        against double-insertion if this is ever called more than once on
        the same tree).

        Parameters
        ----------
        tree : ast.Module
            The finalized module AST, mutated in place.
        """
        if not getattr(self.f2np, "needs_fortran_reshape_helper", False):
            return

        already_present = any(
            isinstance(node, ast.FunctionDef) and node.name == "fortran_reshape"
            for node in tree.body
        )
        if already_present:
            return

        helper_def = build_fortran_reshape_helper()
        class_positions = self._find_positions(tree.body, ast.ClassDef)
        insert_pos = class_positions[0] if class_positions else len(tree.body)
        tree.body.insert(insert_pos, helper_def)

    def update_global_python(
        self, subroutine_key: str, cls_mode: bool = True
    ) -> ast.Module:
        """
        Build and update the global Python AST representation for a given subroutine.

        This method orchestrates the full transformation pipeline:
        - prepares configuration flags
        - extracts declarations from the Fortran side
        - converts them into Python AST nodes
        - builds a class-based structure
        - injects initialization, reading logic, and dependencies
        - attaches child subroutines (DFS traversal)
        - updates metadata (cls_info) and procedure calls

        Parameters
        ----------
        subroutine_key : str
            Name of the isolated subroutine.
        cls_mode : bool
            If the global Python code AST should be in class mode or not.

        Returns
        -------
        tree : ast.Module
            AST tree containing the finalized and updated elements.
        """
        try:
            # Step 1: Configure global transformation state
            self.cls_mode = cls_mode
            self.global_state = True
            self.f2np.needs_fortran_reshape_helper = False
            if not cls_mode:
                raise NotImplementedError(
                    "Only class-based global transformation is currently supported."
                )

            # Step 2: Load base template & extract variable metadata
            code_template = self.out_module_python()
            if code_template is None:
                raise ValueError("Code template is None")

            self.retreive_variable_order()
            self.pre_init_variables(code_template)

            # Step 3: Extract and convert declaration statements
            declaration_stmts = list(self.extractor.dec_global[subroutine_key].values())
            ast_nodes = self.convert_SPECIFICATION_PART(
                declaration_stmts=declaration_stmts, cls_mode=cls_mode
            )
            if ast_nodes is None:
                raise ValueError("AST nodes from specification part are None")

            # Step 4: Resolve dependency relationships
            self.search_dependant_variables(declaration_stmts=declaration_stmts)

            # Step 5: Build class-based AST
            tree = self.transform_to_class(
                ast_nodes=ast_nodes, subroutine_key=subroutine_key
            )

            class_def = next(iter(ast_walk(tree, ast.ClassDef)), None)
            if not class_def:
                raise ValueError("Expected a class-based global module")

            # Step 6: Build class metadata (attributes, methods, instances)
            cls_info, _, _ = self.create_cls_info(
                out_module=tree, subroutine_key=subroutine_key, self_mode=True
            )
            if not cls_info:
                raise ValueError("cls_info generation failed")

            # Step 7: Attach timing decorator/helper
            timer_tree = self.get_timer(subroutine_key=subroutine_key)
            if timer_tree is None:
                raise ValueError("Timer tree(@timer) is None")
            class_def.body.append(timer_tree)

            # Step 8: Collect child subroutines (DFS traversal)
            all_child_subroutines = []
            all_child_subroutines = self.collect_descendants_dfs(subroutine_key)

            if all_child_subroutines:
                all_child_subroutines[-1].decorator_list = (
                    [
                        ast.Name(
                            id=next(ast_walk(timer_tree, ast.FunctionDef)).name,
                            ctx=ast.Load(),
                        )
                    ]
                    if timer_tree
                    else []
                )
                class_def.body.extend(all_child_subroutines)

            # Step 9: Update method metadata + procedure calls
            update_methods(cls_info, all_child_subroutines)
            subroutine_to_stack_index = {
                func.name: idx for idx, func in enumerate(all_child_subroutines)
            }
            main_file_attributes = [
                names.string
                for names in walk(
                    walk(self.extractor.var_dummy[subroutine_key], F23.Entity_Decl),
                    F23.Name,
                )
            ]
            self._process_procedures(
                subroutine_key=subroutine_key,
                subroutine_to_stack_index=subroutine_to_stack_index,
                module_stacks=all_child_subroutines,
                cls_info=cls_info,
                main_file_attributes=main_file_attributes,
            )

            # Step 10: Ensure all methods have `self`
            for func in all_child_subroutines:
                arg_names = [arg.arg for arg in func.args.args]
                if "self" not in arg_names:
                    func.args.args.insert(0, ast.arg(arg="self"))

            # Step 11: Splice in fortran_reshape helper if any translated
            # subroutine (including collected descendants) used RESHAPE with
            # PAD=/ORDER=.
            self._maybe_insert_fortran_reshape_helper(tree)

            return ast.fix_missing_locations(tree)
        except Exception as e:
            self.logger.exception("Error in update_global_python method", e)
            return None

    def collect_descendants_dfs(self, subroutine_key: str) -> list:
        """
        Returns list of AST nodes in processing order: leaves first, parent last using the DFS algorithm.

        Parameters
        ----------
        subroutine_key : str

        Returns
        -------
        order : list
            Sorted subroutine list
        """
        visited = set()
        order = []

        def dfs(key, stack):
            if key in visited:
                return
            if key in stack:
                raise ValueError(
                    f"Cycle detected in subroutine call graph: {' -> '.join(stack + [key])}"
                )
            stack.append(key)
            # iterate called children if any
            called = self.extractor.call_within_sub.get(key) or {}
            for child_key in list(
                called.keys()
            ):  # We recursively call the call statement of internal call statement
                dfs(child_key, stack)
            # Now process this node (get AST)
            subroutines = self.isolator.working_subroutines[key]
            _, _, ast_list = self.f2np.recursive_ast(subroutines)
            if len(ast_list) > 1:
                raise ValueError(
                    f"The length of module stack for {key} AST is greater than 1:{len(ast_list)}"
                )
            order.append(ast_list[-1])
            visited.add(key)
            stack.pop()

        dfs(subroutine_key, [])
        return order

    def prepare_read_code_for_main_template(
        self, assign_nodes: list[ast.AST], subroutine_key: str
    ) -> ast.FunctionDef:
        """
        This method generates a specialized function that reads binary input
        data and initializes variables required by the main execution pipeline.
        It builds the function dynamically using a template and augments it
        with scalar and array read logic derived from assignment AST nodes.

        Parameters
        ----------
        assign_nodes : list[ast.AST]
            Assignment AST nodes representing variables that must be initialized
            or read from binary input.
        subroutine_key : str
            Identifier of the subroutine used to select variable metadata and
            scalar/array classification rules.

        Returns
        -------
        ast.FunctionDef
            AST node representing the fully constructed `read_dummy` function.

        Raises
        ------
        ValueError
            If templates cannot be loaded, parsing fails, or required AST
            components (function definition, scalar nodes) are missing.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            # Load and parse template
            templates = load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")

            for_template_str = templates["Python_templates"][
                "Python_read_dummy_template"
            ]["template"]

            read_code_template = for_template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=subroutine_key,
            )

            read_ast = python_parser(read_code_template).body[0]
            if read_ast is None:
                raise ValueError(
                    "read ast for main template is None due to prior error"
                )

            function_def = next(iter(ast_walk(read_ast, ast.FunctionDef)), None)
            if function_def is None:
                raise ValueError("No FunctionDef found in read_ast")

            # Add arguments (dummy variables)
            dummy_list = []
            for name in self.variable_order:
                function_def.args.args.append(ast.arg(arg=name))
                dummy_list.append(name)

            # Prepare scalar + array metadata
            self.separate_scalar(subroutine_key=subroutine_key)

            assign_map = {}
            for node in assign_nodes:
                target = node.targets[0]
                name = getattr(target, "id", getattr(target, "attr", None))
                if name:
                    assign_map[name] = node

            scalar_nodes = [
                assign_map[scalar] for scalar in self.scalar if scalar in assign_map
            ]
            scalar_read_nodes = self.read_file_ast(
                scalar_nodes
            )  # This will get the read stateemnt for scalars, boolean

            # Insert read logic in correct order
            seen = {
                "arrays": set(),
                "scalars": set(),
                "names": set(),
            }

            var_pos = self._get_insertion_position(read_ast)
            arrays_to_add = self._insert_reads_in_order(
                read_ast, function_def, scalar_read_nodes, assign_map, seen, var_pos
            )
            # Update loop iteration variables
            self._update_for_loop(read_ast, dummy_list, arrays_to_add)
            # Add return statement (if needed)
            self._add_return_if_needed(read_ast, seen, scalar_read_nodes)

            return read_ast
        except Exception as e:
            self.logger.exception("Exception in prepare_read_code_for_main_template", e)
            raise

    def _get_insertion_position(self, read_ast: ast.AST) -> int:
        """
        This method identifies the correct position to insert variable read
        operations by locating the first loop construct (`ast.For`) in the
        AST. If no loop is found, insertion defaults to the beginning.

        Parameters
        ----------
        read_ast : ast.AST
            AST of the read function template.

        Returns
        -------
        int
            Index position where read statements should be inserted.
        """
        for i, node in enumerate(ast.iter_child_nodes(read_ast)):
            if isinstance(node, ast.For):
                return max(i - 1, 0)
        return 0

    def _insert_reads_in_order(
        self,
        read_ast: ast.AST,
        function_def: ast.FunctionDef,
        scalar_read_nodes: list,
        assign_map: dict,
        seen: dict,
        var_pos: int,
    ) -> list:
        """
        This method ensures correct ordering of variable initialization by
        interleaving array reads and scalar reads based on variable precedence
        defined in `self.variable_order`.

        Parameters
        ----------
        read_ast : ast.AST
            AST of the read function being constructed.
        function_def : ast.FunctionDef
            Function definition node to update with required arguments.
        scalar_read_nodes : list
            AST nodes corresponding to scalar variable reads.
        assign_map : dict
            Mapping from variable names to their assignment AST nodes.
        seen : dict
            Tracking structure for processed arrays, scalars, and names.
        var_pos : int
            Current insertion index within the AST body.

        Returns
        -------
        list
            List of array variable names that were inserted into the AST.
        """
        arrays_to_add = []

        scalar_positions = [
            (s, self.variable_order.index(s))
            for s in self.scalar
            if s in self.variable_order
        ]

        for scalar_name, scalar_pos in scalar_positions:
            # Get all arrays before this scalar that are not present in the self.scalar and have not been previously seen/already read.
            arrays_before = [
                name
                for name in self.variable_order[:scalar_pos]
                if name not in self.scalar and name not in seen["arrays"]
            ]

            for name in arrays_before:
                assign = assign_map.get(name)
                if not assign:
                    continue

                new_node = self.read_file_ast([assign])[0]
                self._inject_tuple_args(function_def, new_node, seen)

                read_ast.body.insert(var_pos, new_node)
                var_pos += 1
                seen["arrays"].add(name)

            if scalar_name not in seen["scalars"]:
                scalar_node = self._find_scalar_node(scalar_read_nodes, scalar_name)
                read_ast.body.insert(var_pos, scalar_node)
                var_pos += 1
                seen["scalars"].add(scalar_name)

            arrays_to_add.extend(arrays_before)

        return arrays_to_add

    def _find_scalar_node(
        self, scalar_read_nodes: list, scalar_name: str
    ) -> ast.Assign:
        """
        Searches through precomputed scalar read nodes to locate the AST node
        matching the requested scalar variable name.

        Parameters
        ----------
        scalar_read_nodes : list
            List of AST assignment nodes representing scalar reads.
        scalar_name : str
            Name of the scalar variable to locate.

        Returns
        -------
        ast.Assign
            AST assignment node for the requested scalar.

        Raises
        ------
        ValueError
            If no matching scalar node is found.
        """
        for node in scalar_read_nodes:
            if isinstance(node, ast.Assign) and node.targets:
                name = self._get_assigned_name(node)
                if scalar_name == name:
                    return node

        raise ValueError(f"Could not find scalar node for {scalar_name}")

    def _inject_tuple_args(
        self, function_def: ast.FunctionDef, node: ast.AST, seen: dict
    ) -> None:
        """
        This method inspects AST call nodes that contain tuple unpacking and
        extracts variable names, inserting them as function arguments if they
        have not already been added.

        Parameters
        ----------
        function_def : ast.FunctionDef
            Function definition whose arguments will be modified.
        node : ast.AST
            AST node potentially containing tuple-based variable references.
        seen : dict
            Tracking dictionary to avoid duplicate argument insertion.
        """
        if not (
            isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Tuple)
        ):
            return

        for elt in reversed(node.value.args[0].elts):
            if isinstance(elt.value, ast.Name):
                name = elt.value.id
                if name not in seen["names"]:
                    function_def.args.args.insert(0, ast.arg(arg=name))
                    seen["names"].add(name)

    def _update_for_loop(
        self, read_ast: ast.AST, dummy_list: list, arrays_to_add: list
    ) -> None:
        """
        This method adjusts the loop variables used in the read function based
        on which arrays and scalars are required. If no iteration is needed,
        the loop is removed entirely.

        Parameters
        ----------
        read_ast : ast.AST
            AST of the read function.
        dummy_list : list
            Original list of dummy variables used for iteration.
        arrays_to_add : list
            Array variables that influence loop construction.

        Raises
        ------
        ValueError
            If no loop node is found in the AST.
        """

        for_node = next(iter(ast_walk(read_ast, ast.For)), None)
        if for_node is None:
            raise ValueError("for_node is None")

        table = self.scalar + arrays_to_add if arrays_to_add else self.scalar
        difference = [x for x in dummy_list if x not in table]

        if difference:
            for_node.iter.elts = [ast.Name(id=v, ctx=ast.Load()) for v in difference]
        else:
            read_ast.body = [n for n in read_ast.body if not isinstance(n, ast.For)]

    def _add_return_if_needed(
        self, read_ast: ast.AST, seen: dict, scalar_nodes: list
    ) -> None:
        """
        This method determines whether the generated read function should
        return scalar values or tuples of scalars. It skips return insertion
        when operating in `self` context or when no scalar nodes exist.

        Parameters
        ----------
        read_ast : ast.AST
            AST of the read function being constructed.
        seen : dict
            Tracking structure containing injected argument names.
        scalar_nodes : list
            List of scalar AST nodes used to determine return structure.
        """
        if not scalar_nodes:
            return

        # If operating on self -> no return needed
        if "self" in seen["names"]:
            return

        return_node = ast.Return()

        values = [ast.Name(id=s, ctx=ast.Load()) for s in self.scalar]

        if len(values) == 1:
            return_node.value = values[0]
        else:
            return_node.value = ast.Tuple(elts=values, ctx=ast.Load())

        read_ast.body.append(return_node)

    def update_main_python(
        self, out_module: ast.Module, subroutine_key: str
    ) -> ast.Module:
        """
        Construct and populate the main execution Python AST.

        This method builds a complete executable script by assembling imports,
        class instances, variable declarations, input-reading logic, function
        calls, and optional test execution into a unified `main()` function.

        Parameters
        ----------
        out_module : ast.Module
            The AST module containing the output/global definitions used
            for constructing the main script.
        subroutine_key : str
            Identifier for the target subroutine used to retrieve variable
            specifications and function mappings.

        Returns
        -------
        ast.Module
            A fully constructed and populated AST module representing the
            executable Python script. Returns None if construction fails.

        Raises
        ------
        ValueError
            If required components (main template, AST nodes, class info,
            or function mappings) are missing or invalid.

        """
        self.global_state = False
        self.f2np.needs_fortran_reshape_helper = False
        try:
            # 1. Load main template and locate main()
            out_main_template = self.out_main_python()
            if out_main_template is None:
                raise ValueError("Main template is None")

            self.retreive_variable_order()

            main_function = next(
                (
                    f
                    for f in ast_walk(out_main_template, ast.FunctionDef)
                    if f.name == "main"
                ),
                None,
            )
            if main_function is None:
                raise ValueError("main() not found")

            call_stmts = []  # Keeps the call statements of all the function
            function_stmts = []  # Keeps the all the functions ast in the order that they were created

            # 2. Extract class info (imports + instances)
            idx = len(main_function.body)

            main_cls_info, import_nodes, instance_nodes = self.create_cls_info(
                out_module, subroutine_key=subroutine_key
            )
            if not all((main_cls_info, import_nodes, instance_nodes)):
                raise ValueError(
                    "ONe of these three elements is None:cls_info,import_nodes,instance_nodes"
                )

            # Add the import node inside the main template
            for import_node in import_nodes:
                self.insert_at(
                    idx=None,
                    ast_node=import_node,
                    python_template=out_main_template,
                    method_name=None,
                )

            # Since we add the global instance first since the following variables could depend on this global instance
            for instance_node in instance_nodes:
                self.add_instance(
                    idx,
                    instance_node,
                    main_cls_info,
                    main_function,
                    ["declaration_initialization"],
                )

            # 3. Convert declaration statements -> AST nodes
            declaration_stmts = [
                [elements] for elements in self.extractor.var_dummy[subroutine_key]
            ]

            ast_nodes = self.convert_SPECIFICATION_PART(
                declaration_stmts, fix_loc=True, cls_mode=False
            )
            if ast_nodes is None:
                raise ValueError("Ast_nodes is None")

            assign_nodes = []
            procedure_nodes = []

            for node in ast_nodes:
                if isinstance(node, ast.Import | ast.ImportFrom):
                    procedure_nodes.append(node)
                elif isinstance(node, ast.Assign | ast.Assign):
                    assign_nodes.append(node)

            # Now we need to ensure that we add the procedure_nodes if they exist
            if procedure_nodes:
                for procedure_node in procedure_nodes:
                    self.insert_at(None, procedure_node, out_main_template)

            # 4. Insert assignments (dependency-aware)
            idx = len(main_function.body)
            self.insert_all_assign_nodes(
                assign_nodes,
                main_function,
                method_name="main",
                class_name=list(main_cls_info)[-1],
                method="declaration_initialization",
            )

            # Resolve references to global instances
            identify_replace_all(
                main_function.body, main_cls_info
            )  # THis function allows us to identify and replace them recursivly.

            # 5. Build read + execution calls
            idx = len(main_function.body)
            read_dummy_ast = self.prepare_read_code_for_main_template(
                assign_nodes, subroutine_key=subroutine_key
            )
            read_dummy_ast_call_stmt = self.create_call_statements(read_dummy_ast)
            if read_dummy_ast_call_stmt is None:
                raise ValueError("Read_ast_call_stmt is None")

            call_stmts.append(read_dummy_ast_call_stmt)
            function_stmts.append(read_dummy_ast)

            # We need to search among that of the Global module the function itself,
            instance_key, function_def = self._find_function_in_cls_info(
                main_cls_info, subroutine_key=subroutine_key
            )
            function_def_call_stmt = self.create_call_statements(
                function_def, instance=instance_key
            )
            if function_def_call_stmt is None:
                raise ValueError("Function defintions call statement is None")
            call_stmts.append(function_def_call_stmt)

            # 6. Create test function
            if os.path.exists(
                os.path.join(self.benchmark_dir, subroutine_key, "output.bin")
            ):
                test_subroutine_function = self.create_test_function(
                    main_cls_info, subroutine_key=subroutine_key
                )
                if test_subroutine_function is None:
                    raise ValueError(f"TEST subroutine {subroutine_key} is None")

                test_subroutine_function_call_stmt = self.create_call_statements(
                    test_subroutine_function
                )
                if test_subroutine_function_call_stmt is None:
                    raise ValueError("Test functions call statement is None")
                call_stmts.append(test_subroutine_function_call_stmt)
                function_stmts.append(test_subroutine_function)

            # 7. Insert calls into main()
            for call_stmt in call_stmts:
                if isinstance(call_stmt, ast.AST):
                    self.insert_at(idx, call_stmt, main_function, "main")
                    idx += 1
                else:
                    main_function.body.extend(call_stmt)
                    idx += len(call_stmt)

            # 8. Now we add the functions into the class module
            for functions in function_stmts:
                self.insert_at(None, functions, out_main_template)

            # In the case, we have reshape inside the main script
            self._maybe_insert_fortran_reshape_helper(out_main_template)

            return ast.fix_missing_locations(out_main_template)

        except Exception as e:
            self.logger.exception("Exception error in update_main_python", e)
            return None

    def _find_function_in_cls_info(self, cls_info: dict, subroutine_key: str) -> tuple:
        """
        Locate a function definition inside class instance metadata.

        Searches through nested class-instance mappings to find a function
        associated with the given subroutine key.
        """
        for _, module_content in cls_info.items():
            for instance_name, instance_data in module_content.items():
                methods = instance_data.get("methods", {})
                if subroutine_key in methods:
                    return instance_name, methods[subroutine_key]

        raise ValueError(f"{subroutine_key} not found in cls_info")

    def _process_procedures(
        self,
        subroutine_key: str,
        subroutine_to_stack_index: dict,
        module_stacks: list,
        cls_info: dict,
        main_file_attributes: list,
    ) -> None:
        """
        Recursively process and correct procedure subroutines in dependency order.
        It recursively traverses the call graph, corrects function signatures/usage,
        and applies class-level reference replacements.

        Parameters
        ----------
        subroutine_key : str
            Identifier of the subroutine to process.
        subroutine_to_stack_index : dict
            Mapping from subroutine names to their corresponding module stack index.
        module_stacks : list
            list of AST module stacks corresponding to subroutines.
        cls_info : dict
            Class/instance metadata used for resolving references and corrections.
        main_file_attributes : list
            List of attributes available in the main file for dependency resolution.

        """
        # First, process all sub-subroutines if any
        for child_key in self.extractor.call_within_sub.get(subroutine_key, []):
            self._process_procedures(
                child_key,
                subroutine_to_stack_index,
                module_stacks,
                cls_info,
                main_file_attributes,
            )  # recurse for nested calls

        # Then process the current subroutine
        module_stack_index = subroutine_to_stack_index[subroutine_key]
        self.correct_function(
            module_stacks[module_stack_index],
            cls_info,
            subroutine_key,
            main_file_attributes=main_file_attributes,
        )
        identify_replace_all(
            module_stacks[module_stack_index].body,
            cls_info,
            self.extractor.var_local_names[subroutine_key],
        )

    def run_python_scripts(
        self, base_dir: str, target_dir: str, mode: Literal["CPU", "GPU"] = "CPU"
    ) -> None:
        """
        Validate and execute generated Python scripts with dependency checks.

        Parameters
        ----------
        base_dir : str
            Root directory containing generated modules.
        target_dir : str
            Specific module directory to execute.
        mode : {'CPU', 'GPU'}, optional
            Execution mode (currently informational; defaults to 'CPU').
        """
        if not os.path.isdir(target_dir):
            self.logger.error(f"Target module directory '{target_dir}' not found.")

        subdir_path = os.path.join(base_dir, target_dir)
        subdir = os.path.basename(subdir_path)
        if not os.path.isdir(subdir_path):
            self.logger.warning(f"Skipping non-directory entry: {subdir_path}")
            return

        self.logger.info(f"Processing module: {subdir_path}")
        # Python file checks
        executable_name = os.path.basename(target_dir.rstrip("/"))
        main_file = os.path.join(subdir_path, f"main_{executable_name}.py")
        global_module_file = os.path.join(
            subdir_path, f"global_module_{executable_name}.py"
        )

        missing_files = []
        if not os.path.exists(main_file):
            missing_files.append(f"main_{executable_name}.py")
        if not os.path.exists(global_module_file):
            missing_files.append(f"global_module_{executable_name}.py")

        if missing_files:
            self.logger.warning(
                f"Missing files in '{subdir}': {', '.join(missing_files)}"
            )
            self.logger.info(f"Skipping '{subdir}' due to missing Python files.\n")
            return
        else:
            self.logger.info(f"Required Python files found in '{subdir}'.")

        # Binary file checks
        benchmark_subdir = os.path.join(self.benchmark_dir, subdir)
        dummy_bin = os.path.join(benchmark_subdir, "dummy.bin")
        global_bin = os.path.join(benchmark_subdir, "global.bin")
        output_bin = os.path.join(benchmark_subdir, "output.bin")
        bin_missing = []
        for bin_file in [dummy_bin, global_bin, output_bin]:
            if not os.path.exists(bin_file):
                bin_missing.append(os.path.basename(bin_file))

        if bin_missing:
            self.logger.warning(
                f"Missing binary files for '{subdir}': {', '.join(bin_missing)}"
            )
            self.logger.info(f"Skipping '{subdir}' due to missing binaries.\n")
            return

        self.logger.info(
            f"All binary files found for '{subdir}'. Running unit tests..."
        )
        try:
            result = subprocess.run(
                ["python3", main_file], check=True, capture_output=True, text=True
            )
            self.logger.info(f"Execution output for '{subdir}':\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Error running main_{executable_name}.py for '{subdir}': ", e.stderr
            )
            return
