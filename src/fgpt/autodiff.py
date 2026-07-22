# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
import ast
import os
import stat
import subprocess
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Any, Literal

from fgpt.core.backends.jax_converter.converter import JaxConverter
from fgpt.core.backends.utils import (
    CallEdge,
    RemoveLogging,
    contains_name,
    convert_np_jnp,
    get_class_info_from_ast,
    get_name,
    topo_sort,
)
from fgpt.core.common.logger import Logger
from fgpt.core.common.utils import (
    ast_walk,
    get_instance_name,
    load_code_templates,
    python_parser,
)

DEFAULT_TEMPLATE = Path(__file__).parent / "templates" / "default.yaml"


class AutoDiff:
    """
    Transform a NumPy-based Python module into a JAX/Equinox-compatible one.

    The transformation pipeline covers three main concerns:

    - **Loop conversion** — ``for`` loops are rewritten into ``lax.scan``
      or vectorised with ``vmap`` to make them compatible with JAX tracing
      and XLA compilation.
    - **Conditional conversion** — ``if`` / ``else`` branches are rewritten
      into ``lax.cond`` calls so that JAX gradient functions can trace
      through them correctly.
    - **Vectorisation** — array initialisers and stateful variables are
      lifted to include the appropriate batch axes wherever vectorisation
      is applied.

    The output is a sibling file suffixed with ``_d`` (e.g.
    ``model.py`` → ``model_d.py``) based on the ``mode`` that inherits from
    ``eqx.Module`` and is decorated with ``eqx.filter_jit`` on its
    outermost method.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file containing code templates
        (e.g. the JAX timer template).
    vectorize : list[str], optional
        List of lower-bound loops that the user wants to vectorize.
        Defaults to ``["kjpindex"]``.
    benchmark_dir : str, optional
        Directory where per-module runtime measurements are written.
        Defaults to ``<cwd>/benchmark``, created if absent.
    logger : Logger, optional
        Logger instance for structured output and event timing.
        If ``None``, a default :class:`Logger` is created and a
        ``UserWarning`` is emitted.
    mode : Literal[str]
        Transformation mode. Currently ``'jax'`` (default) produces an
        accelerated JAX module. Reserved values ``'fwd'`` and ``'bwd'``
        are intended for forward- and backward-mode differentiation
        variants respectively.

    Attributes
    ----------
    benchmark_dir : str
        Resolved path to the benchmark output directory.
    logger : Logger
        Active logger instance.
    config_path : str
        Path to the YAML config file.
    vectorize : list[str]
        List of lower-bound loops that can be vectorized.
    mode : str
        Active transformation mode. Controls the suffix applied to output
        files: ``'jax'`` → ``_jax``, ``'fwd'`` → ``_fwd``,
        ``'bwd'`` → ``_bwd``.
    _MODE_SUFFIX : dict[str, str]
        Mapping from mode name to output file suffix.
    removelogging : RemoveLogging
        Transformer that strips ``print`` / ``logging`` calls from
        function bodies before JAX conversion.
    _LIB_ALIASES : set of str
        Array-library prefixes recognised as array constructors
        (``'np'``, ``'jnp'``, ``'torch'``, ``'tf'``).
    _SHAPE_CTORS : set of str
        Constructor function names whose first argument is treated as
        a shape tuple (``'zeros'``, ``'ones'``, ``'empty'``, etc.).
    """

    def __init__(
        self,
        config_path: str,
        vectorize: list[str] = ["kjpindex"],
        benchmark_dir: str | None = None,
        logger: Logger | None = None,
        mode: Literal["jax", "fwd", "bwd"] = "jax",
    ) -> None:
        if benchmark_dir is None:  # THe benchmark directory
            current_dir = os.getcwd()
            self.benchmark_dir = os.path.join(current_dir, "benchmark")
            os.makedirs(self.benchmark_dir, exist_ok=True)
        else:
            self.benchmark_dir = benchmark_dir

        if logger is None:
            import warnings

            warnings.warn("No logger provided; using default Logger()", stacklevel=2)
            self.logger = Logger()
        else:
            self.logger = logger

        if config_path is None:
            self.config_path = DEFAULT_TEMPLATE
        else:
            self.config_path = Path(config_path).resolve()
        self.mode = mode
        self.removelogging = RemoveLogging()

        self._LIB_ALIASES: set = {"np", "jnp", "torch", "tf"}
        self._SHAPE_CTORS: set = {
            "zeros",
            "ones",
            "empty",
            "full",
            "rand",
            "randn",
            "zeros_like",
            "ones_like",
            "empty_like",
        }
        self._MODE_SUFFIX: dict[str, str] = {
            "jax": "_jax",
            "fwd": "_d",
            "bwd": "_b",
        }

        self.logger.show_header("AutoDiff")
        self._convert_function_body = self.logger.log_event(
            "Transforming Python to JAX"
        )(self._convert_function_body)
        self.transform = self.logger.log_event("Create Module")(self.transform)
        self.write_to_file = self.logger.log_event("Transfer to Python File")(
            self.write_to_file
        )
        self.run_python_scripts = self.logger.log_event("Run JAX python scripts")(
            self.run_python_scripts
        )
        # Defines the loop bounds of which are vectorizable
        self.vectorize = vectorize

    def _build_output_path(self, path: Path) -> Path:
        """
        Derive the output file path for *path* based on the active mode.

        Parameters
        ----------
        path : Path
            Original source file path.

        Returns
        -------
        Path
            Sibling file with a mode-specific suffix appended to the stem,
            e.g. ``model.py`` → ``model_jax.py`` / ``model_d.py`` /
            ``model_d.py``.

        Raises
        ------
        ValueError
            If :attr:`mode` is not a recognised transformation mode.
        """
        suffix = self._MODE_SUFFIX.get(self.mode)
        if suffix is None:
            raise ValueError(
                f"Unknown mode {self.mode!r}. "
                f"Expected one of {set(self._MODE_SUFFIX)!r}."
            )
        return path.with_name(path.stem + suffix + path.suffix)

    def _add_jax_imports(
        self, class_module: ast.AST, import_remove: list[str] = ["logging"]
    ) -> None:
        """
        Prepend JAX/Equinox imports and strip unwanted import names from the module.

        Inserts ``import equinox as eqx``, ``import jax.numpy as jnp``, and
        ``import jax`` at the top of *class_module*, then removes any
        ``import`` aliases whose name appears in *import_remove*.

        Parameters
        ----------
        class_module : ast.AST
            Parsed AST module to modify in place.
        import_remove : list[str]
            Import names to strip (e.g. ``['logging', 'time']``).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            new_body = []
            for stmt in class_module.body:
                if isinstance(stmt, ast.Import):
                    stmt.names = [n for n in stmt.names if n.name not in import_remove]
                    if stmt.names:
                        new_body.append(stmt)
                else:
                    new_body.append(stmt)
            class_module.body = new_body

            for i, node in enumerate(
                [
                    ast.Import(names=[ast.alias(name="equinox", asname="eqx")]),
                    ast.Import(names=[ast.alias(name="jax.numpy", asname="jnp")]),
                    ast.Import(names=[ast.alias(name="jax")]),
                ]
            ):
                class_module.body.insert(i, node)

        except Exception as e:
            self.logger.exception("Exception in _add_jax_imports", e)
            raise

    def _prepare_class(self, class_module: ast.AST, main_details: dict) -> tuple:
        """
        Transform a class AST into an Equinox-compatible module.

        Applies five sequential transformations:
        1. Add ``eqx.Module`` inheritance and rename classes with ``_eqx`` suffix.
        2. Inject static/dynamic field declarations.
        3. Strip numpy scalar casts from ``__init__``.
        4. Convert ``declaration_initialization`` for JAX file reading.
        5. Transform procedure bodies to JAX (lax.scan / vmap) via JaxConverter.

        Parameters
        ----------
        class_module : ast.AST
            Parsed AST of the class file.
        main_details : dict
            Analysis dict from :meth:`analyze` (method_calls, attributes_used, …).

        Returns
        -------
        tuple
            ``(class_modif, timer_node)`` where class_modif maps instance/method
            names to their transformed counterparts, and *timer_node* is the
            extracted timer ``FunctionDef`` or ``None``.

        Raises
        ------
        ValueError
            If ``fn_shapes`` is empty or ``class_modif`` cannot be built.
        RuntimeError
            Wraps any unexpected exception for upstream handling.
        """
        try:
            class_defs = [
                node
                for node in ast.walk(class_module)
                if isinstance(node, ast.ClassDef)
            ]

            # Extract timer and strip it from the class body
            timer_node = None
            for stmt in class_module.body:
                if not isinstance(stmt, ast.ClassDef):
                    continue
                new_body = []
                for node in stmt.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "timer":
                        timer_node = node
                    else:
                        new_body.append(node)
                stmt.body = new_body

            # Add JAX/Equinox imports
            self._add_jax_imports(class_module, ["logging", "functools", "time"])
            jax_func_import = ast.ImportFrom(
                module="jax",
                names=[ast.alias(name="jit"), ast.alias(name="lax")],
                level=0,
            )
            class_module.body.insert(0, jax_func_import)
            self.cls_info = get_class_info_from_ast(class_module)

            # Transformation 1 — add eqx.Module base and rename classes
            for class_def in class_defs:
                already_has_eqx = any(
                    isinstance(base, ast.Attribute)
                    and isinstance(base.value, ast.Name)
                    and base.value.id == "eqx"
                    and base.attr == "Module"
                    for base in class_def.bases
                )
                if not already_has_eqx:
                    class_def.bases.append(
                        ast.Attribute(
                            value=ast.Name(id="eqx", ctx=ast.Load()),
                            attr="Module",
                            ctx=ast.Load(),
                        )
                    )
                    class_def.name += "_eqx"

            # Collect __init__ and declaration_initialization functions
            init_functions, declaration_initalization = [], []
            for cls_def in class_defs:
                for node in cls_def.body:
                    if isinstance(node, ast.FunctionDef):
                        if node.name == "__init__":
                            init_functions.append(node)
                        elif node.name == "declaration_initialization":
                            declaration_initalization.append(node)

            # Transformation 2 — static/dynamic field declarations
            method_calls = main_details["method_calls"]
            attributes_used = main_details["attributes_used"]
            first_decl = next(iter(declaration_initalization), None)
            declaration_initialization_used = [
                get_name(n.targets[0])
                for node in (first_decl.body if first_decl else [])
                for n in ast_walk(node, ast.Assign)
            ]
            cls_dynamic_static_fields = self._define_dynamic_and_field_data(
                declaration_initialization_used, attributes_used
            )
            for cls_def, fields in zip(class_defs, cls_dynamic_static_fields):
                cls_def.body = fields + cls_def.body

            # Transformation 3 — strip numpy scalar casts from __init__
            init_functions = [convert_np_jnp(fn) for fn in init_functions]
            for fn in init_functions:
                self._remove_numpy_call(fn, dynamic_field=cls_dynamic_static_fields)

            # Transformation 4 — fix binary-file reading in declaration_initialization
            declaration_initalization = [
                convert_np_jnp(fn) for fn in declaration_initalization
            ]
            for fn in declaration_initalization:
                self._add_jax_for_file_reading(fn)

            # Transformation 5 — convert procedure bodies to JAX
            functions = [
                convert_np_jnp(node)
                for node in ast_walk(class_module, ast.FunctionDef)
                if node.name
                not in {"__init__", "declaration_initialization", "timer", "wrapper"}
            ]
            class_dep = self.get_class_dep(functions)
            fn_index = self._index_functions(functions)
            fn_shapes = self._propagate_shapes(class_dep, fn_index, method_calls)
            if not fn_shapes:
                raise ValueError("fn_shapes is empty — check _propagate_shapes")

            jax_converter = JaxConverter(
                cls_info=self.cls_info,
                vectorize=self.vectorize,
                logger=self.logger,
                mode=self.mode,
            )
            self.transform_procedure(class_dep, fn_index, fn_shapes, jax_converter)

            # Decorate the outermost function with eqx.filter_jit
            outermost_fn = list(fn_index.values())[-1]
            outermost_fn.decorator_list = [
                ast.Attribute(
                    value=ast.Name(id="eqx", ctx=ast.Load()),
                    attr="filter_jit",
                    ctx=ast.Load(),
                )
            ]

            # Build class_modif — maps original names to transformed equivalents
            instance_names = [get_instance_name(cls.name) for cls in class_defs]
            class_modif: dict = {}
            fn_map = {
                n.name: n
                for n in ast.walk(class_module)
                if isinstance(n, ast.FunctionDef)
            }
            for method_call in method_calls:
                stmt = fn_map.get(method_call["method"])
                if not stmt:
                    continue
                for instance_name, class_def in zip(instance_names, class_defs):
                    class_modif[method_call["instance"]] = [instance_name, class_def]
                    call_statement = self._create_call_statement(
                        procedures=stmt, instance_name=instance_name
                    )
                    class_modif[stmt.name] = call_statement

            return (class_modif, timer_node)

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            self.logger.exception("Exception in _prepare_class:", e)
            raise RuntimeError("_prepare_class failed") from e

    def _parse_file(self, path: str) -> ast.Module:
        try:
            with open(path) as f:
                return python_parser(f.read())

        except FileNotFoundError:
            self.logger.error(f"Error: The file {path} was not found.")
            raise
        except SyntaxError as e:
            self.logger.error(f"Syntax error in the file {path}: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise

    def _to_module(self, source: str | ast.Module) -> ast.Module:
        """
        Coerce *source* to a parsed ``ast.Module``.

        Parameters
        ----------
        source : Union[str, ast.Module]
            Either a file path string or an already-parsed ``ast.Module``.

        Returns
        -------
        ast.Module
            Parsed ``ast.Module``.

        Raises
        ------
        TypeError
            If *source* is neither a string path nor an ``ast.Module``.
        """
        if isinstance(source, ast.Module):
            return source
        if isinstance(source, str):
            return self._parse_file(source)
        raise TypeError(
            f"Expected a file path or ast.Module, got {type(source).__name__}"
        )

    def _validate_class_module(self, module: ast.Module) -> None:
        """Validate that a module contains a top-level class definition.

        Parameters
        ----------
        module : ast.Module
            Parsed Python module to validate.

        Raises
        ------
        ValueError
            If the module does not contain a top-level
            :class:`ast.ClassDef` node.
        """
        if not any(isinstance(node, ast.ClassDef) for node in module.body):
            raise ValueError("Expected a module containing a class definition.")

    def _validate_main_module(self, module: ast.Module) -> None:
        """Validate that a module contains a ``__main__`` entry point.

        Parameters
        ----------
        module : ast.Module
            Parsed Python module to validate.

        Raises
        ------
        ValueError
            If the module does not contain an
            ``if __name__ == "__main__":`` block.
        """
        has_main = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in module.body
        )
        if not has_main:
            raise ValueError(
                "Expected a module containing `if __name__ == '__main__':`."
            )

    def transform(
        self,
        main_file: str | ast.Module,
        class_file: str | ast.Module,
        routine_dir: str = None,
    ) -> None:
        """
        Orchestrate the full Python-to-JAX/Equinox transformation pipeline.

        Accepts either file paths or pre-parsed ``ast.Module`` objects for both
        inputs. When paths are given, transformed modules are written to sibling
        files suffixed with ``_d`` (e.g. ``model.py`` -> ``model_d.py``) based
        on the ``mode``.
        When ``ast.Module`` objects are given, output paths are derived from
        the class import statement found in the main module.

        Pipeline steps:
        1. Parse / coerce both inputs to ``ast.Module``.
        2. Inject JAX imports into the main module.
        3. Analyse ``main`` to extract instances, method calls, and test calls.
        4. Correct the main function (64-bit enable, input wrapping, arg renaming).
        5. Transform the class module via :meth:`_prepare_class`.
        6. Patch the timer node into the main module.
        7. Correct test functions and apply ``class_modif`` renaming.
        8. Fix import aliases and module reference in the main module.
        9. Write both transformed modules to `_MODE_SUFFIX`-suffixed output files.

        Parameters
        ----------
        main_file : Union[str, ast.Module]
            Path to the main Python file, or a parsed ``ast.Module``.
        class_file : Union[str, ast.Module]
            Path to the class Python file, or a parsed ``ast.Module``.
        routine_dir : str, optional
            Defines the folder inside which we need to save the files in the
            case when `main_file` and `class_file` is sent as ``ast.Module``.

        Raises
        ------
        ValueError
            If ``class_modif`` or ``timer_node`` resolution fails.
        TypeError
            If *source* is neither a string path nor an ``ast.Module``
            in :meth:`_to_module`
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            main_module = self._to_module(main_file)
            class_module = self._to_module(class_file)

            self._validate_class_module(class_module)
            self._validate_main_module(main_module)

            # Resolve output paths from the original string paths when available,
            # otherwise fall back to the import statement inside main_module.
            if isinstance(main_file, str) and isinstance(class_file, str):
                main_path = Path(main_file)
                class_path = Path(class_file)
            else:
                # Derive paths from the from-import that references the class module
                class_def = next(iter(ast_walk(class_module, ast.ClassDef)), None)
                if class_def is None:
                    raise ValueError(
                        "Cannot derive output paths: no ImportFrom found in main_module"
                    )
                if routine_dir is None:
                    raise ValueError(
                        "When the given main and class files as ast.Modules, path in which \
                        these files needs to be saved"
                    )
                executable_name = os.path.basename(routine_dir.rstrip("/"))

                class_path = Path(
                    os.path.join(routine_dir, f"{class_def.name.lower()}.py")
                )
                main_path = Path(
                    os.path.join(routine_dir, f"main_{executable_name}.py")
                )

            main_new_path = self._build_output_path(main_path)
            class_new_path = self._build_output_path(class_path)

            self._add_jax_imports(main_module)

            main_func = next(
                (
                    node
                    for node in ast_walk(main_module, ast.FunctionDef)
                    if node.name == "main"
                ),
                None,
            )
            main_details = self.extract_main_context(main_func)

            instance = main_details["instances"]
            instance_name = next(iter(instance.values()))

            self.correct_main(main_func, main_details)

            class_modif, timer_node = self._prepare_class(
                class_module=class_module, main_details=main_details
            )
            if not class_modif:
                raise ValueError("class_modif is empty — check _prepare_class")

            # Insert timer node and time import after the last import in main
            last_import_pos = [
                i
                for i, node in enumerate(main_module.body)
                if isinstance(node, ast.Import | ast.ImportFrom)
            ][-1]

            timer_node = self._get_timer(timer_node)
            if timer_node:
                for node in [ast.Import(names=[ast.alias(name="time")]), timer_node]:
                    last_import_pos += 1
                    main_module.body.insert(last_import_pos, node)

            # Correct test functions and apply class renaming
            test_calls = main_details["test_calls"]
            for node in ast_walk(main_module, ast.FunctionDef):
                for test_call in test_calls:
                    if node.name == test_call["test"]:
                        self.correct_test_func(node)
                        self.modify_ast(node, class_modif)

            self.modify_ast(main_func, class_modif)

            # Fix the from-import alias and module name to point at the mode class file
            new_body = []
            for stmt in main_module.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module == class_path.stem:
                    stmt.names = [
                        ast.alias(
                            name=f"{n.name}_eqx" if n.name == instance_name else n.name,
                            asname=n.asname,
                        )
                        for n in stmt.names
                    ]
                    stmt.module = class_new_path.stem
                    if stmt.names:
                        new_body.append(stmt)
                else:
                    new_body.append(stmt)
            main_module.body = new_body

            self.write_to_file(main_new_path, main_module)
            self.write_to_file(class_new_path, class_module)

        except (ValueError, TypeError):
            raise
        except Exception as e:
            self.logger.exception("Exception in transform:", e)
            raise

    def write_to_file(self, file_path: str | Path, tree: ast.Module) -> None:
        """
        Unparse *tree* and write it as an executable Python file.

        Prepends a ``#!/usr/bin/env python3`` shebang and sets owner
        read/write/execute permissions (``rwx------``) on the output file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Destination path (string or ``Path``).
        tree : ast.Module
            Fully resolved ``ast.Module`` (should have had
            ``ast.fix_missing_locations`` applied beforehand).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            file_path = Path(file_path)
            self.logger.info(f"Writing Python file: {file_path}")
            with open(file_path, "w") as f:
                f.write("#!/usr/bin/env python3\n")
                f.write(ast.unparse(ast.fix_missing_locations(tree)))
            os.chmod(file_path, stat.S_IRWXU)
            self.logger.info("File successfully written.")
        except Exception as e:
            self.logger.exception("Exception in write_to_file", e)
            raise

    def _get_timer(
        self, timer_node: ast.FunctionDef | None = None
    ) -> ast.FunctionDef | None:
        """
        Build a JAX-compatible timer function from a config template.

        If *timer_node* is provided, the benchmark path is extracted from its
        body (an assignment ``path = '...'``). Otherwise the path defaults to
        ``<benchmark_dir>/{name}/time.txt``.

        Parameters
        ----------
        timer_node: Optional[ast.FunctionDef]
            Optional existing timer ``FunctionDef`` to extract the
            path from.

        Returns
        -------
        Optional[ast.FunctionDef]
            Parsed ``FunctionDef`` for the timer, or ``None`` if the template
            could not be loaded or parsed.

        Raises
        ------
        ValueError
            If the template config cannot be loaded.
        SyntaxError
            If the rendered template contains invalid Python.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            templates = load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded — check config_path")

            timer_template = templates["JAX_templates"]["JAX_timer_template"][
                "template"
            ]

            path = os.path.join(self.benchmark_dir, "{name}", "time.txt")
            if timer_node:
                for node in ast_walk(timer_node, ast.Assign):
                    if (
                        isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "path"
                    ):
                        path = node.value.value
                        break

            rendered = Template(timer_template).substitute(path=repr(path))
            return ast.parse(rendered).body[0]

        except (ValueError, SyntaxError):
            raise
        except Exception:
            self.logger.exception("Exception in _get_timer")
            raise

    def correct_test_func(self, node: ast.FunctionDef) -> None:
        """
        Patch a test function's array reads for JAX compatibility.

        Wraps all ``read_reals`` / ``read_ints`` call results in the
        appropriate ``jnp.float64`` / ``jnp.int32`` cast, and converts
        ``np`` references to ``jnp`` throughout.

        Parameters
        ----------
        node : ast.FunctionDef
            Test ``FunctionDef`` to modify in place.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            node = convert_np_jnp(node)

            class FortranCaster(ast.NodeTransformer):
                def visit_Assign(self, node: ast.Assign) -> ast.Assign:
                    if not isinstance(node.value, ast.Call):
                        return node

                    call = node.value

                    # chained call: something.read_reals(...).reshape(...)
                    if (
                        isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Call)
                        and isinstance(call.func.value.func, ast.Attribute)
                        and call.func.value.func.attr in {"read_reals", "read_ints"}
                    ):
                        node.value = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()),
                                attr="asarray",
                                ctx=ast.Load(),
                            ),
                            args=[call],
                            keywords=[],
                        )
                        return node

                    # subscript of a read call: something.read_ints(...)[i]
                    if isinstance(node.value, ast.Subscript):
                        inner = node.value.value
                        if isinstance(inner, ast.Call) and isinstance(
                            inner.func, ast.Attribute
                        ):
                            attr = inner.func.attr
                            if attr == "read_ints":
                                cast = "int32"
                            elif attr == "read_reals":
                                cast = "float64"
                            else:
                                return node
                            node.value = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="jnp", ctx=ast.Load()),
                                    attr=cast,
                                    ctx=ast.Load(),
                                ),
                                args=[node.value],
                                keywords=[],
                            )
                    return node

            FortranCaster().visit(node)

        except Exception as e:
            self.logger.exception("Exception in correct_test_func:", e)
            raise

    def get_class_dep(
        self, functions: list[ast.FunctionDef]
    ) -> dict[str, list[CallEdge]] | None:
        """
        Build a call-dependency graph over *functions*.

        Each key is a function name; its value is the list of
        :class:`CallEdge` objects representing calls it makes to
        other functions in the same set.

        Parameters
        ----------
        functions : list[ast.FunctionDef]
            All non-special ``FunctionDef`` nodes from the class module.

        Returns
        -------
        Optional[dict[str, list[CallEdge]]]
            Dependency graph ``{caller: [CallEdge, ...]}``, or ``None`` if an
            error occurs.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            fn_index = self._index_functions(functions)
            graph: dict[str, list[CallEdge]] = {fn.name: [] for fn in functions}
            for fn in functions:
                self._analyze_function(fn, fn_index, graph)
            return graph
        except Exception as e:
            self.logger.exception("Exception in get_class_dep:", e)
            raise

    def _analyze_function(
        self,
        function: ast.FunctionDef,
        fn_index: dict[str, ast.FunctionDef],
        graph: dict[str, list[CallEdge]],
    ) -> None:
        """
        Populate one row of the call-dependency graph for *function*.

        Walks *function*'s AST and records every call to another function
        present in *fn_index* as a :class:`CallEdge` in *graph*.
        Self-recursive calls are ignored.

        Parameters
        ----------
        function : ast.FunctionDef
            The function whose body is being analysed.
        fn_index : dict[str, ast.FunctionDef]
            Name → ``FunctionDef`` map for all class functions.
        graph : dict[str, list[CallEdge]]
            Dependency graph mutated in place;
            ``graph[function.name]`` receives the discovered edges.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            caller_name = function.name
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute | ast.Name):
                    continue

                callee_name = get_name(node.func)
                if callee_name not in fn_index or callee_name == caller_name:
                    continue

                callee_fn = fn_index[callee_name]
                func_args = [a.arg for a in callee_fn.args.args if a.arg != "self"]
                graph[caller_name].append(
                    CallEdge(
                        caller=caller_name,
                        callee=callee_name,
                        call_node=deepcopy(node),
                        arg_shapes={},
                        func_args=func_args,
                    )
                )
        except Exception as e:
            self.logger.exception("Exception in _analyze_function:", e)
            raise

    def correct_main(self, node: ast.FunctionDef, main_details: dict) -> None:
        """
        Patch the ``main`` function for JAX compatibility.

        Applies the following modifications in place:

        - Strips numpy scalar casts via :meth:`_remove_numpy_call`.
        - Inserts ``jax.config.update('jax_enable_x64', True)`` as the
          second statement to enable 64-bit precision.
        - Wraps each method/test input array in the appropriate ``jnp``
          constructor (e.g. ``x_d = jnp.asarray(x)``).
        - Inserts those ``_d`` assignments just before the first method call
          that consumes them.
        - Renames all matching call-site arguments from ``x`` to ``x_d``
          based on the ``mode``.

        Parameters
        ----------
        node : ast.FunctionDef
            The ``main`` ``FunctionDef`` to modify in place.
        main_details : dict
            Analysis dict produced by :meth:`analyze`.

        Raises
        ------
        ValueError
            If no insertion point for the ``_d`` assignments
            can be found in the function body.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            self._remove_numpy_call(node)

            jax_enable = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="jax", ctx=ast.Load()),
                            attr="config",
                            ctx=ast.Load(),
                        ),
                        attr="update",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value="jax_enable_x64"),
                        ast.Constant(value=True),
                    ],
                    keywords=[],
                )
            )
            node.body.insert(1, jax_enable)

            # Collect all array inputs consumed by method calls and test calls
            instances = main_details["instances"]
            input_args = {
                arg
                for call in main_details["method_calls"]
                if call["instance"] in instances
                for arg in call["args"]
            }
            input_args |= {
                arg
                for call in main_details["test_calls"]
                if call["args"]
                for arg in call["args"]
                if arg != call["instance"]
            }

            # Map each input arg to its jnp constructor
            input_attr_map: dict[str, str] = {name: "asarray" for name in input_args}
            for stmt in ast.walk(node):
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if not (isinstance(target, ast.Name) and target.id in input_args):
                        continue
                    if (
                        isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Attribute)
                        and isinstance(stmt.value.func.value, ast.Name)
                        and stmt.value.func.value.id == "np"
                    ):
                        attr = stmt.value.func.attr
                        input_attr_map[target.id] = (
                            "asarray" if attr == "zeros" else attr
                        )

            # Build the x_d = jnp.<ctor>(x) assignment nodes
            jd_assignments = [
                ast.Assign(
                    targets=[
                        ast.Name(
                            id=f"{name}{self._MODE_SUFFIX[self.mode]}", ctx=ast.Store()
                        )
                    ],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr=input_attr_map[name],
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=name, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
                for name in input_args
            ]

            # Find the insertion point: first method call that uses an input arg
            insert_pos = None
            last_assignment_pos = None
            first_consumer_pos = None
            preferred_method_pos = None

            for i, item in enumerate(node.body):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id in input_args:
                            last_assignment_pos = i + 1

                call_node = None

                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                    call_node = item.value

                elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
                    call_node = item.value

                if call_node is None:
                    continue

                consumes_input = any(
                    isinstance(arg, ast.Name) and arg.id in input_args
                    for arg in call_node.args
                ) or any(
                    isinstance(kw.value, ast.Name) and kw.value.id in input_args
                    for kw in call_node.keywords
                )

                if consumes_input and first_consumer_pos is None:
                    first_consumer_pos = i + 1

                if (
                    consumes_input
                    and isinstance(call_node.func, ast.Attribute)
                    and isinstance(call_node.func.value, ast.Name)
                ):
                    instance_name = call_node.func.value.id
                    method_name = call_node.func.attr

                    if (
                        any(
                            mc["instance"] == instance_name
                            and mc["method"] == method_name
                            for mc in main_details["method_calls"]
                        )
                        and preferred_method_pos is None
                    ):
                        preferred_method_pos = i

            if preferred_method_pos is not None:
                insert_pos = preferred_method_pos
            elif first_consumer_pos is not None:
                insert_pos = first_consumer_pos
            else:
                insert_pos = last_assignment_pos

            if insert_pos is None:
                raise ValueError(
                    f"Could not find insertion point for {self._MODE_SUFFIX[self.mode]} assignments in main body"
                )

            node.body[insert_pos:insert_pos] = jd_assignments

            self._rename_call_args(
                node,
                input_args=input_args,
                method_calls=main_details["method_calls"],
                test_calls=main_details["test_calls"],
            )

        except ValueError:
            raise
        except Exception as e:
            self.logger.exception("Exception in correct_main:", e)
            raise

    def modify_ast(self, node: ast.AST, class_modif: dict) -> None:
        """
        Rename class instances, method calls, and arguments throughout ``node``.

        Walks ``node`` using a :class:`ast.NodeTransformer` that applies every
        entry in ``class_modif`` — a mapping produced by :meth:`prepare_class`
        that associates original instance/method names with their ``_eqx``
        counterparts and call-statement replacements.

        A ``timer`` wrapper call is also injected alongside each transformed
        method call (excluding ``declaration_initialization``), guarded by a
        visited-set so each method is wrapped at most once.

        Parameters
        ----------
        node : ast.AST
            AST node to modify in place (typically ``main`` or a test
                function ``FunctionDef``).
        class_modif :
            Mapping produced by :meth:`prepare_class`:
            ``{original_name: [instance_name, class_def]}`` for
            instances, ``{method_name: call_ast}`` for methods.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:

            class Modifier(ast.NodeTransformer):
                def __init__(self, class_modif: dict) -> None:
                    self.class_modif = class_modif
                    self.visited: set = set()

                def _make_timed_pair(
                    self, new_node: ast.AST, func_ref: ast.AST, call_args: list
                ) -> list[ast.AST]:
                    """Return [original_call_expr, timer_assign] for a method call."""
                    timed_call = ast.Assign(
                        targets=new_node.targets,
                        value=ast.Call(
                            func=ast.Name(id="timer", ctx=ast.Load()),
                            args=[func_ref] + call_args,
                            keywords=[],
                        ),
                    )
                    return [ast.Expr(value=new_node.value), timed_call]

                def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
                    if not isinstance(node.value, ast.Call):
                        return self.generic_visit(node)

                    call = node.value

                    # Instance instantiation: MyClass(...) → MyClass_eqx(...)
                    if isinstance(call.func, ast.Name):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id in self.class_modif
                            ):
                                modif = self.class_modif[target.id]
                                call.func.id = modif[1].name
                                target.id = modif[0]
                        return self.generic_visit(node)

                    # Method call: instance.method(...) → transformed + timer
                    if isinstance(call.func, ast.Attribute):
                        method = call.func.attr
                        if (
                            method in self.class_modif
                            and method != "declaration_initialization"
                        ):
                            if method not in self.visited:
                                self.visited.add(method)
                                new_node = self.class_modif[method]
                                return self._make_timed_pair(
                                    new_node=new_node,
                                    func_ref=new_node.value.func,
                                    call_args=call.args,
                                )

                    return self.generic_visit(node)

                def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
                    node = self.generic_visit(node)
                    if not (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                    ):
                        return node

                    method = node.value.func.attr
                    if method not in self.class_modif:
                        return node

                    new_node = deepcopy(self.class_modif[method])
                    if method == "declaration_initialization":
                        return ast.copy_location(new_node, node)

                    if method not in self.visited:
                        self.visited.add(method)
                        return self._make_timed_pair(
                            new_node=new_node,
                            func_ref=node.value.func,
                            call_args=node.value.args,
                        )

                    return node

                def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
                    if (
                        isinstance(node.value, ast.Name)
                        and node.value.id in self.class_modif
                    ):
                        node.value.id = self.class_modif[node.value.id][0]
                    return self.generic_visit(node)

                def visit_Name(self, node: ast.Name) -> ast.Name:
                    if node.id in self.class_modif:
                        node.id = self.class_modif[node.id][0]
                    return node

                def visit_arg(self, node: ast.arg) -> ast.arg:
                    if node.arg in self.class_modif:
                        node.arg = self.class_modif[node.arg][0]
                    return node

            Modifier(class_modif).visit(node)
        except Exception as e:
            self.logger.exception("Exception in modify_ast:", e)
            raise

    def _rename_call_args(
        self, node: ast.AST, input_args: set, method_calls: dict, test_calls: dict
    ) -> None:
        """
        Renames all the arguments within the calls by ensuring
        that they are not special functions.

        Parameters
        ----------
        node : ast.AST
            Call node or Assign node containing the call node
        input_args : set
            Input args sent to the functions
        methods_calls : dict
            Dict containing the methods calls inside the function
        test_calls : dict
            Dict containing the test calls

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Call):
                if self._is_tracked_call(node, method_calls, test_calls):
                    self._rename_args(node.args, input_args)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if self._is_tracked_call(node.value, method_calls, test_calls):
                        self._rename_args([target], input_args)

            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self._rename_call_args(
                                item, input_args, method_calls, test_calls
                            )
                elif isinstance(value, ast.AST):
                    self._rename_call_args(value, input_args, method_calls, test_calls)
        except Exception as e:
            self.logger.exception("Exception in _rename_call_args:", e)
            raise

    def _rename_args(self, args: list, input_args: set) -> None:
        """
        Helper function in charge of transforming
        ``arg`` → ``arg_d`` format

        Parameters
        ----------
        args : list
            List containing the args to modify
        input_args : set
            set of input_args to verify if they are the args
            to modify

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            for arg in args:
                for node in ast.walk(arg):  # walk all nested expressions
                    if isinstance(node, ast.Name) and node.id in input_args:
                        node.id = f"{node.id}{self._MODE_SUFFIX[self.mode]}"
        except Exception as e:
            self.logger.exception("Exception in _rename_args:", e)
            raise

    def _is_tracked_call(
        self, node: ast.AST, method_calls: dict, test_calls: dict
    ) -> bool:
        """
        Checks if the method in question belongs to the called functions
        with the main function

        Parameters
        ----------
        node : ast.AST
            Call node or Assign node containing the call node
        input_args : set
            Input args sent to the functions
        methods_calls : dict
            Dict containing the methods calls inside the function

        Returns
        -------
        bool
            ``True`` if the node in question is present in ``method_calls``
            else ``False``

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node.func, ast.Attribute):
                instance_name = (
                    node.func.value.id
                    if isinstance(node.func.value, ast.Name)
                    else None
                )
                method_name = node.func.attr
                for call in method_calls:
                    if (
                        call.get("instance") == instance_name
                        and call.get("method") == method_name
                    ):
                        return True

            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                for call in test_calls:
                    if call.get("test") == func_name:
                        return True

            return False
        except Exception as e:
            self.logger.exception("Exception in _is_tracked_call:", e)
            raise

    def extract_main_context(self, tree: ast.FunctionDef) -> dict:
        """
        Extract JAX-transformation metadata from the ``main`` function.

        Performs a single-pass walk over *tree* to collect:

        - ``instances`` — mapping of local variable name → class name for
          every class instantiation found (e.g. ``{'model': 'MyModel'}``).
        - ``method_calls`` — list of dicts describing each method called on
          a tracked instance, including argument names and inferred shapes.
        - ``test_calls`` — list of dicts describing each ``test_*`` function
          call, including which instance (if any) is passed.
        - ``attributes_used`` — set of attribute names referenced as array
          dimensions in ``np.zeros`` / ``np.ones`` calls.

        Parameters
        ----------
        tree : ast.FunctionDef
            The ``main`` ``FunctionDef`` node to analyse.

        Returns
        -------
        dict
            Dict with keys ``instances``, ``method_calls``, ``test_calls``,
            and ``attributes_used``.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            instance_to_class: dict[str, str] = {}
            method_calls: list[dict] = []
            test_calls: list[dict] = []
            attributes_used: set = set()

            # Class instantiations and assigned method-call results
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                ):
                    continue
                call = node.value
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if isinstance(call.func, ast.Name):
                        instance_to_class[target.id] = call.func.id
                    elif isinstance(call.func, ast.Attribute):
                        obj = call.func.value
                        if isinstance(obj, ast.Name) and obj.id in instance_to_class:
                            method_calls.append(
                                {
                                    "instance": obj.id,
                                    "class": instance_to_class[obj.id],
                                    "method": call.func.attr,
                                    "args": self._extract_args(call),
                                    "args_shape": self._get_args_shape(
                                        tree, self._extract_args(call)
                                    ),
                                }
                            )

            # Standalone (non-assigned) method calls on tracked instances
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                ):
                    continue
                obj = node.func.value
                if not (isinstance(obj, ast.Name) and obj.id in instance_to_class):
                    continue
                already_recorded = any(
                    mc["instance"] == obj.id and mc["method"] == node.func.attr
                    for mc in method_calls
                )
                if not already_recorded:
                    method_calls.append(
                        {
                            "instance": obj.id,
                            "class": instance_to_class[obj.id],
                            "method": node.func.attr,
                            "args": self._extract_args(node),
                            "args_shape": self._get_args_shape(
                                tree, self._extract_args(node)
                            ),
                        }
                    )

            # test_* function calls
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id.startswith("test_")
                    and node.func.id != "read_dummy"
                ):
                    continue
                tracked = [
                    arg.id
                    for arg in node.args
                    if isinstance(arg, ast.Name) and arg.id in instance_to_class
                ]
                test_calls.append(
                    {
                        "test": node.func.id,
                        "instance": tracked[0] if tracked else None,
                        "args": [
                            arg.id if isinstance(arg, ast.Name) else None
                            for arg in node.args
                        ],
                    }
                )

            # Attribute names used as array dimensions in np.zeros / np.ones
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"
                    and node.func.attr in {"zeros", "ones"}
                    and node.args
                    and isinstance(node.args[0], ast.Tuple)
                ):
                    continue
                for elt in node.args[0].elts:
                    for attr_node in ast.walk(elt):
                        if isinstance(attr_node, ast.Attribute):
                            attributes_used.add(get_name(attr_node))

            return {
                "instances": instance_to_class,
                "method_calls": method_calls,
                "test_calls": test_calls,
                "attributes_used": attributes_used,
            }
        except Exception as e:
            self.logger.exception("Exception in extract_main_content: ", e)
            raise

    # TODO: Need to handle complex arg shape
    def _get_args_shape(self, tree: ast.AST, args: list[str]) -> dict[str, list]:
        """
        Infer the shape of each named argument from array-constructor calls.

        Walks *tree* looking for assignments of the form
        ``x = np.zeros((dim0, dim1))`` where the target name is in *args*,
        and extracts the dimension elements as a list of symbolic names or
        integer constants.

        Only single-argument constructors whose first argument is an
        ``ast.Tuple`` are handled; scalar, keyword-only, and multi-argument
        constructors are skipped.

        Parameters
        ----------
        tree : ast.AST
            AST to walk — typically the ``main`` ``FunctionDef``.
        args : list[str]
            Variable names whose shapes should be resolved.

        Returns
        -------
        dict[str, list]
            Mapping ``{arg_name: [dim0, dim1, ...]}``.  Names absent from
            any constructor call are omitted from the result.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            args_shape: dict[str, list] = {}
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                ):
                    continue
                call = node.value
                if not (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in self._LIB_ALIASES
                ):
                    continue
                for target in node.targets:
                    if not (isinstance(target, ast.Name) and target.id in args):
                        continue
                    for arg in call.args:
                        if isinstance(arg, ast.Tuple):
                            args_shape[target.id] = [get_name(elt) for elt in arg.elts]
            return args_shape
        except Exception as e:
            self.logger.exception("Exception in _get_args_shape:", e)
            raise

    def _extract_args(self, call_node: ast.Call) -> list[str]:
        """
        Extract argument names from a call node.

        Handles three argument forms: plain names (``ast.Name``),
        attribute access (``ast.Attribute``), and arbitrary expressions
        which are serialised via ``ast.dump`` as a fallback.

        Parameters
        ----------
        call_node : ast.Call
            The call node whose positional arguments are extracted.

        Returns
        -------
        list[str]
            Argument names or ``ast.dump`` representations for complex
            expressions.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            args = []
            for arg in call_node.args:
                if isinstance(arg, ast.Name):
                    args.append(arg.id)
                elif isinstance(arg, ast.Attribute):
                    args.append(arg.attr)
                else:
                    args.append(ast.dump(arg))
            return args
        except Exception as e:
            self.logger.exception("Exception in _extract_args:", e)
            raise

    def _index_functions(
        self, functions: list[ast.FunctionDef]
    ) -> dict[str, ast.FunctionDef]:
        """
        Build a name-to-node index over a list of function definitions.

        Parameters
        ----------
        functions : list[ast.FunctionDef]
            Function nodes to index, typically all non-special methods
            extracted from the class module.

        Returns
        -------
        dict[str, ast.FunctionDef]
            Mapping ``{function_name: FunctionDef}``.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            return {fn.name: fn for fn in functions}
        except Exception as e:
            self.logger.exception("Exception in _index_functions:", e)
            raise

    def _propagate_shapes(
        self,
        graph: dict[str, list[CallEdge]],
        fn_index: dict[str, ast.FunctionDef],
        method_calls: list[dict],
    ) -> dict[str, dict]:
        """
        Propagate array shapes from entry-point methods down to all callees.

        Seeds the worklist from *method_calls* (whose shapes come from
        :meth:`_get_args_shape`), then iteratively resolves child argument
        shapes via :meth:`_infer_child_args` until all reachable functions
        have been visited.

        Parameters
        ----------
        graph : dict[str, list[CallEdge]]
            Call-dependency graph produced by :meth:`_get_class_dep`.
        fn_index : dict[str, ast.FunctionDef]
            Name → node index produced by :meth:`_index_functions`.
        method_calls : list[dict]
            Method-call records from :meth:`extract_main_context`, each
            containing at least ``method`` and ``args_shape``.

        Returns
        -------
        dict[str, dict]
            Mapping ``{function_name: {arg_name: shape}}`` for every
            reachable function.

        Raises
        ------
        ValueError
            If a callee referenced in *graph* is absent from *fn_index*.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            fn_shapes: dict[str, dict] = {}
            worklist: list[CallEdge] = []

            # Seed entry points from method calls
            for call in method_calls:
                method = call.get("method")
                if method and method in fn_index:
                    fn_shapes[method] = call["args_shape"]
                    worklist.append(
                        CallEdge(
                            caller=None,
                            callee=method,
                            call_node=None,
                            arg_shapes=call["args_shape"],
                            func_args=[],
                        )
                    )

            while worklist:
                ctx = worklist.pop()
                if ctx.callee not in fn_index:
                    raise ValueError(
                        f"Callee {ctx.callee!r} not found in fn_index — check _get_class_dep"
                    )
                fn = fn_index[ctx.callee]

                for edge in graph.get(ctx.callee, []):
                    callee_fn = fn_index[edge.callee]
                    child_shapes = self._infer_child_args(
                        edge.call_node, ctx.arg_shapes, fn, callee_fn
                    )
                    if (
                        edge.callee not in fn_shapes
                        or fn_shapes[edge.callee] != child_shapes
                    ):
                        fn_shapes[edge.callee] = child_shapes

                    edge.arg_shapes = child_shapes
                    worklist.append(
                        CallEdge(
                            caller=ctx.callee,
                            callee=edge.callee,
                            call_node=deepcopy(edge.call_node),
                            arg_shapes=child_shapes,
                            func_args=[a.arg for a in callee_fn.args.args],
                        )
                    )

            return fn_shapes

        except ValueError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _propagate_shapes:", e)
            raise

    def _infer_child_args(
        self,
        call_node: ast.Call,
        arg_shapes: dict,
        fn: ast.FunctionDef,
        callee_fn: ast.FunctionDef,
    ) -> dict:
        """
        Infer argument shapes for a callee by resolving each call-site argument.

        Resolution is attempted in the following order for each positional arg:

        1. Direct parent-parameter pass-through (arg name is in *arg_shapes*).
        2. Local variable assigned in *fn* body (array-producing assignments).
        3. Inline literal / constructor shape (e.g. ``np.zeros((3, 4))``).
        4. Attribute or subscript access (best-effort from *arg_shapes*).

        Parameters
        ----------
        call_node : ast.Call
            The call site node inside the caller's body.
        arg_shapes : dict
            Known shapes for the caller's own parameters.
        fn : ast.FunctionDef
            The caller function (used for local-variable resolution).
        callee_fn : ast.FunctionDef
            The callee function (provides parameter names).

        Returns
        -------
        dict
            Mapping ``{callee_param_name: shape}`` for every positional
            argument that could be resolved.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            resolved: dict[str, Any] = {}
            callee_args = [a.arg for a in callee_fn.args.args if a.arg != "self"]

            for i, call_arg in enumerate(call_node.args):
                if i >= len(callee_args):
                    break
                param = callee_args[i]
                arg_name = get_name(call_arg)

                # 1. Direct pass-through
                if arg_name and arg_name in arg_shapes:
                    resolved[param] = arg_shapes[arg_name]
                    continue

                # 2. Local variable in caller body
                if arg_name:
                    local_shape = self._resolve_local_array(arg_name, fn, arg_shapes)
                    if local_shape is not None:
                        resolved[param] = local_shape
                        continue

                # 3. Inline constructor (e.g. np.zeros((2, 3)))
                inline_shape = self._resolve_inline_shape(call_arg)
                if inline_shape is not None:
                    resolved[param] = inline_shape
                    continue

                # 4. Attribute / subscript (self.x, batch[0])
                attr_shape = self._resolve_attr_or_subscript(call_arg, arg_shapes)
                if attr_shape is not None:
                    resolved[param] = attr_shape

            return resolved

        except Exception as e:
            self.logger.exception("Exception in _infer_child_args:", e)
            raise

    def _resolve_local_array(
        self,
        var_name: str,
        fn: ast.FunctionDef,
        arg_shapes: dict,
    ) -> Any | None:
        """
        Resolve the shape of a local variable from the caller's function body.

        Walks *fn* and returns the shape inferred from the *last* assignment
        to *var_name* whose right-hand side is one of:

        - a subscript / slice of a known array,
        - an array constructor (``np.zeros``, ``torch.ones``, …),
        - an alias of another known variable.

        Parameters
        ----------
        var_name : str
            Name of the local variable to resolve.
        fn : ast.FunctionDef
            Caller function whose body is searched.
        arg_shapes : dict
            Known shapes for the caller's parameters.

        Returns
        -------
        Optional[Any]
            Inferred shape, or ``None`` if *var_name* is not a resolvable
            local array.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            shape = None
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign | ast.AnnAssign):
                    continue

                targets = (
                    [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                )
                value = node.value
                if value is None:
                    continue

                for target in targets:
                    if not (isinstance(target, ast.Name) and target.id == var_name):
                        continue

                    # Subscript / slice of a known array
                    if isinstance(value, ast.Subscript):
                        parent_name = get_name(value.value)
                        if parent_name and parent_name in arg_shapes:
                            sliced = self._infer_subscript_shape(
                                arg_shapes[parent_name], value.slice
                            )
                            if sliced is not None:
                                shape = sliced
                                continue

                    # Array constructor
                    inline = self._resolve_inline_shape(value)
                    if inline is not None:
                        shape = inline
                        continue

                    # Alias of a known variable
                    rhs_name = get_name(value)
                    if rhs_name and rhs_name in arg_shapes:
                        shape = arg_shapes[rhs_name]

            return shape  # None → not a local array we can resolve

        except Exception as e:
            self.logger.exception("Exception in _resolve_local_array:", e)
            raise

    def _resolve_inline_shape(self, node: ast.AST) -> list | None:
        """
        Detect an array-constructor call and return its shape.

        Recognises common constructors from any library alias
        (``np``, ``jnp``, ``torch``, ``tf``) and bare names.

        Parameters
        ----------
        node : ast.AST
            AST node to inspect — typically the right-hand side of an
            assignment or a call-site argument.

        Returns
        -------
        Optional[list]
            Shape as a list of symbolic names or integers, or ``None`` if
            *node* is not a recognised array constructor call.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not isinstance(node, ast.Call):
                return None

            func = node.func
            is_ctor = (
                isinstance(func, ast.Attribute) and func.attr in self._SHAPE_CTORS
            ) or (isinstance(func, ast.Name) and func.id in self._SHAPE_CTORS)
            if not is_ctor or not node.args:
                return None

            return self._extract_shape_arg(node.args[0])

        except Exception as e:
            self.logger.exception("Exception in _resolve_inline_shape:", e)
            raise

    def _extract_shape_arg(self, node: ast.AST) -> list | None:
        """
        Convert an AST shape argument into a list of symbolic names or ints.

        Parameters
        ----------
        node : ast.AST
            The first argument of an array constructor call, e.g.
            ``(self.kjpindex, nnobio)`` or ``128``.

        Returns
        -------
        Optional[list]
            Examples::

                np.zeros((self.kjpindex, nnobio))  →  ['kjpindex', 'nnobio']
                np.zeros((self.kjpindex, 3))       →  ['kjpindex', 3]
                np.zeros(self.kjpindex)            →  ['kjpindex']
                np.zeros(128)                      →  [128]

            Returns ``None`` if no dimension can be resolved.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Tuple | ast.List):
                return [self._extract_single_dim(elt) for elt in node.elts]

            single = self._extract_single_dim(node)
            return [single] if single is not None else None

        except Exception as e:
            self.logger.exception("Exception in _extract_shape_arg:", e)
            raise

    def _extract_single_dim(self, node: ast.AST) -> Any | None:
        """
        Resolve one dimension element to a symbolic name or integer.

        Parameters
        ----------
        node : ast.AST
            A single dimension element from a shape tuple.

        Returns
        -------
        Optional[Any]
            Resolution results::

                self.kjpindex  →  'kjpindex'   (ast.Attribute)
                nnobio         →  'nnobio'     (ast.Name)
                3              →  3            (ast.Constant int)
                <other>        →  None

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Attribute):
                return node.attr
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            return None

        except Exception as e:
            self.logger.exception("Exception in _extract_single_dim:", e)
            raise

    def _resolve_attr_or_subscript(
        self,
        node: ast.AST,
        arg_shapes: dict,
    ) -> Any | None:
        """
        Best-effort shape resolution for attribute access and subscripts.

        Parameters
        ----------
        node : ast.AST
            Call-site argument node — typically ``self.weights`` or ``x[0]``.
        arg_shapes : dict
            Known shapes for the caller's parameters.

        Returns
        -------
        Optional[Any]
            Resolved shape, or ``None`` if resolution is not possible.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if isinstance(node, ast.Attribute) and node.attr in arg_shapes:
                return arg_shapes[node.attr]

            if isinstance(node, ast.Subscript):
                parent_name = get_name(node.value)
                if parent_name and parent_name in arg_shapes:
                    return self._infer_subscript_shape(
                        arg_shapes[parent_name], node.slice
                    )

            return None

        except Exception as e:
            self.logger.exception("Exception in _resolve_attr_or_subscript:", e)
            raise

    def _infer_subscript_shape(
        self,
        parent_shape: Any,
        slice_node: ast.AST,
    ) -> list | None:
        """
        Infer the shape produced by indexing into *parent_shape*.

        Parameters
        ----------
        parent_shape : Any
            Shape of the array being subscripted — must be a non-empty list
            to proceed.
        slice_node : ast.AST
            The slice node from the subscript expression.

        Returns
        -------
        Optional[list]
            - Integer index → leading dimension dropped.
            - Bare slice    → shape unchanged.
            - Otherwise     → ``None``.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if not (isinstance(parent_shape, list) and parent_shape):
                return None

            if isinstance(slice_node, ast.Constant) and isinstance(
                slice_node.value, int
            ):
                return parent_shape[1:] if len(parent_shape) > 1 else []

            if isinstance(slice_node, ast.Slice):
                return parent_shape

            return None

        except Exception as e:
            self.logger.exception("Exception in _infer_subscript_shape:", e)
            raise

    def transform_procedure(
        self,
        class_dep: dict,
        fn_index: dict,
        fn_shapes: dict,
        converter: JaxConverter,
    ) -> None:
        """
        Apply JAX conversion to all class methods in dependency order.

        Iterates over functions from leaves to root (children before parents)
        using a topological sort of *class_dep*, configures *converter* for
        each function, strips logging calls, then delegates body conversion
        to :meth:`_convert_function_body`.

        Parameters
        ----------
        class_dep : dict
            Call-dependency graph ``{caller: [CallEdge, ...]}``.
        fn_index : dict
            Name → ``FunctionDef`` map produced by :meth:`_index_functions`.
        fn_shapes : dict
            Argument shapes per function from :meth:`_propagate_shapes`.
        converter : JaxConverter
            Configured converter instance shared across all functions.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            for fn_name in topo_sort(class_dep):
                node = fn_index[fn_name]
                self.logger.info(f"Transforming procedure: {fn_name}")

                converter.set_working_function(
                    func_name=fn_name,
                    func_input_dim=fn_shapes.get(fn_name),
                    call_edge=class_dep,
                )
                cleaned = self.removelogging.visit(node)
                ast.copy_location(cleaned, node)
                self._convert_function_body(node, converter)
                converter.reset_all()

        except Exception as e:
            self.logger.exception("Exception in transform_procedure:", e)
            raise

    def _convert_function_body(
        self, node: ast.FunctionDef, converter: JaxConverter
    ) -> None:
        """
        Convert a single function's body to JAX-compatible statements.

        Applies the following steps in order:

        1. Analyses statefulness via ``converter.analyze_function_statefulness``.
        2. Visits and rewrites the AST via ``converter.visit``.
        3. Lifts dynamic variables that require a vectorisation axis into
        correctly-shaped ``jnp.zeros`` / ``jnp.full`` initialisers.
        4. Processes helper functions via ``converter.process_helpers``.
        5. Appends a ``return`` statement when the body lacks one.

        Parameters
        ----------
        node : ast.FunctionDef
            Function node to rewrite in place.
        converter : JaxConverter
            Stateful converter holding shape and variable-modification info.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            converter.analyze_function_statefulness(node)
            new_tree = converter.visit(node)
            ast.fix_missing_locations(new_tree)

            # Snapshot and clear variable-modification sets before lifting
            var_modif_args = converter._var_modif["args"]
            var_modif_attrs = converter._var_modif["attr"]
            converter._var_modif["attr"] = set()
            converter._var_modif["args"] = set()

            # Lift dynamic variables that need a vectorisation axis
            if getattr(converter, "dynamic_variable_lift", None):
                for key, meta in converter.dynamic_variable_lift.items():
                    batched_axes = meta["batched_axis"]
                    vector_loop = meta["vectorized_loop"]

                    for n in ast_walk(new_tree, ast.Assign):
                        if not contains_name(n, key):
                            continue

                        # Case 1 — jnp.zeros(...) initialiser: insert vectorisation axes
                        is_zeros_call = (
                            isinstance(n.value, ast.Call)
                            and isinstance(n.value.func, ast.Attribute)
                            and n.value.func.attr == "zeros"
                        )
                        if is_zeros_call:
                            shape_node = n.value.args[0]
                            shape_elts = (
                                list(shape_node.elts)
                                if isinstance(shape_node, ast.Tuple)
                                else [shape_node]
                            )
                            for axis in sorted(batched_axes):
                                shape_elts.insert(axis, vector_loop)
                            n.value.args[0] = (
                                ast.Tuple(elts=shape_elts, ctx=ast.Load())
                                if len(shape_elts) > 1
                                else shape_elts[0]
                            )
                            continue

                        # Case 2 — stateful scalar/subscript: replace with jnp.full / jnp.zeros
                        if not getattr(converter, "var_state", None):
                            continue
                        var_states = converter.var_state.get(key, ())
                        if not var_states:
                            continue
                        status, state_node = var_states
                        if n is not state_node or status != "stateful":
                            continue

                        dtype = meta["dtype"]
                        if isinstance(n.value, ast.Constant | ast.BinOp):
                            n.value = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="jnp", ctx=ast.Load()),
                                    attr="full",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Tuple(elts=vector_loop, ctx=ast.Load()),
                                    n.value,
                                ],
                                keywords=[
                                    ast.keyword(
                                        arg="dtype",
                                        value=ast.Attribute(
                                            value=ast.Name(id="jnp", ctx=ast.Load()),
                                            attr=dtype,
                                            ctx=ast.Load(),
                                        ),
                                    )
                                ],
                            )
                        elif isinstance(n.value, ast.Subscript):
                            n.value = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="jnp", ctx=ast.Load()),
                                    attr="full",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Tuple(elts=vector_loop, ctx=ast.Load()),
                                    n.value,
                                ],
                                keywords=[],
                            )
                        else:
                            n.value = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="jnp", ctx=ast.Load()),
                                    attr="zeros",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Tuple(elts=vector_loop, ctx=ast.Load())],
                                keywords=[
                                    ast.keyword(
                                        arg="dtype",
                                        value=ast.Attribute(
                                            value=ast.Name(id="jnp", ctx=ast.Load()),
                                            attr=dtype,
                                            ctx=ast.Load(),
                                        ),
                                    )
                                ],
                            )

            converter.process_helpers()

            # Merge variable-modification sets after helper processing
            var_modif_attr = var_modif_attrs | converter._var_modif["attr"]
            var_modif_args = var_modif_args | converter._var_modif["args"]

            if not isinstance(new_tree.body[-1], ast.Return):
                ret_stmt = converter.add_return_stmt(
                    var_modif_args=var_modif_args,
                    var_modif_attr=var_modif_attr,
                )
                if ret_stmt:
                    new_tree.body.append(ret_stmt)

        except Exception as e:
            self.logger.exception("Exception in _convert_function_body:", e)
            raise

    def _create_call_statement(
        self,
        procedures: ast.FunctionDef,
        instance_name: str,
    ) -> ast.Assign:
        """
        Build an ``ast.Assign`` that calls *procedures* on *instance_name*.

        Inspects the last ``return`` statement of *procedures* to determine
        what the call unpacks into, producing one of three forms based on
        the given ``mode``:

        - **Tuple return** ``(arg, ..., self)`` →
        ``(arg_d, ..., instance) = instance.method(arg_d, ...)``
        - **Call return** →
        ``instance = instance.method(arg_d, ...)``
        - **Name return** →
        ``result_d = instance.method(arg_d, ...)``

        Parameters
        ----------
        procedures : ast.FunctionDef
            Transformed method whose return shape drives the assignment form.
        instance_name : str
            Name of the ``_eqx`` instance variable in the caller scope.

        Returns
        -------
        ast.Assign
            Ready-to-insert assignment node.

        Raises
        ------
        NotImplementedError
            If the return type is not one of the three handled forms.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            return_stmt = [node for node in ast_walk(procedures, ast.Return)][-1]
            ret = return_stmt.value

            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=instance_name, ctx=ast.Load()),
                    attr=procedures.name,
                    ctx=ast.Load(),
                ),
                args=[
                    ast.arg(arg=f"{a.arg}{self._MODE_SUFFIX[self.mode]}")
                    for a in procedures.args.args
                    if a.arg != "self"
                ],
                keywords=[],
            )

            if isinstance(ret, ast.Tuple):
                targets = [
                    ast.Name(
                        id=f"{n.id}{self._MODE_SUFFIX[self.mode]}", ctx=ast.Store()
                    )
                    for n in ret.elts
                    if isinstance(n, ast.Name)
                ]
                return ast.Assign(
                    targets=[
                        ast.Tuple(
                            elts=targets + [ast.Name(id=instance_name, ctx=ast.Load())],
                            ctx=ast.Store(),
                        )
                    ],
                    value=call,
                )

            if isinstance(ret, ast.Call):
                return ast.Assign(
                    targets=[ast.Name(id=instance_name, ctx=ast.Store())],
                    value=call,
                )

            if isinstance(ret, ast.Name):
                return ast.Assign(
                    targets=[
                        ast.Name(
                            id=f"{ret.id}{self._MODE_SUFFIX[self.mode]}",
                            ctx=ast.Store(),
                        )
                    ],
                    value=call,
                )

            raise NotImplementedError(
                f"Unhandled return type in _create_call_statement: {type(ret).__name__}"
            )

        except NotImplementedError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _create_call_statement:", e)
            raise

    def _check_dimension_static(self, attributes: dict, attr_name: str) -> bool:
        """
        Determine whether a dimension attribute can be treated as a JAX static field.

        An attribute is considered dynamic (not static) when it carries a
        ``dep_value`` key that itself resolves to another known attribute,
        meaning its value cannot be determined at trace time.

        Parameters
        ----------
        attributes : dict
            Attribute metadata dict from :attr:`cls_info`.
        attr_name : str
            Name of the attribute to check.

        Returns
        -------
        bool
            ``True`` if the attribute is static, ``False`` if dynamic.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            attr_info = attributes.get(attr_name)
            if not attr_info:
                return True
            dependency = attr_info.get("dep_value")
            return not (dependency and dependency in attributes)
        except Exception as e:
            self.logger.exception("Exception in _check_dimension_static:", e)
            raise

    def _define_dynamic_and_field_data(
        self,
        declaration_initialization_used: list[str],
        attributes_used: set,
    ) -> list[list[ast.AnnAssign]]:
        """
        Build ``eqx.field(static=True)`` and plain type-annotation nodes for
        each class attribute.

        Separates attributes into:

        - **Static fields** — scalar ``int`` / ``float`` / ``bool`` attributes
        whose dimensions are fully known at trace time, annotated with
        ``eqx.field(static=True)``.
        - **Dynamic fields** — ``jnp.ndarray`` and remaining scalars,
        annotated with their type only.

        Parameters
        ----------
        declaration_initialization_used : list[str]
            Attribute names initialised inside ``declaration_initialization``;
            these cannot be static.
        attributes_used : set
            Attribute names referenced as array dimensions in the main module
            (always added as static ``int`` fields if not already present).

        Returns
        -------
        list[list[ast.AnnAssign]]
            One inner list of ``AnnAssign`` nodes per class in
            :attr:`cls_info`, ordered static fields first.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            cls_dynamic_fields = []

            for _, cls_attributes in self.cls_info.items():
                attributes = cls_attributes["attributes"]
                local_method_arrays = cls_attributes["methods"]

                declaration_initialization_used = [
                    n for n in declaration_initialization_used if n in attributes
                ]

                # Collect dimension names that are statically known
                arr_dim: set = set()
                for attr_info in attributes.values():
                    for dim in attr_info.get("dimensions", []):
                        if isinstance(dim, str) and self._check_dimension_static(
                            attributes, dim
                        ):
                            arr_dim.add(dim)

                if local_method_arrays:
                    for method_values in local_method_arrays.values():
                        for attr in method_values.get("local_arr", {}).values():
                            arr_dim.update(attr.get("dimensions", []))

                # Attributes initialised in declaration_initialization are dynamic
                arr_dim -= set(declaration_initialization_used)

                static_attributes, dynamic_attributes = [], []
                for key, value in attributes.items():
                    if value["type"] in {"int", "float", "bool"} and key in arr_dim:
                        static_attributes.append((key, value["type"]))
                    elif value["type"] in {"jnp.ndarray", "int", "float", "bool"}:
                        dynamic_attributes.append((key, value["type"]))

                # Ensure all dimension-referenced attributes are present as static int
                static_keys = {v[0] for v in static_attributes}
                for arg_member in attributes_used:
                    if arg_member not in static_keys:
                        static_attributes.append((arg_member, "int"))

                dynamic_static_fields: list[ast.AnnAssign] = []

                for name, type_str in static_attributes:
                    dynamic_static_fields.append(
                        ast.AnnAssign(
                            target=ast.Name(id=name, ctx=ast.Store()),
                            annotation=ast.Name(id=type_str, ctx=ast.Load()),
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="eqx", ctx=ast.Load()),
                                    attr="field",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg="static", value=ast.Constant(value=True)
                                    )
                                ],
                            ),
                            simple=1,
                        )
                    )

                for name, type_str in dynamic_attributes:
                    annotation = (
                        ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="ndarray",
                            ctx=ast.Load(),
                        )
                        if type_str == "jnp.ndarray"
                        else ast.Name(id=type_str, ctx=ast.Load())
                    )
                    dynamic_static_fields.append(
                        ast.AnnAssign(
                            target=ast.Name(id=name, ctx=ast.Store()),
                            annotation=annotation,
                            simple=1,
                        )
                    )

                cls_dynamic_fields.append(dynamic_static_fields)

            return cls_dynamic_fields

        except Exception as e:
            self.logger.exception("Exception in _define_dynamic_and_field_data:", e)
            raise

    def _remove_numpy_call(
        self,
        init_function: ast.AST,
        dynamic_field: list[list[ast.AnnAssign]] | None = None,
    ) -> None:
        """
        Strip numpy scalar casts from ``__init__`` and fill missing attributes.

        Two operations are performed:

        1. For every ``self.x = np.int32(v)`` / ``np.float64(v)`` assignment
        whose target is a known static (``eqx.field``) attribute, the cast
        is unwrapped to ``self.x = v``.
        2. Attributes declared in *dynamic_field* but absent from
        ``__init__`` are inserted as ``self.x = None`` placeholders so
        that Equinox's ``__init__`` contract is satisfied.

        Parameters
        ----------
        init_function : ast.AST
            The ``__init__`` ``FunctionDef`` (or ``main`` function) to patch
            in place.
        dynamic_field : Optional[list[list[ast.AnnAssign]]]
            Field declarations produced by :meth:`_define_dynamic_and_field_data`.
            When ``None``, only numpy-cast stripping is performed.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        _NUMPY_CASTS = {"int32", "float64", "bool_", "bool"}

        try:
            dynamic_names: set = set()
            temp_names: set = set()

            if dynamic_field:
                for dynamic_elements in dynamic_field:
                    for ann in dynamic_elements:
                        if not (
                            isinstance(ann, ast.AnnAssign)
                            and isinstance(ann.target, ast.Name)
                        ):
                            continue
                        is_static_field = (
                            isinstance(ann.value, ast.Call)
                            and isinstance(ann.value.func, ast.Attribute)
                            and isinstance(ann.value.func.value, ast.Name)
                            and ann.value.func.value.id == "eqx"
                            and ann.value.func.attr == "field"
                        )
                        if is_static_field:
                            dynamic_names.add(ann.target.id)
                        else:
                            temp_names.add(ann.target.id)

            target_names: set = set()
            for node in init_function.body:
                if not (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                ):
                    continue
                target_name = node.targets[0].attr
                target_names.add(target_name)

                if target_name not in dynamic_names:
                    continue
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr in _NUMPY_CASTS
                    and node.value.args
                ):
                    node.value = node.value.args[0]

            # Insert self.x = None for declared-but-missing attributes
            missing = temp_names - target_names
            if missing:
                insert_pos = [
                    i
                    for i, n in enumerate(init_function.body)
                    if isinstance(n, ast.Assign)
                ][-1]
                for attr in missing:
                    init_function.body.insert(
                        insert_pos,
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr=attr,
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Constant(value=None),
                        ),
                    )
                    insert_pos += 1

        except Exception as e:
            self.logger.exception("Exception in _remove_numpy_call:", e)
            raise

    def _find_parent_body(
        self,
        root: ast.AST,
        target: ast.AST,
    ) -> list[ast.AST] | None:
        """
        Find the list-valued body that directly contains *target*.

        Parameters
        ----------
        root : ast.AST
            Root node to search from.
        target : ast.AST
            Node whose containing list is sought.

        Returns
        -------
        Optional[list[ast.AST]]
            The ``body`` or ``orelse`` list containing *target*, or ``None``
            if not found.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            for node in ast.walk(root):
                for _, value in ast.iter_fields(node):
                    # Look for list-valued fields: .body, .orelse
                    if isinstance(value, list) and target in value:
                        return value
            return None
        except Exception as e:
            self.logger.exception("Exception in _find_parent_body:", e)
            raise

    def _get_read_call(
        self,
        node: ast.AST,
    ) -> tuple[ast.AST | None, str | None]:
        """
        Detect a ``read_ints`` or ``read_reals`` call in *node*.

        Handles both a direct call and a subscript of a call
        (e.g. ``f.read_ints(...)[0]``).

        Parameters
        ----------
        node : ast.AST
            AST node to inspect — typically the right-hand side of an
            assignment.

        Returns
        -------
        tuple[Optional[ast.AST], Optional[str]]
            ``(call_node, dtype)`` where *dtype* is ``'int32'`` or
            ``'float64'``, or ``(None, None)`` if no read call is found.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        _READ_FUNCS = {"read_ints": "int32", "read_reals": "float64"}

        try:
            # Direct call: f.read_ints(...)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                dtype = _READ_FUNCS.get(node.func.attr)
                if dtype:
                    return node, dtype

            # Subscript of call: f.read_ints(...)[0]
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
            ):
                dtype = _READ_FUNCS.get(node.value.func.attr)
                if dtype:
                    return node, dtype

            return None, None

        except Exception as e:
            self.logger.exception("Exception in _get_read_call:", e)
            raise

    def _check_read_rhs_assign_value(self, node: ast.AST) -> bool:
        """
        Check whether *node* contains a ``read_ints`` or ``read_reals`` call.

        Recurses into ``ast.Call`` argument lists to detect nested read calls.

        Parameters
        ----------
        node : ast.AST
            Right-hand side of an assignment to inspect.

        Returns
        -------
        bool
            ``True`` if a read call is found anywhere in *node*.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            call, _ = self._get_read_call(node)
            if call:
                return True
            if isinstance(node, ast.Call):
                return any(self._check_read_rhs_assign_value(arg) for arg in node.args)
            return False
        except Exception as e:
            self.logger.exception("Exception in _check_read_rhs_assign_value:", e)
            raise

    def _correct_name(
        self,
        node: ast.AST,
        value: list[str],
    ) -> ast.AST:
        """
        Replace ``self.<attr>`` with a bare ``<attr>`` name node.

        Only rewrites ``ast.Attribute`` nodes of the form ``self.x`` where
        ``x`` is in *value*. Recurses into ``ast.BinOp`` left and right
        operands.

        Parameters
        ----------
        node : ast.AST
            Node to rewrite — returned as-is when no substitution applies.
        value : list[str]
            Attribute names eligible for substitution.

        Returns
        -------
        ast.AST
            Rewritten node (may be the same object if unchanged).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in value
            ):
                return ast.Name(id=node.attr, ctx=node.ctx)

            if isinstance(node, ast.BinOp):
                node.left = self._correct_name(node.left, value)
                node.right = self._correct_name(node.right, value)

            return node

        except Exception as e:
            self.logger.exception("Exception in _correct_name:", e)
            raise

    def _correct_tuple_shape(
        self,
        tuple_shape: ast.Tuple,
        value: list[str],
    ) -> ast.Tuple:
        """
        Apply :meth:`_correct_name` to every element of a shape tuple.

        Parameters
        ----------
        tuple_shape : ast.Tuple
            Shape tuple from an array constructor call.
        value : list[str]
            Attribute names to replace with bare name nodes.

        Returns
        -------
        ast.Tuple
            The same tuple with elements rewritten in place.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            tuple_shape.elts = [
                self._correct_name(elt, value) for elt in tuple_shape.elts
            ]
            return tuple_shape
        except Exception as e:
            self.logger.exception("Exception in _correct_tuple_shape:", e)
            raise

    def _identify_transform_scalar(
        self,
        decl_init_function: ast.FunctionDef,
    ) -> tuple[ast.Dict | None, bool]:
        """
        Identify and transform scalar / array reads in ``declaration_initialization``.

        Walks *decl_init_function* and applies two rewrites:

        - **Array initialisers** (``self.x = np.zeros(...)`` with no read
        call on the RHS): the target is renamed from ``self.x`` to ``x``
        and the shape tuple is corrected via :meth:`_correct_tuple_shape`.
        - **Read assignments** (``self.x = f.read_reals(...)``): the target
        is renamed to ``x`` and the RHS is wrapped in ``jnp.float64(...)``
        or ``jnp.int32(...)`` as appropriate.

        Parameters
        ----------
        decl_init_function : ast.FunctionDef
            The ``declaration_initialization`` method to rewrite in place.

        Returns
        -------
        tuple[Optional[ast.Dict], bool]
            ``(updated_dict, inside_method)`` where *updated_dict* maps
            transformed names to their local variable nodes (used later to
            build the ``eqx.tree_at`` return), and *inside_method* is
            ``True`` when at least one array initialiser was found.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            keys: list[ast.Constant] = []
            values: list[ast.Name] = []
            inside_method = False

            for assign in ast_walk(decl_init_function, ast.Assign):
                has_read = self._check_read_rhs_assign_value(assign.value)

                if not has_read:
                    # Array initialiser without a read call
                    if not (
                        isinstance(assign.value, ast.Call)
                        and isinstance(assign.value.func, ast.Attribute)
                        and assign.value.func.attr == "zeros"
                    ):
                        continue

                    target = assign.targets[0]
                    if not isinstance(target, ast.Attribute):
                        continue

                    name = target.attr
                    assign.targets[0] = ast.Name(id=name, ctx=target.ctx)

                    # Replace self.<x> references in the shape tuple
                    known_names = [n.id for n in values if isinstance(n, ast.Name)]
                    for i, arg in enumerate(assign.value.args):
                        if isinstance(arg, ast.Tuple):
                            assign.value.args[i] = self._correct_tuple_shape(
                                arg, known_names
                            )

                    inside_method = True
                    keys.append(ast.Constant(value=name))
                    values.append(ast.Name(id=name, ctx=ast.Load()))

                else:
                    # Read assignment: wrap RHS in jnp cast
                    target = assign.targets[0]
                    if not isinstance(target, ast.Attribute):
                        continue

                    name = target.attr
                    assign.targets[0] = ast.Name(id=name, ctx=target.ctx)
                    keys.append(ast.Constant(value=name))
                    values.append(ast.Name(id=name, ctx=ast.Load()))

                    call_node, dtype = self._get_read_call(assign.value)
                    if call_node:
                        assign.value = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="jnp", ctx=ast.Load()),
                                attr=dtype,
                                ctx=ast.Load(),
                            ),
                            args=[assign.value],
                            keywords=[],
                        )

            if keys:
                return ast.Dict(keys=keys, values=values), inside_method

            return None, inside_method

        except Exception as e:
            self.logger.exception("Exception in _identify_transform_scalar:", e)
            raise

    def _add_jax_for_file_reading(self, decl_init_function: ast.FunctionDef) -> None:
        """
        Rewrite ``declaration_initialization`` for JAX/Equinox compatibility.

        Applies the following transformations in order:

        1. Inserts an ``updated = {}`` dict (or a pre-populated one from
        :meth:`_identify_transform_scalar`) before the first ``for`` loop.
        2. Replaces ``setattr(self, attr_name, value)`` calls inside the
        loop body with ``jax_arr = jnp.array(value)`` followed by
        ``updated[attr_name] = jax_arr``.
        3. Appends an ``eqx.tree_at(...)`` return statement that applies
        all accumulated updates to ``self`` in one immutable update.

        Parameters
        ----------
        decl_init_function : ast.FunctionDef
            The ``declaration_initialization`` method to rewrite in place.

        Raises
        ------
        ValueError
            If a ``setattr`` call is expected inside the loop but not found.
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            updated, inside_method = self._identify_transform_scalar(
                decl_init_function=decl_init_function
            )
            updated_dict = ast.Assign(
                targets=[ast.Name(id="updated", ctx=ast.Store())],
                value=ast.Dict(keys=[], values=[]) if not updated else updated,
            )

            for_elements = [
                (i, elem)
                for i, elem in enumerate(decl_init_function.body)
                if isinstance(elem, ast.For)
            ]

            # No for loop — only scalar/boolean reads; insert dict at the end
            if not for_elements:
                decl_init_function.body.append(updated_dict)
            else:
                for_pos, for_loop = for_elements[-1]
                decl_init_function.body.insert(for_pos, updated_dict)
                for_loop_var = for_loop.target.id

                # Replace setattr(self, attr_name, value) with dict-update pair
                call_setattr = next(
                    (
                        node
                        for node in ast_walk(decl_init_function, ast.Expr)
                        if (
                            isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "setattr"
                        )
                    ),
                    None,
                )
                if call_setattr is None:
                    raise ValueError(
                        "setattr call not found in declaration_initialization"
                    )

                # Wrap jnp.prod arguments in jnp.array to avoid tracing issues
                class ProdArrayTransformer(ast.NodeTransformer):
                    def visit_Call(self, node: ast.Call) -> ast.Call:
                        if (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "prod"
                            and node.args
                        ):
                            node.args[0] = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="jnp", ctx=ast.Load()),
                                    attr="array",
                                    ctx=ast.Load(),
                                ),
                                args=[node.args[0]],
                                keywords=[],
                            )
                        return self.generic_visit(node)

                for if_node in ast_walk(decl_init_function, ast.If):
                    if (
                        len(if_node.body) == 1
                        and isinstance(if_node.body[0], ast.Continue)
                        and not if_node.orelse
                    ):
                        ProdArrayTransformer().visit(if_node)
                        ast.fix_missing_locations(if_node)

                np_to_jax = ast.Assign(
                    targets=[ast.Name(id="jax_arr", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="array",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id="value", ctx=ast.Load())],
                        keywords=[],
                    ),
                )
                dict_update = ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id="updated", ctx=ast.Load()),
                            slice=ast.Name(id=for_loop_var, ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Name(id="jax_arr", ctx=ast.Load()),
                )

                parent_body = self._find_parent_body(decl_init_function, call_setattr)
                if parent_body is None:
                    raise ValueError(
                        "Could not locate parent body for setattr replacement"
                    )

                idx = parent_body.index(call_setattr)
                parent_body[idx : idx + 1] = [np_to_jax, dict_update]

            # Append eqx.tree_at(...) return to apply all updates immutably
            return_stmt = ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="eqx", ctx=ast.Load()),
                        attr="tree_at",
                        ctx=ast.Load(),
                    ),
                    args=[
                        # lambda m: tuple(getattr(m, k) for k in updated.keys())
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[ast.arg(arg="m")],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                            ),
                            body=ast.Call(
                                func=ast.Name(id="tuple", ctx=ast.Load()),
                                args=[
                                    ast.GeneratorExp(
                                        elt=ast.Call(
                                            func=ast.Name(id="getattr", ctx=ast.Load()),
                                            args=[
                                                ast.Name(id="m", ctx=ast.Load()),
                                                ast.Name(id="k", ctx=ast.Load()),
                                            ],
                                            keywords=[],
                                        ),
                                        generators=[
                                            ast.comprehension(
                                                target=ast.Name(
                                                    id="k", ctx=ast.Store()
                                                ),
                                                iter=ast.Call(
                                                    func=ast.Attribute(
                                                        value=ast.Name(
                                                            id="updated", ctx=ast.Load()
                                                        ),
                                                        attr="keys",
                                                        ctx=ast.Load(),
                                                    ),
                                                    args=[],
                                                    keywords=[],
                                                ),
                                                ifs=[],
                                                is_async=0,
                                            )
                                        ],
                                    )
                                ],
                                keywords=[],
                            ),
                        ),
                        ast.Name(id="self", ctx=ast.Load()),
                        # tuple(updated.values())
                        ast.Call(
                            func=ast.Name(id="tuple", ctx=ast.Load()),
                            args=[
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="updated", ctx=ast.Load()),
                                        attr="values",
                                        ctx=ast.Load(),
                                    ),
                                    args=[],
                                    keywords=[],
                                )
                            ],
                            keywords=[],
                        ),
                    ],
                    keywords=[
                        ast.keyword(
                            arg="is_leaf",
                            value=ast.Lambda(
                                args=ast.arguments(
                                    posonlyargs=[],
                                    args=[ast.arg(arg="x")],
                                    kwonlyargs=[],
                                    kw_defaults=[],
                                    defaults=[],
                                ),
                                body=ast.Compare(
                                    left=ast.Name(id="x", ctx=ast.Load()),
                                    ops=[ast.Is()],
                                    comparators=[ast.Constant(value=None)],
                                ),
                            ),
                        )
                    ],
                )
            )

            self._transform_declaration_init(decl_init_function)
            decl_init_function.body.append(return_stmt)

        except ValueError:
            raise
        except Exception as e:
            self.logger.exception("Exception in _add_jax_for_file_reading:", e)
            raise

    def _transform_declaration_init(self, decl_init_function: ast.FunctionDef) -> None:
        """
        Patch ``for`` loops in ``declaration_initialization`` to handle
        attributes not yet present in the Equinox pytree.

        For each loop body that follows the pattern::

            attribute = getattr(self, attr_name)
            if isinstance(attribute, jnp.ndarray):
                arr_shape = attribute.shape
                <remaining stmts>

        the ``if`` branch is extended with an ``orelse`` that falls back to
        ``updated.get(attr_name)`` and re-derives ``arr_shape`` from there,
        ensuring that attributes initialised inside the loop (and therefore
        absent from ``self`` during tracing) are handled correctly.

        Parameters
        ----------
        decl_init_function : ast.FunctionDef
            The ``declaration_initialization`` method to rewrite in place.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:

            class AttributeLoopTransformer(ast.NodeTransformer):
                def visit_For(self, node: ast.For) -> ast.For:
                    self.generic_visit(node)
                    new_body: list[ast.AST] = []
                    i = 0

                    while i < len(node.body):
                        stmt = node.body[i]

                        # Match: attribute = getattr(self, attr_name)
                        # followed by: if isinstance(attribute, jnp.ndarray): ...
                        is_getattr_assign = (
                            isinstance(stmt, ast.Assign)
                            and len(stmt.targets) == 1
                            and isinstance(stmt.targets[0], ast.Name)
                            and stmt.targets[0].id == "attribute"
                        )
                        has_isinstance_next = (
                            i + 1 < len(node.body)
                            and isinstance(node.body[i + 1], ast.If)
                            and isinstance(node.body[i + 1].test, ast.Call)
                            and isinstance(node.body[i + 1].test.func, ast.Name)
                            and node.body[i + 1].test.func.id == "isinstance"
                        )

                        if not (is_getattr_assign and has_isinstance_next):
                            new_body.append(stmt)
                            i += 1
                            continue

                        if_stmt = node.body[i + 1]
                        arr_shape_assign = if_stmt.body[0]
                        remaining_stmts = if_stmt.body[1:]

                        new_if = ast.If(
                            test=if_stmt.test,
                            body=[arr_shape_assign],
                            orelse=[
                                # attribute = updated.get(attr_name)
                                ast.Assign(
                                    targets=[ast.Name(id="attribute", ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Attribute(
                                            value=ast.Name(
                                                id="updated", ctx=ast.Load()
                                            ),
                                            attr="get",
                                            ctx=ast.Load(),
                                        ),
                                        args=[ast.Name(id="attr_name", ctx=ast.Load())],
                                        keywords=[],
                                    ),
                                ),
                                # arr_shape = attribute.shape
                                ast.Assign(
                                    targets=[ast.Name(id="arr_shape", ctx=ast.Store())],
                                    value=ast.Attribute(
                                        value=ast.Name(id="attribute", ctx=ast.Load()),
                                        attr="shape",
                                        ctx=ast.Load(),
                                    ),
                                ),
                            ],
                        )

                        new_body.append(stmt)
                        new_body.append(new_if)
                        new_body.extend(remaining_stmts)
                        i += 2

                    node.body = new_body
                    return node

            transformer = AttributeLoopTransformer()
            transformer.visit(decl_init_function)
            ast.fix_missing_locations(decl_init_function)

        except Exception as e:
            self.logger.exception("Exception in _transform_declaration_init:", e)
            raise

    def run_python_scripts(
        self, base_dir: str, target_dir: str, mode: Literal["CPU", "GPU"] = "CPU"
    ) -> None:
        """
        Validate and execute generated JAX Python scripts with dependency checks.

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
        main_file = os.path.join(subdir_path, f"main_{executable_name}_{self.mode}.py")
        global_module_file = os.path.join(
            subdir_path, f"global_module_{executable_name}_{self.mode}.py"
        )

        missing_files = []
        if not os.path.exists(main_file):
            missing_files.append(f"main_{executable_name}_{self.mode}.py")
        if not os.path.exists(global_module_file):
            missing_files.append(f"global_module_{executable_name}_{self.mode}.py")

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
                f"Error running main_{executable_name}_{self.mode}.py for '{subdir}': ",
                e.stderr,
            )
            return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform a NumPy-based Python module into a JAX/Equinox-compatible one."
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config file containing code templates (e.g. template.yaml).",
    )

    parser.add_argument(
        "--class_file",
        type=str,
        required=True,
        help="Path to the global module Python file (e.g. global_module_hydrol_soil.py).",
    )

    parser.add_argument(
        "--main_file",
        type=str,
        required=True,
        help="Path to the main driver Python file (e.g. main_hydrol_soil.py).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["jax", "fwd", "bwd"],
        default="jax",
        help="Transformation mode: 'jax' (default), 'fwd' (forward-mode AD), 'bwd' (reverse-mode AD).",
    )

    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Directory for benchmark outputs. Defaults to <cwd>/benchmark if not set.",
    )

    parser.add_argument(
        "--vectorize",
        nargs="+",
        metavar="LOOP_BOUND",
        default=["kjpindex"],
        help=(
            "Loop upper-bound variables to vectorize. "
            "Example: --vectorize kjpindex nvm npts"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    autodiff = AutoDiff(
        config_path=args.config_path,
        benchmark_dir=args.benchmark_dir,
        mode=args.mode,
        vectorize=args.vectorize,
    )
    autodiff.transform(
        class_file=args.class_file,
        main_file=args.main_file,
    )


if __name__ == "__main__":
    main()
