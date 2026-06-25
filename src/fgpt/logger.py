import ast
import functools
import inspect
import logging
import time
import traceback
from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


class Logger:
    """
    Flexible logging utility for FGPT with console, file, and progress tracking support.

    The Logger class provides a unified interface for structured logging, progress
    visualization, and optional persistent log storage. It is designed to support
    both development debugging (rich console output) and reproducible execution
    tracking (file logging).

    It integrates Python's standard logging module with Rich-based console output
    and progress bars to provide enhanced readability during long-running
    transformations such as AST parsing and code rewriting.

    Parameters
    ----------
    console_output : bool, default=True
        Enables rich console output for interactive logging.
    file_output : bool, default=False
        Enables writing logs to a file.
    log_file : str, default="Fgpt_log_file.log"
        Path to the log file when file_output is enabled.
    pretty_print : bool, default=True
        Enables formatted and structured log display in console.
    record : bool, default=False
        Enables Rich console recording for exportable logs.

    Attributes
    ----------
    console : rich.console.Console
        Rich console instance used for formatted output.
    logger : logging.Logger
        Standard Python logger configured for FGPT.
    progress : rich.progress.Progress
        Progress bar manager for long-running tasks.
    metrics : dict
        Internal tracking dictionary for execution statistics, including:
        - start_time: execution start timestamp
        - end_time: execution end timestamp
        - node_sequence: AST traversal order
        - steps_completed: number of completed processing steps
        - node_count: frequency of processed AST nodes

    Notes
    -----
    This logger is designed for pipeline transparency in FGPT, enabling both
    human-readable debugging and reproducible execution logging. It is especially
    useful during AST transformations, where tracking execution order and node
    traversal is critical.
    """

    def __init__(
        self,
        console_output=True,
        file_output=False,
        log_file="Fgpt_log_file.log",
        pretty_print=True,
        record=False,
    ):
        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self.pretty_print = pretty_print
        self.record = record

        self.console = Console(record=self.record)
        self.logger = logging.getLogger("ModuleLogger")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Plain text handler for file output only (no RichHandler for console)
        if self.file_output:
            file_handler = logging.FileHandler(
                self.log_file, mode="w", encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            # Use a simple formatter for file output
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
            transient=True,
        )

        self.metrics = {
            "start_time": None,
            "end_time": None,
            "node_sequence": [],
            "steps_completed": 0,
            "node_count": {},
        }

    def clear_logs(self):
        """Clear the stored Rich logs if record=True."""
        if self.record and hasattr(self, "console") and self.console:
            self.console.clear()

    def show_header(self, module_name):
        """Display startup banner."""
        self.module_name = module_name
        if self.console_output:
            self.console.print(
                Panel(
                    f"[bold red]🚀 Starting Module:[/bold red] [cyan]{self.module_name}[/cyan]",
                    title="Fortran General purpose Transformer (Fgpt)",
                    border_style="bright_blue",
                )
            )
        # Also log to file
        if self.file_output:
            self.logger.info(f"🚀 Starting Module: {self.module_name}")

    def start_task(self, task_name: str, description: str = "", **meta):
        """Display a clearly formatted 'task start' message with good spacing."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Construct sections with spacing between them
        header = Text(f"🚀 {task_name}", style="bold cyan")
        desc = Text(f"📝 {description}", style="yellow") if description else None
        time_text = Text(f"🕒 {timestamp}", style="dim")

        meta_lines = []
        for key, value in meta.items():
            meta_lines.append(f"🔹 [white]{key.upper()}:[/white] {value}")

        components = [header]
        if desc:
            components.append(desc)
        components.append(Text(""))  # blank line
        components.append(time_text)
        components.append(Text(""))  # blank line
        if meta_lines:
            components.extend(Text.from_markup(line) for line in meta_lines)
            components.append(Text(""))

        content = Group(*components)

        if self.console_output:
            self.console.print(
                Panel(
                    content,
                    title="[bold red]TASK STARTED[/bold red]",
                    border_style="red",
                    expand=False,
                    padding=(1, 4),  # (top-bottom, left-right)
                )
            )

        # Log to file
        if self.file_output:
            meta_str = ", ".join([f"{k.upper()}: {v}" for k, v in meta.items()])
            self.logger.info(f"TASK STARTED: {task_name} - {description} - {meta_str}")

    def log_event(self, node_name: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.metrics["node_sequence"].append(node_name)
                self.metrics["node_count"][node_name] = (
                    self.metrics["node_count"].get(node_name, 0) + 1
                )
                start_time = time.time()
                self.metrics["start_time"] = start_time
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                args_info = ", ".join(
                    f"[bold cyan]{name}[/bold cyan]=[green]{value!r}[/green]"
                    for name, value in bound_args.arguments.items()
                )

                # Log to file
                if self.file_output:
                    self.logger.info(
                        f"Entering node: {node_name} - Function: {func.__name__}({args_info})"
                    )

                if func.__name__ == "isolate_procedure":
                    parent_name = bound_args.arguments.get(
                        "parent_procedure", "UnknownParent"
                    )
                    child_name = bound_args.arguments.get(
                        "child_procedure", "UnknownChild"
                    )

                    if self.console_output:
                        entry_group = Group(
                            f"🧬 [bold yellow]Entering node:[/bold yellow] [magenta]{node_name}[/magenta]",
                            f"🔹 Parent → Child: [magenta]{parent_name}[/magenta] → [cyan]{child_name}[/cyan]",
                        )
                        self.console.print(
                            Panel(
                                entry_group,
                                title="🚀 Node Start",
                                border_style="bright_blue",
                            )
                        )

                    if self.file_output:
                        self.logger.info(
                            f"Entering node: {node_name} - Child → Parent: {child_name} → {parent_name}"
                        )

                elif func.__name__ == "_convert_function_body":
                    # extract just the function name from the ast.FunctionDef node
                    node = bound_args.arguments.get("node")
                    fn_name = (
                        node.name
                        if isinstance(node, ast.FunctionDef)
                        else "UnknownFunction"
                    )

                    if self.console_output:
                        entry_group = Group(
                            f"[bold yellow]Entering node:[/bold yellow] [magenta]{node_name}[/magenta]",
                            f"🔁 [bold white]Converting function:[/bold white] [cyan]{fn_name}[/cyan]",
                        )
                        self.console.print(
                            Panel(
                                entry_group,
                                title="🚀 Node Start",
                                border_style="bright_blue",
                            )
                        )

                    if self.file_output:
                        self.logger.info(
                            f"Entering node: {node_name} - Converting function: {fn_name}"
                        )

                else:
                    if self.console_output:
                        entry_group = Group(
                            f"[bold yellow]Entering node:[/bold yellow] [magenta]{node_name}[/magenta]",
                            f"[bold white]Function:[/bold white] {func.__name__}({args_info})",
                        )
                        self.console.print(
                            Panel(
                                entry_group,
                                title="🚀 Node Start",
                                border_style="bright_blue",
                            )
                        )

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    self.error(f"Error in node '{node_name}'", e)
                    raise
                else:
                    end_time = time.time()
                    self.metrics["end_time"] = end_time
                    duration = end_time - start_time
                    self.metrics.setdefault("node_times", {})
                    self.metrics["node_times"][node_name] = (
                        self.metrics["node_times"].get(node_name, 0) + duration
                    )
                    self.metrics["steps_completed"] += 1

                    # Log to file
                    if self.file_output:
                        self.logger.info(
                            f"Exiting node: {node_name} - Duration: {duration:.2f}s"
                        )

                    if func.__name__ == "isolate_procedure":
                        child_name = bound_args.arguments.get(
                            "child_procedure", "UnknownChild"
                        )

                        if self.console_output:
                            exit_group = Group(
                                f"[bold green]Exiting node:[/bold green] [magenta]{node_name}[/magenta]",
                                f"[dim]Duration:[/dim] {duration:.2f}s",
                                f"✅ Done isolating → [cyan]{child_name}[/cyan]",
                            )
                            self.console.print(
                                Panel(
                                    exit_group,
                                    title="✅ Node Complete",
                                    border_style="green",
                                )
                            )
                    elif func.__name__ == "_convert_function_body":
                        node = bound_args.arguments.get("node")
                        fn_name = (
                            node.name
                            if isinstance(node, ast.FunctionDef)
                            else "UnknownFunction"
                        )

                        if self.console_output:
                            exit_group = Group(
                                f"[bold green]Exiting node:[/bold green] [magenta]{node_name}[/magenta]",
                                f"[dim]Duration:[/dim] {duration:.2f}s",
                                f"✅ [cyan]{fn_name}[/cyan] converted to JAX",
                            )
                            self.console.print(
                                Panel(
                                    exit_group,
                                    title="✅ Node Complete",
                                    border_style="green",
                                )
                            )

                    else:
                        if self.console_output:
                            exit_group = Group(
                                f"[bold green]Exiting node:[/bold green] [magenta]{node_name}[/magenta]",
                                f"[dim]Duration:[/dim] {duration:.2f}s",
                            )
                            self.console.print(
                                Panel(
                                    exit_group,
                                    title="✅ Node Complete",
                                    border_style="green",
                                )
                            )

                    return result

            return wrapper

        return decorator

    def log_metrics(self):
        """Log pipeline metrics"""
        if self.console_output:
            table = Table(title="📊 Pipeline Metrics", show_lines=True)
            table.add_column("Node", style="cyan")
            table.add_column("Count", justify="center")
            table.add_column("Total Time (s)", justify="right")

            for node, count in self.metrics["node_count"].items():
                total_time = self.metrics["node_times"].get(node, 0)
                table.add_row(node, str(count), f"{total_time:.2f}")
            table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{sum(self.metrics['node_count'].values())}[/bold]",
                f"[bold]{sum(self.metrics['node_times'].values()):.2f}[/bold]",
            )

            self.console.print(
                Panel(table, title="Metrics Summary", border_style="bright_blue")
            )

        # Log metrics to file
        if self.file_output:
            self.logger.info("Pipeline Metrics Summary:")
            for node, count in self.metrics["node_count"].items():
                total_time = self.metrics["node_times"].get(node, 0)
                self.logger.info(
                    f"  {node}: Count={count}, Total Time={total_time:.2f}s"
                )
            self.logger.info(
                f"  Total: Count={sum(self.metrics['node_count'].values())}, Total Time={sum(self.metrics['node_times'].values()):.2f}s"
            )

    def info(self, message):
        """Formatted info message"""
        if self.console_output:
            self.console.print(f"[bold cyan][INFO][/bold cyan] {message}")
        if self.file_output:
            self.logger.info(message)

    def warning(self, message):
        """Formatted warning message"""
        if self.console_output:
            self.console.print(
                f"[bold yellow][WARNING][/bold yellow] :warning: {message}"
            )
        if self.file_output:
            self.logger.warning(message)

    def success(self, message):
        """Custom success level (not default logging level)"""
        if self.console_output:
            self.console.print(f":white_check_mark: [bold green]{message}[/bold green]")
        if self.file_output:
            self.logger.info(f"SUCCESS: {message}")

    def step(self, step_name, message):
        """Highlight pipeline step events"""
        if self.console_output:
            self.console.print(
                f"[bold magenta]▶ Step: {step_name}[/bold magenta] — {message}"
            )
        if self.file_output:
            self.logger.info(f"Step: {step_name} - {message}")

    def _format_traceback_panels(self, exception: Exception):
        """Format traceback as a series of Rich panels for readability."""
        tb = exception.__traceback__
        extracted_tb = traceback.extract_tb(tb)
        panels = []

        for i, frame in enumerate(extracted_tb):
            file_name = frame.filename
            line_no = frame.lineno
            func_name = frame.name
            code_line = (frame.line or "").strip()

            # Create base Text block (no markup parsing)
            header = Text()
            header.append(f"File: {file_name}\n", style="bold cyan")
            header.append(f"Line: {line_no} | Function: {func_name}\n", style="dim")

            if code_line:
                # Use from_markup for the highlighted code
                code_text = Text.from_markup(
                    f"Code: [italic yellow]{code_line}[/italic yellow]"
                )
                header.append(code_text)

            frame_panel = Panel(
                header,
                title=f"[Frame {i + 1}]",
                border_style="bright_blue",
                expand=False,
            )
            panels.append(frame_panel)

        exception_info = Panel(
            Text.from_markup(
                f"[bold red]{type(exception).__name__}[/bold red]: {exception}"
            ),
            title="[bold red]Exception Raised[/bold red]",
            border_style="red",
        )

        return Panel(
            Group(*panels, exception_info),
            title="[bold red]Traceback[/bold red]",
            border_style="red",
            expand=False,
        )

    def exception(self, message, exception=None):
        """Display a formatted exception message with visual stack trace."""
        if exception:
            if self.file_output:
                self.logger.error(f"{message} - {exception}")
            if self.console_output:
                tb_panels = self._format_traceback_panels(exception)
                main_panel = Panel(
                    Group(
                        Text.from_markup(f"[bold red]{message}[/bold red]\n"), tb_panels
                    ),
                    title="[bold red]EXCEPTION[/bold red]",
                    border_style="red",
                )
                self.console.print(main_panel)
        else:
            if self.file_output:
                self.logger.error(message)
            if self.console_output:
                self.console.print(
                    Panel(
                        f"[bold red]{message}[/bold red]",
                        title="[bold red]EXCEPTION[/bold red]",
                        border_style="red",
                    )
                )

    def error(self, message, exception=None):
        """Display a formatted error log, optionally including exception trace."""
        if exception:
            if self.file_output:
                self.logger.error(f"{message} - {exception}")
            if self.console_output:
                tb = traceback.format_exc()
                self.console.print(
                    Panel(
                        f"[bold red]{message}[/bold red]\n\n"
                        f"[red]Error:[/red] [bold]{type(exception).__name__}[/bold]: {str(exception)}\n\n"
                        f"[dim]{tb}[/dim]",
                        title="[bold red]ERROR[/bold red]",
                        border_style="red",
                    )
                )
        else:
            if self.file_output:
                self.logger.error(message)
            if self.console_output:
                self.console.print(
                    Panel(
                        f"[bold red]{message}[/bold red]",
                        title="[bold red]ERROR[/bold red]",
                        border_style="red",
                    )
                )
