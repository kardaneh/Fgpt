import logging, time, json
import traceback
import functools
import inspect
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.traceback import install
from rich.logging import RichHandler


class Logger:
    def __init__(self, console_output=True, file_output=False, log_file="module_log_file.log", pretty_print=True, Module_name="Transformer", record=False):

        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self.pretty_print = pretty_print
        self.module_name = Module_name
        self.record = record

        self.console = Console(record=self.record)
        self.logger = logging.getLogger(f"ModuleLogger.{self.module_name}")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler = RichHandler(console=self.console, show_path=False, show_time=True)
        self.logger.addHandler(handler)

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
            "total_time": None,
            "node_sequence": [],
            "steps_completed": 0,
            "node_count": {}
        }

    def clear_logs(self):
        """Clear the stored Rich logs if record=True."""
        if self.record and hasattr(self, "console") and self.console:
            self.console.clear()

    def show_header(self):
        """Display startup banner."""
        self.console.print(Panel(
            f"[bold red]🚀 Starting Module:[/bold red] [cyan]{self.module_name}[/cyan]",
            title="Fortran → Python Transformer", border_style="bright_blue"
        ))

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

        self.console.print(
            Panel(
                content,
                title="[bold red]TASK STARTED[/bold red]",
                border_style="red",
                expand=False,
                padding=(1, 4),  # (top-bottom, left-right)
            )
        )

    def log_event(self, node_name: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.metrics["node_sequence"].append(node_name)
                self.metrics["node_count"][node_name] = self.metrics["node_count"].get(node_name, 0) + 1
                start_time = time.time()

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                args_info = ", ".join(
                    f"[bold cyan]{name}[/bold cyan]=[green]{value!r}[/green]" 
                    for name, value in bound_args.arguments.items()
                )

                entry_group = Group(
                    f"[bold yellow]Entering node:[/bold yellow] [magenta]{node_name}[/magenta]",
                    f"[bold white]Function:[/bold white] {func.__name__}({args_info})"
                )
                self.console.print(Panel(entry_group, title=f"🚀 Node Start", border_style="bright_blue"))

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    self.log_error(f"Error in node '{node_name}'", e)
                    raise
                else:
                    duration = time.time() - start_time
                    self.metrics.setdefault("node_times", {})
                    self.metrics["node_times"][node_name] = (
                        self.metrics["node_times"].get(node_name, 0) + duration
                    )
                    self.metrics["steps_completed"] += 1

                    exit_group = Group(
                        f"[bold green]Exiting node:[/bold green] [magenta]{node_name}[/magenta]",
                        f"[dim]Duration:[/dim] {duration:.2f}s"
                    )
                    self.console.print(Panel(exit_group, title="✅ Node Complete", border_style="green"))

                    return result

            return wrapper
        return decorator
    
    def log_metrics(self):
        table = Table(title="📊 Pipeline Metrics", show_lines=True)
        table.add_column("Node", style="cyan")
        table.add_column("Count", justify="center")
        table.add_column("Total Time (s)", justify="right")

        for node, count in self.metrics["node_count"].items():
            total_time = self.metrics["node_times"].get(node, 0)
            table.add_row(node, str(count), f"{total_time:.2f}")

        self.console.print(Panel(table, title="Metrics Summary", border_style="bright_blue"))

    def info(self, message):
        """Formatted info message"""
        self.console.print(f"[bold cyan][INFO][/bold cyan] {message}")

    def warning(self, message):
        """Formatted warning message"""
        self.console.print(f"[bold yellow][WARNING][/bold yellow] :warning: {message}")

    def success(self, message):
        """Custom success level (not default logging level)"""
        self.console.print(f":white_check_mark: [bold green]{message}[/bold green]")

    def step(self, step_name, message):
        """Highlight pipeline step events"""
        self.console.print(f"[bold magenta]▶ Step: {step_name}[/bold magenta] — {message}")

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
                code_text = Text.from_markup(f"Code: [italic yellow]{code_line}[/italic yellow]")
                header.append(code_text)

            frame_panel = Panel(
                header,
                title=f"[Frame {i+1}]",
                border_style="bright_blue",
                expand=False,
            )
            panels.append(frame_panel)

        exception_info = Panel(
            Text.from_markup(f"[bold red]{type(exception).__name__}[/bold red]: {exception}"),
            title="[bold red]Exception Raised[/bold red]",
            border_style="red",
        )

        return Panel(
            Group(*panels, exception_info),
            title="[bold red]Traceback[/bold red]",
            border_style="red",
            expand=False
        )

    def exception(self, message, exception=None):
        """Display a formatted exception message with visual stack trace."""
        if exception:
            tb_panels = self._format_traceback_panels(exception)
            main_panel = Panel(
                Group(
                    Text.from_markup(f"[bold red]{message}[/bold red]\n"),
                    tb_panels
                ),
                title="[bold red]EXCEPTION[/bold red]",
                border_style="red"
            )
            self.console.print(main_panel)
        else:
            self.console.print(Panel(f"[bold red]{message}[/bold red]", title="[bold red]EXCEPTION[/bold red]", border_style="red"))

    def log_error(self, message, exception=None):
        """Display a formatted error log, optionally including exception trace."""
        if exception:
            tb = traceback.format_exc()
            exception_type = type(exception).__name__
            exception_msg = str(exception)
            self.console.print(Panel(
                f"[bold red]{message}[/bold red]\n\n"
                f"[red]Error:[/red] [bold]{exception_type}[/bold]: {exception_msg}\n\n"
                f"[dim]{tb}[/dim]",
                title="[bold red]ERROR[/bold red]",
                border_style="red"
            ))
        else:
            self.console.print(Panel(f"[bold red]{message}[/bold red]", title="[bold red]ERROR[/bold red]", border_style="red"))
    