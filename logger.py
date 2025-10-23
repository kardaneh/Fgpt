import logging, time, json, traceback
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.traceback import install
from rich.syntax import Syntax
from rich.tree import Tree
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

    def log_event(self, event_type:str, message:str, style="bold white", **kwargs):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[dim]{timestamp}[/dim] [bold cyan]{event_type}[/bold cyan] — [{style}]{message}[/]"
        self.console.print(formatted)

        if self.file_output:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"event": event_type, "message": message, **kwargs}) + "\n")

    def info(self, message):
        """Formatted info message"""
        self.console.print(f"[bold cyan][INFO][/bold cyan] {message}")

    def warning(self, message):
        """Formatted warning message"""
        self.console.print(f":warning: [bold yellow][WARNING][/bold yellow] {message}")

    def success(self, message):
        """Custom success level (not default logging level)"""
        self.console.print(f":white_check_mark: [bold green]{message}[/bold green]")

    def step(self, step_name, message):
        """Highlight pipeline step events"""
        self.console.print(f"[bold magenta]▶ Step: {step_name}[/bold magenta] — {message}")

    def exception(self, message, exception=None):
        """Display a formatted exception message with optional traceback."""
        if exception:
            tb = traceback.format_exc()
            exception_type = type(exception).__name__
            exception_msg = str(exception)
            self.console.print(Panel(
                f"[bold red]{message}[/bold red]\n\n"
                f"[red]Exception:[/red] [bold]{exception_type}[/bold]: {exception_msg}\n\n"
                f"[dim]{tb}[/dim]",
                title="[bold red]EXCEPTION[/bold red]",
                border_style="red"
            ))
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
    