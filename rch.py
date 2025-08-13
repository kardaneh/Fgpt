import logging
import os
from collections.abc import Callable
from collections.abc import Iterable
from datetime import datetime
from logging import Handler
from logging import LogRecord
from types import ModuleType
from typing import Any
from typing import ClassVar

from rich.console import Console as RichConsole
from rich.console import ConsoleRenderable
from rich.console import JustifyMethod
from rich.console import NewLine
from rich.highlighter import Highlighter
from rich.highlighter import ReprHighlighter
from rich.panel import Panel
from rich.scope import render_scope
from rich.segment import Segment
from rich.style import Style
from rich.styled import Styled
from rich.text import Text
from rich.traceback import Traceback

FormatTimeCallable = Callable[[datetime], Text]

"""An example for monkey patching Rich to allow fluid output in terminal and CI and also keep pretty highlighting.

File links are removed since those break in Gitlab.

Gitlab issue that will never get fixed: https://gitlab.com/gitlab-org/gitlab-runner/-/issues/29365
"""

class Console(RichConsole):
    """A modified version of Rich Console to remove file links in output for Gitlab.

    From https://github.com/Textualize/rich/blob/master/rich/console.py#L593

    """

    def log(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        style: str | Style | None = None,
        justify: JustifyMethod | None = None,
        emoji: bool | None = None,
        markup: bool | None = None,
        highlight: bool | None = None,
        log_locals: bool = False,
        _stack_offset: int = 1,
    ) -> None:
        """Log rich content to the terminal.

        Args:
            objects: Objects to log to the terminal.
            sep: String to write between print data. Defaults to " ".
            end: String to write at end of print data. Defaults to "\\\\n".
            style: A style to apply to output. Defaults to None.
            justify: One of "left", "right", "center", or "full". Defaults to ``None``.
            emoji: Enable emoji code, or ``None`` to use console default. Defaults to None.
            markup: Enable markup, or ``None`` to use console default. Defaults to None.
            highlight: Enable automatic highlighting, or ``None`` to use console default. Defaults to None.
            log_locals: Boolean to enable logging of locals where ``log()``
                was called. Defaults to False.
            _stack_offset: Offset of caller from end of call stack. Defaults to 1.
        """
        if not objects:
            objects = (NewLine(),)

        render_hooks = self._render_hooks[:]

        with self:
            renderables: list[ConsoleRenderable] = self._collect_renderables(
                objects,
                sep,
                end,
                justify=justify,
                emoji=emoji,
                markup=markup,
                highlight=highlight,
            )
            if style is not None:
                renderables = [Styled(renderable, style) for renderable in renderables]

            filename, line_no, locals = self._caller_frame_info(_stack_offset)
            # link_path = None if filename.startswith("<") else os.path.abspath(filename)
            path = filename.rpartition(os.sep)[-1]
            if log_locals:
                locals_map = {key: value for key, value in locals.items() if not key.startswith("__")}
                renderables.append(render_scope(locals_map, title="[i]locals"))

            renderables = [
                self._log_render(
                    self,
                    renderables,
                    log_time=self.get_datetime(),
                    path=path,
                    line_no=line_no,
                    # all because of this
                    # link_path=link_path,
                )
            ]
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            new_segments: list[Segment] = []
            extend = new_segments.extend
            render = self.render
            render_options = self.options
            for renderable in renderables:
                extend(render(renderable, render_options))
            buffer_extend = self._buffer.extend
            for line in Segment.split_and_crop_lines(new_segments, self.width, pad=False):
                buffer_extend(line)


class FluidRichHandler(Handler):
    LEVEL_MAPPING = {
        logging.DEBUG: "[blue]DBUG[/blue]",
        logging.INFO: "[green]INFO[/green]",
        logging.WARNING: "[yellow]WARN[/yellow]",
        logging.ERROR: "[red]ERRR[/red]",
        logging.CRITICAL: "[bold red]CRIT[/bold red]",
    }

    HIGHLIGHTER_CLASS: ClassVar[type[Highlighter]] = ReprHighlighter

    KEYWORDS: ClassVar[list[str] | None] = [
        "GET",
        "POST",
        "HEAD",
        "PUT",
        "DELETE",
        "OPTIONS",
        "TRACE",
        "PATCH",
    ]

    def __init__(
        self,
        console: Console,
        level: int | str = logging.NOTSET,
        *,
        enable_link_path: bool = True,
        highlighter: Highlighter | None = None,
        markup: bool = False,
        rich_tracebacks: bool = False,
        tracebacks_width: int | None = None,
        tracebacks_code_width: int = 88,
        tracebacks_extra_lines: int = 3,
        tracebacks_theme: str | None = None,
        tracebacks_word_wrap: bool = True,
        tracebacks_show_locals: bool = False,
        tracebacks_suppress: Iterable[str | ModuleType] = (),
        tracebacks_max_frames: int = 100,
        locals_max_length: int = 10,
        locals_max_string: int = 80,
        columns: int = -1,
        keywords: list[str] | None = None,
    ) -> None:
        super().__init__(level=level)
        self.console = console
        self.highlighter = highlighter or self.HIGHLIGHTER_CLASS()
        self.markup = markup
        self.enable_link_path = enable_link_path
        self.markup = markup
        self.rich_tracebacks = rich_tracebacks
        self.tracebacks_width = tracebacks_width
        self.tracebacks_extra_lines = tracebacks_extra_lines
        self.tracebacks_theme = tracebacks_theme
        self.tracebacks_word_wrap = tracebacks_word_wrap
        self.tracebacks_show_locals = tracebacks_show_locals
        self.tracebacks_suppress = tracebacks_suppress
        self.tracebacks_max_frames = tracebacks_max_frames
        self.tracebacks_code_width = tracebacks_code_width
        self.locals_max_length = locals_max_length
        self.locals_max_string = locals_max_string
        self.keywords = keywords
        if columns > 0:
            self.columns(columns)

    def columns(self, size: int) -> None:
        self.columns_size = size
        # rich will default to 80 columns if there isn't a tty
        # https://github.com/Textualize/rich/blob/master/docs/source/console.rst#terminal-detection
        os.environ["COLUMNS"] = str(size)

    def emit(self, record):
        message = self.format(record)
        traceback = None
        if self.rich_tracebacks and record.exc_info and record.exc_info != (None, None, None):
            exc_type, exc_value, exc_traceback = record.exc_info
            assert exc_type is not None
            assert exc_value is not None
            traceback = Traceback.from_exception(
                exc_type,
                exc_value,
                exc_traceback,
                width=self.tracebacks_width,
                code_width=self.tracebacks_code_width,
                extra_lines=self.tracebacks_extra_lines,
                theme=self.tracebacks_theme,
                word_wrap=self.tracebacks_word_wrap,
                show_locals=self.tracebacks_show_locals,
                locals_max_length=self.locals_max_length,
                locals_max_string=self.locals_max_string,
                suppress=self.tracebacks_suppress,
                max_frames=self.tracebacks_max_frames,
            )
            message = record.getMessage()

        message_renderable = self.render_message(
            record,
            [message] if not traceback else [message, traceback],
        )
        try:
            self.console.print(message_renderable, soft_wrap=True)
        except Exception:
            self.handleError(record)

    def render_message(self, record: LogRecord, message: list[str | Traceback]) -> "ConsoleRenderable":
        """Render message text in to Text.

        Args:
            record (LogRecord): logging Record.
            message (str): String containing log message.

        Returns:
            ConsoleRenderable: Renderable to display log message.
        """
        message_text = Text()
        for elem in message:
            if isinstance(elem, str):
                use_markup = getattr(record, "markup", self.markup)
                message_text = Text.from_markup(elem) if use_markup else Text(record.msg)
            else:
                message_text += elem

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        if self.keywords is None:
            self.keywords = self.KEYWORDS

        if self.keywords:
            message_text.highlight_words(self.keywords, "logging.keyword")

        return message_text

    def format(self, record):
        record.levelname = self.LEVEL_MAPPING[getattr(logging, str(record.levelname))]
        return super().format(record)


def banner(text: str) -> None:
    """Print a banner"""
    console.log(Panel(text), _stack_offset=2)


if __name__ == "__main__":
    console = Console(
        soft_wrap=True,
        color_system="256",
        force_terminal=True,
    )
    format = "%(asctime)s %(levelname)s \\[%(filename)s:%(lineno)s] :: %(message)s"
    sink = FluidRichHandler(
        console=console,
        level=logging.DEBUG,
        markup=True,
        rich_tracebacks=True,
        enable_link_path=False,
    )
    if os.getenv("CI") == "true":
        sink.columns(150)
    logging.basicConfig(level=logging.INFO, format=format, datefmt="[%X]", handlers=[sink])

    banner("link to https://alexandrite.app/")
    logging.info("A really long line with a long link that should be clickable (alexandrite.app): http://www.reallylong.link/rll/4QMpSmNQFPAzm6xssWDkgcED_nBklJunV2HwJKvBSjLOOIq3Pn/RvmOrS6lZmzNonqJ0X4Y_P4iBuLoMX3et/F9lebVVSnufNSssnWAiZUL1Q7aj9mnII9aQWQ2mhuPidyTYaOunxFGgKO8/NEzPfVjNDbNsZ61xhx0tQGfEerdwFtRagcaM0vI0XDLD2VwrRw/OOGxnPFH9HL_c5oe3w15WfFfALrqAP58/hI6Pm82wTzucsF08ak5Eh5A/6kt3q3FXPetygBQqxkVHqBt52dK0fYz4AShrsn4Rfj9Qc6KKN/HzJsUOV7nU6CSlo5BALkgUt5Uh0YhC_G5OQzGyNWC/qSZbTfL9mulULs8FM0s__JbLb949SP9nevrdMrJeznIPhPw9IHvbuJJtPNuaAcIVklk9r2mrf/8WoE3H8l_UKh175r_hvQaP4mOG2u91trDDlWiC5HeO2gemnpqhLX3GDYQuWUO8imD9hY3GtiqUMQ65STCluNolWkpJEO/6vXisSkC7RpoBqTSGBc3a9AeuK/F/NvcxZdZPLdkpcMnihcZjgFE31oed5Le25YxBd5EFEm_aXTXsjIJuUMuTZgcpiumYDnDosI9xxAVYeKt5z0AGeVr2tl3HH27Oi451qAgbMOgszdMT8Bpe0smUGXiKsPPKtwOh8RNqjX7XeiCDwlS_YN0MikEkqoB42hK87rOfQHm7ubPbVVIk8uv9d6sz66e/7lSsNg7ySfPMyhFAqSDvLdT5V2DWD7MAPyBJpqmcPyZPc3J9882fYlREgelmuKQRTGMzc36tV4WtOCnlH/Gq86Dc9Gq2ogZp2gzR9gNqiK1HaSYMiVkNdIpPYRGoPeUMAhJzaCfGFpGqvD6rl3bkOFv1q4DaAIynDJkUpgcrevR2Ds77rl9GeHbyHDRn4cqXI5PpVZKh5lVSiZ_PViXKQLW_e_w2UwCVxu5bSHiM28c/dRoZfU3VHBgYCp2PPgIYuMfS2reGhdn8ppHgSpXppn15iGw/_zVaLWJ2F0XbzoOBA4pRI5wmRxvIbNHEj3prswk4Ya_uXe3urInAaBK5oy5XmuO_hqr/3uyqb8v9ZmvnT_h_fxL3er_IVfsEb0Nx1ramqKZmkACbbnQ3nqibFmzs9EflzPvEudNPyim8QbIGDJ9ZflPSCFjOuqp_Na7SJKbKRMVOP1xsV_K_xMC6me6g1s0A79TXZPfNKJ1jH5VbmlwgnIoRGoFtLp10ecLuCfeYR2/yhsqlXz3kM7xNv09LyPOA5EmV9_64wlDjRCgxhZOZNvS4P5QP4vbnSU8PC6JKS/TVgywW/UZyyfPoUsi9ePeDXzRrvfKFpZnrPFySLz6VNiYjnEV09Zv9CXMIJaNbWGY98oQcDXwx3kHmDiYRaTF4SPgfyeJDl5hjt8LOQWZYMz_x07KW/5DZo1Dexf/3k4wNRxTytEsWlLVDJ20eN/lRDCckix_6CToi8oQA7jZP9j5elkjkPXDwm231_SSjgRZe52vWI1vErr8KgpZ5GKJLSVyUkPOjVK07jj/7A0mDGUdI_x7lyeUjIwTlZvtEf0YWpsZJpOoZ7Kb2kKApAVRV2YdZspMRmjP6suWnThSgQaLO9m5dFH/7i2oaLRBAK8158k0JhPMMUweI8UNq/8UwzHqBYw6IZ1eRKgQMyFsTaLi44SYeZiGIMDBPksKNJAuv_qNqklD9wNIYpouYQxz3_cHPzEdzyUHc7ub82oBw9joACFrnLR1IJZaG3QsULwsNqAcU7Vpz/ZPyaYgwim9H3lAGnjLI8uSzFoaNUt49g_fSaHX/ozqAPjyeX8g1LFXPJHwOXPQSwRHBLh4Av_qaeL7Xd/ateWXnp_WtrXzPW0Bf8UhMi9y9Y75cTM2AzT2Qfa91NXka/vg1Uf2a2BWCGcADTgHUl8yDYrUvcpgXFWHpr2ZEm49SwM24H91GYYqVbTUMcvoPvMq")
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")
    banner("end https://alexandrite.app/")
