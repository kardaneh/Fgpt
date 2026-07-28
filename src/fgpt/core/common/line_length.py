# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import re
from typing import Literal


class FortLineLength:
    """
    This class processes a free-format Fortran code string, ensuring that any lines
    longer than the specified maximum line length are wrapped appropriately, using
    continuation characters based on the line type (e.g., statement, OpenMP, OpenACC, comment).

    Attributes
    ----------
        _line_length (int): The maximum allowed line length for Fortran code.
        _cont_start (dict): The starting continuation characters for different line types.
        _cont_end (dict): The ending continuation characters for different line types.
        _key_lists (dict): List of break-point keywords for different line types.
        _stat (re.Pattern): Regex to match standard Fortran statements.
        _omp (re.Pattern): Regex to match OpenMP directives.
        _acc (re.Pattern): Regex to match OpenACC directives.
        _comment (re.Pattern): Regex to match comments in the Fortran code.
    """

    def __init__(self, line_length: int = 132):
        """
        Initializes the FortLineLength class with a specified maximum line length.

        Parameters
        ----------
        line_length : int
            Maximum allowed line length, default is 132.
        """
        self._line_length = line_length
        self._cont_start = {
            "statement": "&",
            "openmp_directive": "!$OMP& ",
            "openacc_directive": "!$ACC& ",
            "comment": "!& ",
            "unknown": "&",
        }
        self._cont_end = {
            "statement": "&",
            "openmp_directive": " &",
            "openacc_directive": " &",
            "comment": "",
            "unknown": "&",
        }
        self._key_lists = {
            "statement": [", ", ",", " "],
            "openmp_directive": [" ", ",", ")", "="],
            "openacc_directive": [" ", ",", ")", "="],
            "comment": [" ", ".", ",", "="],
            "unknown": [" ", ",", "=", "+", ")"],
        }
        self._stat = re.compile(
            r"^\s*(INTEGER|REAL|TYPE|CALL|SUBROUTINE|USE)", flags=re.I
        )
        self._omp = re.compile(r"^\s*!\$OMP", flags=re.I)
        self._acc = re.compile(r"^\s*!\$ACC", flags=re.I)
        self._comment = re.compile(r"^\s*!")

    def find_break_point(self, line: str, max_index: int, key_list: list) -> int:
        """
        Finds the most appropriate break point for a Fortran line based on a list of keywords.

        Parameters
        ----------
            line (str): The Fortran code line.
            max_index (int): The maximum index up to which the break point should be found.
            key_list (list): List of keywords that can serve as break points.

        Returns
        -------
        int
            The index where the break point occurs.

        Raises
        ------
        Exception
            If no suitable break point is found within the allowed range.
        """
        for key in key_list:
            idx = line.rfind(key, 0, max_index)
            if idx > 0:
                return idx + len(key)
        raise Exception(
            "Error in find_break_point. No suitable break point found for line '"
            + line[:max_index]
            + "' and keys '"
            + str(key_list)
            + "'"
        )

    def long_lines(self, fortran_in: str) -> bool:
        """
        Checks if any line in the input Fortran code exceeds the maximum allowed line length.

        Parameters
        ----------
            fortran_in (str): The Fortran code as a string.

        Returns
        -------
        bool
            True if at least one line is longer than the allowed length, False otherwise.
        """
        for line in fortran_in.split("\n"):
            if len(line) > self._line_length:
                return True
        return False

    def process(self, fortran_in: str) -> str:
        """
        Processes the input Fortran code and wraps lines that exceed the maximum allowed length.

        Parameters
        ----------
        fortran_in : str
            The input Fortran code as a string.

        Returns
        -------
        str
            The processed Fortran code with long lines wrapped appropriately.
        """
        fortran_out = ""
        for line in fortran_in.split("\n"):
            if len(line) > self._line_length:
                line_type = self._get_line_type(line)
                c_start = self._cont_start[line_type]
                c_end = self._cont_end[line_type]
                key_list = self._key_lists[line_type]

                break_point = self.find_break_point(
                    line, self._line_length - len(c_end), key_list
                )
                fortran_out += line[:break_point] + c_end + "\n"
                line = line[break_point:]

                while len(line) + len(c_start) > self._line_length:
                    break_point = self.find_break_point(
                        line, self._line_length - len(c_end) - len(c_start), key_list
                    )
                    fortran_out += c_start + line[:break_point] + c_end + "\n"
                    line = line[break_point:]

                if line:
                    fortran_out += c_start + line + "\n"
            else:
                fortran_out += line + "\n"

        # Remove the extra newline at the end
        return fortran_out[:-1]

    def _get_line_type(
        self, line: str
    ) -> Literal[
        "statement", "openmp_directive", "openacc_directive", "comment", "unknown"
    ]:
        """
        Determines the type of a Fortran line (statement, OpenMP directive, OpenACC directive, or comment).

        Parameters
        ----------
        line : str
            A single Fortran code line.

        Returns
        -------
        str
            The line type, either 'statement', 'openmp_directive', 'openacc_directive', 'comment', or 'unknown'.
        """
        if self._stat.match(line):
            return "statement"
        if self._omp.match(line):
            return "openmp_directive"
        if self._acc.match(line):
            return "openacc_directive"
        if self._comment.match(line):
            return "comment"
        return "unknown"
