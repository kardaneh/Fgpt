# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/


class _Scope:
    """
    Lexical-scope and unique-name-generation helpers for ``JaxConverter``.

    Composes onto ``JaxConverter`` to manage the local-variable scope
    stack (:attr:`_local_defined_stack`, owned by the base class) used
    during vectorised body visits and helper-function generation, plus
    the monotonic counter (:attr:`counter`, also owned by the base
    class) used to generate unique ``lax.cond`` helper names.
    """

    def _push_scope(self) -> None:
        """
        Push a new empty local-variable scope onto the definition stack.

        Called at the start of every vectorised ``for`` body, ``while``
        body, and helper-function visit to isolate variable definitions
        from the enclosing scope.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            self._local_defined_stack.append({})
        except Exception as e:
            self.logger.exception("Exception in _push_scope:", e)
            raise

    def _pop_scope(self) -> None:
        """
        Pop the innermost local-variable scope from the definition stack.

        Should always be paired with a preceding :meth:`_push_scope`
        call. Typically placed in a ``finally`` block to guarantee
        cleanup even when the body visit raises.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            self._local_defined_stack.pop()
        except Exception as e:
            self.logger.exception("Exception in _pop_scope:", e)
            raise

    def _add_local(self, name: str, shape: tuple) -> None:
        """
        Register *name* with *shape* in the current (innermost) scope.

        Used during ``visit_Assign`` to record scalar variables that
        have been promoted to arrays under vectorisation so that later
        references in the same block can resolve their shape.

        Parameters
        ----------
        name : str
            Variable name to register.
        shape : tuple
            Inferred shape (may be an empty tuple for unresolved
            scalars).

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            self._local_defined_stack[-1][name] = shape
        except Exception as e:
            self.logger.exception("Exception in _add_local:", e)
            raise

    def _is_local(self, name: str) -> bool:
        """
        Return ``True`` if *name* is defined in any active scope.

        Searches the scope stack from innermost to outermost, mirroring
        standard lexical-scoping lookup order.

        Parameters
        ----------
        name : str
            Variable name to look up.

        Returns
        -------
        bool
            ``True`` if *name* appears in at least one active scope
            dict.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            return any(name in scope for scope in reversed(self._local_defined_stack))
        except Exception as e:
            self.logger.exception("Exception in _is_local:", e)
            raise

    def _fresh_names(self) -> tuple[str, str]:
        """
        Generate a unique pair of helper-function names for one ``if``
        node.

        Increments :attr:`counter` so that successive calls never
        produce the same names within a single function transformation
        pass.

        Returns
        -------
        Tuple[str, str]
            ``(_if_true_N, _if_false_N)`` where *N* is the current
            counter value before incrementing.

        Raises
        ------
        Exception
            Re-raises any unexpected error after logging.
        """
        try:
            n = self.counter
            self.counter += 1
            return f"_if_true_{n}", f"_if_false_{n}"
        except Exception as e:
            self.logger.exception("Exception in _fresh_names:", e)
            raise
