"""Useful functions with different objectives.

Usage:
    -

Author:
    A.J. Sanchez-Fernandez - 22/03/2023
"""


def is_notebook() -> bool:
    """Checks whether the current Python environment is running inside a
    Jupyter Notebook or a Python script.

    Returns:
        bool: True if the environment is a Jupyter Notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__

        # Jupyter notebook or qtconsole.
        if shell == 'ZMQInteractiveShell':
            return True

        # Terminal running IPython.
        elif shell == 'TerminalInteractiveShell':
            return False

        # Other type (?).
        else:
            return False

    # Probably standard Python interpreter.
    except NameError:
        return False
