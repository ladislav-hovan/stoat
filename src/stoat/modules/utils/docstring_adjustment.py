### Imports ###
from typing import Callable

### Decorators ###
def format_docstring(
    **kwargs,
) -> Callable:
    """
    Formats the docstring of the function with the provided keyword
    arguments.

    Returns
    -------
    Callable
        Decorator which will format function docstrings using the
        provided keyword arguments
    """

    def inner_decorator(
        fxn: Callable,
    ) -> Callable:
        """
        Replaces the docstring of the provided function with the
        formatted version.

        Parameters
        ----------
        fxn : Callable
            Function whose docstring should be formatted

        Returns
        -------
        Callable
            Function with the properly formatted docstring
        """

        fxn.__doc__ = fxn.__doc__.format(**kwargs)

        return fxn

    return inner_decorator