"""Compatibility wrappers for API names awaiting validated native Python backends."""

from __future__ import annotations

from ._rbridge import call_r_function


def make_r_bridge_wrapper(name: str):
    def wrapper(*args, **kwargs):
        return call_r_function(name, *args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = (
        f"Compatibility wrapper for R gp3tools `{name}()`. "
        "This API name is frozen in the Python alpha build; when no validated "
        "native backend is available, calls delegate through optional rpy2."
    )
    wrapper._gp3tools_status = "r-bridge"
    return wrapper


def r_aliases(function, /, **aliases):
    """Add explicit R-compatible keyword aliases without swallowing unknown arguments.

    Each keyword maps an R argument name to the existing Python argument name.
    Supplying both names for the same value is an error.
    """
    import inspect
    from functools import wraps

    original_signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        positional = original_signature.bind_partial(*args)

        for alias, target in aliases.items():
            if alias not in kwargs:
                continue

            if target in kwargs or target in positional.arguments:
                raise TypeError(
                    f"{function.__name__} received both "
                    f"'{target}' and its R-compatible alias '{alias}'"
                )

            kwargs[target] = kwargs.pop(alias)

        return function(*args, **kwargs)

    # Expose aliases explicitly to introspection without weakening the
    # runtime validation performed by the wrapped Python function.
    parameters = list(original_signature.parameters.values())

    var_keyword = [p for p in parameters if p.kind == inspect.Parameter.VAR_KEYWORD]

    parameters = [p for p in parameters if p.kind != inspect.Parameter.VAR_KEYWORD]

    existing = {p.name for p in parameters}

    for alias in aliases:
        if alias not in existing:
            parameters.append(
                inspect.Parameter(
                    alias,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                )
            )

    parameters.extend(var_keyword)

    wrapped.__signature__ = original_signature.replace(parameters=parameters)
    wrapped.__r_aliases__ = dict(aliases)

    return wrapped
