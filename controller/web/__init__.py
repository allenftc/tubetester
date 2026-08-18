"""Web control surface for the test tube scanner.

Imports are intentionally lazy so utility modules such as the console event
store remain usable before optional runtime dependencies are installed.
"""

__all__ = ["create_control_app", "serve_control_server"]


def __getattr__(name: str):
    if name in __all__:
        from .server import create_control_app, serve_control_server

        return {"create_control_app": create_control_app, "serve_control_server": serve_control_server}[name]
    raise AttributeError(name)