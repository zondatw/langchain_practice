__all__ = ["RustProjectAssistant", "load_settings"]


def __getattr__(name: str):
    if name == "RustProjectAssistant":
        from .assistant import RustProjectAssistant
        return RustProjectAssistant
    if name == "load_settings":
        from .settings import load_settings
        return load_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
