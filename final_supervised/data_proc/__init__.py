"""Data loading helpers for the supervised PatchTST experiment."""

__all__ = ["TimeSeriesDataset", "build_loaders", "load_weather"]


def __getattr__(name):
    if name == "TimeSeriesDataset":
        from .dataset import TimeSeriesDataset

        return TimeSeriesDataset
    if name == "build_loaders":
        from .loaders import build_loaders

        return build_loaders
    if name == "load_weather":
        from .weather import load_weather

        return load_weather
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
