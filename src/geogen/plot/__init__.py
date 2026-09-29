# from .GeoWordPlotter import GeoWordPlotter
# from .ModelReviewerJupyter import *
from .plot import *

def __getattr__(name: str):
    """Load optional plotting interfaces only when requested."""

    if name == "GeoWordPlotter":
        from .GeoWordPlotter import GeoWordPlotter

        globals()[name] = GeoWordPlotter
        return GeoWordPlotter

    if name == "ModelReviewer":
        from .ModelReviewerJupyter import ModelReviewer

        globals()[name] = ModelReviewer
        return ModelReviewer

    raise AttributeError(
        f"Module {__name__!r} has no attribute {name!r}"
    )
