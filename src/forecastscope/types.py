"""Small numerical result type, independent of any application artifact schema."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodForecast:
    """Named point path and outer interval; units are those of the supplied history."""
    name: str
    family: str
    point: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
