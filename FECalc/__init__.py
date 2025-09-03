"""Convenience imports for FECalc package."""

from .FECalc import FECalc as _FECalc
from .PCCBuilder import PCCBuilder as _PCCBuilder
from .TargetMOL import TargetMOL as _TargetMOL

__all__ = ["FECalc", "PCCBuilder", "TargetMOL"]


def __getattr__(name):
    if name == "FECalc":
        return _FECalc
    if name == "PCCBuilder":
        return _PCCBuilder
    if name == "TargetMOL":
        return _TargetMOL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
