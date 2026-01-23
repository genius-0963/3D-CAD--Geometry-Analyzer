"""
Manufacturing process analysis for 3D CAD models.

This module provides tools for analyzing 3D models for manufacturability
across different manufacturing processes.
"""

from .base import ManufacturingProcess, ManufacturingResult
from .cnc import CNCProcess
from .injection_molding import InjectionMoldingProcess
from .printing.fdm import FDMProcess
from .printing.sla import SLAProcess
from .printing.sls import SLSProcess

__all__ = [
    'ManufacturingProcess',
    'ManufacturingResult',
    'CNCProcess',
    'InjectionMoldingProcess',
    'FDMProcess',
    'SLAProcess',
    'SLSProcess'
]
