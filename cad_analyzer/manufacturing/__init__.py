"""
Manufacturing process analysis for 3D CAD models.

This module provides tools for analyzing 3D models for manufacturability
across different manufacturing processes.
"""

from .base import ManufacturingProcess, ManufacturingResult
from .cnc import CNCProcess
from .injection_molding import InjectionMoldingProcess
from .drc import (
    DesignRule,
    MinWallThicknessRule,
    OverhangAngleRule,
    HoleToleranceRule,
    DRCEngine,
    Violation,
    ViolationSeverity,
    ViolationType
)

__all__ = [
    'ManufacturingProcess',
    'ManufacturingResult',
    'CNCProcess',
    'InjectionMoldingProcess',
    'DesignRule',
    'MinWallThicknessRule',
    'OverhangAngleRule',
    'HoleToleranceRule',
    'DRCEngine',
    'Violation',
    'ViolationSeverity',
    'ViolationType'
]
