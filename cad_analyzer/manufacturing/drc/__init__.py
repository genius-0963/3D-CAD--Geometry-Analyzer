"""
Design Rule Checking (DRC) for manufacturing validation.

This module provides tools for validating 3D models against manufacturing
constraints and design rules for various manufacturing processes.
"""

from .rules import DesignRule, MinWallThicknessRule, OverhangAngleRule, HoleToleranceRule
from .engine import DRCEngine
from .violation import Violation, ViolationSeverity, ViolationType

__all__ = [
    'DesignRule',
    'MinWallThicknessRule',
    'OverhangAngleRule',
    'HoleToleranceRule',
    'DRCEngine',
    'Violation',
    'ViolationSeverity',
    'ViolationType',
]
