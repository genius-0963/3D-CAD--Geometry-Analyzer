"""
Design Rule Checking (DRC) for manufacturing validation.

This module provides tools for validating 3D models against manufacturing
constraints and design rules for various manufacturing processes.
"""

from .rules import DesignRule, MinWallThicknessRule, OverhangAngleRule, HoleToleranceRule
from .engine import DRCEngine
from .violation import Violation, ViolationSeverity, ViolationType
from .visualization import DRCVisualizer
from .presets import get_process_preset, ProcessPreset, create_engine_for_process, compare_processes
from .optimization import OptimizedDRCEngine, PerformanceConfig, SpatialIndex

__all__ = [
    'DesignRule',
    'MinWallThicknessRule',
    'OverhangAngleRule',
    'HoleToleranceRule',
    'DRCEngine',
    'Violation',
    'ViolationSeverity',
    'ViolationType',
    'DRCVisualizer',
    'get_process_preset',
    'ProcessPreset',
    'create_engine_for_process',
    'compare_processes',
    'OptimizedDRCEngine',
    'PerformanceConfig',
    'SpatialIndex',
]
