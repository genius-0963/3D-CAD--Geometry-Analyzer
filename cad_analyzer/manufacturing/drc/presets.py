"""
Process-specific rule presets for different manufacturing processes.
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

from .rules import (
    DesignRule,
    MinWallThicknessRule,
    OverhangAngleRule,
    HoleToleranceRule,
)
from .violation import ViolationSeverity

class ProcessPreset(Enum):
    """Available manufacturing process presets."""
    CNC = "cnc"
    FDM = "fdm"
    SLA = "sla"
    SLS = "sls"
    INJECTION_MOLDING = "injection_molding"
    CNC_PRECISION = "cnc_precision"
    FDM_FINE = "fdm_fine"

@dataclass
class ProcessConfig:
    """Configuration for a manufacturing process."""
    name: str
    description: str
    rules: List[DesignRule]
    metadata: Dict[str, Any]

# Process-specific configurations
PROCESS_CONFIGS = {
    ProcessPreset.CNC: ProcessConfig(
        name="CNC Machining",
        description="Standard CNC milling and turning processes",
        rules=[
            MinWallThicknessRule(min_thickness=1.5, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=90, severity=ViolationSeverity.WARNING),  # CNC can handle any angle
            HoleToleranceRule(min_diameter=1.0, severity=ViolationSeverity.ERROR),
        ],
        metadata={
            "min_tool_diameter": 1.0,
            "typical_tolerance": 0.1,
            "material_removal_rate": "medium",
        }
    ),
    
    ProcessPreset.CNC_PRECISION: ProcessConfig(
        name="Precision CNC Machining",
        description="High-precision CNC machining with tight tolerances",
        rules=[
            MinWallThicknessRule(min_thickness=0.8, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=90, severity=ViolationSeverity.WARNING),
            HoleToleranceRule(min_diameter=0.5, severity=ViolationSeverity.ERROR),
        ],
        metadata={
            "min_tool_diameter": 0.5,
            "typical_tolerance": 0.01,
            "material_removal_rate": "low",
        }
    ),
    
    ProcessPreset.FDM: ProcessConfig(
        name="FDM 3D Printing",
        description="Fused Deposition Modeling 3D printing",
        rules=[
            MinWallThicknessRule(min_thickness=0.8, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=45, severity=ViolationSeverity.ERROR),
            HoleToleranceRule(min_diameter=0.6, severity=ViolationSeverity.WARNING),
        ],
        metadata={
            "layer_height": 0.2,
            "nozzle_diameter": 0.4,
            "typical_tolerance": 0.2,
        }
    ),
    
    ProcessPreset.FDM_FINE: ProcessConfig(
        name="Fine FDM 3D Printing",
        description="High-resolution FDM printing with small nozzle",
        rules=[
            MinWallThicknessRule(min_thickness=0.4, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=50, severity=ViolationSeverity.ERROR),
            HoleToleranceRule(min_diameter=0.3, severity=ViolationSeverity.WARNING),
        ],
        metadata={
            "layer_height": 0.1,
            "nozzle_diameter": 0.25,
            "typical_tolerance": 0.1,
        }
    ),
    
    ProcessPreset.SLA: ProcessConfig(
        name="SLA 3D Printing",
        description="Stereolithography 3D printing",
        rules=[
            MinWallThicknessRule(min_thickness=0.6, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=60, severity=ViolationSeverity.WARNING),
            HoleToleranceRule(min_diameter=0.3, severity=ViolationSeverity.WARNING),
        ],
        metadata={
            "layer_height": 0.05,
            "laser_spot_size": 0.1,
            "typical_tolerance": 0.05,
        }
    ),
    
    ProcessPreset.SLS: ProcessConfig(
        name="SLS 3D Printing",
        description="Selective Laser Sintering 3D printing",
        rules=[
            MinWallThicknessRule(min_thickness=1.0, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=70, severity=ViolationSeverity.WARNING),
            HoleToleranceRule(min_diameter=0.8, severity=ViolationSeverity.WARNING),
        ],
        metadata={
            "layer_height": 0.1,
            "laser_spot_size": 0.5,
            "typical_tolerance": 0.15,
        }
    ),
    
    ProcessPreset.INJECTION_MOLDING: ProcessConfig(
        name="Injection Molding",
        description="Plastic injection molding process",
        rules=[
            MinWallThicknessRule(min_thickness=2.0, severity=ViolationSeverity.ERROR),
            OverhangAngleRule(max_angle=3, severity=ViolationSeverity.ERROR),  # Draft angle requirement
            HoleToleranceRule(min_diameter=0.8, severity=ViolationSeverity.WARNING),
        ],
        metadata={
            "min_draft_angle": 1.0,
            "typical_tolerance": 0.05,
            "cycle_time": "medium",
        }
    ),
}

def get_process_preset(preset: ProcessPreset) -> ProcessConfig:
    """Get the configuration for a specific manufacturing process.
    
    Args:
        preset: The process preset to retrieve
        
    Returns:
        ProcessConfig for the specified preset
        
    Raises:
        ValueError: If the preset is not found
    """
    if preset not in PROCESS_CONFIGS:
        raise ValueError(f"Unknown process preset: {preset}")
    
    return PROCESS_CONFIGS[preset]

def get_all_presets() -> Dict[ProcessPreset, ProcessConfig]:
    """Get all available process presets.
    
    Returns:
        Dictionary mapping presets to their configurations
    """
    return PROCESS_CONFIGS.copy()

def create_engine_for_process(preset: ProcessPreset) -> 'DRCEngine':
    """Create a DRC engine configured for a specific manufacturing process.
    
    Args:
        preset: The manufacturing process preset
        
    Returns:
        Configured DRCEngine instance
    """
    from .engine import DRCEngine
    
    config = get_process_preset(preset)
    engine = DRCEngine()
    
    # Register all rules for the process
    for rule in config.rules:
        engine.register_rule(rule)
    
    return engine

def compare_processes(presets: List[ProcessPreset]) -> Dict[str, Any]:
    """Compare different manufacturing processes for the same design.
    
    Args:
        presets: List of process presets to compare
        
    Returns:
        Comparison results with recommendations
    """
    comparison = {
        "processes": [],
        "recommendations": [],
        "analysis": {}
    }
    
    for preset in presets:
        config = get_process_preset(preset)
        process_info = {
            "name": config.name,
            "description": config.description,
            "rules_count": len(config.rules),
            "metadata": config.metadata
        }
        comparison["processes"].append(process_info)
    
    # Generate recommendations based on process characteristics
    if ProcessPreset.CNC_PRECISION in presets:
        comparison["recommendations"].append({
            "process": "Precision CNC",
            "reason": "Best for tight tolerances and complex geometries",
            "use_case": "High-precision parts, metal components"
        })
    
    if ProcessPreset.FDM_FINE in presets:
        comparison["recommendations"].append({
            "process": "Fine FDM",
            "reason": "Good balance of detail and cost for prototyping",
            "use_case": "Detailed prototypes, functional testing"
        })
    
    if ProcessPreset.SLA in presets:
        comparison["recommendations"].append({
            "process": "SLA",
            "reason": "Highest detail and surface finish",
            "use_case": "Visual models, detailed prototypes"
        })
    
    return comparison

def get_process_suggestions(part_size: float, complexity: str, 
                          tolerance_requirement: float) -> List[ProcessPreset]:
    """Get suggested manufacturing processes based on part requirements.
    
    Args:
        part_size: Approximate part size in mm
        complexity: Part complexity ('simple', 'medium', 'complex')
        tolerance_requirement: Required tolerance in mm
        
    Returns:
        List of suggested process presets
    """
    suggestions = []
    
    # Simple logic for process selection
    if tolerance_requirement < 0.05:
        # Very tight tolerance - precision processes
        suggestions.extend([ProcessPreset.CNC_PRECISION, ProcessPreset.SLA])
    elif tolerance_requirement < 0.1:
        # Tight tolerance - precision or standard processes
        suggestions.extend([ProcessPreset.CNC, ProcessPreset.SLA, ProcessPreset.INJECTION_MOLDING])
    else:
        # Loose tolerance - most processes work
        suggestions.extend([ProcessPreset.FDM, ProcessPreset.SLS, ProcessPreset.CNC])
    
    # Consider part size
    if part_size > 200:
        # Large parts - favor processes that can handle large builds
        suggestions = [p for p in suggestions if p in [ProcessPreset.CNC, ProcessPreset.FDM, ProcessPreset.SLS]]
    elif part_size < 10:
        # Small parts - favor high-detail processes
        suggestions = [p for p in suggestions if p in [ProcessPreset.SLA, ProcessPreset.CNC_PRECISION]]
    
    # Consider complexity
    if complexity == 'complex':
        # Complex geometries - favor additive manufacturing
        suggestions = [p for p in suggestions if p in [ProcessPreset.SLA, ProcessPreset.SLS, ProcessPreset.FDM_FINE]]
    
    return list(set(suggestions))  # Remove duplicates
