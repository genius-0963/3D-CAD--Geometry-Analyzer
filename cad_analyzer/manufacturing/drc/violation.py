"""
Violation handling for Design Rule Checking (DRC).
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class ViolationSeverity(Enum):
    """Severity levels for design rule violations."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ViolationType(Enum):
    """Types of design rule violations."""
    WALL_THICKNESS = "wall_thickness"
    OVERHANG_ANGLE = "overhang_angle"
    HOLE_TOLERANCE = "hole_tolerance"
    EDGE_RADIUS = "edge_radius"
    INTERFERENCE = "interference"
    SURFACE_FINISH = "surface_finish"
    OTHER = "other"

@dataclass
class Violation:
    """Represents a design rule violation."""
    violation_type: ViolationType
    severity: ViolationSeverity
    location: Union[Tuple[float, float, float], List[Tuple[float, float, float]]]
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    element_id: Optional[Union[int, List[int]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to a dictionary for serialization."""
        return {
            'type': self.violation_type.value,
            'severity': self.severity.name,
            'location': self.location,
            'message': self.message,
            'data': self.data,
            'element_id': self.element_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Violation':
        """Create a Violation from a dictionary."""
        return cls(
            violation_type=ViolationType(data['type']),
            severity=ViolationSeverity[data['severity']],
            location=data['location'],
            message=data['message'],
            data=data.get('data', {}),
            element_id=data.get('element_id')
        )
