"""
Design rule implementations for manufacturing validation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..mesh import Mesh
from .violation import Violation, ViolationSeverity, ViolationType

class DesignRule(ABC):
    """Abstract base class for design rules."""
    
    @abstractmethod
    def check(self, mesh: Mesh, **kwargs) -> List[Violation]:
        """Check the mesh against this design rule.
        
        Args:
            mesh: The 3D mesh to validate.
            **kwargs: Additional parameters for the check.
            
        Returns:
            List of violations found.
        """
        pass

@dataclass
class MinWallThicknessRule(DesignRule):
    """Check for minimum wall thickness in the model."""
    
    min_thickness: float = 1.0  # in mm
    severity: ViolationSeverity = ViolationSeverity.ERROR
    sample_density: float = 1.0  # points per mm²
    
    def check(self, mesh: Mesh, **kwargs) -> List[Violation]:
        """Check for walls thinner than the minimum allowed thickness."""
        violations = []
        
        # Calculate the number of points to sample based on surface area
        surface_area = mesh.area
        num_samples = max(100, int(surface_area * self.sample_density))
        
        # Sample points on the surface
        points, face_indices = mesh.sample_surface(num_samples, return_index=True)
        
        for i, point in enumerate(points):
            face_normal = mesh.face_normals[face_indices[i]]
            
            # Cast a ray in the opposite direction of the normal
            ray_origin = point + face_normal * 0.01  # Slightly offset from surface
            ray_direction = -face_normal
            
            # Perform ray casting to find the opposite face
            hit = mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction],
                multiple_hits=False
            )
            
            if hit[0].size > 0:  # If we hit something
                wall_thickness = np.linalg.norm(hit[0][0] - point)
                
                if wall_thickness < self.min_thickness:
                    violations.append(Violation(
                        violation_type=ViolationType.WALL_THICKNESS,
                        severity=self.severity,
                        location=point.tolist(),
                        message=f"Wall thickness {wall_thickness:.2f}mm is below minimum {self.min_thickness}mm",
                        data={
                            "measured_thickness": float(wall_thickness),
                            "min_threshold": self.min_thickness,
                            "face_index": int(face_indices[i])
                        }
                    ))
        
        return violations

@dataclass
class OverhangAngleRule(DesignRule):
    """Check for overhangs steeper than the maximum allowed angle."""
    
    max_angle: float = 45.0  # in degrees
    severity: ViolationSeverity = ViolationSeverity.WARNING
    gravity_vector: Tuple[float, float, float] = (0, 0, -1)  # Default: negative Z is down
    
    def check(self, mesh: Mesh, **kwargs) -> List[Violation]:
        """Check for overhangs exceeding the maximum allowed angle."""
        violations = []
        
        # Normalize the gravity vector
        gravity = np.array(self.gravity_vector, dtype=np.float64)
        gravity = gravity / np.linalg.norm(gravity)
        
        for i, normal in enumerate(mesh.face_normals):
            # Calculate the angle between the face normal and the gravity vector
            angle_rad = np.arccos(np.clip(np.dot(normal, gravity), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            # The overhang angle is 90° - the angle between normal and gravity
            overhang_angle = 90.0 - angle_deg
            
            if overhang_angle > self.max_angle:
                # Get face center for the violation location
                face_vertices = mesh.vertices[mesh.faces[i]]
                face_center = face_vertices.mean(axis=0)
                
                violations.append(Violation(
                    violation_type=ViolationType.OVERHANG_ANGLE,
                    severity=self.severity,
                    location=face_center.tolist(),
                    message=f"Overhang angle {overhang_angle:.1f}° exceeds maximum {self.max_angle}°",
                    data={
                        "measured_angle": float(overhang_angle),
                        "max_threshold": self.max_angle,
                        "face_index": i
                    }
                ))
        
        return violations

@dataclass
class HoleToleranceRule(DesignRule):
    """Check that holes meet minimum diameter requirements."""
    
    min_diameter: float = 1.0  # in mm
    severity: ViolationSeverity = ViolationSeverity.ERROR
    
    def check(self, mesh: Mesh, **kwargs) -> List[Violation]:
        """Check for holes smaller than the minimum allowed diameter."""
        violations = []
        
        # This is a simplified implementation. In practice, you would:
        # 1. Detect cylindrical surfaces in the mesh
        # 2. Calculate their diameters
        # 3. Check against the minimum diameter
        
        # For demonstration, we'll just check the bounding box dimensions
        # as a proxy for hole detection
        bbox = mesh.bounding_box
        bbox_dims = bbox.extents
        
        # Check each dimension for potential holes
        for dim in range(3):
            if bbox_dims[dim] < self.min_diameter:
                # This is a simplification - in reality, we'd check actual holes
                violations.append(Violation(
                    violation_type=ViolationType.HOLE_TOLERANCE,
                    severity=self.severity,
                    location=bbox.center.tolist(),
                    message=f"Feature in dimension {['X','Y','Z'][dim]} is too small: {bbox_dims[dim]:.2f}mm < {self.min_diameter}mm",
                    data={
                        "measured_diameter": float(bbox_dims[dim]),
                        "min_threshold": self.min_diameter,
                        "dimension": ['X','Y','Z'][dim]
                    }
                ))
        
        return violations
