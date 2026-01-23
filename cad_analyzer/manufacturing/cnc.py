"""
CNC Machining Process Analysis Module
"""
import numpy as np
from typing import Dict, Any, List
from .base import ManufacturingProcess, ManufacturingResult

class CNCProcess(ManufacturingProcess):
    """Analyzes 3D models for CNC machining feasibility."""
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for CNC machining analysis."""
        return {
            'min_wall_thickness': 1.0,  # mm
            'min_hole_diameter': 1.0,   # mm
            'max_depth_diameter_ratio': 10.0,
            'min_corner_radius': 0.5,   # mm
            'tool_access_angle': 45.0,  # degrees
            'undercut_tolerance': 0.1,  # mm
        }
    
    def analyze(self, mesh, features: Dict[str, Any]) -> ManufacturingResult:
        """Analyze the mesh for CNC machining feasibility."""
        result = ManufacturingResult(
            process_name="CNC Machining",
            feasibility_score=1.0,
            is_feasible=True
        )
        
        # Check wall thickness
        if 'wall_thickness' in features:
            min_thickness = features['wall_thickness'].min()
            result.metrics['min_wall_thickness'] = min_thickness
            if min_thickness < self.config['min_wall_thickness']:
                result.feasibility_score *= 0.7
                result.is_feasible = False
                result.issues.append({
                    'type': 'thin_wall',
                    'severity': 'high',
                    'message': f"Minimum wall thickness {min_thickness:.2f}mm is below required {self.config['min_wall_thickness']}mm"
                })
        
        # Check for deep cavities
        if 'cavity_depths' in features:
            for i, depth in enumerate(features['cavity_depths']):
                if depth > self.config['max_depth_diameter_ratio'] * self.config['min_hole_diameter']:
                    result.feasibility_score *= 0.8
                    result.issues.append({
                        'type': 'deep_cavity',
                        'severity': 'medium',
                        'cavity_id': i,
                        'depth': depth,
                        'message': f"Cavity {i} is too deep for available tooling"
                    })
        
        # Check for sharp internal corners
        if 'internal_corners' in features:
            sharp_corners = [c for c in features['internal_corners'] 
                           if c < self.config['min_corner_radius']]
            if sharp_corners:
                result.feasibility_score *= 0.9
                result.issues.append({
                    'type': 'sharp_corners',
                    'severity': 'low',
                    'count': len(sharp_corners),
                    'min_radius': min(sharp_corners),
                    'message': f"Found {len(sharp_corners)} sharp internal corners below minimum radius"
                })
        
        # Add recommendations
        if result.feasibility_score < 0.9:
            result.recommendations = [
                "Consider increasing minimum wall thickness",
                "Add draft angles to vertical walls",
                "Increase internal corner radii where possible"
            ]
        
        return result
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for CNC analysis."""
        return [
            'wall_thickness',
            'cavity_depths',
            'internal_corners',
            'undercuts'
        ]
