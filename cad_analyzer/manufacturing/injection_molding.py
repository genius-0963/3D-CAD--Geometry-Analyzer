"""
Injection Molding Process Analysis Module
"""
import numpy as np
from typing import Dict, Any, List
from .base import ManufacturingProcess, ManufacturingResult

class InjectionMoldingProcess(ManufacturingProcess):
    """Analyzes 3D models for injection molding feasibility."""
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for injection molding analysis."""
        return {
            'min_wall_thickness': 0.8,  # mm
            'max_wall_thickness': 4.0,  # mm
            'min_draft_angle': 1.0,     # degrees
            'max_thickness_ratio': 2.0, # Max ratio between thick and thin sections
            'min_corner_radius': 0.5,   # mm
            'undercut_tolerance': 0.1,  # mm
            'sink_mark_risk_threshold': 0.8,  # Risk score 0-1
        }
    
    def analyze(self, mesh, features: Dict[str, Any]) -> ManufacturingResult:
        """Analyze the mesh for injection molding feasibility."""
        result = ManufacturingResult(
            process_name="Injection Molding",
            feasibility_score=1.0,
            is_feasible=True
        )
        
        # Check wall thickness
        if 'wall_thickness' in features:
            thickness = features['wall_thickness']
            min_thickness = thickness.min()
            max_thickness = thickness.max()
            thickness_ratio = max_thickness / min_thickness if min_thickness > 0 else float('inf')
            
            result.metrics.update({
                'min_wall_thickness': min_thickness,
                'max_wall_thickness': max_thickness,
                'thickness_ratio': thickness_ratio
            })
            
            # Check minimum wall thickness
            if min_thickness < self.config['min_wall_thickness']:
                result.feasibility_score *= 0.7
                result.is_feasible = False
                result.issues.append({
                    'type': 'thin_wall',
                    'severity': 'high',
                    'value': min_thickness,
                    'threshold': self.config['min_wall_thickness'],
                    'message': f"Minimum wall thickness {min_thickness:.2f}mm is below required {self.config['min_wall_thickness']}mm"
                })
            
            # Check maximum wall thickness
            if max_thickness > self.config['max_wall_thickness']:
                result.feasibility_score *= 0.8
                result.issues.append({
                    'type': 'thick_wall',
                    'severity': 'medium',
                    'value': max_thickness,
                    'threshold': self.config['max_wall_thickness'],
                    'message': f"Maximum wall thickness {max_thickness:.2f}mm exceeds recommended {self.config['max_wall_thickness']}mm"
                })
            
            # Check thickness variation
            if thickness_ratio > self.config['max_thickness_ratio']:
                result.feasibility_score *= 0.9
                result.issues.append({
                    'type': 'thickness_variation',
                    'severity': 'medium',
                    'ratio': thickness_ratio,
                    'threshold': self.config['max_thickness_ratio'],
                    'message': f"Wall thickness variation ratio {thickness_ratio:.1f}x exceeds recommended {self.config['max_thickness_ratio']}x"
                })
        
        # Check draft angles
        if 'draft_angles' in features:
            low_draft_areas = features['draft_angles'] < np.radians(self.config['min_draft_angle'])
            low_draft_count = np.sum(low_draft_areas)
            
            if low_draft_count > 0:
                low_draft_percent = 100 * low_draft_count / len(features['draft_angles'])
                result.feasibility_score *= 0.8
                result.issues.append({
                    'type': 'low_draft_angle',
                    'severity': 'high',
                    'affected_area_percent': low_draft_percent,
                    'min_angle_deg': self.config['min_draft_angle'],
                    'message': f"{low_draft_percent:.1f}% of surfaces have draft angles below {self.config['min_draft_angle']}°"
                })
        
        # Check for sink marks
        if 'sink_mark_risk' in features and 'wall_thickness' in features:
            high_risk_areas = features['sink_mark_risk'] > self.config['sink_mark_risk_threshold']
            high_risk_count = np.sum(high_risk_areas)
            
            if high_risk_count > 0:
                risk_percent = 100 * high_risk_count / len(features['sink_mark_risk'])
                result.feasibility_score *= 0.9
                result.issues.append({
                    'type': 'sink_mark_risk',
                    'severity': 'medium',
                    'affected_area_percent': risk_percent,
                    'message': f"Potential sink mark risk in {risk_percent:.1f}% of thick sections"
                })
        
        # Add recommendations
        if result.feasibility_score < 0.9:
            result.recommendations = [
                "Ensure uniform wall thickness between 0.8-4.0mm",
                f"Add draft angles of at least {self.config['min_draft_angle']}° to vertical walls",
                "Add fillets to sharp corners to improve material flow",
                "Consider using ribs instead of thick sections to prevent sink marks"
            ]
        
        return result
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for injection molding analysis."""
        return [
            'wall_thickness',
            'draft_angles',
            'undercuts',
            'sink_mark_risk',
            'corner_radii'
        ]
