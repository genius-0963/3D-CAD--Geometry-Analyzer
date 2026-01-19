""
Main analyzer class for the 3D CAD Geometry Analyzer.
"""
from typing import Dict, Any, Optional, Union
import os
from pathlib import Path
import numpy as np

from ..file_handling import STLLoader, BaseCADLoader
from ..geometry.mesh import Mesh
from ..geometry.analysis import GeometryAnalyzer, WallThicknessAnalysis, CurvatureAnalysis, UndercutAnalysis

class CADAnalyzer:
    """
    Main class for analyzing 3D CAD models for manufacturability.
    
    This class serves as the main entry point for the analysis pipeline,
    coordinating between file loading, geometry processing, and analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CAD analyzer with optional configuration.
        
        Args:
            config: Optional configuration dictionary. Can include:
                - sample_density: Density for wall thickness sampling
                - min_thickness: Minimum acceptable wall thickness
                - angle_threshold: Angle threshold for undercut detection (degrees)
                - build_direction: Build direction vector for undercut analysis
        """
        self.config = {
            'sample_density': 0.1,
            'min_thickness': 1.0,
            'angle_threshold': 45.0,
            'build_direction': [0, 0, 1],
            **(config or {})
        }
        self.mesh: Optional[Mesh] = None
        self.analyzer: Optional[GeometryAnalyzer] = None
        self.analysis_results: Dict[str, Any] = {}
    
    def load_file(self, file_path: Union[str, os.PathLike]) -> None:
        """Load a CAD file for analysis.
        
        Args:
            file_path: Path to the CAD file
            
        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and use appropriate loader
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.stl':
            loader: BaseCADLoader = STLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Load the mesh
        self.mesh = loader.to_mesh()
        self.analyzer = GeometryAnalyzer(self.mesh)
    
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """Run all available analyses on the loaded model.
        
        Args:
            **kwargs: Override default analysis parameters
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.analyzer is None:
            raise RuntimeError("No model loaded. Call load_file() first.")
        
        # Update config with any overrides
        config = {**self.config, **kwargs}
        
        # Run analyses
        self.analysis_results = self.analyzer.analyze_manufacturability(
            sample_density=config['sample_density'],
            min_threshold=config['min_thickness'],
            angle_threshold=config['angle_threshold'],
            build_direction=config['build_direction']
        )
        
        return self.analysis_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results.
        
        Returns:
            Dictionary containing a summary of the analysis
            
        Raises:
            RuntimeError: If no analysis has been performed
        """
        if not self.analysis_results:
            raise RuntimeError("No analysis results available. Call analyze() first.")
        
        wall = self.analysis_results.get('wall_thickness', {})
        undercuts = self.analysis_results.get('undercuts', {})
        
        return {
            'wall_thickness': {
                'min': getattr(wall, 'min_thickness', None),
                'max': getattr(wall, 'max_thickness', None),
                'avg': getattr(wall, 'avg_thickness', None),
                'thin_regions_count': len(getattr(wall, 'thin_regions', [])),
                'is_acceptable': getattr(wall, 'min_thickness', 0) >= self.config['min_thickness']
            },
            'undercuts': {
                'count': len(getattr(undercuts, 'undercut_faces', [])),
                'max_severity': float(np.max(getattr(undercuts, 'undercut_severity', [0.0]))) if hasattr(undercuts, 'undercut_severity') else 0.0,
                'is_acceptable': len(getattr(undercuts, 'undercut_faces', [])) == 0
            },
            'overall_manufacturability': self._assess_manufacturability()
        }
    
    def _assess_manufacturability(self) -> Dict[str, Any]:
        """Assess overall manufacturability based on analysis results."""
        if not self.analysis_results:
            return {
                'status': 'not_analyzed',
                'score': 0.0,
                'issues': []
            }
        
        issues = []
        score = 1.0
        
        # Check wall thickness
        wall = self.analysis_results.get('wall_thickness', {})
        if hasattr(wall, 'min_thickness') and wall.min_thickness < self.config['min_thickness']:
            severity = min(1.0, (self.config['min_thickness'] - wall.min_thickness) / self.config['min_thickness'])
            issues.append({
                'type': 'thin_walls',
                'severity': float(severity),
                'message': f"Minimum wall thickness ({wall.min_thickness:.2f} units) is below threshold ({self.config['min_thickness']} units)"
            })
            score *= (1.0 - 0.5 * severity)  # Reduce score based on severity
        
        # Check undercuts
        undercuts = self.analysis_results.get('undercuts', {})
        if hasattr(undercuts, 'undercut_faces') and len(undercuts.undercut_faces) > 0:
            severity = np.mean(undercuts.undercut_severity) if hasattr(undercuts, 'undercut_severity') else 0.5
            issues.append({
                'type': 'undercuts',
                'severity': float(severity),
                'count': len(undercuts.undercut_faces),
                'message': f"Found {len(undercuts.undercut_faces)} undercut faces"
            })
            score *= (1.0 - 0.3 * severity)  # Reduce score based on severity
        
        # Determine status
        if not issues:
            status = 'excellent'
        elif score > 0.7:
            status = 'good'
        elif score > 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'score': float(score),
            'issues': issues
        }
    
    def export_analysis(self, format: str = 'json') -> Dict[str, Any]:
        """Export analysis results in the specified format.
        
        Args:
            format: Output format ('json' or 'dict')
            
        Returns:
            Analysis results in the requested format
            
        Raises:
            ValueError: If an unsupported format is requested
        """
        if format.lower() == 'json':
            return self._to_json_serializable(self.analysis_results)
        elif format.lower() == 'dict':
            return self.analysis_results
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _to_json_serializable(self, obj):
        """Recursively convert analysis results to JSON-serializable types."""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return {k: self._to_json_serializable(v) for k, v in vars(obj).items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)
