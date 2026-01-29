"""
Simplified geometry analysis module for manufacturability assessment.
"""
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import open3d as o3d
from scipy import spatial
from dataclasses import dataclass
from .mesh import Mesh

@dataclass
class WallThicknessAnalysis:
    """Container for wall thickness analysis results."""
    min_thickness: float
    max_thickness: float 
    avg_thickness: float
    thickness_distribution: np.ndarray
    thin_regions: np.ndarray
    
@dataclass
class CurvatureAnalysis:
    """Container for surface curvature analysis results."""
    gaussian_curvature: np.ndarray
    mean_curvature: np.ndarray
    curvature_distribution: Dict[str, np.ndarray]
    
@dataclass
class UndercutAnalysis:
    """Container for undercut analysis results."""
    undercut_faces: np.ndarray
    undercut_severity: np.ndarray
    build_direction: np.ndarray

class GeometryAnalyzer:
    """
    Analyzes 3D geometry for manufacturability characteristics.
    """
    
    def __init__(self, mesh: Mesh):
        """Initialize with a mesh to analyze."""
        self.mesh = mesh
        self._kd_tree = None
        
    @property
    def kd_tree(self) -> spatial.KDTree:
        """Lazily build and return a KD-tree for spatial queries."""
        if self._kd_tree is None:
            self._kd_tree = spatial.KDTree(self.mesh.vertices)
        return self._kd_tree
    
    def analyze_wall_thickness(self, 
                             sample_density: float = 0.1,
                             min_threshold: float = 1.0) -> WallThicknessAnalysis:
        """Analyze wall thickness distribution using simplified method."""
        # For now, return a simple analysis based on triangle sizes
        triangles = self.mesh.vertices[self.mesh.triangles]
        
        # Calculate edge lengths
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        edge3 = triangles[:, 2] - triangles[:, 1]
        
        edge_lengths = np.array([
            np.linalg.norm(edge1, axis=1),
            np.linalg.norm(edge2, axis=1),
            np.linalg.norm(edge3, axis=1)
        ])
        
        # Use average edge length as thickness approximation
        thickness = np.mean(edge_lengths, axis=0)
        
        # Filter out invalid measurements
        valid_thickness = thickness[thickness > 0]
        
        if len(valid_thickness) == 0:
            return WallThicknessAnalysis(
                min_thickness=0,
                max_thickness=0,
                avg_thickness=0,
                thickness_distribution=np.array([]),
                thin_regions=np.array([])
            )
        
        # Calculate statistics
        min_t = np.min(valid_thickness)
        max_t = np.max(valid_thickness)
        avg_t = np.mean(valid_thickness)
        
        # Identify thin regions
        thin_mask = thickness < min_threshold
        thin_indices = np.where(thin_mask)[0]
        
        # Create histogram of thickness values
        hist, bin_edges = np.histogram(valid_thickness, bins=50, density=True)
        
        return WallThicknessAnalysis(
            min_thickness=float(min_t),
            max_thickness=float(max_t),
            avg_thickness=float(avg_t),
            thickness_distribution=hist,
            thin_regions=thin_indices
        )
    
    def analyze_curvature(self) -> CurvatureAnalysis:
        """Analyze surface curvature characteristics."""
        # Simple curvature estimation based on normal variation
        k1 = np.zeros(len(self.mesh.vertices))
        k2 = np.zeros(len(self.mesh.vertices))
        
        # For each vertex, compute curvature based on neighbor variation
        for i, vertex in enumerate(self.mesh.vertices):
            # Find nearest neighbors
            _, idx = self.kd_tree.query(vertex, k=min(10, len(self.mesh.vertices)))
            neighbors = self.mesh.vertices[idx]
            
            # Center points
            center = np.mean(neighbors, axis=0)
            centered = neighbors - center
            
            # Simple curvature estimation
            if len(centered) > 3:
                # Fit a plane to neighbors
                _, _, vh = np.linalg.svd(centered)
                normal = vh[2, :]
                
                # Calculate deviation from plane
                distances = np.abs(centered @ normal)
                k1[i] = np.mean(distances)
                k2[i] = np.std(distances)
        
        # Calculate Gaussian and mean curvature
        gaussian_curvature = k1 * k2
        mean_curvature = 0.5 * (k1 + k2)
        
        # Create histograms
        hist_gaussian, _ = np.histogram(gaussian_curvature, bins=50, density=True)
        hist_mean, _ = np.histogram(mean_curvature, bins=50, density=True)
        
        return CurvatureAnalysis(
            gaussian_curvature=gaussian_curvature,
            mean_curvature=mean_curvature,
            curvature_distribution={
                'gaussian': hist_gaussian,
                'mean': hist_mean
            }
        )
    
    def analyze_undercuts(self, build_direction: np.ndarray = None,
                         angle_threshold: float = 45.0) -> UndercutAnalysis:
        """Analyze model for potential undercuts."""
        if build_direction is None:
            build_direction = np.array([0, 0, 1], dtype=np.float32)
        
        # Normalize build direction
        build_direction = build_direction / np.linalg.norm(build_direction)
        
        # Calculate triangle normals if not already present
        if not hasattr(self.mesh, 'triangle_normals') or len(self.mesh.triangle_normals) == 0:
            triangles = self.mesh.vertices[self.mesh.triangles]
            edge1 = triangles[:, 1] - triangles[:, 0]
            edge2 = triangles[:, 2] - triangles[:, 0]
            normals = np.cross(edge1, edge2)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            self.mesh.triangle_normals = normals / norms
        
        face_normals = self.mesh.triangle_normals
        
        # Calculate angle between face normals and build direction
        dot_products = np.dot(face_normals, build_direction)
        angles = np.rad2deg(np.arccos(np.clip(dot_products, -1.0, 1.0)))
        
        # Identify undercut faces (facing downward relative to build direction)
        undercut_mask = angles > (90 + angle_threshold)
        undercut_indices = np.where(undercut_mask)[0]
        
        # Calculate severity (0-1) based on angle
        severity = (angles[undercut_mask] - (90 + angle_threshold)) / (90 - angle_threshold)
        severity = np.clip(severity, 0, 1)
        
        return UndercutAnalysis(
            undercut_faces=undercut_indices,
            undercut_severity=severity,
            build_direction=build_direction
        )
    
    def analyze_manufacturability(self, **kwargs) -> Dict:
        """Run all manufacturability analyses."""
        return {
            'wall_thickness': self.analyze_wall_thickness(
                sample_density=kwargs.get('sample_density', 0.1),
                min_threshold=kwargs.get('min_thickness', 1.0)
            ),
            'curvature': self.analyze_curvature(),
            'undercuts': self.analyze_undercuts(
                build_direction=kwargs.get('build_direction'),
                angle_threshold=kwargs.get('angle_threshold', 45.0)
            )
        }
