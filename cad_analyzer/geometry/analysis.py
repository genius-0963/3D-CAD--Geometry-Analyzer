"""
Geometry analysis module for manufacturability assessment.
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
    thickness_distribution: np.ndarray  # Histogram of thickness values
    thin_regions: np.ndarray  # Indices of triangles with thickness below threshold
    
@dataclass
class CurvatureAnalysis:
    """Container for surface curvature analysis results."""
    gaussian_curvature: np.ndarray  # Per-vertex Gaussian curvature
    mean_curvature: np.ndarray     # Per-vertex mean curvature
    curvature_distribution: Dict[str, np.ndarray]  #Histograms of curvature values
    
@dataclass
class UndercutAnalysis:
    """Container for undercut analysis results."""
    undercut_faces: np.ndarray  # Indices of undercut triangles
    undercut_severity: np.ndarray  # Severity of undercuts (0-1)
    build_direction: np.ndarray  # Build direction used for analysis [0, 0, 1] by default

class GeometryAnalyzer:
    """
    Analyzes 3D geometry for manufacturability characteristics.
    """
    
    def __init__(self, mesh: Mesh):
        """Initialize with a mesh to analyze.
        
        Args:
            mesh: Mesh object containing the 3D geometry
        """
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
        """Analyze wall thickness distribution.
        
        Args:
            sample_density: Distance between sample points as a fraction of bounding box diagonal
            min_threshold: Minimum acceptable wall thickness
            
        Returns:
            WallThicknessAnalysis object containing thickness metrics
        """
        # Get bounding box diagonal for scaling
        bbox = self.mesh.get_bounding_box()
        bbox_diag = np.linalg.norm(bbox['max'] - bbox['min'])
        sample_dist = sample_density * bbox_diag
        
        # Convert to Open3D for ray casting
        mesh = self.mesh.to_open3d()
        
        # Sample points on the surface
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=int(10000 * (1.0 / sample_density)),
            init_factor=5
        )
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Cast rays in both directions of the normal
        ray_directions = np.vstack([normals, -normals])
        ray_origins = np.vstack([points, points])
        
        # Perform ray casting
        scene = o3d.t.geometry.RaycastingScene()
        mesh_id = scene.add_triangles(
            o3d.core.Tensor.from_dlpack(
                o3d.core.DLPackTensor(self.mesh.vertices.astype(np.float32))
            ),
            o3d.core.Tensor.from_dlpack(
                o3d.core.DLPackTensor(self.mesh.triangles.astype(np.uint32))
            )
        )
        
        rays = np.hstack([ray_origins, ray_directions]).astype(np.float32)
        ans = scene.cast_rays(rays)
        
        # Process ray casting results
        hit_distances = ans['t_hit'].numpy()
        hit_distances[hit_distances == np.inf] = 0
        
        # Calculate thickness at each point
        thickness = np.zeros(len(points))
        for i in range(len(points)):
            d1 = hit_distances[i] if hit_distances[i] < np.inf else 0
            d2 = hit_distances[i + len(points)] if hit_distances[i + len(points)] < np.inf else 0
            thickness[i] = d1 + d2
        
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
        """Analyze surface curvature characteristics.
        
        Returns:
            CurvatureAnalysis object containing curvature metrics
        """
        # Use Open3D to compute curvature
        mesh = self.mesh.to_open3d()
        
        # Estimate curvature using vertex normals
        mesh.compute_vertex_normals()
        
        # Compute curvature using normal variation
        # This is a simplified approach - more sophisticated methods could be used
        k1 = np.zeros(len(mesh.vertices))
        k2 = np.zeros(len(mesh.vertices))
        
        # For each vertex, compute curvature based on normal variation
        for i, vertex in enumerate(mesh.vertices):
            # Find nearest neighbors (including self)
            _, idx = self.kd_tree.query(vertex, k=10)
            neighbors = self.mesh.vertices[idx]
            normals = self.mesh.vertex_normals[idx]
            
            # Center points and normals
            center = np.mean(neighbors, axis=0)
            centered = neighbors - center
            
            # Simple curvature estimation
            if len(centered) > 3:
                # Fit a plane to the normals
                _, _, vh = np.linalg.svd(centered)
                normal = vh[2, :]
                
                # Project normals to tangent plane
                proj_normals = normals - np.outer(normals @ normal, normal)
                k1[i] = np.mean(np.linalg.norm(proj_normals, axis=1))
                
                # Crude approximation of second principal curvature
                k2[i] = np.std(np.linalg.norm(centered, axis=1))
        
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
        """Analyze model for potential undercuts.
        
        Args:
            build_direction: Build direction vector (default: [0, 0, 1])
            angle_threshold: Angle threshold in degrees to consider a face an undercut
            
        Returns:
            UndercutAnalysis object containing undercut information
        """
        if build_direction is None:
            build_direction = np.array([0, 0, 1], dtype=np.float32)
        
        # Normalize build direction
        build_direction = build_direction / np.linalg.norm(build_direction)
        
        # Calculate angle between face normals and build direction
        face_normals = self.mesh.triangle_normals
        dot_products = np.dot(face_normals, build_direction)
        angles = np.rad2deg(np.arccos(np.clip(dot_products, -1.0, 1.0)))
        
        # Identify undercut faces (facing downward)
        undercut_mask = angles > (180 - angle_threshold)
        undercut_indices = np.where(undercut_mask)[0]
        
        # Calculate severity (0-1) based on angle
        severity = (angles[undercut_mask] - (180 - angle_threshold)) / angle_threshold
        severity = np.clip(severity, 0, 1)
        
        return UndercutAnalysis(
            undercut_faces=undercut_indices,
            undercut_severity=severity,
            build_direction=build_direction
        )
    
    def analyze_manufacturability(self, **kwargs) -> Dict:
        """Run all manufacturability analyses.
        
        Args:
            **kwargs: Additional arguments to pass to individual analysis methods
            
        Returns:
            Dictionary containing all analysis results
        """
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
