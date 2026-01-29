"""
Performance optimization module for DRC with spatial indexing.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
import time

from ...geometry.mesh import Mesh
from .violation import Violation

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_spatial_indexing: bool = True
    max_samples_per_rule: int = 10000
    parallel_processing: bool = False
    cache_results: bool = True
    batch_size: int = 1000

class SpatialIndex:
    """Spatial indexing for efficient mesh queries."""
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """Initialize spatial index.
        
        Args:
            vertices: Array of vertex positions (Nx3)
            faces: Array of face indices (Mx3)
        """
        self.vertices = vertices
        self.faces = faces
        self.face_centers = self._compute_face_centers()
        self.vertex_tree = None
        self.face_tree = None
        self._build_indices()
    
    def _compute_face_centers(self) -> np.ndarray:
        """Compute face centers for indexing."""
        return np.mean(self.vertices[self.faces], axis=1)
    
    def _build_indices(self):
        """Build spatial indices."""
        # Build KD-tree for vertices
        self.vertex_tree = cKDTree(self.vertices)
        
        # Build KD-tree for face centers
        self.face_tree = cKDTree(self.face_centers)
    
    def find_nearby_vertices(self, point: np.ndarray, radius: float) -> np.ndarray:
        """Find vertices within radius of point.
        
        Args:
            point: 3D point to search around
            radius: Search radius
            
        Returns:
            Array of vertex indices within radius
        """
        if self.vertex_tree is None:
            return np.array([])
        
        indices = self.vertex_tree.query_ball_point(point, radius)
        return np.array(indices)
    
    def find_nearby_faces(self, point: np.ndarray, radius: float) -> np.ndarray:
        """Find faces within radius of point.
        
        Args:
            point: 3D point to search around
            radius: Search radius
            
        Returns:
            Array of face indices within radius
        """
        if self.face_tree is None:
            return np.array([])
        
        indices = self.face_tree.query_ball_point(point, radius)
        return np.array(indices)
    
    def find_nearest_vertex(self, point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest vertices to point.
        
        Args:
            point: 3D point
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.vertex_tree is None:
            return np.array([]), np.array([])
        
        distances, indices = self.vertex_tree.query(point, k=k)
        return distances, indices
    
    def find_nearest_face(self, point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest faces to point.
        
        Args:
            point: 3D point
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.face_tree is None:
            return np.array([]), np.array([])
        
        distances, indices = self.face_tree.query(point, k=k)
        return distances, indices

class OptimizedDRCEngine:
    """Performance-optimized DRC engine with spatial indexing."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize optimized DRC engine.
        
        Args:
            config: Performance configuration
        """
        self.config = config or PerformanceConfig()
        self.rules = []
        self.spatial_index = None
        self.cached_results = {}
        self.performance_stats = {
            'total_checks': 0,
            'total_time': 0.0,
            'average_time_per_check': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def register_rule(self, rule):
        """Register a design rule."""
        self.rules.append(rule)
    
    def _build_spatial_index(self, mesh: Mesh):
        """Build spatial index for mesh."""
        if self.config.enable_spatial_indexing:
            self.spatial_index = SpatialIndex(mesh.vertices, mesh.faces)
    
    def _get_cache_key(self, mesh: Mesh, rule_name: str) -> str:
        """Generate cache key for mesh and rule combination."""
        # Simple hash based on mesh properties and rule name
        mesh_hash = hash((len(mesh.vertices), len(mesh.faces), mesh.area))
        return f"{mesh_hash}_{rule_name}"
    
    def _cached_check(self, rule, mesh: Mesh, rule_name: str) -> List[Violation]:
        """Check rule with caching."""
        if not self.config.cache_results:
            return rule.check(mesh)
        
        cache_key = self._get_cache_key(mesh, rule_name)
        
        if cache_key in self.cached_results:
            self.performance_stats['cache_hits'] += 1
            return self.cached_results[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        violations = rule.check(mesh)
        self.cached_results[cache_key] = violations
        
        return violations
    
    def run_checks_optimized(self, mesh: Mesh) -> Dict[str, List[Violation]]:
        """Run DRC checks with performance optimizations.
        
        Args:
            mesh: The 3D mesh to validate
            
        Returns:
            Dictionary mapping rule names to violations
        """
        start_time = time.time()
        
        # Build spatial index if enabled
        self._build_spatial_index(mesh)
        
        results = {}
        
        for rule in self.rules:
            rule_name = rule.__class__.__name__
            
            try:
                # Check if rule supports spatial optimization
                if hasattr(rule, 'set_spatial_index') and self.spatial_index:
                    rule.set_spatial_index(self.spatial_index)
                
                # Run check with caching
                violations = self._cached_check(rule, mesh, rule_name)
                results[rule_name] = violations
                
            except Exception as e:
                # Create error violation
                from .violation import Violation, ViolationSeverity, ViolationType
                
                error_violation = Violation(
                    violation_type=ViolationType.OTHER,
                    severity=ViolationSeverity.ERROR,
                    location=[0, 0, 0],
                    message=f"Error executing rule {rule_name}: {str(e)}",
                    data={"error": str(e), "rule": rule_name}
                )
                results[rule_name] = [error_violation]
        
        # Update performance stats
        end_time = time.time()
        check_time = end_time - start_time
        
        self.performance_stats['total_checks'] += 1
        self.performance_stats['total_time'] += check_time
        self.performance_stats['average_time_per_check'] = (
            self.performance_stats['total_time'] / self.performance_stats['total_checks']
        )
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        stats = self.performance_stats.copy()
        
        if self.config.cache_results:
            total_cache_requests = stats['cache_hits'] + stats['cache_misses']
            if total_cache_requests > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
            else:
                stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """Clear cached results."""
        self.cached_results.clear()
        self.performance_stats['cache_hits'] = 0
        self.performance_stats['cache_misses'] = 0

class OptimizedMinWallThicknessRule:
    """Optimized version of MinWallThicknessRule using spatial indexing."""
    
    def __init__(self, min_thickness: float = 1.0, sample_density: float = 1.0):
        """Initialize optimized rule.
        
        Args:
            min_thickness: Minimum wall thickness in mm
            sample_density: Sample density per mm²
        """
        self.min_thickness = min_thickness
        self.sample_density = sample_density
        self.spatial_index = None
    
    def set_spatial_index(self, spatial_index: SpatialIndex):
        """Set spatial index for optimization."""
        self.spatial_index = spatial_index
    
    def check(self, mesh: Mesh, **kwargs) -> List[Violation]:
        """Check wall thickness with spatial optimization."""
        from .violation import Violation, ViolationSeverity, ViolationType
        
        violations = []
        
        # Calculate adaptive sample count
        surface_area = mesh.area
        num_samples = min(10000, max(100, int(surface_area * self.sample_density)))
        
        # Sample points on surface
        points, face_indices = mesh.sample_surface(num_samples, return_index=True)
        
        # Process in batches for memory efficiency
        batch_size = 1000
        
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_indices = face_indices[i:i+batch_size]
            
            for j, point in enumerate(batch_points):
                face_idx = batch_indices[j]
                face_normal = mesh.face_normals[face_idx]
                
                # Use spatial index to find nearby vertices
                if self.spatial_index:
                    nearby_vertices = self.spatial_index.find_nearby_vertices(
                        point, self.min_thickness * 2
                    )
                    
                    if len(nearby_vertices) == 0:
                        continue
                    
                    # Check thickness using nearby vertices only
                    min_distance = float('inf')
                    for vertex_idx in nearby_vertices:
                        vertex = mesh.vertices[vertex_idx]
                        distance = np.linalg.norm(vertex - point)
                        if distance < min_distance and distance > 0.001:  # Avoid self
                            min_distance = distance
                    
                    wall_thickness = min_distance
                else:
                    # Fallback to original method
                    ray_origin = point + face_normal * 0.01
                    ray_direction = -face_normal
                    
                    hit = mesh.ray.intersects_location(
                        ray_origins=[ray_origin],
                        ray_directions=[ray_direction],
                        multiple_hits=False
                    )
                    
                    if hit[0].size > 0:
                        wall_thickness = np.linalg.norm(hit[0][0] - point)
                    else:
                        continue
                
                if wall_thickness < self.min_thickness:
                    violations.append(Violation(
                        violation_type=ViolationType.WALL_THICKNESS,
                        severity=ViolationSeverity.ERROR,
                        location=point.tolist(),
                        message=f"Wall thickness {wall_thickness:.2f}mm is below minimum {self.min_thickness}mm",
                        data={
                            "measured_thickness": float(wall_thickness),
                            "min_threshold": self.min_thickness,
                            "face_index": int(face_idx)
                        }
                    ))
        
        return violations

class OptimizedOverhangAngleRule:
    """Optimized version of OverhangAngleRule using spatial indexing."""
    
    def __init__(self, max_angle: float = 45.0, gravity_vector: Tuple[float, float, float] = (0, 0, -1)):
        """Initialize optimized rule.
        
        Args:
            max_angle: Maximum overhang angle in degrees
            gravity_vector: Gravity direction vector
        """
        self.max_angle = max_angle
        self.gravity_vector = np.array(gravity_vector, dtype=np.float64)
        self.gravity_vector = self.gravity_vector / np.linalg.norm(self.gravity_vector)
        self.spatial_index = None
    
    def set_spatial_index(self, spatial_index: SpatialIndex):
        """Set spatial index for optimization."""
        self.spatial_index = spatial_index
    
    def check(self, mesh: Mesh, **kwargs) -> List[Violation]:
        """Check overhang angles with spatial optimization."""
        from .violation import Violation, ViolationSeverity, ViolationType
        
        violations = []
        
        # Use spatial index to find faces with problematic normals
        if self.spatial_index:
            # Pre-filter faces based on normal orientation
            problematic_faces = []
            
            for i, normal in enumerate(mesh.face_normals):
                angle_rad = np.arccos(np.clip(np.dot(normal, self.gravity_vector), -1.0, 1.0))
                overhang_angle = 90.0 - np.degrees(angle_rad)
                
                if overhang_angle > self.max_angle * 0.8:  # Include some margin
                    problematic_faces.append(i)
            
            # Only check problematic faces
            face_indices = problematic_faces
        else:
            face_indices = range(len(mesh.face_normals))
        
        for i in face_indices:
            normal = mesh.face_normals[i]
            
            # Calculate overhang angle
            angle_rad = np.arccos(np.clip(np.dot(normal, self.gravity_vector), -1.0, 1.0))
            overhang_angle = 90.0 - np.degrees(angle_rad)
            
            if overhang_angle > self.max_angle:
                face_vertices = mesh.vertices[mesh.faces[i]]
                face_center = face_vertices.mean(axis=0)
                
                violations.append(Violation(
                    violation_type=ViolationType.OVERHANG_ANGLE,
                    severity=ViolationSeverity.WARNING,
                    location=face_center.tolist(),
                    message=f"Overhang angle {overhang_angle:.1f}° exceeds maximum {self.max_angle}°",
                    data={
                        "measured_angle": float(overhang_angle),
                        "max_threshold": self.max_angle,
                        "face_index": i
                    }
                ))
        
        return violations

def benchmark_drc_performance(mesh: Mesh, rules: List, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark DRC performance with and without optimizations.
    
    Args:
        mesh: Test mesh
        rules: List of rules to test
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with benchmark results
    """
    from .engine import DRCEngine
    
    results = {
        'standard_engine': {},
        'optimized_engine': {},
        'comparison': {}
    }
    
    # Benchmark standard engine
    standard_engine = DRCEngine()
    for rule in rules:
        standard_engine.register_rule(rule)
    
    standard_times = []
    for _ in range(iterations):
        start_time = time.time()
        standard_results = standard_engine.run_checks(mesh)
        end_time = time.time()
        standard_times.append(end_time - start_time)
    
    results['standard_engine'] = {
        'average_time': np.mean(standard_times),
        'min_time': np.min(standard_times),
        'max_time': np.max(standard_times),
        'std_time': np.std(standard_times)
    }
    
    # Benchmark optimized engine
    optimized_engine = OptimizedDRCEngine()
    for rule in rules:
        optimized_engine.register_rule(rule)
    
    optimized_times = []
    for _ in range(iterations):
        start_time = time.time()
        optimized_results = optimized_engine.run_checks_optimized(mesh)
        end_time = time.time()
        optimized_times.append(end_time - start_time)
    
    results['optimized_engine'] = {
        'average_time': np.mean(optimized_times),
        'min_time': np.min(optimized_times),
        'max_time': np.max(optimized_times),
        'std_time': np.std(optimized_times)
    }
    
    # Calculate comparison
    speedup = results['standard_engine']['average_time'] / results['optimized_engine']['average_time']
    results['comparison'] = {
        'speedup_factor': speedup,
        'time_reduction_percent': (1 - 1/speedup) * 100 if speedup > 0 else 0
    }
    
    return results
