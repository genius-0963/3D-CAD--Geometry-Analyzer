""
Core mesh class for 3D geometry representation and manipulation.
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import open3d as o3d
from dataclasses import dataclass

@dataclass
class Mesh:
    """
    A class to represent and manipulate 3D mesh data.
    
    Attributes:
        vertices: Array of vertex positions (Nx3)
        triangles: Array of triangle indices (Mx3)
        vertex_normals: Optional array of vertex normals (Nx3)
        triangle_normals: Optional array of triangle normals (Mx3)
        is_watertight: Boolean indicating if the mesh is watertight
    """
    vertices: np.ndarray
    triangles: np.ndarray
    vertex_normals: Optional[np.ndarray] = None
    triangle_normals: Optional[np.ndarray] = None
    is_watertight: bool = False
    
    def __post_init__(self):
        """Validate and preprocess mesh data."""
        self._validate_inputs()
        self._compute_normals_if_needed()
    
    def _validate_inputs(self) -> None:
        """Validate input arrays."""
        if not isinstance(self.vertices, np.ndarray) or self.vertices.shape[1] != 3:
            raise ValueError("Vertices must be an Nx3 numpy array")
        if not isinstance(self.triangles, np.ndarray) or self.triangles.shape[1] != 3:
            raise ValueError("Triangles must be an Mx3 numpy array")
        if self.vertex_normals is not None and self.vertex_normals.shape != self.vertices.shape:
            raise ValueError("Vertex normals must have the same shape as vertices")
        if self.triangle_normals is not None and self.triangle_normals.shape[0] != len(self.triangles):
            raise ValueError("Number of triangle normals must match number of triangles")
    
    def _compute_normals_if_needed(self) -> None:
        """Compute normals if they're not provided."""
        if self.vertex_normals is None:
            self.compute_vertex_normals()
        if self.triangle_normals is None:
            self.compute_triangle_normals()
    
    def compute_vertex_normals(self) -> None:
        """Compute vertex normals using Open3D."""
        mesh = self.to_open3d()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    
    def compute_triangle_normals(self) -> None:
        """Compute triangle normals using cross products."""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        
        # Compute face normals
        face_normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norm[norm == 0] = 1.0  # Avoid division by zero for degenerate triangles
        self.triangle_normals = face_normals / norm
    
    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        """Convert to Open3D TriangleMesh.
        
        Returns:
            Open3D TriangleMesh object
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        
        if self.vertex_normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.triangle_normals is not None:
            mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals)
            
        return mesh
    
    def get_bounding_box(self) -> Dict[str, np.ndarray]:
        """Get the axis-aligned bounding box of the mesh.
        
        Returns:
            Dictionary with 'min' and 'max' keys containing the corner points
        """
        return {
            'min': np.min(self.vertices, axis=0),
            'max': np.max(self.vertices, axis=0)
        }
    
    def normalize(self) -> 'Mesh':
        """Normalize mesh to fit in unit cube centered at origin.
        
        Returns:
            Normalized Mesh instance
        """
        bbox = self.get_bounding_box()
        center = (bbox['max'] + bbox['min']) / 2
        scale = np.max(bbox['max'] - bbox['min'])
        
        if scale > 0:
            normalized_vertices = (self.vertices - center) / scale
        else:
            normalized_vertices = self.vertices.copy()
            
        return Mesh(
            vertices=normalized_vertices,
            triangles=self.triangles,
            vertex_normals=self.vertex_normals,
            triangle_normals=self.triangle_normals,
            is_watertight=self.is_watertight
        )
    
    def simplify(self, target_triangles: int) -> 'Mesh':
        """Simplify mesh to target number of triangles.
        
        Args:
            target_triangles: Target number of triangles
            
        Returns:
            Simplified Mesh instance
        """
        if len(self.triangles) <= target_triangles:
            return self
            
        mesh = self.to_open3d()
        simplified = mesh.simplify_quadric_decimation(target_triangles)
        
        return Mesh(
            vertices=np.asarray(simplified.vertices, dtype=np.float32),
            triangles=np.asarray(simplified.triangles, dtype=np.int32),
            vertex_normals=np.asarray(simplified.vertex_normals, dtype=np.float32) 
                       if simplified.has_vertex_normals() else None,
            is_watertight=simplified.is_watertight()
        )
