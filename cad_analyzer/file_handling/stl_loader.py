""
STL file loader implementation.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import open3d as o3d

from . import BaseCADLoader
from ..geometry.mesh import Mesh

class STLLoader(BaseCADLoader):
    """Loader for STL (STereoLithography) files."""
    
    @classmethod
    def supported_formats(cls) -> list:
        """Return supported file extensions."""
        return ['.stl']
    
    def load(self) -> Dict[str, Any]:
        """Load STL file using Open3D.
        
        Returns:
            Dictionary containing vertices, triangles, and other mesh data
        """
        mesh = o3d.io.read_triangle_mesh(str(self.file_path))
        
        return {
            'vertices': np.asarray(mesh.vertices, dtype=np.float32),
            'triangles': np.asarray(mesh.triangles, dtype=np.int32),
            'vertex_normals': np.asarray(mesh.vertex_normals, dtype=np.float32) 
                           if mesh.has_vertex_normals() else None,
            'triangle_normals': np.asarray(mesh.triangle_normals, dtype=np.float32)
                             if mesh.has_triangle_normals() else None,
            'is_watertight': mesh.is_watertight()
        }
    
    def to_mesh(self) -> Mesh:
        """Convert STL data to a Mesh object.
        
        Returns:
            Mesh object containing the STL geometry
        """
        data = self.load()
        return Mesh(
            vertices=data['vertices'],
            triangles=data['triangles'],
            vertex_normals=data['vertex_normals'],
            triangle_normals=data['triangle_normals'],
            is_watertight=data['is_watertight']
        )
