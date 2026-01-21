"""
Create a simple cube STL file for testing purposes.
"""
import numpy as np
import open3d as o3d

def create_cube_stl(output_file: str = 'test_cube.stl', size: float = 1.0):
    """Create a simple cube STL file.
    
    Args:
        output_file: Path to save the STL file
        size: Size of the cube
    """
    # Create a triangle mesh of a cube
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    
    # Add vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Save as STL
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"Created test cube STL file: {output_file}")

if __name__ == "__main__":
    # Create a test cube in the examples directory
    create_cube_stl('test_cube.stl')
    print("You can now run: python basic_usage.py")
