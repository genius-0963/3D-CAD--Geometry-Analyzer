import open3d as o3d
import numpy as np

def create_test_cube(output_file="test_cube.stl"):
    print("Creating a test cube...")
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"Test cube saved to {output_file}")

if __name__ == "__main__":
    create_test_cube()
