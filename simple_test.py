import open3d as o3d
print("Creating a simple cube...")
mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
print("Mesh created successfully")
print(f"Vertices: {len(mesh.vertices)}")
print(f"Triangles: {len(mesh.triangles)}")
