import open3d as o3d
print("Open3D version:", o3d.__version__)
print("Open3D test: Creating a simple point cloud...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
print("Point cloud created successfully!")
print("Number of points:", len(pcd.points))
