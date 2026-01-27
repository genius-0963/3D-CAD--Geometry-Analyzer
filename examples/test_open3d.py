# test_open3d.py
import open3d as o3d
import numpy as np

def test_open3d():
    print("Testing Open3D...")
    print(f"Open3D version: {o3d.__version__}")
    
    # Create a simple point cloud
    print("Creating a point cloud...")
    pcd = o3d.geometry.PointCloud()
    points = np.random.rand(100, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Created point cloud with {len(points)} points")
    
    # Try to visualize
    try:
        print("Trying to visualize...")
        o3d.visualization.draw_geometries([pcd])
        print("Visualization successful!")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    test_open3d()
    