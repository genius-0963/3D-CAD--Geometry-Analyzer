import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports one by one
print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")

try:
    import open3d as o3d
    print("✓ open3d imported")
except Exception as e:
    print(f"✗ open3d failed: {e}")

try:
    from cad_analyzer.file_handling import STLLoader
    print("✓ STLLoader imported")
except Exception as e:
    print(f"✗ STLLoader failed: {e}")

try:
    from cad_analyzer.core.analyzer import CADAnalyzer
    print("✓ CADAnalyzer imported")
except Exception as e:
    print(f"✗ CADAnalyzer failed: {e}")

# Test loading STL file
print("\nTesting STL loading...")
try:
    loader = STLLoader("examples/test_cube.stl")
    mesh_data = loader.load()
    print("✓ STL loaded successfully")
    print(f"  Vertices: {len(mesh_data.get('vertices', []))}")
    print(f"  Triangles: {len(mesh_data.get('triangles', []))}")
except Exception as e:
    print(f"✗ STL loading failed: {e}")

# Test analyzer initialization
print("\nTesting analyzer...")
try:
    analyzer = CADAnalyzer()
    print("✓ Analyzer initialized")
except Exception as e:
    print(f"✗ Analyzer initialization failed: {e}")

# Test file loading in analyzer
print("\nTesting analyzer file loading...")
try:
    analyzer = CADAnalyzer()
    analyzer.load_file("examples/test_cube.stl")
    print("✓ File loaded in analyzer")
except Exception as e:
    print(f"✗ Analyzer file loading failed: {e}")

print("\nTesting analysis step-by-step...")
try:
    analyzer = CADAnalyzer()
    analyzer.load_file("examples/test_cube.stl")
    
    # Try to access the mesh
    if hasattr(analyzer, 'mesh') and analyzer.mesh:
        print("✓ Mesh accessible")
    else:
        print("✗ Mesh not accessible")
        
    # Try to run analysis
    print("Attempting analysis...")
    results = analyzer.analyze()
    print("✓ Analysis completed")
    
except Exception as e:
    print(f"✗ Analysis failed: {e}")
    import traceback
    traceback.print_exc()
