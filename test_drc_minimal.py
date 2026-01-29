#!/usr/bin/env python3
"""
Minimal test for DRC without complex mesh operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cad_analyzer.manufacturing.drc import (
    DRCEngine,
    MinWallThicknessRule,
    OverhangAngleRule,
    HoleToleranceRule,
    Violation,
    ViolationSeverity,
    ViolationType
)

class MockMesh:
    """Mock mesh class for testing DRC rules without Open3D."""
    
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.area = 6.0  # Simple cube surface area
        self.face_normals = self._compute_face_normals()
        self.edges_unique = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])  # Mock edges
        self.edges_face = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])  # Mock edge faces
        self.is_watertight = True
        self.bounding_box = MockBoundingBox()
        
    def _compute_face_normals(self):
        """Compute simple face normals."""
        # Return mock normals for a cube
        return np.array([
            [0, 0, -1], [0, 0, -1],  # bottom
            [0, 0, 1], [0, 0, 1],    # top
            [0, -1, 0], [0, -1, 0],  # front
            [0, 1, 0], [0, 1, 0],    # back
            [-1, 0, 0], [-1, 0, 0],  # left
            [1, 0, 0], [1, 0, 0]     # right
        ])
    
    def sample_surface(self, num_samples, return_index=False):
        """Mock surface sampling."""
        points = np.random.rand(num_samples, 3)
        if return_index:
            indices = np.random.randint(0, len(self.faces), num_samples)
            return points, indices
        return points
    
    @property
    def ray(self):
        """Mock ray casting."""
        return MockRay()

class MockRay:
    """Mock ray casting."""
    
    def intersects_location(self, ray_origins, ray_directions, multiple_hits=False):
        """Mock ray intersection."""
        # Return empty intersection to avoid complex calculations
        return (np.array([]), np.array([]))

class MockBoundingBox:
    """Mock bounding box."""
    
    def __init__(self):
        self.center = np.array([0.5, 0.5, 0.5])
        self.min_bound = np.array([0, 0, 0])
        self.max_bound = np.array([1, 1, 1])
        self.extents = np.array([1, 1, 1])

def test_drc_minimal():
    """Test DRC with minimal mock mesh."""
    print("Testing DRC with minimal mock mesh...")
    
    try:
        # Create mock mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
        ])
        
        mesh = MockMesh(vertices, faces)
        print(f"✓ Mock mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Create DRC engine
        engine = DRCEngine()
        
        # Add rules
        engine.register_rule(MinWallThicknessRule(min_thickness=0.1))
        engine.register_rule(OverhangAngleRule(max_angle=90))
        engine.register_rule(HoleToleranceRule(min_diameter=0.1))
        
        print("✓ DRC engine initialized with rules")
        
        # Run checks
        results = engine.run_checks(mesh)
        
        print("✓ DRC checks completed")
        print(f"  Rules checked: {len(results)}")
        
        for rule_name, violations in results.items():
            print(f"  {rule_name}: {len(violations)} violations")
            for violation in violations[:2]:
                print(f"    - {violation.message}")
        
        # Generate summary
        summary = engine.get_summary(results)
        print(f"✓ Summary: {summary['total_violations']} total violations")
        
        # Test report generation
        report = engine.generate_report(results)
        print(f"✓ Report generated with {len(report)} sections")
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal DRC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Minimal DRC Test")
    print("=" * 50)
    
    success = test_drc_minimal()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Minimal DRC test passed!")
    else:
        print("✗ Minimal DRC test failed!")
    print("=" * 50)
