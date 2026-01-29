#!/usr/bin/env python3
"""
Safe test script for the DRC module with basic mesh.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cad_analyzer.geometry.mesh import Mesh
from cad_analyzer.manufacturing.drc import (
    DRCEngine,
    MinWallThicknessRule,
    OverhangAngleRule,
    HoleToleranceRule
)

def create_simple_mesh():
    """Create a simple test mesh."""
    # Create a simple cube
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2]   # right
    ])
    
    return Mesh(vertices, faces)

def test_drc_with_simple_mesh():
    """Test DRC with a simple mesh."""
    print("Testing DRC with simple mesh...")
    
    try:
        # Create simple mesh
        mesh = create_simple_mesh()
        print(f"✓ Simple mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Create DRC engine
        engine = DRCEngine()
        
        # Add rules with conservative settings
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
            for violation in violations[:2]:  # Show first 2 violations
                print(f"    - {violation.message}")
        
        # Generate summary
        summary = engine.get_summary(results)
        print(f"✓ Summary: {summary['total_violations']} total violations")
        
        return True
        
    except Exception as e:
        print(f"✗ DRC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rule_registration():
    """Test rule registration and management."""
    print("\nTesting rule registration...")
    
    try:
        engine = DRCEngine()
        
        # Test adding rules
        rule1 = MinWallThicknessRule(min_thickness=1.0)
        rule2 = OverhangAngleRule(max_angle=45)
        
        engine.register_rule(rule1)
        engine.register_rule(rule2)
        
        print(f"✓ Rules registered: {len(engine.rules)}")
        
        # Test clearing rules
        engine.clear_rules()
        print(f"✓ Rules cleared: {len(engine.rules)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Rule registration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Safe DRC Test")
    print("=" * 50)
    
    success1 = test_rule_registration()
    success2 = test_drc_with_simple_mesh()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ All safe DRC tests passed!")
    else:
        print("✗ Some safe DRC tests failed!")
    print("=" * 50)
