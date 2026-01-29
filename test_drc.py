#!/usr/bin/env python3
"""
Test script for the Design Rule Checking (DRC) module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cad_analyzer.file_handling import STLLoader
from cad_analyzer.geometry.mesh import Mesh
from cad_analyzer.manufacturing.drc import (
    DRCEngine,
    MinWallThicknessRule,
    OverhangAngleRule,
    HoleToleranceRule
)

def test_drc_basic():
    """Test basic DRC functionality."""
    print("Testing DRC module...")
    
    try:
        # Load a test mesh
        loader = STLLoader("test_cube.stl")
        mesh_data = loader.load()
        mesh = Mesh(mesh_data['vertices'], mesh_data['triangles'])
        
        print(f"✓ Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Create DRC engine
        engine = DRCEngine()
        
        # Add rules
        engine.register_rule(MinWallThicknessRule(min_thickness=0.5))
        engine.register_rule(OverhangAngleRule(max_angle=60))
        engine.register_rule(HoleToleranceRule(min_diameter=0.8))
        
        print("✓ DRC engine initialized with rules")
        
        # Run checks
        results = engine.run_checks(mesh)
        
        print("✓ DRC checks completed")
        print(f"  Rules checked: {len(results)}")
        
        for rule_name, violations in results.items():
            print(f"  {rule_name}: {len(violations)} violations")
            for violation in violations[:3]:  # Show first 3 violations
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

if __name__ == "__main__":
    print("=" * 50)
    print("DRC Module Test")
    print("=" * 50)
    
    success = test_drc_basic()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ DRC test passed!")
    else:
        print("✗ DRC test failed!")
    print("=" * 50)
