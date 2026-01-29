#!/usr/bin/env python3
"""
Simple test script for the DRC module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if DRC modules can be imported."""
    print("Testing DRC imports...")
    
    try:
        from cad_analyzer.manufacturing.drc import DRCEngine
        print("✓ DRCEngine imported")
    except Exception as e:
        print(f"✗ DRCEngine import failed: {e}")
        return False
    
    try:
        from cad_analyzer.manufacturing.drc import MinWallThicknessRule
        print("✓ MinWallThicknessRule imported")
    except Exception as e:
        print(f"✗ MinWallThicknessRule import failed: {e}")
        return False
    
    try:
        from cad_analyzer.manufacturing.drc import OverhangAngleRule
        print("✓ OverhangAngleRule imported")
    except Exception as e:
        print(f"✗ OverhangAngleRule import failed: {e}")
        return False
    
    try:
        from cad_analyzer.manufacturing.drc import HoleToleranceRule
        print("✓ HoleToleranceRule imported")
    except Exception as e:
        print(f"✗ HoleToleranceRule import failed: {e}")
        return False
    
    return True

def test_engine_creation():
    """Test if DRC engine can be created."""
    print("\nTesting DRC engine creation...")
    
    try:
        from cad_analyzer.manufacturing.drc import DRCEngine
        engine = DRCEngine()
        print("✓ DRC engine created")
        return True
    except Exception as e:
        print(f"✗ DRC engine creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple DRC Test")
    print("=" * 50)
    
    success1 = test_imports()
    success2 = test_engine_creation()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ Simple DRC tests passed!")
    else:
        print("✗ Some simple DRC tests failed!")
    print("=" * 50)
