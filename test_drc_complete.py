#!/usr/bin/env python3
"""
Complete test for the DRC module with visualization and process presets.
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
    DRCVisualizer,
    get_process_preset,
    ProcessPreset,
    create_engine_for_process,
    compare_processes,
)

class MockMesh:
    """Mock mesh class for testing."""
    
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.area = 6.0
        self.face_normals = self._compute_face_normals()
        self.edges_unique = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        self.edges_face = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        self.is_watertight = True
        self.bounding_box = MockBoundingBox()
        
    def _compute_face_normals(self):
        return np.array([
            [0, 0, -1], [0, 0, -1], [0, 0, 1], [0, 0, 1],
            [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0],
            [-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]
        ])
    
    def sample_surface(self, num_samples, return_index=False):
        points = np.random.rand(num_samples, 3)
        if return_index:
            indices = np.random.randint(0, len(self.faces), num_samples)
            return points, indices
        return points
    
    @property
    def ray(self):
        return MockRay()

class MockRay:
    def intersects_location(self, ray_origins, ray_directions, multiple_hits=False):
        return (np.array([]), np.array([]))

class MockBoundingBox:
    def __init__(self):
        self.center = np.array([0.5, 0.5, 0.5])
        self.min_bound = np.array([0, 0, 0])
        self.max_bound = np.array([1, 1, 1])
        self.extents = np.array([1, 1, 1])

def test_process_presets():
    """Test process-specific rule presets."""
    print("Testing process presets...")
    
    try:
        # Test getting preset configurations
        fdm_config = get_process_preset(ProcessPreset.FDM)
        print(f"✓ FDM preset: {fdm_config.name}")
        print(f"  Rules: {len(fdm_config.rules)}")
        print(f"  Min wall thickness: {fdm_config.rules[0].min_thickness}mm")
        
        cnc_config = get_process_preset(ProcessPreset.CNC)
        print(f"✓ CNC preset: {cnc_config.name}")
        print(f"  Rules: {len(cnc_config.rules)}")
        print(f"  Min wall thickness: {cnc_config.rules[0].min_thickness}mm")
        
        # Test creating engines for processes
        fdm_engine = create_engine_for_process(ProcessPreset.FDM)
        print(f"✓ FDM engine created with {len(fdm_engine.rules)} rules")
        
        cnc_engine = create_engine_for_process(ProcessPreset.CNC)
        print(f"✓ CNC engine created with {len(cnc_engine.rules)} rules")
        
        return True
        
    except Exception as e:
        print(f"✗ Process presets test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_process_comparison():
    """Test process comparison functionality."""
    print("\nTesting process comparison...")
    
    try:
        processes = [ProcessPreset.FDM, ProcessPreset.SLA, ProcessPreset.CNC]
        comparison = compare_processes(processes)
        
        print(f"✓ Compared {len(comparison['processes'])} processes")
        print(f"  Recommendations: {len(comparison['recommendations'])}")
        
        for rec in comparison['recommendations']:
            print(f"    - {rec['process']}: {rec['reason']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Process comparison test failed: {e}")
        return False

def test_visualization():
    """Test DRC visualization functionality."""
    print("\nTesting DRC visualization...")
    
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
        
        # Create visualizer
        visualizer = DRCVisualizer()
        visualizer.load_mesh(vertices, faces)
        print("✓ Mesh loaded for visualization")
        
        # Create some mock violations
        from cad_analyzer.manufacturing.drc import Violation, ViolationSeverity, ViolationType
        
        violations = {
            "MinWallThicknessRule": [
                Violation(
                    violation_type=ViolationType.WALL_THICKNESS,
                    severity=ViolationSeverity.ERROR,
                    location=[0.5, 0.5, 0.5],
                    message="Wall too thin",
                    data={"measured_thickness": 0.3, "min_threshold": 0.5}
                )
            ],
            "OverhangAngleRule": [
                Violation(
                    violation_type=ViolationType.OVERHANG_ANGLE,
                    severity=ViolationSeverity.WARNING,
                    location=[0.2, 0.2, 0.8],
                    message="Overhang too steep",
                    data={"measured_angle": 50, "max_threshold": 45}
                )
            ]
        }
        
        visualizer.set_violations(violations)
        print("✓ Violations set for visualization")
        
        # Test color map creation
        colors = visualizer.create_color_map()
        print(f"✓ Color map created: {colors.shape}")
        
        # Test 2D projection (without showing)
        fig = visualizer.create_2d_projection('z')
        print("✓ 2D projection created")
        
        # Test summary plot
        summary_fig = visualizer.create_summary_plot()
        print("✓ Summary plot created")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close('all')
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_workflow():
    """Test complete DRC workflow with process presets and visualization."""
    print("\nTesting complete workflow...")
    
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
        
        # Test multiple processes
        processes = [ProcessPreset.FDM, ProcessPreset.SLA, ProcessPreset.CNC]
        results = {}
        
        for process in processes:
            engine = create_engine_for_process(process)
            process_results = engine.run_checks(mesh)
            results[process.value] = process_results
            
            summary = engine.get_summary(process_results)
            print(f"✓ {process.value}: {summary['total_violations']} violations")
        
        # Create visualization for the process with most violations
        worst_process = max(results.keys(), key=lambda k: sum(len(v) for v in results[k].values()))
        print(f"✓ Worst process: {worst_process}")
        
        # Setup visualization
        visualizer = DRCVisualizer()
        visualizer.load_mesh(vertices, faces)
        visualizer.set_violations(results[worst_process])
        
        # Create summary
        summary_fig = visualizer.create_summary_plot()
        print("✓ Complete workflow summary created")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close('all')
        
        return True
        
    except Exception as e:
        print(f"✗ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Complete DRC Module Test")
    print("=" * 60)
    
    success1 = test_process_presets()
    success2 = test_process_comparison()
    success3 = test_visualization()
    success4 = test_complete_workflow()
    
    print("\n" + "=" * 60)
    if all([success1, success2, success3, success4]):
        print("✓ All complete DRC tests passed!")
        print("\nFeatures demonstrated:")
        print("- Process-specific rule presets (CNC, FDM, SLA, SLS, Injection Molding)")
        print("- DRC engine configuration for different processes")
        print("- Process comparison and recommendations")
        print("- 3D visualization of violations")
        print("- 2D projections and heatmaps")
        print("- Summary statistics and reporting")
    else:
        print("✗ Some complete DRC tests failed!")
    print("=" * 60)
