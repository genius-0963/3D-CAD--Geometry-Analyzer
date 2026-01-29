"""
Comprehensive unit tests for the DRC module.
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_analyzer.manufacturing.drc import (
    DRCEngine,
    MinWallThicknessRule,
    OverhangAngleRule,
    HoleToleranceRule,
    Violation,
    ViolationSeverity,
    ViolationType,
    get_process_preset,
    ProcessPreset,
    create_engine_for_process,
    compare_processes,
)

class MockMesh:
    """Mock mesh for testing."""
    
    def __init__(self, vertices=None, faces=None):
        if vertices is None:
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
        if faces is None:
            faces = np.array([
                [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
                [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
            ])
        
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

class TestDRCEngine(unittest.TestCase):
    """Test cases for the DRC engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = DRCEngine()
        self.mesh = MockMesh()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertEqual(len(self.engine.rules), 0)
    
    def test_rule_registration(self):
        """Test rule registration."""
        rule = MinWallThicknessRule(min_thickness=1.0)
        self.engine.register_rule(rule)
        self.assertEqual(len(self.engine.rules), 1)
    
    def test_multiple_rule_registration(self):
        """Test registering multiple rules."""
        rules = [
            MinWallThicknessRule(min_thickness=1.0),
            OverhangAngleRule(max_angle=45),
            HoleToleranceRule(min_diameter=0.5)
        ]
        
        for rule in rules:
            self.engine.register_rule(rule)
        
        self.assertEqual(len(self.engine.rules), 3)
    
    def test_rule_clearing(self):
        """Test clearing all rules."""
        self.engine.register_rule(MinWallThicknessRule())
        self.engine.register_rule(OverhangAngleRule())
        self.assertEqual(len(self.engine.rules), 2)
        
        self.engine.clear_rules()
        self.assertEqual(len(self.engine.rules), 0)
    
    def test_run_checks_empty_engine(self):
        """Test running checks with no rules."""
        results = self.engine.run_checks(self.mesh)
        self.assertEqual(len(results), 0)
    
    def test_run_checks_with_rules(self):
        """Test running checks with rules."""
        self.engine.register_rule(MinWallThicknessRule(min_thickness=0.1))
        self.engine.register_rule(OverhangAngleRule(max_angle=90))
        
        results = self.engine.run_checks(self.mesh)
        self.assertEqual(len(results), 2)
        
        # Check that all rules have results
        self.assertIn('MinWallThicknessRule', results)
        self.assertIn('OverhangAngleRule', results)
    
    def test_get_summary(self):
        """Test summary generation."""
        self.engine.register_rule(MinWallThicknessRule(min_thickness=0.1))
        results = self.engine.run_checks(self.mesh)
        
        summary = self.engine.get_summary(results)
        
        self.assertIn('total_violations', summary)
        self.assertIn('severity_counts', summary)
        self.assertIn('type_counts', summary)
        self.assertIn('rules_checked', summary)
        self.assertIn('rules_with_violations', summary)
        
        self.assertEqual(summary['rules_checked'], 1)
    
    def test_generate_report_dict(self):
        """Test report generation in dict format."""
        self.engine.register_rule(MinWallThicknessRule(min_thickness=0.1))
        results = self.engine.run_checks(self.mesh)
        
        report = self.engine.generate_report(results, format='dict')
        
        self.assertIn('summary', report)
        self.assertIn('violations', report)
        self.assertIn('metadata', report)
    
    def test_generate_report_json(self):
        """Test report generation in JSON format."""
        self.engine.register_rule(MinWallThicknessRule(min_thickness=0.1))
        results = self.engine.run_checks(self.mesh)
        
        report = self.engine.generate_report(results, format='json')
        
        self.assertIsInstance(report, str)
        # Should be valid JSON
        import json
        parsed = json.loads(report)
        self.assertIn('summary', parsed)

class TestDesignRules(unittest.TestCase):
    """Test cases for individual design rules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mesh = MockMesh()
    
    def test_min_wall_thickness_rule_initialization(self):
        """Test MinWallThicknessRule initialization."""
        rule = MinWallThicknessRule(min_thickness=1.5, severity=ViolationSeverity.ERROR)
        self.assertEqual(rule.min_thickness, 1.5)
        self.assertEqual(rule.severity, ViolationSeverity.ERROR)
    
    def test_min_wall_thickness_rule_check(self):
        """Test MinWallThicknessRule check."""
        rule = MinWallThicknessRule(min_thickness=0.1)
        violations = rule.check(self.mesh)
        
        self.assertIsInstance(violations, list)
        # Each violation should be a Violation object
        for violation in violations:
            self.assertIsInstance(violation, Violation)
            self.assertEqual(violation.violation_type, ViolationType.WALL_THICKNESS)
    
    def test_overhang_angle_rule_initialization(self):
        """Test OverhangAngleRule initialization."""
        rule = OverhangAngleRule(max_angle=45, severity=ViolationSeverity.WARNING)
        self.assertEqual(rule.max_angle, 45)
        self.assertEqual(rule.severity, ViolationSeverity.WARNING)
    
    def test_overhang_angle_rule_check(self):
        """Test OverhangAngleRule check."""
        rule = OverhangAngleRule(max_angle=90)
        violations = rule.check(self.mesh)
        
        self.assertIsInstance(violations, list)
        for violation in violations:
            self.assertIsInstance(violation, Violation)
            self.assertEqual(violation.violation_type, ViolationType.OVERHANG_ANGLE)
    
    def test_hole_tolerance_rule_initialization(self):
        """Test HoleToleranceRule initialization."""
        rule = HoleToleranceRule(min_diameter=1.0, severity=ViolationSeverity.ERROR)
        self.assertEqual(rule.min_diameter, 1.0)
        self.assertEqual(rule.severity, ViolationSeverity.ERROR)
    
    def test_hole_tolerance_rule_check(self):
        """Test HoleToleranceRule check."""
        rule = HoleToleranceRule(min_diameter=0.1)
        violations = rule.check(self.mesh)
        
        self.assertIsInstance(violations, list)
        for violation in violations:
            self.assertIsInstance(violation, Violation)
            self.assertEqual(violation.violation_type, ViolationType.HOLE_TOLERANCE)

class TestProcessPresets(unittest.TestCase):
    """Test cases for process presets."""
    
    def test_get_process_preset(self):
        """Test getting process presets."""
        fdm_config = get_process_preset(ProcessPreset.FDM)
        
        self.assertEqual(fdm_config.name, "FDM 3D Printing")
        self.assertGreater(len(fdm_config.rules), 0)
        self.assertIsInstance(fdm_config.metadata, dict)
    
    def test_all_process_presets(self):
        """Test all process presets are accessible."""
        presets = [
            ProcessPreset.CNC,
            ProcessPreset.CNC_PRECISION,
            ProcessPreset.FDM,
            ProcessPreset.FDM_FINE,
            ProcessPreset.SLA,
            ProcessPreset.SLS,
            ProcessPreset.INJECTION_MOLDING
        ]
        
        for preset in presets:
            config = get_process_preset(preset)
            self.assertIsNotNone(config)
            self.assertGreater(len(config.rules), 0)
    
    def test_create_engine_for_process(self):
        """Test creating engines for processes."""
        fdm_engine = create_engine_for_process(ProcessPreset.FDM)
        
        self.assertIsInstance(fdm_engine, DRCEngine)
        self.assertGreater(len(fdm_engine.rules), 0)
    
    def test_process_comparison(self):
        """Test process comparison."""
        processes = [ProcessPreset.FDM, ProcessPreset.SLA, ProcessPreset.CNC]
        comparison = compare_processes(processes)
        
        self.assertIn('processes', comparison)
        self.assertIn('recommendations', comparison)
        self.assertEqual(len(comparison['processes']), 3)

class TestViolation(unittest.TestCase):
    """Test cases for Violation class."""
    
    def test_violation_creation(self):
        """Test violation creation."""
        violation = Violation(
            violation_type=ViolationType.WALL_THICKNESS,
            severity=ViolationSeverity.ERROR,
            location=[1.0, 2.0, 3.0],
            message="Test violation",
            data={"test": "data"}
        )
        
        self.assertEqual(violation.violation_type, ViolationType.WALL_THICKNESS)
        self.assertEqual(violation.severity, ViolationSeverity.ERROR)
        self.assertEqual(violation.location, [1.0, 2.0, 3.0])
        self.assertEqual(violation.message, "Test violation")
        self.assertEqual(violation.data["test"], "data")
    
    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = Violation(
            violation_type=ViolationType.WALL_THICKNESS,
            severity=ViolationSeverity.ERROR,
            location=[1.0, 2.0, 3.0],
            message="Test violation"
        )
        
        violation_dict = violation.to_dict()
        
        self.assertEqual(violation_dict['type'], 'wall_thickness')
        self.assertEqual(violation_dict['severity'], 'ERROR')
        self.assertEqual(violation_dict['location'], [1.0, 2.0, 3.0])
        self.assertEqual(violation_dict['message'], "Test violation")
    
    def test_violation_from_dict(self):
        """Test violation deserialization."""
        data = {
            'type': 'wall_thickness',
            'severity': 'ERROR',
            'location': [1.0, 2.0, 3.0],
            'message': "Test violation",
            'data': {"test": "data"}
        }
        
        violation = Violation.from_dict(data)
        
        self.assertEqual(violation.violation_type, ViolationType.WALL_THICKNESS)
        self.assertEqual(violation.severity, ViolationSeverity.ERROR)
        self.assertEqual(violation.location, [1.0, 2.0, 3.0])
        self.assertEqual(violation.message, "Test violation")
        self.assertEqual(violation.data["test"], "data")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete DRC system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mesh = MockMesh()
    
    def test_complete_workflow(self):
        """Test complete DRC workflow."""
        # Create engine for FDM process
        engine = create_engine_for_process(ProcessPreset.FDM)
        
        # Run checks
        results = engine.run_checks(self.mesh)
        
        # Generate summary
        summary = engine.get_summary(results)
        
        # Generate report
        report = engine.generate_report(results)
        
        # Verify all components work together
        self.assertIsInstance(results, dict)
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('violations', report)
    
    def test_multiple_process_comparison(self):
        """Test comparing multiple processes on the same mesh."""
        processes = [ProcessPreset.FDM, ProcessPreset.SLA, ProcessPreset.CNC]
        process_results = {}
        
        for process in processes:
            engine = create_engine_for_process(process)
            results = engine.run_checks(self.mesh)
            process_results[process.value] = engine.get_summary(results)
        
        # Verify we have results for all processes
        self.assertEqual(len(process_results), 3)
        
        # Verify each process has summary data
        for process_name, summary in process_results.items():
            self.assertIn('total_violations', summary)
            self.assertIn('rules_checked', summary)
    
    def test_custom_rule_configuration(self):
        """Test custom rule configuration."""
        engine = DRCEngine()
        
        # Add custom rules with specific parameters
        engine.register_rule(MinWallThicknessRule(
            min_thickness=0.5,
            severity=ViolationSeverity.WARNING
        ))
        engine.register_rule(OverhangAngleRule(
            max_angle=30,
            severity=ViolationSeverity.ERROR
        ))
        
        # Verify rules are configured correctly
        self.assertEqual(len(engine.rules), 2)
        
        wall_rule = engine.rules[0]
        self.assertEqual(wall_rule.min_thickness, 0.5)
        self.assertEqual(wall_rule.severity, ViolationSeverity.WARNING)
        
        overhang_rule = engine.rules[1]
        self.assertEqual(overhang_rule.max_angle, 30)
        self.assertEqual(overhang_rule.severity, ViolationSeverity.ERROR)

if __name__ == '__main__':
    unittest.main()
