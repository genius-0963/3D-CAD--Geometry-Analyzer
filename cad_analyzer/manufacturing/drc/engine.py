"""
Design Rule Checking (DRC) engine for validating 3D models against manufacturing constraints.
"""
from typing import Dict, List, Optional, Type, Any
import numpy as np

from .violation import Violation, ViolationSeverity, ViolationType
from ...geometry.mesh import Mesh

class DRCEngine:
    """
    Engine for running design rule checks on 3D models.
    
    This class manages a collection of design rules and coordinates the validation process.
    """
    
    def __init__(self):
        """Initialize the DRC engine with an empty set of rules."""
        self.rules = []
    
    def register_rule(self, rule: 'DesignRule') -> None:
        """Register a design rule with the engine.
        
        Args:
            rule: The design rule to register.
        """
        self.rules.append(rule)
    
    def unregister_rule(self, rule_type: Type['DesignRule']) -> bool:
        """Unregister a design rule from the engine.
        
        Args:
            rule_type: The type of the rule to unregister.
            
        Returns:
            bool: True if the rule was found and removed, False otherwise.
        """
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if not isinstance(r, rule_type)]
        return len(self.rules) < initial_count
    
    def clear_rules(self) -> None:
        """Remove all registered rules from the engine."""
        self.rules = []
    
    def run_checks(self, mesh: 'Mesh', **kwargs) -> Dict[str, List[Violation]]:
        """Run all registered design rule checks on the given mesh.
        
        Args:
            mesh: The 3D mesh to validate.
            **kwargs: Additional arguments to pass to the rule checks.
            
        Returns:
            Dict mapping rule names to lists of violations.
        """
        results = {}
        
        for rule in self.rules:
            rule_name = rule.__class__.__name__
            try:
                violations = rule.check(mesh, **kwargs)
                results[rule_name] = violations
            except Exception as e:
                # Create a violation for the error
                error_violation = Violation(
                    violation_type=ViolationType.OTHER,
                    severity=ViolationSeverity.ERROR,
                    location=(0, 0, 0),  # Default location
                    message=f"Error executing rule {rule_name}: {str(e)}",
                    data={"error": str(e), "rule": rule_name}
                )
                results[rule_name] = [error_violation]
        
        return results
    
    def get_summary(self, results: Dict[str, List[Violation]]) -> Dict[str, Any]:
        """Generate a summary of the validation results.
        
        Args:
            results: The results from run_checks()
            
        Returns:
            A dictionary containing summary statistics.
        """
        total_violations = sum(len(v) for v in results.values())
        
        severity_counts = {severity.name: 0 for severity in ViolationSeverity}
        type_counts = {vtype.value: 0 for vtype in ViolationType}
        
        for violations in results.values():
            for violation in violations:
                severity_counts[violation.severity.name] += 1
                type_counts[violation.violation_type.value] += 1
        
        return {
            "total_violations": total_violations,
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "rules_checked": len(results),
            "rules_with_violations": sum(1 for v in results.values() if v),
        }
    
    def generate_report(self, results: Dict[str, List[Violation]], format: str = 'dict') -> Any:
        """Generate a report of the validation results.
        
        Args:
            results: The results from run_checks()
            format: The output format ('dict' or 'json')
            
        Returns:
            The report in the requested format.
        """
        summary = self.get_summary(results)
        
        # Convert violations to serializable format
        serialized_results = {
            rule_name: [v.to_dict() for v in violations]
            for rule_name, violations in results.items()
        }
        
        report = {
            "summary": summary,
            "violations": serialized_results,
            "metadata": {
                "engine_version": "1.0.0",
            }
        }
        
        if format.lower() == 'json':
            import json
            return json.dumps(report, indent=2)
        
        return report
