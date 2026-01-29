"""
Visualization module for DRC violations.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import open3d as o3d

from .violation import Violation, ViolationSeverity

class DRCVisualizer:
    """Visualizer for DRC violations on 3D meshes."""
    
    # Color mapping for violation severities
    SEVERITY_COLORS = {
        ViolationSeverity.INFO: (0.0, 1.0, 0.0),      # Green
        ViolationSeverity.WARNING: (1.0, 0.5, 0.0),   # Orange
        ViolationSeverity.ERROR: (1.0, 0.0, 0.0),     # Red
        ViolationSeverity.CRITICAL: (0.5, 0.0, 0.5),   # Purple
    }
    
    def __init__(self):
        """Initialize the visualizer."""
        self.mesh = None
        self.violations = {}
    
    def load_mesh(self, vertices: np.ndarray, faces: np.ndarray):
        """Load mesh data for visualization.
        
        Args:
            vertices: Array of vertex positions (Nx3)
            faces: Array of face indices (Mx3)
        """
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh.compute_vertex_normals()
    
    def set_violations(self, violations: Dict[str, List[Violation]]):
        """Set the violations to visualize.
        
        Args:
            violations: Dictionary mapping rule names to violation lists
        """
        self.violations = violations
    
    def create_color_map(self) -> np.ndarray:
        """Create a color map for the mesh based on violations.
        
        Returns:
            Array of RGB colors for each vertex
        """
        if not self.mesh:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        
        num_vertices = len(self.mesh.vertices)
        colors = np.zeros((num_vertices, 3))
        
        # Default color: light gray
        colors[:] = (0.8, 0.8, 0.8)
        
        # Color vertices based on nearby violations
        for rule_name, violation_list in self.violations.items():
            for violation in violation_list:
                color = self.SEVERITY_COLORS.get(violation.severity, (1.0, 0.0, 0.0))
                
                # Find vertices near the violation location
                if isinstance(violation.location, list) and len(violation.location) == 3:
                    # Single point violation
                    self._color_nearby_vertices(colors, violation.location, color)
                elif isinstance(violation.location, list) and len(violation.location) > 0:
                    # Multiple points violation
                    for point in violation.location:
                        self._color_nearby_vertices(colors, point, color)
        
        return colors
    
    def _color_nearby_vertices(self, colors: np.ndarray, location: List[float], 
                             color: Tuple[float, float, float], radius: float = 0.1):
        """Color vertices near a specific location.
        
        Args:
            colors: Array to modify
            location: 3D point to color around
            color: RGB color to apply
            radius: Radius around location to color
        """
        vertices = np.asarray(self.mesh.vertices)
        distances = np.linalg.norm(vertices - np.array(location), axis=1)
        
        # Color vertices within radius
        mask = distances < radius
        colors[mask] = color
    
    def visualize_3d(self, show_frame: bool = True) -> o3d.geometry.TriangleMesh:
        """Create a 3D visualization with colored violations.
        
        Args:
            show_frame: Whether to show coordinate frame
            
        Returns:
            Open3D mesh with violation colors
        """
        if not self.mesh:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        
        # Apply colors to mesh
        colors = self.create_color_map()
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        return self.mesh
    
    def show_interactive(self, window_name: str = "DRC Violations"):
        """Show interactive 3D visualization.
        
        Args:
            window_name: Name of the visualization window
        """
        colored_mesh = self.visualize_3d()
        
        # Create coordinate frame if requested
        geometries = [colored_mesh]
        if show_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            geometries.append(coordinate_frame)
        
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
    
    def create_2d_projection(self, projection_axis: str = 'z') -> plt.Figure:
        """Create a 2D projection plot of violations.
        
        Args:
            projection_axis: Axis to project onto ('x', 'y', or 'z')
            
        Returns:
            Matplotlib figure
        """
        if not self.mesh:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Get vertices and project to 2D
        vertices = np.asarray(self.mesh.vertices)
        if projection_axis.lower() == 'x':
            coords = vertices[:, [1, 2]]  # Y-Z projection
            xlabel, ylabel = 'Y', 'Z'
        elif projection_axis.lower() == 'y':
            coords = vertices[:, [0, 2]]  # X-Z projection
            xlabel, ylabel = 'X', 'Z'
        else:  # 'z'
            coords = vertices[:, [0, 1]]  # X-Y projection
            xlabel, ylabel = 'X', 'Y'
        
        # Plot mesh outline (simplified)
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.5)
        
        # Plot violations
        for rule_name, violation_list in self.violations.items():
            for violation in violation_list:
                color = self.SEVERITY_COLORS.get(violation.severity, (1.0, 0.0, 0.0))
                
                if isinstance(violation.location, list) and len(violation.location) == 3:
                    point = np.array(violation.location)
                    if projection_axis.lower() == 'x':
                        coord = [point[1], point[2]]
                    elif projection_axis.lower() == 'y':
                        coord = [point[0], point[2]]
                    else:
                        coord = [point[0], point[1]]
                    
                    ax.scatter(coord[0], coord[1], c=[color], s=50, marker='o', 
                             edgecolors='black', linewidth=1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'DRC Violations - {projection_axis.upper()} Projection')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = []
        for severity, color in self.SEVERITY_COLORS.items():
            legend_elements.append(plt.scatter([], [], c=[color], s=50, 
                                              label=severity.name))
        ax.legend(handles=legend_elements)
        
        return fig
    
    def save_violation_heatmap(self, filename: str, projection_axis: str = 'z'):
        """Save a 2D heatmap of violations to file.
        
        Args:
            filename: Output filename
            projection_axis: Axis to project onto
        """
        fig = self.create_2d_projection(projection_axis)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def create_summary_plot(self) -> plt.Figure:
        """Create a summary plot of violation statistics.
        
        Returns:
            Matplotlib figure with violation summary
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Violations by rule
        rule_counts = {rule: len(violations) for rule, violations in self.violations.items()}
        rules = list(rule_counts.keys())
        counts = list(rule_counts.values())
        
        ax1.bar(rules, counts)
        ax1.set_title('Violations by Rule')
        ax1.set_ylabel('Number of Violations')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Violations by severity
        severity_counts = {}
        for violations in self.violations.values():
            for violation in violations:
                severity = violation.severity.name
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            severities = list(severity_counts.keys())
            sev_counts = list(severity_counts.values())
            colors = [self.SEVERITY_COLORS[ViolationSeverity[s]] for s in severities]
            
            ax2.pie(sev_counts, labels=severities, colors=colors, autopct='%1.1f%%')
            ax2.set_title('Violations by Severity')
        else:
            ax2.text(0.5, 0.5, 'No Violations', ha='center', va='center')
            ax2.set_title('Violations by Severity')
        
        # 3. Severity distribution by rule
        rule_severity_data = {}
        for rule, violations in self.violations.items():
            rule_severity_data[rule] = {}
            for violation in violations:
                severity = violation.severity.name
                rule_severity_data[rule][severity] = rule_severity_data[rule].get(severity, 0) + 1
        
        # Create stacked bar chart
        all_severities = set()
        for data in rule_severity_data.values():
            all_severities.update(data.keys())
        
        if all_severities:
            bottom = np.zeros(len(rules))
            for severity in sorted(all_severities):
                counts = [rule_severity_data.get(rule, {}).get(severity, 0) for rule in rules]
                color = self.SEVERITY_COLORS[ViolationSeverity[severity]]
                ax3.bar(rules, counts, bottom=bottom, label=severity, color=color)
                bottom += counts
            
            ax3.set_title('Severity Distribution by Rule')
            ax3.set_ylabel('Number of Violations')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Summary statistics
        total_violations = sum(len(v) for v in self.violations.values())
        total_rules = len(self.violations)
        rules_with_violations = sum(1 for v in self.violations.values() if v)
        
        stats_text = f"""
        Total Violations: {total_violations}
        Total Rules Checked: {total_rules}
        Rules with Violations: {rules_with_violations}
        
        Most Critical Rule: {max(rule_counts.items(), key=lambda x: x[1])[0] if rule_counts else 'N/A'}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
