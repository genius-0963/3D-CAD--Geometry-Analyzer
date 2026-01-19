"""
Basic usage example for the 3D CAD Geometry Analyzer.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_analyzer.core.analyzer import CADAnalyzer

def main():
    # Initialize the analyzer with default configuration
    analyzer = CADAnalyzer()
    
    # Path to the example STL file (you'll need to provide your own STL file)
    example_file = "path/to/your/model.stl"
    
    if not os.path.exists(example_file):
        print(f"Error: Example file not found at {example_file}")
        print("Please provide a valid path to an STL file.")
        return
    
    try:
        # Load the CAD file
        print(f"Loading {example_file}...")
        analyzer.load_file(example_file)
        
        # Run the analysis
        print("Analyzing model...")
        analysis_results = analyzer.analyze()
        
        # Get a summary of the analysis
        summary = analyzer.get_summary()
        
        # Print the summary
        print("\n=== Analysis Summary ===")
        print(f"Overall Status: {summary['overall_manufacturability']['status'].upper()}")
        print(f"Manufacturability Score: {summary['overall_manufacturability']['score']:.2f}/1.00")
        
        print("\n=== Wall Thickness ===")
        wall = summary['wall_thickness']
        print(f"Minimum: {wall['min']:.4f} units")
        print(f"Average: {wall['avg']:.4f} units")
        print(f"Maximum: {wall['max']:.4f} units")
        print(f"Thin regions: {wall['thin_regions_count']}")
        print(f"Status: {'PASS' if wall['is_acceptable'] else 'FAIL'}")
        
        print("\n=== Undercuts ===")
        undercuts = summary['undercuts']
        print(f"Number of undercut faces: {undercuts['count']}")
        print(f"Maximum severity: {undercuts['max_severity']:.2f}")
        print(f"Status: {'PASS' if undercuts['is_acceptable'] else 'FAIL'}")
        
        # Print any issues found
        if summary['overall_manufacturability']['issues']:
            print("\n=== Issues Found ===")
            for i, issue in enumerate(summary['overall_manufacturability']['issues'], 1):
                print(f"{i}. {issue['message']} (Severity: {issue['severity']:.2f})")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
