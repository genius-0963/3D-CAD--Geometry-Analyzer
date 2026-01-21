"""
Basic usage example for the 3D CAD Geometry Analyzer.
"""
import os
import sys
import traceback
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    try:
        # Import here to catch import errors
        from cad_analyzer.core.analyzer import CADAnalyzer
        
        # Initialize the analyzer with default configuration
        analyzer = CADAnalyzer()
        
        # Path to the test cube STL file
        example_file = os.path.join(os.path.dirname(__file__), "test_cube.stl")
        
        if not os.path.exists(example_file):
            print(f"Error: Example file not found at {example_file}")
            print("Please provide a valid path to an STL file.")
            return
        
        try:
            # Load the CAD file
            print(f"Loading {example_file}...")
            analyzer.load_file(example_file)
            print("File loaded successfully.")
            
            # Run the analysis
            print("Analyzing model...")
            analysis_results = analyzer.analyze()
            print("Analysis completed successfully.")
            
            # Get a summary of the analysis
            summary = analyzer.get_summary()
            print("\nAnalysis Summary:")
            print("-----------------")
            for key, value in summary.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            print("\nStack trace:")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"Import error: {str(e)}")
        print("\nPlease make sure all dependencies are installed correctly.")
        print("Try running: pip install -r requirements.txt")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        
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
