"""
Base class for CAD file loaders.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

class BaseCADLoader(ABC):
    """Abstract base class for CAD file loaders."""
    
    def __init__(self, file_path: str):
        """Initialize the CAD loader with a file path.
        
        Args:
            file_path: Path to the CAD file
        """
        self.file_path = Path(file_path)
        self.mesh_data = None
        
    @classmethod
    @abstractmethod
    def supported_formats(cls) -> List[str]:
        """Return a list of supported file extensions.
        
        Returns:
            List of supported file extensions (e.g., ['.stl', '.obj'])
        """
        pass
        
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load and parse the CAD file.
        
        Returns:
            Dictionary containing the loaded mesh data
        """
        pass
        
    def validate_file(self) -> bool:
        """Validate that the file exists and has a supported extension.
        
        Returns:
            bool: True if the file is valid, False otherwise
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        if self.file_path.suffix.lower() not in self.supported_formats():
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats())}"
            )
        return True
