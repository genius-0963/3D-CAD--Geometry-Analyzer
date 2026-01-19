""
File handling module for CAD file operations.

This module provides base classes and implementations for loading and processing
various CAD file formats.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

class BaseCADLoader(ABC):
    """Abstract base class for CAD file loaders."""
    
    def __init__(self, file_path: str):
        """Initialize the CAD loader with a file path.
        
        Args:
            file_path: Path to the CAD file
        """
        self.file_path = Path(file_path)
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate that the file exists and has the correct extension."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.suffix.lower() in self.supported_formats():
            raise ValueError(
                f"Unsupported file format. Supported formats: {', '.join(self.supported_formats())}"
            )
    
    @classmethod
    @abstractmethod
    def supported_formats(cls) -> list:
        """Return a list of supported file extensions (including .)."""
        pass
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load and parse the CAD file.
        
        Returns:
            Dictionary containing the parsed CAD data
        """
        pass
    
    @abstractmethod
    def to_mesh(self) -> 'Mesh':
        """Convert the CAD data to a mesh representation.
        
        Returns:
            Mesh object containing the geometry data
        """
        pass
