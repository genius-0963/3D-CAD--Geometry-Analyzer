"""
File handling module for CAD file operations.

This module provides base classes and implementations for loading and processing
various CAD file formats.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Type, Dict, Any

from .base_loader import BaseCADLoader
from .stl_loader import STLLoader

# List of available loaders
AVAILABLE_LOADERS: List[Type[BaseCADLoader]] = [
    STLLoader,
]

def get_loader_for_file(file_path: str) -> Type[BaseCADLoader]:
    """Get the appropriate loader for the given file.
    
    Args:
        file_path: Path to the CAD file
        
    Returns:
        A loader class that can handle the file
        
    Raises:
        ValueError: If no loader is found for the file extension
    """
    path = Path(file_path)
    for loader in AVAILABLE_LOADERS:
        if path.suffix.lower() in loader.supported_formats():
            return loader
    
    raise ValueError(
        f"No loader found for file: {file_path}. "
        f"Supported formats: {', '.join(fmt for loader in AVAILABLE_LOADERS for fmt in loader.supported_formats())}"
    )
