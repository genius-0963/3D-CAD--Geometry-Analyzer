"""
Base classes for manufacturing process analysis.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class ManufacturingResult:
    """Container for manufacturing analysis results."""
    process_name: str
    feasibility_score: float  # 0-1, higher is better
    is_feasible: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return asdict(self)

class ManufacturingProcess(ABC):
    """Abstract base class for manufacturing process analysis.
    
    Subclasses must implement the analyze() method to perform process-specific
    manufacturability analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the manufacturing process analyzer.
        
        Args:
            config: Configuration dictionary for the manufacturing process
        """
        self.config = self.get_default_config()
        if config:
            self.config.update(config)
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get the default configuration for this manufacturing process.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        pass
    
    @abstractmethod
    def analyze(self, mesh, features: Dict[str, Any]) -> ManufacturingResult:
        """Analyze the mesh for manufacturability with this process.
        
        Args:
            mesh: The 3D mesh to analyze
            features: Precomputed geometric features
            
        Returns:
            ManufacturingResult containing analysis results
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return True
    
    def get_required_features(self) -> List[str]:
        """Get the list of required feature names for this process.
        
        Returns:
            List of required feature names
        """
        return []
