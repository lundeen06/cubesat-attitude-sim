# cubesat_sim/components/component.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple

class Component(ABC):
    """
    Abstract base class for all CubeSat components.
    
    This class defines the interface that all components must implement,
    including basic physical properties and methods for mass properties.
    """
    
    def __init__(
        self,
        name: str,
        mass: float,
        position: np.ndarray,
        orientation: np.ndarray
    ):
        """
        Initialize a component.
        
        Args:
            name: Unique identifier for the component
            mass: Mass in kg
            position: Position vector [x,y,z] in m relative to CubeSat center of mass
            orientation: Orientation vector [rx,ry,rz] in radians
        """
        self.name = name
        self._mass = mass
        self._position = np.array(position, dtype=float)
        self._orientation = np.array(orientation, dtype=float)
        
    @property
    def mass(self) -> float:
        """Get component mass in kg."""
        return self._mass
    
    @property
    def position(self) -> np.ndarray:
        """Get component position relative to CubeSat center of mass."""
        return self._position
    
    @property
    def orientation(self) -> np.ndarray:
        """Get component orientation in radians."""
        return self._orientation
    
    @abstractmethod
    def calculate_inertia_tensor(self) -> np.ndarray:
        """
        Calculate the inertia tensor for this component.
        
        Returns:
            3x3 inertia tensor matrix in kg⋅m²
        """
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Get component-specific properties.
        
        Returns:
            Dictionary of component properties
        """
        pass
    
    def contribute_to_inertia(self) -> np.ndarray:
        """
        Calculate this component's contribution to the total spacecraft inertia tensor,
        including the parallel axis theorem adjustment.
        
        Returns:
            3x3 inertia tensor matrix in kg⋅m²
        """
        # Get the component's local inertia tensor
        local_inertia = self.calculate_inertia_tensor()
        
        # Apply parallel axis theorem
        r = self.position
        parallel_axis_term = self.mass * (
            np.dot(r, r) * np.eye(3) - np.outer(r, r)
        )
        
        return local_inertia + parallel_axis_term