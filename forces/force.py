# cubesat_sim/forces/force.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class Force(ABC):
    """
    Abstract base class for all forces and torques acting on the CubeSat.
    
    This class defines the interface that all force/torque generators must implement,
    whether they are environmental (drag, gravity gradient) or actuator-based
    (reaction wheels, magnetorquers).
    """
    
    def __init__(self, name: str):
        """
        Initialize force/torque generator.
        
        Args:
            name: Unique identifier for this force/torque
        """
        self.name = name
        self._enabled = True
        self._last_force = np.zeros(3)
        self._last_torque = np.zeros(3)
    
    @property
    def enabled(self) -> bool:
        """Get whether this force/torque is currently enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Set whether this force/torque is enabled."""
        self._enabled = value
    
    @property
    def last_force(self) -> np.ndarray:
        """Get last calculated force vector."""
        return self._last_force
    
    @property
    def last_torque(self) -> np.ndarray:
        """Get last calculated torque vector."""
        return self._last_torque
    
    @abstractmethod
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate force and torque for current state.
        
        Args:
            state: Current state dictionary containing at minimum:
                  - position: np.ndarray (3,) Position in m
                  - velocity: np.ndarray (3,) Velocity in m/s
                  - quaternion: np.ndarray (4,) Attitude quaternion
                  - angular_velocity: np.ndarray (3,) Angular velocity in rad/s
            time: Current simulation time in seconds
            
        Returns:
            Tuple of (force, torque) where each is a 3D numpy array in N and N⋅m
        """
        pass
    
    def update(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update force and torque calculations.
        
        Args:
            state: Current state dictionary
            time: Current simulation time in seconds
            
        Returns:
            Tuple of (force, torque) vectors
        """
        if not self.enabled:
            self._last_force = np.zeros(3)
            self._last_torque = np.zeros(3)
            return self._last_force, self._last_torque
        
        self._last_force, self._last_torque = self.calculate_force_and_torque(state, time)
        return self._last_force, self._last_torque
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get properties of this force/torque generator.
        
        Returns:
            Dictionary of properties
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'last_force': self.last_force,
            'last_torque': self.last_torque,
            'type': self.__class__.__name__
        }