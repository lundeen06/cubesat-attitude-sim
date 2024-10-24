# cubesat_sim/forces/magnetorquer.py

from typing import Dict, Any, Tuple
import numpy as np
from .actuator import Actuator

class Magnetorquer(Actuator):
    """
    Magnetorquer actuator.
    
    Provides torque by generating magnetic dipole to interact with Earth's field.
    Accounts for:
    - Coil alignment
    - Maximum dipole
    - Coil dynamics
    """
    
    def __init__(
        self,
        name: str,
        axis: np.ndarray,
        max_dipole: float,  # A⋅m²
        time_constant: float = 0.1  # seconds
    ):
        """
        Initialize magnetorquer.
        
        Args:
            name: Unique identifier for this magnetorquer
            axis: Coil axis unit vector in body frame
            max_dipole: Maximum magnetic dipole in A⋅m²
            time_constant: Coil response time constant in seconds
        """
        super().__init__(name, max_dipole, time_constant)
        self.axis = np.array(axis) / np.linalg.norm(axis)
        
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetorquer force and torque.
        
        Args:
            state: Current state dictionary containing:
                  - magnetic_field: np.ndarray (3,) Magnetic field vector in body frame
            time: Current simulation time
            
        Returns:
            Tuple of (force, torque) vectors
        """
        # Update coil dynamics
        self._update_output(time)
        
        # Get magnetic field in body frame
        B_body = state.get('magnetic_field', np.zeros(3))
        
        # Calculate magnetic dipole moment
        dipole = self._current_output * self.axis
        
        # Calculate torque T = m × B
        torque = np.cross(dipole, B_body)
        
        # Magnetorquers produce no translational force
        force = np.zeros(3)
        
        return force, torque
    
    def get_properties(self) -> Dict[str, Any]:
        """Get magnetorquer properties."""
        properties = super().get_properties()
        properties.update({
            'axis': self.axis,
            'current_dipole': self._current_output * self.axis
        })
        return properties