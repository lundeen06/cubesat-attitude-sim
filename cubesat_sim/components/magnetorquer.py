# cubesat_sim/components/magnetorquer.py

import numpy as np
from typing import Dict, Any, Tuple
from ..forces.actuator import Actuator

class Magnetorquer(Actuator):
    """
    Magnetorquer actuator.
    
    Provides torque by generating magnetic dipole to interact with Earth's field.
    """
    
    def __init__(
        self,
        name: str,
        axis: np.ndarray,
        max_dipole: float,  # A⋅m²
        mass: float = 0.050,  # 50g typical for CubeSat MTQ
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        time_constant: float = 0.1  # seconds
    ):
        """
        Initialize magnetorquer.
        
        Args:
            name: Unique identifier for this magnetorquer
            axis: Coil axis unit vector in body frame
            max_dipole: Maximum magnetic dipole in A⋅m²
            mass: Mass in kg
            position: Position vector [x,y,z] relative to COM
            orientation: Orientation vector [rx,ry,rz] in radians
            time_constant: Coil response time constant in seconds
        """
        # Set defaults for position and orientation
        if position is None:
            position = np.zeros(3)
        if orientation is None:
            orientation = np.zeros(3)
            
        super().__init__(name, max_dipole, time_constant, mass, position, orientation)
        
        # Normalize axis vector
        self.axis = np.array(axis) / np.linalg.norm(axis)
        
        # Physical dimensions (typical for CubeSat MTQ)
        self.length = 0.08  # 8cm length
        self.width = 0.01   # 1cm width/height
        
    def calculate_inertia_tensor(self) -> np.ndarray:
        """
        Calculate inertia tensor for the magnetorquer.
        
        Approximates the MTQ as a thin rod along its axis.
        
        Returns:
            3x3 inertia tensor matrix in kg⋅m²
        """
        # Get axis aligned inertia (rod along z-axis)
        mass = self.mass
        l = self.length
        w = self.width
        
        # Rod along z-axis inertia
        Ixx = mass * (3*w**2 + l**2) / 12
        Iyy = Ixx
        Izz = mass * w**2 / 2
        
        I = np.diag([Ixx, Iyy, Izz])
        
        # Rotate to align with actual axis
        # This is a simplified rotation - could be made more accurate
        if np.allclose(self.axis, [1, 0, 0]):  # X-axis
            I = np.array([[Izz, 0, 0],
                         [0, Ixx, 0],
                         [0, 0, Ixx]])
        elif np.allclose(self.axis, [0, 1, 0]):  # Y-axis
            I = np.array([[Ixx, 0, 0],
                         [0, Izz, 0],
                         [0, 0, Ixx]])
        
        return I
    
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
            'axis': self.axis.tolist(),
            'current_dipole': (self._current_output * self.axis).tolist(),
            'length': self.length,
            'width': self.width
        })
        return properties