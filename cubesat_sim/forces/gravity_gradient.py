# cubesat_sim/forces/gravity_gradient.py

from typing import Dict, Any, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from ..utils.constants import Constants  # Update import
from .force import Force

class GravityGradient(Force):
    """
    Gravity gradient torque calculator.
    
    Calculates torques due to gravity gradient effects in Earth orbit.
    This torque occurs because different parts of the spacecraft are
    at slightly different distances from Earth.
    """
    
    def __init__(
        self,
        name: str = "gravity_gradient",
        earth_mu: float = Constants.EARTH_MU,  # Update reference
        min_altitude: float = 100000  # meters, for safety checks
    ):
        """
        Initialize gravity gradient calculator.
        
        Args:
            name: Unique identifier for this force
            earth_mu: Earth's gravitational parameter (m³/s²)
            min_altitude: Minimum allowed altitude for calculations
        """
        self.name = name
        super().__init__(name)
        self.earth_mu = earth_mu
        self.min_altitude = min_altitude
        
        # Store last calculated values
        self._last_orbital_radius = 0.0
        self._last_nadir_vector = np.zeros(3)
    
    def get_properties(self) -> Dict[str, Any]:
        """Get properties of the gravity gradient calculator."""
        properties = super().get_properties()
        properties.update({
            'earth_mu': self.earth_mu,
            'last_orbital_radius': self._last_orbital_radius,
            'last_nadir_vector': self._last_nadir_vector
        })
        return properties
    
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gravity gradient force and torque.
        
        Args:
            state: Current state dictionary containing:
                  - position: np.ndarray (3,) Position in m (ECI frame)
                  - quaternion: np.ndarray (4,) Attitude quaternion
                  - inertia_tensor: np.ndarray (3,3) Spacecraft inertia tensor
            time: Current simulation time (not used for gravity gradient)
            
        Returns:
            Tuple of (force, torque) where force is zeros (gravity gradient only produces torque)
        """
        position = state['position']
        quaternion = state['quaternion']
        inertia = state['inertia_tensor']
        
        # Calculate orbital radius and check altitude
        self._last_orbital_radius = np.linalg.norm(position)
        altitude = self._last_orbital_radius - Constants.EARTH_RADIUS
        if altitude < self.min_altitude:
            raise ValueError(f"Altitude {altitude/1000:.1f}km below minimum {self.min_altitude/1000:.1f}km")
        
        # Calculate nadir vector (points from spacecraft to Earth center) in ECI
        self._last_nadir_vector = -position / self._last_orbital_radius
        
        # Transform nadir vector to body frame
        rot_matrix = Rotation.from_quat(quaternion).as_matrix()
        nadir_body = rot_matrix @ self._last_nadir_vector
        
        # Calculate gravity gradient torque
        # T = (3μ/2r³) * (n × I·n)
        # where n is nadir unit vector, I is inertia tensor
        coefficient = 3 * self.earth_mu / (2 * self._last_orbital_radius**3)
        torque = coefficient * np.cross(nadir_body, inertia @ nadir_body)
        
        # Gravity gradient produces no net force, only torque
        force = np.zeros(3)
        
        return force, torque