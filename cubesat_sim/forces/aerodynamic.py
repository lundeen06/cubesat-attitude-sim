from typing import Dict, Any, Tuple
import numpy as np
from scipy.spatial.transform import Rotation  # Added this import
from ..utils.constants import Constants
from .force import Force


class AerodynamicForce(Force):
    def __init__(
        self,
        name: str = "aerodynamic",
        drag_coefficient: float = 2.2,
        reference_altitude: float = 400000,  # meters
        custom_density: float = None,
    ):
        # Call parent class constructor first
        super().__init__(name)
        
        # Then initialize our own attributes
        self.drag_coefficient = drag_coefficient
        self.reference_altitude = reference_altitude
        
        # Use custom density if provided, otherwise look up from constants
        if custom_density is not None:
            self._density = custom_density
        else:
            # Find closest altitude in our density table
            altitudes = np.array(list(Constants.ATMOSPHERIC_DENSITY.keys()))
            idx = np.abs(altitudes - reference_altitude).argmin()
            self._density = Constants.ATMOSPHERIC_DENSITY[altitudes[idx]]
            
        # Store the last calculated values for visualization/analysis
        self._last_dynamic_pressure = 0.0
        self._last_projected_area = 0.0
        
    @property
    def density(self) -> float:
        """Get current atmospheric density."""
        return self._density
    
    def get_properties(self) -> Dict[str, Any]:
        """Get properties of the aerodynamic force calculator."""
        properties = super().get_properties()
        properties.update({
            'drag_coefficient': self.drag_coefficient,
            'density': self.density,
            'reference_altitude': self.reference_altitude,
            'last_dynamic_pressure': self._last_dynamic_pressure,
            'last_projected_area': self._last_projected_area
        })
        return properties
    
    def _calculate_projected_area(
        self,
        velocity_body: np.ndarray,
        components: list
    ) -> float:
        """
        Calculate projected area based on velocity direction and geometry.
        
        Args:
            velocity_body: Velocity vector in body frame
            components: List of spacecraft components
            
        Returns:
            Projected area in m²
        """
        # This is a simplified version - in reality would need to:
        # 1. Consider each component's orientation
        # 2. Handle shadowing effects
        # 3. Account for complex geometries
        
        # For now, return a simple approximation
        # TODO: Implement proper projected area calculation
        return 0.02  # 20 cm² (typical for 2U CubeSat)
    
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aerodynamic force and torque.
        
        Args:
            state: Current state dictionary containing:
                  - velocity: np.ndarray (3,) Velocity in m/s (inertial frame)
                  - quaternion: np.ndarray (4,) Attitude quaternion
                  - components: list of spacecraft components
            time: Current simulation time (not used for aero)
            
        Returns:
            Tuple of (force, torque) where each is a 3D numpy array
        """
        # Transform velocity to body frame
        velocity_inertial = state['velocity']
        quaternion = state['quaternion']
        rot_matrix = Rotation.from_quat(quaternion).as_matrix()
        velocity_body = rot_matrix @ velocity_inertial
        
        # Calculate dynamic pressure
        velocity_magnitude = np.linalg.norm(velocity_body)
        self._last_dynamic_pressure = 0.5 * self.density * velocity_magnitude**2
        
        # Get projected area
        self._last_projected_area = self._calculate_projected_area(
            velocity_body,
            state.get('components', [])
        )
        
        # Calculate force magnitude
        force_magnitude = (self._last_dynamic_pressure * 
                         self.drag_coefficient * 
                         self._last_projected_area)
        
        # Force direction is opposite to velocity
        force_direction = -velocity_body / velocity_magnitude
        force = force_magnitude * force_direction
        
        # Calculate torque (simplified - assumes center of pressure offset)
        # TODO: Implement proper torque calculation based on component geometry
        cp_offset = np.array([0, 0.05, 0])  # 5cm offset in Y
        torque = np.cross(cp_offset, force)
        
        return force, torque