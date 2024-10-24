# cubesat_sim/forces/magnetic.py

from typing import Dict, Any, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from ..utils.constants import Constants  # Update import
from .force import Force

class MagneticForce(Force):
    """
    Magnetic force and torque calculator.
    
    Calculates forces and torques due to interaction with Earth's magnetic field.
    This includes both:
    1. Natural interactions with residual spacecraft magnetic dipole
    2. Intentional interactions from magnetorquer actuators
    """
    
    def __init__(
        self,
        name: str = "magnetic",
        residual_dipole: np.ndarray = None,
        earth_dipole: float = Constants.EARTH_MAGNETIC_DIPOLE,  # Update reference
        custom_field_model: callable = None
    ):
        """
        Initialize magnetic force calculator.
        
        Args:
            name: Unique identifier for this force
            residual_dipole: Residual magnetic dipole vector in A·m² (default: small Z-axis dipole)
            earth_dipole: Earth's magnetic dipole moment in T·m³
            custom_field_model: Optional custom magnetic field model function
                              Takes position vector, returns B-field vector
        """
        self.name = name
        super().__init__(name)
        
        # Set residual dipole (default: small Z-axis dipole)
        if residual_dipole is None:
            residual_dipole = np.array([0.0, 0.0, 0.001])  # 0.001 A·m² in Z
        self.residual_dipole = np.array(residual_dipole)
        
        self.earth_dipole = earth_dipole
        self.custom_field_model = custom_field_model
        
        # Store last calculated values
        self._last_magnetic_field = np.zeros(3)
        self._last_control_dipole = np.zeros(3)
    
    def get_properties(self) -> Dict[str, Any]:
        """Get properties of the magnetic force calculator."""
        properties = super().get_properties()
        properties.update({
            'residual_dipole': self.residual_dipole,
            'earth_dipole': self.earth_dipole,
            'last_magnetic_field': self._last_magnetic_field,
            'last_control_dipole': self._last_control_dipole
        })
        return properties
    
    def _calculate_magnetic_field(
        self,
        position: np.ndarray,
        time: float
    ) -> np.ndarray:
        """
        Calculate magnetic field vector at given position.
        
        Args:
            position: Position vector in ECI frame
            time: Current simulation time
            
        Returns:
            Magnetic field vector in Tesla
        """
        if self.custom_field_model is not None:
            return self.custom_field_model(position, time)
            
        # Simple dipole model in ECI frame
        r = np.linalg.norm(position)
        r_unit = position / r
        
        # B = (μ0/4π) * (M/r³) * (3(m·r̂)r̂ - m)
        # where M is Earth's dipole moment, m is dipole direction (Z-axis in ECI)
        m = np.array([0, 0, 1])  # Dipole aligned with Z-axis
        
        # Calculate field magnitude and direction
        B_magnitude = self.earth_dipole / r**3
        m_dot_r = np.dot(m, r_unit)
        B = B_magnitude * (3 * m_dot_r * r_unit - m)
        
        return B
    
    def add_control_dipole(self, dipole: np.ndarray):
        """
        Add control dipole from magnetorquers.
        
        Args:
            dipole: Control dipole vector in A·m²
        """
        self._last_control_dipole = np.array(dipole)
    
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetic force and torque.
        
        Args:
            state: Current state dictionary containing:
                  - position: np.ndarray (3,) Position in m (ECI frame)
                  - quaternion: np.ndarray (4,) Attitude quaternion
            time: Current simulation time
            
        Returns:
            Tuple of (force, torque) where force is zeros (magnetic primarily produces torque)
        """
        position = state['position']
        quaternion = state['quaternion']
        
        # Calculate magnetic field in ECI
        B_eci = self._calculate_magnetic_field(position, time)
        
        # Transform to body frame
        rot_matrix = Rotation.from_quat(quaternion).as_matrix()
        B_body = rot_matrix @ B_eci
        self._last_magnetic_field = B_body
        
        # Total dipole is residual plus any control dipole
        total_dipole = self.residual_dipole + self._last_control_dipole
        
        # Calculate torque T = m × B
        torque = np.cross(total_dipole, B_body)
        
        # Magnetic forces are negligible for CubeSat applications
        force = np.zeros(3)
        
        return force, torque