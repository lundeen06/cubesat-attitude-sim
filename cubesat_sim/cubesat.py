# cubesat_sim/cubesat.py

from typing import Dict, Any, List, Optional, Union
import numpy as np
from scipy.spatial.transform import Rotation

from .components.component import Component
from .forces.force import Force
from .utils.quaternion import Quaternion
from .utils.constants import CUBESAT_UNIT

class CubeSat:
    """
    Main CubeSat class representing the complete spacecraft.
    
    Handles:
    - Physical configuration and properties
    - Component management
    - Force/torque calculations
    - State propagation
    - Mass properties
    """
    
    def __init__(
        self,
        size_units: Union[int, tuple] = 1,
        name: str = "CubeSat",
        initial_orbit: Dict[str, float] = None,
        initial_attitude: Dict[str, float] = None
    ):
        """
        Initialize CubeSat.
        
        Args:
            size_units: CubeSat size in U (1=1U, 2=2U, (1,2,3)=1U×2U×3U)
            name: Name identifier for the spacecraft
            initial_orbit: Initial orbital parameters (if None, uses defaults)
            initial_attitude: Initial attitude parameters (if None, uses defaults)
        """
        self.name = name
        
        # Set dimensions based on size_units
        if isinstance(size_units, (int, float)):
            self.dimensions = np.array([1, 1, size_units]) * CUBESAT_UNIT
        else:
            self.dimensions = np.array(size_units) * CUBESAT_UNIT
            
        # Initialize component and force lists
        self.components: List[Component] = []
        self.forces: List[Force] = []
        
        # Initialize state with defaults or provided values
        self._initialize_state(initial_orbit, initial_attitude)
        
        # Mass properties
        self._mass = 0.0
        self._center_of_mass = np.zeros(3)
        self._inertia_tensor = np.zeros((3, 3))
        self._inertia_inverse = None
        
        # Cached properties (updated when components change)
        self._properties_dirty = True
        self._last_update_time = 0.0
        
    def _initialize_state(
        self,
        orbit_params: Optional[Dict[str, float]] = None,
        attitude_params: Optional[Dict[str, float]] = None
    ):
        """Initialize orbital and attitude state."""
        # Default orbit (circular 400km equatorial)
        if orbit_params is None:
            orbit_params = {
                'semi_major_axis': 6771000,  # meters (400km altitude)
                'eccentricity': 0.0,
                'inclination': 0.0,
                'raan': 0.0,
                'arg_perigee': 0.0,
                'true_anomaly': 0.0
            }
            
        # Default attitude (aligned with orbital frame)
        if attitude_params is None:
            attitude_params = {
                'quaternion': [1, 0, 0, 0],  # scalar-first
                'angular_velocity': [0, 0, 0]
            }
            
        # Initialize state dictionary
        self.state = {
            # Orbital state
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orbit_params': orbit_params,
            
            # Attitude state
            'quaternion': np.array(attitude_params['quaternion']),
            'angular_velocity': np.array(attitude_params['angular_velocity']),
            
            # Environmental state
            'magnetic_field': np.zeros(3),
            'sun_vector': np.zeros(3),
            
            # Time
            'time': 0.0
        }
        
    def add_component(self, component: Component):
        """Add a component to the spacecraft."""
        self.components.append(component)
        self._properties_dirty = True
        
    def add_force(self, force: Force):
        """Add a force/torque generator."""
        self.forces.append(force)
        
    def update_mass_properties(self):
        """Update mass properties based on current components."""
        if not self._properties_dirty:
            return
            
        # Reset properties
        self._mass = 0.0
        com_numerator = np.zeros(3)
        self._inertia_tensor = np.zeros((3, 3))
        
        # Sum contributions from all components
        for component in self.components:
            self._mass += component.mass
            com_numerator += component.mass * component.position
            
        # Calculate center of mass
        if self._mass > 0:
            self._center_of_mass = com_numerator / self._mass
            
            # Calculate inertia tensor about COM
            for component in self.components:
                # Get component's inertia contribution
                self._inertia_tensor += component.contribute_to_inertia()
                
            # Calculate inverse if possible
            try:
                self._inertia_inverse = np.linalg.inv(self._inertia_tensor)
            except np.linalg.LinAlgError:
                self._inertia_inverse = None
                raise ValueError("Singular inertia tensor - check component configuration")
        
        self._properties_dirty = False
        
    def get_properties(self) -> Dict[str, Any]:
        """Get current spacecraft properties."""
        self.update_mass_properties()
        
        return {
            'name': self.name,
            'dimensions': self.dimensions,
            'mass': self._mass,
            'center_of_mass': self._center_of_mass,
            'inertia_tensor': self._inertia_tensor,
            'num_components': len(self.components),
            'num_forces': len(self.forces)
        }
        
    def calculate_net_force_and_torque(self, state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate net force and torque from all sources."""
        net_force = np.zeros(3)
        net_torque = np.zeros(3)
        
        # Update mass properties if needed
        self.update_mass_properties()
        
        # Add inertia tensor to state for forces that need it
        state_with_inertia = state.copy()
        state_with_inertia['inertia_tensor'] = self._inertia_tensor
        
        # Sum contributions from all enabled forces
        for force in self.forces:
            if force.enabled:
                force_vec, torque_vec = force.calculate_force_and_torque(
                    state_with_inertia,
                    state['time']
                )
                net_force += force_vec
                net_torque += torque_vec
                
        return net_force, net_torque
        
    def propagate_state(self, dt: float):
        """
        Propagate spacecraft state forward in time.
        
        Args:
            dt: Time step in seconds
        """
        # Update mass properties if needed
        self.update_mass_properties()
        
        # Get current state values
        q = Quaternion(self.state['quaternion'])  # current quaternion
        w = self.state['angular_velocity']        # current angular velocity
        
        # Calculate net force and torque
        net_force, net_torque = self.calculate_net_force_and_torque(self.state)
        
        # Angular acceleration
        if self._inertia_inverse is not None:
            w_dot = self._inertia_inverse @ (
                net_torque - np.cross(w, self._inertia_tensor @ w)
            )
        else:
            w_dot = np.zeros(3)
            
        # Update angular velocity
        w_new = w + w_dot * dt
        
        # Update quaternion
        q_dot = Quaternion.derivative(q.scalar_last, w)
        q_new = q.scalar_last + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)  # normalize
        
        # Update linear motion (simplified - no orbital dynamics yet)
        a = net_force / self._mass if self._mass > 0 else np.zeros(3)
        v_new = self.state['velocity'] + a * dt
        p_new = self.state['position'] + v_new * dt
        
        # Update state
        self.state.update({
            'quaternion': q_new,
            'angular_velocity': w_new,
            'velocity': v_new,
            'position': p_new,
            'time': self.state['time'] + dt,
            'net_force': net_force,
            'net_torque': net_torque
        })
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state dictionary."""
        return self.state.copy()
        
    def set_state(self, new_state: Dict[str, Any]):
        """Set state values from dictionary."""
        self.state.update(new_state)
        
    def command_actuators(self, commands: Dict[str, float]):
        """
        Send commands to actuators.
        
        Args:
            commands: Dictionary mapping actuator names to commanded values
        """
        for force in self.forces:
            if force.name in commands and hasattr(force, 'command'):
                force.command(commands[force.name])