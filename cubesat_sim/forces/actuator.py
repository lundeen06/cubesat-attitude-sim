# cubesat_sim/forces/actuator.py

from typing import Dict, Any, Tuple
import numpy as np
from .force import Force
from ..components.component import Component
from abc import abstractmethod

class Actuator(Component, Force):
    """Base class for actuators (reaction wheels, magnetorquers)."""
    
    def __init__(
        self,
        name: str,
        axis: np.ndarray,
        max_output: float,
        mass: float,
        position: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.zeros(3),
        time_constant: float = 0.1  # seconds
    ):
        """
        Initialize actuator.
        
        Args:
            name: Unique identifier
            axis: Actuation axis unit vector
            max_output: Maximum output (N⋅m for wheels, A⋅m² for MTQs)
            mass: Actuator mass in kg
            position: Position relative to COM
            orientation: Orientation angles
            time_constant: Response time constant
        """
        Component.__init__(self, name, mass, position, orientation)
        Force.__init__(self, name)
        
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.max_output = max_output
        self.time_constant = time_constant
        
        self._commanded_output = 0.0
        self._current_output = 0.0
        self._last_update_time = 0.0
    
    def command(self, output: float):
        """Set commanded output (with saturation)."""
        self._commanded_output = np.clip(output, -self.max_output, self.max_output)
    
    def _update_output(self, time: float):
        """Update actual output based on dynamics."""
        dt = time - self._last_update_time
        if dt > 0:
            # First-order response
            alpha = 1 - np.exp(-dt / self.time_constant)
            self._current_output += alpha * (self._commanded_output - self._current_output)
            self._last_update_time = time
        
    def get_properties(self) -> Dict[str, Any]:
        """Get actuator properties."""
        props = Component.get_properties(self)
        props.update(Force.get_properties(self))
        props.update({
            'axis': self.axis.tolist(),
            'max_output': self.max_output,
            'time_constant': self.time_constant,
            'commanded_output': self._commanded_output,
            'current_output': self._current_output
        })
        return props
    
    def calculate_inertia_tensor(self) -> np.ndarray:
        """Calculate actuator's inertia tensor."""
        # Simple rod model along axis
        length = 0.05  # 5cm typical length
        radius = 0.01  # 1cm typical radius
        
        # Parallel axis contributions
        I_parallel = self.mass * length**2 / 12
        I_perpendicular = self.mass * (3*radius**2 + length**2) / 12
        
        # Create diagonal tensor
        I = np.eye(3) * I_perpendicular
        I += np.outer(self.axis, self.axis) * (I_parallel - I_perpendicular)
        
        return I
    
    @abstractmethod
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate force and torque (to be implemented by subclasses)."""
        pass