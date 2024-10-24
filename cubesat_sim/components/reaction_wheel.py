# cubesat_sim/components/reaction_wheel.py

from typing import Dict, Any, Tuple
import numpy as np
from ..forces.actuator import Actuator  # Updated import

class ReactionWheel(Actuator):
    """
    Reaction wheel actuator.
    
    Provides torque by changing wheel spin rate. Accounts for:
    - Wheel inertia
    - Maximum torque
    - Maximum speed
    - Motor dynamics
    """
    
    def __init__(
        self,
        name: str,
        axis: np.ndarray,
        max_torque: float,  # N⋅m
        max_speed: float,   # rad/s
        wheel_inertia: float,  # kg⋅m²
        time_constant: float = 0.1  # seconds
    ):
        """
        Initialize reaction wheel.
        
        Args:
            name: Unique identifier for this wheel
            axis: Wheel spin axis unit vector in body frame
            max_torque: Maximum wheel torque in N⋅m
            max_speed: Maximum wheel speed in rad/s
            wheel_inertia: Wheel moment of inertia in kg⋅m²
            time_constant: Motor response time constant in seconds
        """
        super().__init__(name, max_torque, time_constant)
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.max_speed = max_speed
        self.wheel_inertia = wheel_inertia
        
        self._speed = 0.0  # Current wheel speed
        
    @property
    def speed(self) -> float:
        """Get current wheel speed in rad/s."""
        return self._speed
    
    def calculate_force_and_torque(
        self,
        state: Dict[str, Any],
        time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reaction wheel force and torque.
        
        Args:
            state: Current state dictionary (not used for reaction wheel)
            time: Current simulation time
            
        Returns:
            Tuple of (force, torque) vectors
        """
        # Update wheel dynamics
        self._update_output(time)
        
        # Calculate change in wheel speed
        dt = time - self._last_update_time
        if dt > 0:
            dw = (self._current_output / self.wheel_inertia) * dt
            new_speed = self._speed + dw
            
            # Limit wheel speed
            if abs(new_speed) <= self.max_speed:
                self._speed = new_speed
            else:
                # Wheel is saturated
                self._speed = np.clip(new_speed, -self.max_speed, self.max_speed)
                self._current_output = 0.0  # No more torque when saturated
        
        # Torque is along wheel axis
        torque = -self._current_output * self.axis  # Negative by reaction
        
        # Reaction wheels produce no translational force
        force = np.zeros(3)
        
        return force, torque
    
    def get_properties(self) -> Dict[str, Any]:
        """Get wheel properties."""
        properties = super().get_properties()
        properties.update({
            'axis': self.axis,
            'max_speed': self.max_speed,
            'wheel_inertia': self.wheel_inertia,
            'current_speed': self._speed
        })
        return properties