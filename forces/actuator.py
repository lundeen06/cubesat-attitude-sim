# cubesat_sim/forces/actuator.py

from abc import abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from .force import Force

class Actuator(Force):
    """
    Base class for all spacecraft actuators.
    
    Actuators are special types of forces that can be commanded
    and have physical limitations on their output.
    """
    
    def __init__(
        self,
        name: str,
        max_output: float,
        time_constant: float = 0.1  # seconds
    ):
        """
        Initialize actuator.
        
        Args:
            name: Unique identifier for this actuator
            max_output: Maximum output (units depend on actuator type)
            time_constant: Response time constant in seconds
        """
        super().__init__(name)
        self.max_output = max_output
        self.time_constant = time_constant
        
        self._commanded_output = 0.0
        self._current_output = 0.0
        self._last_update_time = 0.0
    
    @property
    def commanded_output(self) -> float:
        """Get currently commanded output."""
        return self._commanded_output
    
    @property
    def current_output(self) -> float:
        """Get current actual output."""
        return self._current_output
    
    def command(self, output: float):
        """
        Command actuator to desired output.
        
        Args:
            output: Desired output (will be clamped to ±max_output)
        """
        self._commanded_output = np.clip(output, -self.max_output, self.max_output)
    
    def _update_output(self, time: float):
        """
        Update current output based on time constant.
        
        Args:
            time: Current simulation time
        """
        dt = time - self._last_update_time
        if dt > 0:
            # First-order response to commanded value
            alpha = 1 - np.exp(-dt / self.time_constant)
            self._current_output += alpha * (self._commanded_output - self._current_output)
            self._last_update_time = time
    
    def get_properties(self) -> Dict[str, Any]:
        """Get actuator properties."""
        properties = super().get_properties()
        properties.update({
            'max_output': self.max_output,
            'time_constant': self.time_constant,
            'commanded_output': self._commanded_output,
            'current_output': self._current_output
        })
        return properties