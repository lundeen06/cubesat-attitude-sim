# cubesat_sim/simulator.py

from typing import Dict, Any, List, Optional, Union
import numpy as np
from dataclasses import dataclass
import time
from pathlib import Path
import json

from .cubesat import CubeSat
from .forces import Force
from .components import Component
from .utils.constants import Constants

@dataclass
class SimulationConfig:
    """Configuration parameters for simulation."""
    dt: float = 0.1                    # Time step (seconds)
    duration: float = 300.0            # Total simulation time (seconds)
    save_frequency: int = 1            # Save every N steps
    enable_progress: bool = True       # Show progress bar
    save_path: Optional[Path] = None   # Where to save results
    scenarios: List[str] = None        # Specific scenarios to run

@dataclass
class SimulationResults:
    """Container for simulation results."""
    time: np.ndarray
    states: List[Dict[str, Any]]
    config: SimulationConfig
    events: List[Dict[str, Any]]
    
    def save(self, path: Path):
        """Save results to file."""
        data = {
            'time': self.time.tolist(),
            'states': self.states,
            'config': vars(self.config),
            'events': self.events
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> 'SimulationResults':
        """Load results from file."""
        data = json.loads(path.read_text())
        config = SimulationConfig(**data['config'])
        return cls(
            time=np.array(data['time']),
            states=data['states'],
            config=config,
            events=data['events']
        )

class Simulator:
    """
    Main simulation engine for CubeSat missions.
    
    Handles:
    - Time stepping and integration
    - State management
    - Event handling
    - Data logging
    - Scenario management
    """
    
    def __init__(
        self,
        cubesat: CubeSat,
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize simulator.
        
        Args:
            cubesat: CubeSat instance to simulate
            config: Simulation configuration (or use defaults)
        """
        self.cubesat = cubesat
        self.config = config or SimulationConfig()
        
        # Initialize storage
        self.states: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.time_points: List[float] = []
        
        # Simulation state
        self._step_count = 0
        self._running = False
        self._paused = False
        self._start_time = None
        
        # Register default scenarios
        self.scenarios = {
            'detumble': self._scenario_detumble,
            'momentum_dump': self._scenario_momentum_dump,
            'sun_pointing': self._scenario_sun_pointing,
            'nadir_pointing': self._scenario_nadir_pointing
        }
    
    def add_scenario(self, name: str, scenario_func: callable):
        """
        Add custom scenario.
        
        Args:
            name: Scenario identifier
            scenario_func: Function that implements the scenario
        """
        self.scenarios[name] = scenario_func
    
    def save_state(self):
        """Save current state and time."""
        if self._step_count % self.config.save_frequency == 0:
            self.states.append(self.cubesat.get_state())
            self.time_points.append(self.cubesat.state['time'])
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a simulation event."""
        self.events.append({
            'time': self.cubesat.state['time'],
            'step': self._step_count,
            'type': event_type,
            'data': data
        })
    
    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step.
        
        Args:
            dt: Optional time step (otherwise uses config.dt)
        """
        dt = dt or self.config.dt
        
        try:
            # Pre-step event handlers would go here
            
            # Propagate state
            self.cubesat.propagate_state(dt)
            
            # Post-step event handlers would go here
            
            # Save state if needed
            self.save_state()
            
            self._step_count += 1
            
        except Exception as e:
            self.log_event('error', {'error': str(e)})
            raise
    
    def run(
        self,
        scenarios: Optional[List[str]] = None,
        callback: Optional[callable] = None
    ) -> SimulationResults:
        """
        Run full simulation.
        
        Args:
            scenarios: List of scenarios to run (or use config)
            callback: Optional callback function(simulator, step_number)
        
        Returns:
            SimulationResults object
        """
        self._running = True
        self._start_time = time.time()
        
        try:
            # Run specified scenarios
            scenarios = scenarios or self.config.scenarios
            if scenarios:
                for scenario in scenarios:
                    if scenario in self.scenarios:
                        self.scenarios[scenario]()
                    else:
                        self.log_event('warning', {
                            'message': f'Unknown scenario: {scenario}'
                        })
            
            # Main simulation loop
            while (self.cubesat.state['time'] < self.config.duration and 
                   self._running and not self._paused):
                
                self.step()
                
                if callback:
                    callback(self, self._step_count)
                
            # Create results
            results = SimulationResults(
                time=np.array(self.time_points),
                states=self.states,
                config=self.config,
                events=self.events
            )
            
            # Save if path specified
            if self.config.save_path:
                results.save(self.config.save_path)
            
            return results
            
        finally:
            self._running = False
            
    def pause(self):
        """Pause simulation."""
        self._paused = True
        
    def resume(self):
        """Resume simulation."""
        self._paused = False
        
    def stop(self):
        """Stop simulation."""
        self._running = False
    
    def setup_tumbling_state(
    self,
    max_angular_velocity: float = 0.1,  # rad/s
    random_attitude: bool = True,
    altitude: float = 400000  # Default 400km altitude
):
        """
        Set up an initial tumbling state for the spacecraft.
        
        Args:
            max_angular_velocity: Maximum angular velocity magnitude (rad/s)
            random_attitude: Whether to randomize initial attitude
            altitude: Initial orbital altitude in meters
        """
        # Generate random angular velocity
        angular_velocity = np.random.uniform(
            -max_angular_velocity,
            max_angular_velocity,
            3
        )
        
        # Generate random quaternion if requested
        if random_attitude:
            # Random rotation axis
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            # Random rotation angle
            angle = np.random.uniform(0, 2*np.pi)
            
            # Convert to quaternion (scalar last format)
            quaternion = np.array([
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2),
                np.cos(angle/2)
            ])
        else:
            quaternion = np.array([0, 0, 0, 1])  # Identity quaternion
        
        # Set initial position at specified altitude
        position = np.array([0, 0, Constants.EARTH_RADIUS + altitude])  # Start above north pole
        velocity = np.array([7700, 0, 0])  # Approximate orbital velocity for LEO
        
        # Update spacecraft state
        self.cubesat.state.update({
            'angular_velocity': angular_velocity,
            'quaternion': quaternion,
            'position': position,
            'velocity': velocity,
            'magnetic_field': np.array([0, 0.2e-4, -0.4e-4])  # Approximate field strength in Tesla
        })
        
        self.log_event('tumbling_initialized', {
            'angular_velocity': angular_velocity.tolist(),
            'angular_velocity_magnitude': np.linalg.norm(angular_velocity),
            'quaternion': quaternion.tolist(),
            'altitude': altitude,
            'position': position.tolist(),
            'velocity': velocity.tolist()
    })

    def _scenario_detumble(self):
        """
        Detumbling scenario using B-dot control with magnetorquers.
        
        Algorithm creates magnetic dipole moment opposite to the change in 
        magnetic field (in body frame) to reduce angular rates.
        """
        self.log_event('scenario_start', {'name': 'detumble', 'type': 'bdot'})
        
        # Get magnetorquers from components
        magnetorquers = [c for c in self.cubesat.components 
                        if c.__class__.__name__ == 'Magnetorquer']
        
        if not magnetorquers:
            self.log_event('error', {
                'scenario': 'detumble',
                'message': 'No magnetorquers found'
            })
            return
            
        # Initialize variables for B-dot calculation
        last_b_body = np.zeros(3)
        last_time = 0.0
        k_detumble = 5e-4  # Control gain (tune this based on magnetorquer strength)
        
        # Add callback to implement B-dot control
        def detumble_callback(sim: Simulator, step: int):
            nonlocal last_b_body, last_time
            
            current_state = sim.cubesat.state
            current_time = current_state['time']
            current_b_body = current_state['magnetic_field']
            
            dt = current_time - last_time
            if dt > 0:
                # Calculate B-dot in body frame
                b_dot = (current_b_body - last_b_body) / dt
                
                # Calculate required magnetic dipole moment (negative for stability)
                dipole_moment = -k_detumble * b_dot
                
                # Command each magnetorquer based on its axis
                for mtq in magnetorquers:
                    # Project desired dipole onto magnetorquer axis
                    command = np.dot(dipole_moment, mtq.axis)
                    mtq.command(command)
                
                # Update last values
                last_b_body = current_b_body.copy()
                last_time = current_time
                
                # Log angular rates periodically
                if step % 100 == 0:
                    angular_velocity = current_state['angular_velocity']
                    sim.log_event('detumble_status', {
                        'time': current_time,
                        'angular_velocity': angular_velocity.tolist(),
                        'angular_velocity_norm': np.linalg.norm(angular_velocity),
                        'b_dot_norm': np.linalg.norm(b_dot),
                        'dipole_moment': dipole_moment.tolist()
                    })
        
        # Store original config
        original_duration = self.config.duration
        
        # Set up detumbling simulation parameters
        self.config.duration = 3000  # 50 minutes should be enough for initial detumble
        target_rate = 0.01  # rad/s target angular velocity
        
        # Run simulation with callback
        try:
            while self._running:
                self.step()
                detumble_callback(self, self._step_count)
                
                # Check if detumbling is complete
                current_rate = np.linalg.norm(self.cubesat.state['angular_velocity'])
                if current_rate < target_rate:
                    self.log_event('detumble_complete', {
                        'time': self.cubesat.state['time'],
                        'final_rate': current_rate,
                        'iterations': self._step_count
                    })
                    break
                    
                # Check timeout
                if self.cubesat.state['time'] >= self.config.duration:
                    self.log_event('detumble_timeout', {
                        'time': self.cubesat.state['time'],
                        'final_rate': current_rate
                    })
                    break
                    
        finally:
            # Restore original config
            self.config.duration = original_duration
            
            # Turn off magnetorquers
            for mtq in magnetorquers:
                mtq.command(0.0)
            
    def _scenario_momentum_dump(self):
        """Momentum dumping using magnetorquers."""
        self.log_event('scenario_start', {'name': 'momentum_dump'})
        # Implementation would go here
        
    def _scenario_sun_pointing(self):
        """Sun pointing demonstration."""
        self.log_event('scenario_start', {'name': 'sun_pointing'})
        # Implementation would go here
        
    def _scenario_nadir_pointing(self):
        """Nadir pointing demonstration."""
        self.log_event('scenario_start', {'name': 'nadir_pointing'})
        # Implementation would go here