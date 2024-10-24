# cubesat_sim/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from ..utils.quaternion import Quaternion

class SimulationPlotter:
    """Generates various plots from simulation results."""
    
    @staticmethod
    def plot_angular_velocity(results: Dict[str, Any], ax=None):
        """Plot angular velocity components and magnitude."""
        if ax is None:
            _, ax = plt.subplots()
            
        time = results['time']
        angular_velocities = np.array([s['angular_velocity'] 
                                     for s in results['states']])
        
        ax.plot(time, angular_velocities[:,0], 'r-', label='ωx')
        ax.plot(time, angular_velocities[:,1], 'g-', label='ωy')
        ax.plot(time, angular_velocities[:,2], 'b-', label='ωz')
        ax.plot(time, np.linalg.norm(angular_velocities, axis=1), 
                'k--', label='|ω|')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.grid(True)
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_magnetic_field(results: Dict[str, Any], ax=None):
        """Plot magnetic field components in body frame."""
        if ax is None:
            _, ax = plt.subplots()
            
        time = results['time']
        b_fields = np.array([s['magnetic_field'] 
                            for s in results['states']])
        
        ax.plot(time, b_fields[:,0], 'r-', label='Bx')
        ax.plot(time, b_fields[:,1], 'g-', label='By')
        ax.plot(time, b_fields[:,2], 'b-', label='Bz')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnetic Field (T)')
        ax.grid(True)
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_magnetorquer_commands(results: Dict[str, Any], ax=None):
        """Plot magnetorquer command history."""
        if ax is None:
            _, ax = plt.subplots()
            
        time = results['time']
        commands = np.array([s.get('mtq_commands', [0,0,0]) 
                           for s in results['states']])
        
        ax.plot(time, commands[:,0], 'r-', label='MTQ-X')
        ax.plot(time, commands[:,1], 'g-', label='MTQ-Y')
        ax.plot(time, commands[:,2], 'b-', label='MTQ-Z')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Dipole Moment (A⋅m²)')
        ax.grid(True)
        ax.legend()
        
        return ax
    
    @staticmethod
    def create_detumbling_summary(results: Dict[str, Any], 
                                save_path: str = None):
        """Create comprehensive detumbling visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Angular velocity
        SimulationPlotter.plot_angular_velocity(results, axes[0])
        axes[0].set_title('Angular Velocity During Detumbling')
        
        # Magnetic field
        SimulationPlotter.plot_magnetic_field(results, axes[1])
        axes[1].set_title('Magnetic Field in Body Frame')
        
        # Magnetorquer commands
        SimulationPlotter.plot_magnetorquer_commands(results, axes[2])
        axes[2].set_title('Magnetorquer Commands')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig, axes