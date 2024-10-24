# cubesat_sim/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import matplotlib.animation as animation
from typing import Dict, Any, List, Optional, Tuple
from ..simulator import SimulationResults  # Add this import

class SimulationPlotter:
    """Generates various plots from simulation results."""
    
    @staticmethod
    def plot_angular_velocity(results: SimulationResults, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot angular velocity components and magnitude."""
        if ax is None:
            _, ax = plt.subplots()
            
        time = results.time
        angular_velocities = np.array([s['angular_velocity'] 
                                     for s in results.states])
        
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
    def plot_magnetic_field(results: SimulationResults, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot magnetic field components in body frame."""
        if ax is None:
            _, ax = plt.subplots()
            
        time = results.time
        b_fields = np.array([s.get('magnetic_field', np.zeros(3)) 
                            for s in results.states])
        
        ax.plot(time, b_fields[:,0], 'r-', label='Bx')
        ax.plot(time, b_fields[:,1], 'g-', label='By')
        ax.plot(time, b_fields[:,2], 'b-', label='Bz')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnetic Field (T)')
        ax.grid(True)
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_magnetorquer_commands(results: SimulationResults, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot magnetorquer command history."""
        if ax is None:
            _, ax = plt.subplots()
            
        time = results.time
        commands = np.array([s.get('mtq_commands', np.zeros(3)) 
                            for s in results.states])
        
        ax.plot(time, commands[:,0], 'r-', label='MTQ-X')
        ax.plot(time, commands[:,1], 'g-', label='MTQ-Y')
        ax.plot(time, commands[:,2], 'b-', label='MTQ-Z')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Dipole Moment (A⋅m²)')
        ax.grid(True)
        ax.legend()
        
        return ax

    def create_detumbling_summary(self, results: SimulationResults, 
                                cubesat_dimensions: np.ndarray,
                                save_path: Optional[str] = None):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Angular velocity
        self.plot_angular_velocity(results, axes[0])
        axes[0].set_title('Angular Velocity During Detumbling')
        
        # Magnetic field
        self.plot_magnetic_field(results, axes[1])
        axes[1].set_title('Magnetic Field in Body Frame')
        
        # Magnetorquer commands
        self.plot_magnetorquer_commands(results, axes[2])
        axes[2].set_title('Magnetorquer Commands')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        # Create animation
        anim_fig, anim = create_detumbling_animation(results, cubesat_dimensions)
        
        return fig, axes, anim_fig, anim
    
def create_detumbling_animation(simulation_results, cubesat_dimensions):
    fig = plt.figure(figsize=(16, 8))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1])
    
    # Create CubeSat geometry faces
    faces = create_cubesat_geometry(cubesat_dimensions)
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Get state for this frame
        state = simulation_results.states[frame]
        time = simulation_results.time[:frame+1]
        quaternion = state['quaternion']
        angular_velocity = state['angular_velocity']
        
        # Calculate rotation matrix
        R = Rotation.from_quat(quaternion).as_matrix()
        
        # Plot CubeSat
        theme = {
            'body': '#3498DB',
            'panels': '#95A5A6',
            'bg': '#FFFFFF',
            'text': '#2C3E50'
        }
        plot_faces(faces, ax1, R, theme['body'])
        
        # Draw axes and vectors
        for axis, color in zip(np.eye(3), ['r', 'g', 'b']):
            draw_vector(np.zeros(3), R @ axis, ax1, color=color)
        
        if 'magnetic_field' in state:
            draw_vector(np.zeros(3), state['magnetic_field'], ax1, color=theme['text'])
        
        # Style 3D plot
        ax1.set_box_aspect([1,1,1])
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot angular velocities
        angular_velocities = np.array([s['angular_velocity'] 
                                       for s in simulation_results.states[:frame+1]])
        
        ax2.plot(time, angular_velocities[:,0], 'r-', label='ωx')
        ax2.plot(time, angular_velocities[:,1], 'g-', label='ωy')
        ax2.plot(time, angular_velocities[:,2], 'b-', label='ωz')
        ax2.plot(time, np.linalg.norm(angular_velocities, axis=1), 
                 'k--', label='|ω|')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    anim = animation.FuncAnimation(fig, update,
                                   frames=len(simulation_results.states),
                                   interval=50,
                                   repeat=True)
    
    return fig, anim

def create_cubesat_geometry(dimensions):
        print(f'Calculating CubeSat geometry... X = {dimensions[0]}, Y = {dimensions[1]}, Z = {dimensions[2]}')
        width, height, depth = dimensions
        
        vertices = np.array([
            [-width/2, -height/2, -depth/2],
            [width/2, -height/2, -depth/2],
            [width/2, height/2, -depth/2],
            [-width/2, height/2, -depth/2],
            [-width/2, -height/2, depth/2],
            [width/2, -height/2, depth/2],
            [width/2, height/2, depth/2],
            [-width/2, height/2, depth/2],
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]
        
        return faces

def plot_faces(faces, ax, rotation_matrix, color, alpha=0.3):
    rotated_faces = [[vertex @ rotation_matrix.T for vertex in face] 
                    for face in faces]
    collection = Poly3DCollection(rotated_faces, facecolors=color, alpha=alpha)
    collection.set_edgecolor('black')
    ax.add_collection3d(collection)
    return collection

def draw_vector(start, vector, ax, color='r', scale=0.1):
    vector = vector * scale
    ax.quiver(start[0], start[1], start[2],
            vector[0], vector[1], vector[2],
            color=color, alpha=0.8, arrow_length_ratio=0.2)