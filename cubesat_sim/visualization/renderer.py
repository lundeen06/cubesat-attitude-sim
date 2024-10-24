# cubesat_sim/visualization/renderer.py

import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

class CubeSatRenderer:
    """
    3D renderer for CubeSat visualization.
    Adapts visualization code from original viz.py.
    """
    
    def __init__(self, theme=None):
        if theme is None:
            self.theme = {
                'body': '#3498DB',
                'panels': '#95A5A6',
                'bg': '#FFFFFF',
                'text': '#2C3E50'
            }
        else:
            self.theme = theme
            
    def _create_cubesat_geometry(self, dimensions):
        """Create CubeSat geometry vectors."""
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

    def _plot_faces(self, ax, faces, rotation_matrix, color, alpha=0.3):
        """Plot CubeSat faces with given rotation."""
        rotated_faces = [[vertex @ rotation_matrix.T for vertex in face] 
                        for face in faces]
        collection = Poly3DCollection(rotated_faces, facecolors=color, alpha=alpha)
        collection.set_edgecolor('black')
        ax.add_collection3d(collection)
        return collection

    def _draw_vector(self, ax, start, vector, color='r', scale=0.1, 
                    arrow_length_ratio=0.2):
        """Draw a vector arrow."""
        vector = vector * scale
        ax.quiver(start[0], start[1], start[2],
                 vector[0], vector[1], vector[2],
                 color=color, alpha=0.8, 
                 arrow_length_ratio=arrow_length_ratio)

    def create_animation(self, simulation_results: Dict[str, Any], 
                        cubesat_dimensions: np.ndarray):
        """Create animation of CubeSat motion."""
        fig = plt.figure(figsize=(16, 8))
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
        
        # 3D visualization
        ax1 = fig.add_subplot(gs[0], projection='3d')
        # Angular velocity plot
        ax2 = fig.add_subplot(gs[1])
        
        faces = self._create_cubesat_geometry(cubesat_dimensions)
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Get state for this frame
            state = simulation_results['states'][frame]
            time = simulation_results['time'][:frame+1]
            quaternion = state['quaternion']
            angular_velocity = state['angular_velocity']
            
            # Calculate rotation matrix
            R = Quaternion(quaternion).to_matrix()
            
            # Plot CubeSat
            self._plot_faces(ax1, faces, R, self.theme['body'])
            
            # Draw axes and vectors
            for axis, color in zip(np.eye(3), ['r', 'g', 'b']):
                self._draw_vector(ax1, [0,0,0], R @ axis, color=color)
            
            if 'magnetic_field' in state:
                self._draw_vector(ax1, [0,0,0], 
                                state['magnetic_field'], 
                                color=self.theme['text'])
            
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
                                         for s in simulation_results['states'][:frame+1]])
            
            ax2.plot(time, angular_velocities[:,0], 'r-', label='ωx')
            ax2.plot(time, angular_velocities[:,1], 'g-', label='ωy')
            ax2.plot(time, angular_velocities[:,2], 'b-', label='ωz')
            ax2.plot(time, np.linalg.norm(angular_velocities, axis=1), 
                    'k--', label='|ω|')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(simulation_results['states']),
            interval=50,
            repeat=True
        )
        
        return fig, anim