# cubesat_sim/visualization/animate.py

from IPython.display import HTML
import matplotlib.animation as animation
from .renderer import CubeSatRenderer

def create_detumbling_animation(simulation_results, cubesat):
    """Create animation of detumbling process."""
    renderer = CubeSatRenderer()
    fig, anim = renderer.create_animation(
        simulation_results,
        cubesat.dimensions
    )
    
    return HTML(anim.to_jshtml())

def save_animation(anim, filename, fps=30):
    """Save animation to file."""
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='CubeSatSim'),
                   bitrate=1800)
    anim.save(filename, writer=writer)