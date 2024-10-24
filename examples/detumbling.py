# examples/detumbling.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cubesat_sim import CubeSat, Simulator, SimulationConfig
from cubesat_sim.components import Magnetorquer
from cubesat_sim.forces import AerodynamicForce, GravityGradient, MagneticForce
from cubesat_sim.visualization import SimulationPlotter, create_detumbling_animation

def main():
    """Run a complete detumbling simulation with visualization."""
    
    # Create and configure CubeSat
    print("Initializing 2U CubeSat...")
    cubesat = CubeSat(size_units=2)  # 2U CubeSat
    
    # Add magnetorquers
    print("Adding magnetorquers...")
    mtqs = [
        Magnetorquer("MTQ_X", axis=[1,0,0], max_dipole=0.2),
        Magnetorquer("MTQ_Y", axis=[0,1,0], max_dipole=0.2),
        Magnetorquer("MTQ_Z", axis=[0,0,1], max_dipole=0.05)
    ]
    for mtq in mtqs:
        cubesat.add_component(mtq)
    
    # Set up environmental forces
    print("Configuring environmental forces...")
    forces = [
        AerodynamicForce(name="aero", reference_altitude=400000),
        GravityGradient(name="gravity"),
        MagneticForce(name="magnetic_dist",
                     residual_dipole=[0.001, 0.001, 0.001])
    ]
    for force in forces:
        # force.enabled = True if isinstance(force, MagneticForce) else False
        cubesat.add_force(force)
    
    # Configure and create simulator
    print("Setting up simulator...")
    config = SimulationConfig(
        dt=0.1,           # 100ms time step
        duration=1.0,  
        save_frequency=1  # Save every step
    )
    sim = Simulator(cubesat, config)
    
    # Set up initial tumbling state
    print("Initializing tumbling state...")
    sim.setup_tumbling_state(
        max_angular_velocity=0.2,  # 0.2 rad/s initial tumbling
        random_attitude=True
    )
    
    # Run simulation
    print("Running detumbling simulation...")
    results = sim.run(scenarios=['detumble'])


    # Create visualizations
    print("Generating visualizations...")
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create summary plots
    plotter = SimulationPlotter()
    fig, axes, anim_fig, anim = plotter.create_detumbling_summary(
        results,
        cubesat.dimensions,
        save_path="detumbling_summary.png"
    )
    plt.show()
    
    # Show plots
    plt.show()
    
    return results, fig

if __name__ == "__main__":
    results, fig = main()