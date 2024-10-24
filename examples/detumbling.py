import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cubesat_sim import CubeSat, Simulator, SimulationConfig
from cubesat_sim.components import Magnetorquer, SolarPanel
from cubesat_sim.forces import AerodynamicForce, GravityGradient, MagneticForce
from cubesat_sim.visualization import SimulationPlotter, create_detumbling_animation

def run_detumbling_simulation():
    """Run a complete detumbling simulation with visualization."""
    
    # 1. Create and configure CubeSat
    print("Initializing 3U CubeSat...")
    cubesat = CubeSat(size_units=3)  # 3U CubeSat
    
    # Add magnetorquers (typical values for CubeSat magnetorquers)
    print("Adding magnetorquers...")
    mtqs = [
        Magnetorquer("MTQ_X", axis=[1,0,0], max_dipole=0.2),  # 0.2 A⋅m² is typical
        Magnetorquer("MTQ_Y", axis=[0,1,0], max_dipole=0.2),
        Magnetorquer("MTQ_Z", axis=[0,0,1], max_dipole=0.2)
    ]
    for mtq in mtqs:
        cubesat.add_component(mtq)
    
    # Add deployable solar panels (optional - adds interesting dynamics)
    print("Adding solar panels...")
    panels = [
        SolarPanel("Panel_X+", 
                  height=0.3,    # 30cm height
                  width=0.1,     # 10cm width
                  thickness=0.002,# 2mm thickness
                  num_folds=3,
                  mounting_position=[0.05, 0, 0],  # Slight offset from center
                  mounting_orientation=[0, 0, 0]),
        SolarPanel("Panel_X-",
                  height=0.3,
                  width=0.1,
                  thickness=0.002,
                  num_folds=3,
                  mounting_position=[-0.05, 0, 0],
                  mounting_orientation=[0, np.pi, 0])
    ]
    for panel in panels:
        cubesat.add_component(panel)
    
    # 2. Set up environmental forces
    print("Configuring environmental forces...")
    forces = [
        AerodynamicForce(name="aero", 
                        reference_altitude=400000),  # 400km
        GravityGradient(name="gravity"),
        MagneticForce(name="magnetic_dist",
                     residual_dipole=[0.001, 0.001, 0.001])  # Small residual dipole
    ]
    for force in forces:
        cubesat.add_force(force)
    
    # 3. Configure and create simulator
    print("Setting up simulator...")
    config = SimulationConfig(
        dt=0.1,           # 100ms time step
        duration=300.0,   # 5 minutes
        save_frequency=1  # Save every step
    )
    sim = Simulator(cubesat, config)
    
    # 4. Set up initial tumbling state
    print("Initializing tumbling state...")
    sim.setup_tumbling_state(
        max_angular_velocity=0.2,  # 0.2 rad/s initial tumbling
        random_attitude=True
    )
    
    # 5. Run simulation
    print("Running detumbling simulation...")
    results = sim.run(scenarios=['detumble'])
    
    # 6. Create visualizations
    print("Generating visualizations...")
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create summary plots
    plotter = SimulationPlotter()
    fig, axes = plotter.create_detumbling_summary(
        results,
        save_path=output_dir / "detumbling_summary.png"
    )
    
    # Create animation
    anim = create_detumbling_animation(results, cubesat)
    anim.save(output_dir / "detumbling.mp4")
    
    print(f"Results saved to {output_dir}")
    return results, fig, anim

if __name__ == "__main__":
    results, fig, anim = run_detumbling_simulation()
    plt.show()