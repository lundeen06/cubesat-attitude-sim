# cubesat-attitude-sim
 
```mermaid
%%{init: {'theme': 'dark'}}%%
classDiagram
    class CubeSat {
        +dimensions
        +mass_properties
        +components[]
        +state
        +calculate_inertia_tensor()
        +get_state()
        +update_state()
    }
    
    class Component {
        <<abstract>>
        +mass
        +position
        +orientation
        +get_properties()
    }
    
    class SolarPanel {
        +dimensions
        +num_folds
        +deployment_angle
    }
    
    class ReactionWheel {
        +max_torque
        +moment_of_inertia
        +current_speed
    }
    
    class Magnetorquer {
        +max_dipole
        +orientation
    }
    
    class Force {
        <<abstract>>
        +calculate_torque()
        +calculate_force()
    }
    
    class AerodynamicForce {
        +density
        +velocity
        +calculate_drag()
    }
    
    class GravityGradient {
        +orbital_radius
        +earth_mu
    }
    
    class MagneticForce {
        +magnetic_field
        +calculate_magnetic_torque()
    }
    
    class Simulator {
        +cubesat
        +forces[]
        +step()
        +run_simulation()
    }
    
    class Visualizer {
        +render_cubesat()
        +render_forces()
        +animate()
    }

    CubeSat --> Component
    Component <|-- SolarPanel
    Component <|-- ReactionWheel
    Component <|-- Magnetorquer
    Force <|-- AerodynamicForce
    Force <|-- GravityGradient
    Force <|-- MagneticForce
    Simulator --> CubeSat
    Simulator --> Force
    Visualizer --> Simulator
```
