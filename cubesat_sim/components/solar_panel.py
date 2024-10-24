# cubesat_sim/components/solar_panel.py

from typing import Dict, Any
import numpy as np
from ..utils.constants import Constants  # Update import
from .component import Component

class SolarPanel(Component):
    """
    Solar panel component with deployable accordion-style panels.
    """
    
    def __init__(
        self,
        name: str,
        height: float,
        width: float,
        thickness: float,
        num_folds: int,
        mounting_position: np.ndarray,
        mounting_orientation: np.ndarray,
        deployment_angle: float = None,
        mass_per_area: float = Constants.SOLAR_PANEL_DENSITY  # Update reference
    ):
        """
        Initialize deployable solar panel.
        
        Args:
            name: Unique identifier for this panel
            height: Panel height (Z dimension) in meters
            width: Panel width (Y dimension) in meters
            thickness: Panel thickness (X dimension) in meters
            num_folds: Number of accordion folds
            mounting_position: Position vector [x,y,z] relative to COM
            mounting_orientation: Orientation vector [rx,ry,rz] in radians
            deployment_angle: Current deployment angle (None = fully deployed)
            mass_per_area: Panel mass per unit area (kg/m²)
        """
        # Calculate total mass
        panel_area = height * width * num_folds
        total_mass = panel_area * mass_per_area
        
        super().__init__(name, total_mass, mounting_position, mounting_orientation)
        
        self.height = height
        self.width = width
        self.thickness = thickness
        self.num_folds = num_folds
        self.deployment_angle = deployment_angle
        self.mass_per_area = mass_per_area
        
    def calculate_inertia_tensor(self) -> np.ndarray:
        """Calculate inertia tensor for the panel."""
        if self.deployment_angle is None:
            # Fully deployed - treat as single rectangular plate
            # Assumes deployment along Y axis
            Ixx = (self.mass/12) * (self.height**2 + (self.width * self.num_folds)**2)
            Iyy = (self.mass/12) * (self.height**2 + self.thickness**2)
            Izz = (self.mass/12) * ((self.width * self.num_folds)**2 + self.thickness**2)
            
            return np.diag([Ixx, Iyy, Izz])
        else:
            # TODO: Implement partially deployed inertia tensor
            # Would need to consider accordion fold geometry
            raise NotImplementedError("Partial deployment not yet supported")
    
    def get_properties(self) -> Dict[str, Any]:
        """Get panel properties."""
        properties = super().get_properties()
        properties.update({
            'height': self.height,
            'width': self.width,
            'thickness': self.thickness,
            'num_folds': self.num_folds,
            'deployment_angle': self.deployment_angle,
            'mass_per_area': self.mass_per_area,
            'total_area': self.height * self.width * self.num_folds
        })
        return properties