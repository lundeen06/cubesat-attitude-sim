# cubesat_sim/utils/constants.py

"""Physical constants for CubeSat simulation"""

# Standard Earth Parameters
EARTH_RADIUS = 6371000  # m
EARTH_MU = 3.986004418e14  # m³/s² (Earth's gravitational parameter)
EARTH_J2 = 1.082635854e-3  # Earth's J2 coefficient

# Orbital Environment
LOW_EARTH_ORBIT_HEIGHT = 400000  # m (typical LEO altitude)
ORBITAL_VELOCITY_LEO = 7700  # m/s (typical)

# Atmospheric Properties (by altitude)
ATMOSPHERIC_DENSITY = {
    200000: 2.54e-10,  # kg/m³ at 200 km
    300000: 1.95e-11,  # kg/m³ at 300 km
    400000: 1.66e-12,  # kg/m³ at 400 km
}

# CubeSat Standard Unit
CUBESAT_UNIT = 0.1  # m (10 cm)
CUBESAT_UNIT_MASS = 1.33  # kg (typical 1U mass)

# Material Properties
SOLAR_PANEL_DENSITY = 2.7  # kg/m² (typical)
ALUMINUM_DENSITY = 2700  # kg/m³

# Magnetic Field
EARTH_MAGNETIC_DIPOLE = 7.96e15  # T⋅m³