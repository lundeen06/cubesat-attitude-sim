# cubesat_sim/utils/transformations.py

import numpy as np
from typing import Tuple
from .constants import Constants

class Transformations:
    """
    Coordinate transformation utilities for spacecraft applications.
    
    Handles conversions between:
    - ECI (Earth-Centered Inertial)
    - ECEF (Earth-Centered Earth-Fixed)
    - LLA (Latitude, Longitude, Altitude)
    - NED (North, East, Down)
    - Orbital frame (radial, along-track, cross-track)
    """
    
    @staticmethod
    def eci_to_ecef(position_eci: np.ndarray, velocity_eci: np.ndarray, 
                    time_since_epoch: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ECI to ECEF coordinates.
        
        Args:
            position_eci: Position vector in ECI frame [x,y,z]
            velocity_eci: Velocity vector in ECI frame [vx,vy,vz]
            time_since_epoch: Time since epoch in seconds
            
        Returns:
            Tuple of (position_ecef, velocity_ecef)
        """
        # Earth's rotation rate (rad/s)
        earth_rate = 7.2921150e-5
        
        # Rotation angle since epoch
        theta = earth_rate * time_since_epoch
        
        # Rotation matrix from ECI to ECEF
        R = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Angular velocity vector of ECEF frame
        omega_earth = np.array([0, 0, earth_rate])
        
        # Transform position
        position_ecef = R @ position_eci
        
        # Transform velocity (including Coriolis effect)
        velocity_ecef = R @ velocity_eci - np.cross(omega_earth, position_ecef)
        
        return position_ecef, velocity_ecef
    
    @staticmethod
    @staticmethod
    def ecef_to_lla(position_ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to LLA (Latitude, Longitude, Altitude).
        
        Args:
            position_ecef: Position vector in ECEF frame [x,y,z]
            
        Returns:
            Tuple of (latitude_rad, longitude_rad, altitude_m)
        """
        x, y, z = position_ecef
        
        # WGS84 parameters
        a = Constants.EARTH_RADIUS  # Update reference
        e = 0.0818191908425  # eccentricity
        
        # Calculate longitude
        longitude = np.arctan2(y, x)
        
        # Initialize values for iteration
        p = np.sqrt(x**2 + y**2)
        latitude = np.arctan2(z, p * (1 - e**2))
        
        # Iterate to find latitude and altitude
        for _ in range(10):  # Usually converges in 2-3 iterations
            N = a / np.sqrt(1 - (e * np.sin(latitude))**2)
            altitude = p / np.cos(latitude) - N
            latitude = np.arctan2(z, p * (1 - e**2 * N/(N + altitude)))
        
        return latitude, longitude, altitude
    
    @staticmethod
    def lla_to_ned(
        lat: float, lon: float, alt: float,
        ref_lat: float, ref_lon: float, ref_alt: float
    ) -> np.ndarray:
        """
        Convert LLA to NED coordinates relative to reference point.
        
        Args:
            lat, lon, alt: Position in radians and meters
            ref_lat, ref_lon, ref_alt: Reference position in radians and meters
            
        Returns:
            Position vector in NED frame [north,east,down]
        """
        # Earth radius at reference
        R = Constants.EARTH_RADIUS  # Update reference
        
        
        # Convert to meters
        d_lat = (lat - ref_lat) * R
        d_lon = (lon - ref_lon) * R * np.cos(ref_lat)
        d_alt = ref_alt - alt
        
        return np.array([d_lat, d_lon, d_alt])
    
    @staticmethod
    def orbital_frame(
        position: np.ndarray,
        velocity: np.ndarray
    ) -> np.ndarray:
        """
        Calculate orbital reference frame vectors.
        
        Args:
            position: Position vector in ECI [x,y,z]
            velocity: Velocity vector in ECI [vx,vy,vz]
            
        Returns:
            3x3 matrix where rows are [radial, alongtrack, crosstrack] unit vectors
        """
        # Radial unit vector (R)
        R = position / np.linalg.norm(position)
        
        # Cross-track unit vector (H)
        H = np.cross(position, velocity)
        H = H / np.linalg.norm(H)
        
        # Along-track unit vector (S)
        S = np.cross(H, R)
        
        return np.vstack([R, S, H])