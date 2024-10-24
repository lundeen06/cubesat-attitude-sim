# cubesat_sim/state.py

from typing import Dict, Any, Optional
import numpy as np
from scipy.spatial.transform import Rotation

from .utils.quaternion import Quaternion

class AttitudeState:
    """
    Manages spacecraft attitude state.
    
    Handles:
    - Attitude representation (quaternions)
    - Angular velocity
    - State conversions and transformations
    - State validation and normalization
    """
    
    def __init__(
        self,
        quaternion: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
        time: float = 0.0
    ):
        """
        Initialize attitude state.
        
        Args:
            quaternion: Initial quaternion [x,y,z,w] (scalar last)
                       Defaults to identity quaternion
            angular_velocity: Initial angular velocity [ωx,ωy,ωz] rad/s
                            Defaults to zero angular velocity
            time: Initial time in seconds
        """
        # Initialize attitude quaternion (defaults to identity)
        if quaternion is None:
            self._quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self._quaternion = np.array(quaternion)
            self._normalize_quaternion()
            
        # Initialize angular velocity (defaults to zero)
        if angular_velocity is None:
            self._angular_velocity = np.zeros(3)
        else:
            self._angular_velocity = np.array(angular_velocity)
            
        self._time = time
        
        # Cache for derived quantities
        self._rotation_matrix = None
        self._euler_angles = None
        self._cached_quaternion = None
        
    def _normalize_quaternion(self):
        """Ensure quaternion is normalized."""
        norm = np.linalg.norm(self._quaternion)
        if norm < 1e-10:
            raise ValueError("Zero quaternion encountered")
        self._quaternion = self._quaternion / norm
        
    def _invalidate_cache(self):
        """Invalidate cached rotation representations."""
        self._rotation_matrix = None
        self._euler_angles = None
        self._cached_quaternion = None
        
    @property
    def quaternion(self) -> np.ndarray:
        """Get attitude quaternion (scalar last)."""
        return self._quaternion
    
    @quaternion.setter
    def quaternion(self, value: np.ndarray):
        """Set attitude quaternion."""
        self._quaternion = np.array(value)
        self._normalize_quaternion()
        self._invalidate_cache()
        
    @property
    def angular_velocity(self) -> np.ndarray:
        """Get angular velocity vector."""
        return self._angular_velocity
    
    @angular_velocity.setter
    def angular_velocity(self, value: np.ndarray):
        """Set angular velocity vector."""
        self._angular_velocity = np.array(value)
        
    @property
    def time(self) -> float:
        """Get current state time."""
        return self._time
    
    @time.setter
    def time(self, value: float):
        """Set state time."""
        self._time = float(value)
        
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix representation (cached)."""
        if (self._rotation_matrix is None or 
            not np.array_equal(self._cached_quaternion, self._quaternion)):
            self._cached_quaternion = self._quaternion.copy()
            self._rotation_matrix = Quaternion(self._quaternion).to_matrix()
        return self._rotation_matrix
    
    def get_euler_angles(self, sequence: str = 'ZYX') -> np.ndarray:
        """
        Get Euler angles for specified rotation sequence.
        
        Args:
            sequence: Rotation sequence (e.g., 'ZYX', 'XYZ')
            
        Returns:
            Array of Euler angles in radians
        """
        return Quaternion(self._quaternion).to_euler(sequence)
    
    def from_euler_angles(self, angles: np.ndarray, sequence: str = 'ZYX'):
        """
        Set attitude from Euler angles.
        
        Args:
            angles: Euler angles in radians
            sequence: Rotation sequence (e.g., 'ZYX', 'XYZ')
        """
        self.quaternion = Quaternion.from_euler(angles, sequence).scalar_last
        
    def propagate(self, dt: float):
        """
        Propagate state forward by dt seconds using current angular velocity.
        
        Args:
            dt: Time step in seconds
        """
        # Calculate quaternion derivative
        q_dot = Quaternion.derivative(self._quaternion, self._angular_velocity)
        
        # Integrate quaternion
        self._quaternion += q_dot * dt
        self._normalize_quaternion()
        self._invalidate_cache()
        
        # Update time
        self._time += dt
        
    def get_angular_momentum(self, inertia_tensor: np.ndarray) -> np.ndarray:
        """
        Calculate angular momentum vector.
        
        Args:
            inertia_tensor: 3x3 inertia tensor
            
        Returns:
            Angular momentum vector in kg⋅m²/s
        """
        return inertia_tensor @ self._angular_velocity
    
    def get_kinetic_energy(self, inertia_tensor: np.ndarray) -> float:
        """
        Calculate rotational kinetic energy.
        
        Args:
            inertia_tensor: 3x3 inertia tensor
            
        Returns:
            Rotational kinetic energy in Joules
        """
        return 0.5 * self._angular_velocity @ (inertia_tensor @ self._angular_velocity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'quaternion': self._quaternion.tolist(),
            'angular_velocity': self._angular_velocity.tolist(),
            'time': self._time,
            'rotation_matrix': self.rotation_matrix.tolist(),
            'euler_angles_zyx': self.get_euler_angles('ZYX').tolist()
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'AttitudeState':
        """Create state from dictionary."""
        return cls(
            quaternion=np.array(state_dict['quaternion']),
            angular_velocity=np.array(state_dict['angular_velocity']),
            time=state_dict['time']
        )
    
    def __str__(self) -> str:
        """String representation."""
        euler = np.rad2deg(self.get_euler_angles())
        return (f"Time: {self._time:.2f}s\n"
                f"Euler (deg): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]\n"
                f"Angular Velocity (rad/s): {self._angular_velocity}")