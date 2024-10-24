# cubesat_sim/utils/quaternion.py

import numpy as np
from typing import Union, Tuple
from scipy.spatial.transform import Rotation

class Quaternion:
    """
    Quaternion utilities for attitude representation and operations.
    
    Convention used: scalar-last [x, y, z, w] for internal representation
    but provides methods for scalar-first [w, x, y, z] conversion.
    """
    
    def __init__(self, vec: Union[np.ndarray, list]):
        """
        Initialize quaternion. Accepts either [x,y,z,w] or [w,x,y,z] format
        and converts to internal [x,y,z,w] representation.
        
        Args:
            vec: Quaternion values in either [x,y,z,w] or [w,x,y,z] format
        """
        vec = np.array(vec, dtype=float)
        if len(vec) != 4:
            raise ValueError("Quaternion must have exactly 4 elements")
            
        # Normalize the quaternion
        vec = vec / np.linalg.norm(vec)
        self._q = vec
        
    @classmethod
    def from_scalar_first(cls, vec: Union[np.ndarray, list]) -> 'Quaternion':
        """Create from [w,x,y,z] format."""
        vec = np.array(vec)
        return cls([vec[1], vec[2], vec[3], vec[0]])
    
    @classmethod
    def from_scalar_last(cls, vec: Union[np.ndarray, list]) -> 'Quaternion':
        """Create from [x,y,z,w] format."""
        return cls(vec)
    
    @classmethod
    def from_euler(cls, angles: Union[np.ndarray, list], sequence: str = 'ZYX') -> 'Quaternion':
        """
        Create quaternion from euler angles.
        
        Args:
            angles: Euler angles in radians [phi, theta, psi]
            sequence: Rotation sequence (default: 'ZYX')
        """
        rot = Rotation.from_euler(sequence, angles)
        quat = rot.as_quat()  # Returns scalar-last
        return cls(quat)
    
    @property
    def scalar_last(self) -> np.ndarray:
        """Get quaternion in [x,y,z,w] format."""
        return self._q
    
    @property
    def scalar_first(self) -> np.ndarray:
        """Get quaternion in [w,x,y,z] format."""
        return np.array([self._q[3], self._q[0], self._q[1], self._q[2]])
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        return Rotation.from_quat(self._q).as_matrix()
    
    def to_euler(self, sequence: str = 'ZYX') -> np.ndarray:
        """Convert to euler angles in specified sequence."""
        return Rotation.from_quat(self._q).as_euler(sequence)
    
    def conjugate(self) -> 'Quaternion':
        """Return conjugate quaternion."""
        return Quaternion([-self._q[0], -self._q[1], -self._q[2], self._q[3]])
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Quaternion multiplication (composition of rotations).
        
        Args:
            other: Another quaternion
            
        Returns:
            New quaternion representing combined rotation
        """
        x1, y1, z1, w1 = self._q
        x2, y2, z2, w2 = other._q
        
        return Quaternion([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    @staticmethod
    def derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Calculate quaternion derivative given angular velocity.
        
        Args:
            q: Current quaternion [x,y,z,w]
            omega: Angular velocity vector [ωx,ωy,ωz]
            
        Returns:
            Quaternion derivative [ẋ,ẏ,ż,ẇ]
        """
        omega_matrix = np.array([
            [0, -omega[2], omega[1], omega[0]],
            [omega[2], 0, -omega[0], omega[1]],
            [-omega[1], omega[0], 0, omega[2]],
            [-omega[0], -omega[1], -omega[2], 0]
        ])
        
        return 0.5 * omega_matrix @ q