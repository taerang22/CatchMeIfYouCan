#!/usr/bin/env python3
"""
BallPredictor - Ballistic trajectory prediction for ball catching.

Collects position samples and predicts where/when the ball will cross
a vertical plane using projectile motion equations.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


class BallPredictor:
    """
    Predicts ball trajectory using ballistic (projectile) motion.
    
    Usage:
        predictor = BallPredictor(g=9.81)
        predictor.add_sample(t, pos)  # Add position samples
        result = predictor.predict_hit_for_vertical_plane(x_catch)
        if result:
            t_hit, p_hit = result
    """

    def __init__(self, g: float = 9.81, max_samples: int = 10):
        """
        Args:
            g: Gravitational acceleration (m/s²), positive value
            max_samples: Maximum samples to keep in buffer
        """
        self.g = g
        self.max_samples = max_samples
        
        # Sample buffer: (timestamp, position)
        self._samples: deque = deque(maxlen=max_samples)
    
    @property
    def num_samples(self) -> int:
        """Number of samples in buffer."""
        return len(self._samples)
    
    def add_sample(self, t: float, pos: np.ndarray) -> None:
        """
        Add a position sample.
        
        Args:
            t: Timestamp (seconds, any reference frame - typically ROS clock)
            pos: Position array [x, y, z] in meters
        """
        pos = np.asarray(pos, dtype=float).flatten()
        if pos.shape[0] != 3:
            raise ValueError("pos must be a 3-element array [x, y, z]")
        self._samples.append((float(t), pos.copy()))
    
    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()
    
    def get_latest_state(self) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Get latest time, position, and estimated velocity.
        
        Returns:
            (t, pos, vel) tuple, or None if < 2 samples
        """
        if len(self._samples) < 2:
            return None
        
        # Use last two samples for velocity estimation
        t1, p1 = self._samples[-2]
        t2, p2 = self._samples[-1]
        
        dt = t2 - t1
        if dt <= 1e-6:
            return None
        
        # Estimate velocity (finite difference)
        vel = (p2 - p1) / dt
        
        return t2, p2, vel
    
    def predict_hit_for_vertical_plane(
        self, 
        x_plane: float
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Predict when and where the ball crosses a vertical plane at x = x_plane.
        
        Uses ballistic motion equations:
            x(t) = x0 + vx * dt
            y(t) = y0 + vy * dt
            z(t) = z0 + vz * dt - 0.5 * g * dt²
        
        Args:
            x_plane: X-coordinate of the catch plane (meters)
        
        Returns:
            (t_hit, p_hit) where:
                t_hit: Absolute time when ball crosses plane (same reference as input)
                p_hit: Position [x_plane, y, z] at intersection
            Returns None if:
                - Not enough samples
                - Ball not moving toward plane
                - Ball won't reach plane (moving away or parallel)
        """
        state = self.get_latest_state()
        if state is None:
            return None
        
        t0, p0, v0 = state
        x0, y0, z0 = p0
        vx, vy, vz = v0
        
        # Check if ball is moving toward the plane
        dx = x_plane - x0
        
        # If velocity in x is too small or wrong direction, can't predict
        if abs(vx) < 1e-6:
            return None
        
        # Time to reach x_plane
        dt = dx / vx
        
        # Must be positive (ball moving toward plane, not away)
        if dt <= 0:
            return None
        
        # Limit prediction horizon (e.g., max 3 seconds)
        if dt > 3.0:
            return None
        
        # Predict y and z using ballistic equations
        y_hit = y0 + vy * dt
        z_hit = z0 + vz * dt - 0.5 * self.g * dt * dt
        
        # Absolute time of hit
        t_hit = t0 + dt
        
        # Position at hit
        p_hit = np.array([x_plane, y_hit, z_hit], dtype=float)
        
        return t_hit, p_hit
    
    def predict_position_at_time(self, t_target: float) -> Optional[np.ndarray]:
        """
        Predict ball position at a specific future time.
        
        Args:
            t_target: Target time (absolute, same reference as samples)
        
        Returns:
            Predicted position [x, y, z], or None if cannot predict
        """
        state = self.get_latest_state()
        if state is None:
            return None
        
        t0, p0, v0 = state
        dt = t_target - t0
        
        if dt < 0:
            return None  # Can't predict past
        
        x0, y0, z0 = p0
        vx, vy, vz = v0
        
        # Ballistic prediction
        x = x0 + vx * dt
        y = y0 + vy * dt
        z = z0 + vz * dt - 0.5 * self.g * dt * dt
        
        return np.array([x, y, z], dtype=float)

