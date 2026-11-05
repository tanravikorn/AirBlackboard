"""
filters.py - Real-time smoothing filters for hand landmark tracking and drawing.

Includes:
- OneEuroFilter: Adaptive low-pass filter (Casiez et al., CHI 2012)
  Provides near-zero lag during fast movement and heavy jitter removal during stationary/slow movement.
- ExponentialMovingAverage: Simple low-pass filter for complementary smoothing.
"""

import math
import numpy as np
from typing import Optional, Union


def _smoothing_factor(t_e: float, cutoff: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate low-pass alpha smoothing coefficient."""
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(alpha: Union[float, np.ndarray],
                           x: Union[float, np.ndarray],
                           x_prev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Apply exponential low-pass smoothing: alpha * x + (1 - alpha) * x_prev."""
    return alpha * x + (1.0 - alpha) * x_prev


class LowPassFilter:
    """Standard 1st-order IIR Low-Pass Filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[Union[float, np.ndarray]] = None

    def reset(self):
        self.s = None

    def filter(self, x: Union[float, np.ndarray], alpha: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = np.copy(x) if isinstance(x, np.ndarray) else x
            return self.s
        self.s = _exponential_smoothing(self.alpha, x, self.s)
        return self.s


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter for smooth, low-latency tracking.

    References:
        Casiez, G., Roussel, N. and Vogel, F. (2012).
        1 € Filter: A Simple Speed-based Low-pass Filter for Noisy Input in HCI.
        Proc. CHI '12, ACM, pp. 2527-2530.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        Parameters
        ----------
        min_cutoff : float
            Minimum cutoff frequency in Hz. Lower values eliminate more jitter
            at low velocities / stationary gestures.
        beta : float
            Speed coefficient. Higher values adapt quicker to high speeds,
            eliminating lag during rapid strokes.
        d_cutoff : float
            Cutoff frequency for the derivative filter in Hz (typically 1.0 Hz).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.t_prev: Optional[float] = None
        self.x_prev: Optional[Union[float, np.ndarray]] = None

    def reset(self):
        """Reset filter state."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.t_prev = None
        self.x_prev = None

    def filter(
        self,
        x: Union[float, np.ndarray, list, tuple],
        timestamp: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Filter an input value or array at a given timestamp (in seconds).
        If timestamp is None, an assumed delta t = 1/30s is used.
        """
        is_array = isinstance(x, (np.ndarray, list, tuple))
        if is_array:
            x_arr = np.asarray(x, dtype=np.float64)
        else:
            x_arr = float(x)

        if self.t_prev is None or self.x_prev is None:
            self.t_prev = timestamp
            self.x_prev = np.copy(x_arr) if is_array else x_arr
            self.x_filter.filter(x_arr, alpha=1.0)
            self.dx_filter.filter(np.zeros_like(x_arr) if is_array else 0.0, alpha=1.0)
            return np.copy(x_arr) if is_array else x_arr

        if timestamp is not None and timestamp > self.t_prev:
            t_e = timestamp - self.t_prev
        else:
            t_e = 1.0 / 30.0  # fallback ~30 fps

        # Prevent division by zero or extreme timesteps
        t_e = max(t_e, 1e-4)

        # Estimate derivative (speed)
        dx = (x_arr - self.x_prev) / t_e

        # Filter the derivative
        alpha_d = _smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.dx_filter.filter(dx, alpha=alpha_d)

        # Calculate adaptive cutoff frequency
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal
        alpha = _smoothing_factor(t_e, cutoff)
        x_hat = self.x_filter.filter(x_arr, alpha=alpha)

        self.t_prev = timestamp
        self.x_prev = np.copy(x_hat) if is_array else x_hat

        return x_hat


class ExponentialMovingAverage:
    """Exponential Moving Average filter with controllable alpha factor."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self.val: Optional[Union[float, np.ndarray]] = None

    def reset(self):
        self.val = None

    def filter(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.val is None:
            self.val = np.copy(x) if isinstance(x, np.ndarray) else x
            return self.val
        self.val = self.alpha * x + (1.0 - self.alpha) * self.val
        return self.val
