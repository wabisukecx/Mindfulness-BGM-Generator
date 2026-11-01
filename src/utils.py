"""
Utility functions for Mindfulness BGM Generator
"""

from typing import Tuple
import math


def parse_instrument_interval(value: str) -> Tuple[float, float]:
    """
    Parse instrument interval settings from command line argument
    
    Args:
        value: String value in format "min-max", "value", or "0"
        
    Returns:
        Tuple of (min_interval, max_interval)
    """
    value = value.strip()
    try:
        if "-" in value:
            low, high = map(float, value.split("-"))
        else:
            val = float(value)
            low, high = val * 0.8, val * 1.2
    except ValueError as exc:
        raise ValueError("Use a nonnegative interval or min-max range") from exc
    if not all(math.isfinite(v) for v in (low, high)) or low < 0 or high < low:
        raise ValueError("Intervals must be finite and satisfy 0 <= min <= max")
    return low, high


def parse_ambient_value(value: str) -> float:
    """
    Parse ambient sound ratio from command line argument
    
    Args:
        value: String value for ambient ratio
        
    Returns:
        Float value between 0.0 and 1.0
    """
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("Ambient ratio must be finite")
    return max(0.0, min(1.0, val))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between minimum and maximum
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))