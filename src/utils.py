"""
Utility functions for Mindfulness BGM Generator
"""

from typing import Tuple


def parse_instrument_interval(value: str) -> Tuple[float, float]:
    """
    Parse instrument interval settings from command line argument
    
    Args:
        value: String value in format "min-max", "value", or "0"
        
    Returns:
        Tuple of (min_interval, max_interval)
    """
    if value == "0":
        return 0, 0
    
    if "-" in value:
        min_val, max_val = value.split("-")
        return float(min_val), float(max_val)
    else:
        val = float(value)
        return val * 0.8, val * 1.2


def parse_ambient_value(value: str) -> float:
    """
    Parse ambient sound ratio from command line argument
    
    Args:
        value: String value for ambient ratio
        
    Returns:
        Float value between 0.0 and 1.0
    """
    val = float(value)
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