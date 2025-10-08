"""
This module provides a function to flatten a nested dictionary.
"""
from typing import Dict, Any

def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested dictionary, creating new keys by concatenating nested keys with '/'.

    Args:
        d (Dict[str, Any]): The nested dictionary to flatten.

    Returns:
        Dict[str, Any]: A new, single-level dictionary.
    """
    out_dict = {}
    _flatten_recursive(d, out_dict, "")
    return out_dict

def _flatten_recursive(d: Dict[str, Any], out_dict: Dict[str, Any], tag: str):
    """
    Helper function to recursively flatten the dictionary.
    """
    for key, val in d.items():
        if isinstance(val, dict):
            new_tag = f"{tag}{key}/"
            _flatten_recursive(val, out_dict, new_tag)
        else:
            new_key = f"{tag}{key}"
            out_dict[new_key] = val 