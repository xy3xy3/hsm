"""
VLM Utilities

Utility functions for VLM operations.
"""

import numpy as np


def round_nested_values(data, decimals=4):
    """
    Recursively round all float values in nested dictionaries, lists, and tuples to specified decimal places.

    Args:
        data: The input data structure (dict, list, tuple, or primitive type)
        decimals: Number of decimal places to round to (default: 4)

    Returns:
        The data structure with all float values rounded
    """
    if isinstance(data, dict):
        return {key: round_nested_values(value, decimals) for key, value in data.items()}
    elif isinstance(data, list):
        return [round_nested_values(item, decimals) for item in data]
    elif isinstance(data, tuple):
        return tuple(round_nested_values(item, decimals) for item in data)
    elif isinstance(data, (float, np.float32, np.float64)):
        return round(float(data), decimals)
    return data


def extract_program(response: str, description: str):
    """
    Extract the program from the response of the VLM.

    Args:
        response: string, the response from the VLM
        description: string, the description of the program

    Returns:
        program: Program, the program extracted from the response
    """
    from hsm_core.scene_motif.programs.program import Program

    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    code = response.split("\n")
    program = Program(code, description)

    return program


def extract_code(response: str) -> str:
    """
    Extract the code from the response of the VLM.

    Args:
        response: string, the response from the VLM

    Returns:
        code: string, the code extracted from the response
    """

    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response


def extract_json(response: str) -> str:
    """
    Extract the JSON string from the response of the VLM.

    Args:
        response: string, the response from the VLM

    Returns:
        str: The extracted JSON string
    """

    if "```json" in response:
        response = response.split("```json\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response
