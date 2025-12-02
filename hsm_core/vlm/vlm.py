"""
VLM Factory Module

This module provides a unified interface for creating VLM sessions
"""

from typing import Dict, Callable

_VLM_REGISTRY = {}

def register_vlm_backend(name: str, factory_func: Callable):
    """Register a VLM backend factory function."""
    _VLM_REGISTRY[name] = factory_func

def create_session(
    prompts_path: str, 
    model_type: str = "gpt", 
    model_name: str = "gpt-4o-2024-08-06",
    temperature: float = 0.7, 
    output_dir: str = "", 
    prompt_info: Dict[str, str] = {},
    **kwargs
):
    """
    Factory function to create VLM sessions using dynamic registry.
    """
    if model_type not in _VLM_REGISTRY:
        available = ', '.join(_VLM_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")
    
    factory = _VLM_REGISTRY[model_type]
    kwargs.pop('model', None)
    return factory(prompts_path, model=model_name, temperature=temperature, 
                   output_dir=output_dir, prompt_info=prompt_info, **kwargs)

def _create_gpt_session(prompts_path: str, model=None, **kwargs):
    from hsm_core.vlm.gpt import Session
    # Remove model from kwargs to avoid duplicate argument
    kwargs.pop('model', None)
    return Session(prompts_path, model=model, **kwargs)

register_vlm_backend("gpt", _create_gpt_session)

try:
    from hsm_core.vlm.qwen import register_qwen_backends
    register_qwen_backends()
except ImportError:
    pass  # Qwen not available
