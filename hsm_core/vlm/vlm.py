"""
VLM Factory Module

This module provides a unified interface for creating VLM sessions
"""

from __future__ import annotations

from typing import Any, Callable, Dict

_VLM_REGISTRY = {}

def register_vlm_backend(name: str, factory_func: Callable):
    """Register a VLM backend factory function."""
    _VLM_REGISTRY[name] = factory_func


def get_session_config(cfg: Any | None = None, default_model_type: str = "gpt") -> Dict[str, str | None]:
    """Extract session configuration from a config object."""
    llm_config = getattr(cfg, "llm", None) if cfg is not None else None
    return {
        "model_type": getattr(llm_config, "model_type", default_model_type) if llm_config else default_model_type,
        "model_name": getattr(llm_config, "model_name", None) if llm_config else None,
        "base_url": getattr(llm_config, "base_url", None) if llm_config else None,
    }

def create_session(
    prompts_path: str, 
    model_type: str = "gpt", 
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7, 
    output_dir: str = "", 
    prompt_info: Dict[str, str] | None = None,
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
    kwargs.pop('base_url', None)
    return factory(prompts_path, model=model_name, temperature=temperature, 
                   base_url=base_url, output_dir=output_dir, prompt_info=prompt_info or {}, **kwargs)

def _create_gpt_session(prompts_path: str, model=None, base_url=None, **kwargs):
    from hsm_core.vlm.gpt import Session
    # Remove model from kwargs to avoid duplicate argument
    kwargs.pop('model', None)
    kwargs.pop('base_url', None)
    return Session(prompts_path, model=model, base_url=base_url, **kwargs)

register_vlm_backend("gpt", _create_gpt_session)

try:
    from hsm_core.vlm.qwen import register_qwen_backends
    register_qwen_backends()
except ImportError:
    pass  # Qwen not available
