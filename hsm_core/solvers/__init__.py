"""
Modular Spatial Optimization System

This package provides a flexible, configurable spatial optimization system for 3D scenes.
The system uses an advanced mesh-based collision detection engine with room geometry integration.

Key Components:
- UnifiedSceneSpatialOptimizer: The primary, mesh-based spatial optimizer.
- SpatialOptimizerConfig: A simple configuration system for the optimizer.

Usage:
    from hsm_core.solvers import UnifiedSceneSpatialOptimizer, SpatialOptimizerConfig
    
    config = SpatialOptimizerConfig()
    optimizer = UnifiedSceneSpatialOptimizer(scene, config)
    optimized_objects = optimizer.optimize_objects(objects)
"""

# Scene Spatial optimizer
from .scene_spatial_optimizer import SceneSpatialOptimizer

# Scene Motif Spatial optimizer
from hsm_core.scene_motif.spatial import optimize_sm

# Configuration
from .config import SceneSpatialOptimizerConfig

# Public API
__all__ = [
    # Scene Spatial optimizer
    'SceneSpatialOptimizer',

    # Scene Motif Spatial optimizer
    'optimize_sm',

    # Configuration
    'SceneSpatialOptimizerConfig',
]
