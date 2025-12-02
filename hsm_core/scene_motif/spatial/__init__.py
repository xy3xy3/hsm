"""
Spatial Optimization for scene motifs
"""

from .hierarchical_optimizer import optimize_hierarchical
from .spatial_optimizer import optimize

def optimize_sm(arrangement, hierarchy=None, **kwargs):
    """
    Spatial optimization entry point

    Args:
        arrangement: The arrangement to optimize
        hierarchy: Optional MotifHierarchy for hierarchical optimization
        **kwargs: Additional optimization parameters

    Returns:
        Optimized arrangement
    """
    # Check if any objects have meshes before attempting optimization
    objs_with_meshes = [obj for obj in arrangement.objs if hasattr(obj, 'has_mesh') and obj.has_mesh]
    if not objs_with_meshes:
        from hsm_core.utils import get_logger
        logger = get_logger('scene_motif.spatial')
        logger.debug(f"Skipping spatial optimization - no objects have meshes. Objects: {[obj.label for obj in arrangement.objs]}")
        return arrangement  # Return unchanged arrangement

    if hierarchy and hierarchy.root:
        optimized_hierarchy = optimize_hierarchical(hierarchy, **kwargs)
        
        if optimized_hierarchy.root and optimized_hierarchy.root.arrangement:
            result_arrangement = optimized_hierarchy.root.arrangement
            # Attach hierarchy for later extraction
            setattr(result_arrangement, '_hierarchy', optimized_hierarchy)
            return result_arrangement
        else:
            from hsm_core.utils import get_logger
            logger = get_logger('scene_motif.spatial')
            logger.warning("Hierarchical optimization failed, falling back to standard optimization")
            return optimize(arrangement, **kwargs)
    else:
        return optimize(arrangement, **kwargs)

__all__ = [
    'optimize_sm'
]
