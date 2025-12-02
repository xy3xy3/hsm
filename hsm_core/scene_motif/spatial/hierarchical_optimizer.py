import time
import trimesh
import numpy as np
from copy import deepcopy

from .spatial_optimizer import optimize as optimize_single_arrangement
from ..core.obj import Obj
from ..core.bounding_box import BoundingBox
import logging
logger = logging.getLogger('scene_motif.spatial')


def optimize_hierarchical(
    hierarchy,
    resolve_collisions: bool = True,
    collision_move_step: float = 0.005,
    collision_max_iters: int = 1000,
    make_tight: bool = False,
    make_tight_iters: int = 10,
    approximate_gravity: bool = True,
    global_min_y: float | None = None,
):
    """Optimize arrangements hierarchically"""
    
    if not hierarchy.root:
        logger.warning("Empty hierarchy provided to hierarchical optimizer")
        return hierarchy
    
    logger.info(f"Hierarchical spatial optimization starting...")
    overall_start_time = time.time()
    
    # Phase 1: Optimize all leaf nodes (including root if it's a leaf)
    leaf_nodes = hierarchy.get_leaf_nodes()
    logger.info(f"Phase 1: Optimizing {len(leaf_nodes)} leaf nodes")
    
    for leaf_node in leaf_nodes:
        if leaf_node.arrangement and leaf_node.arrangement.objs:
            logger.info(f"  Optimizing leaf node: {leaf_node.motif_type} (depth={leaf_node.depth}) with {len(leaf_node.arrangement.objs)} objects")
            node_start_time = time.time()
            
            optimized_arrangement = optimize_single_arrangement(
                leaf_node.arrangement,
                resolve_collisions=resolve_collisions,
                collision_move_step=collision_move_step,
                collision_max_iters=collision_max_iters,
                make_tight=make_tight,
                make_tight_iters=make_tight_iters,
                approximate_gravity=approximate_gravity,
                global_min_y=global_min_y
            )
            
            hierarchy.set_arrangement(leaf_node, optimized_arrangement)
            logger.info(f"    Leaf optimization completed in {(time.time() - node_start_time):.3f}s")
        else:
            logger.info(f"  Leaf node '{leaf_node.motif_type}' at depth {leaf_node.depth} - no arrangement to optimize")
    
    # Phase 2: Optimize only non-leaf nodes (parent motifs) by respecting children as super-objects
    logger.info("Phase 2: Optimizing non-leaf arrangements (parent motifs)")
    non_leaf_nodes = [node for node in hierarchy.traverse_bottom_up() if node.children]
    
    if not non_leaf_nodes:
        logger.info("  No non-leaf nodes to optimize (simple arrangement)")
    
    for node in non_leaf_nodes:
        if node.arrangement and node.arrangement.objs:
            logger.info(f"  Optimizing parent motif: {node.motif_type} (depth={node.depth})")
            node_start_time = time.time()
            
            # Build a combined arrangement that treats each child motif as a super-object
            # and includes parent-only objects (e.g., items not part of any child motif)
            combined_with_parent = _create_parent_combined_arrangement(node)
            if combined_with_parent and len(combined_with_parent.objs) > 1:
                optimized_combined_parent = optimize_single_arrangement(
                    combined_with_parent,
                    resolve_collisions=resolve_collisions,
                    collision_move_step=collision_move_step,
                    collision_max_iters=collision_max_iters,
                    make_tight=make_tight,
                    make_tight_iters=make_tight_iters,
                    approximate_gravity=approximate_gravity,
                    global_min_y=global_min_y
                )
                _apply_parent_transforms(node, combined_with_parent, optimized_combined_parent)
            
            logger.info(f"    Completed in {(time.time() - node_start_time):.3f}s")
    
    # Phase 3: Optimize motif-to-motif relationships at each depth level
    logger.info("Phase 3: Inter-motif optimization (by depth level)")
    for depth in range(hierarchy.root.depth, max(node.depth for node in hierarchy.execution_order) + 1):
        nodes_at_depth = hierarchy.get_nodes_at_depth(depth)
        if len(nodes_at_depth) <= 1:
            continue
            
        logger.info(f"  Optimizing {len(nodes_at_depth)} motifs at depth {depth}")
        depth_start_time = time.time()
        
        combined_arrangement = _create_combined_arrangement(nodes_at_depth)
        if combined_arrangement and len(combined_arrangement.objs) > 1:
            optimized_combined = optimize_single_arrangement(
                combined_arrangement,
                resolve_collisions=resolve_collisions,
                collision_move_step=collision_move_step * 2,
                collision_max_iters=collision_max_iters // 2,
                make_tight=make_tight,
                approximate_gravity=approximate_gravity,
                global_min_y=global_min_y
            )

            _distribute_transforms_to_motifs(nodes_at_depth, combined_arrangement, optimized_combined)
        logger.info(f"    Completed depth {depth} in {(time.time() - depth_start_time):.3f}s")
    
    # Phase 4: Cache world transforms for all objects in hierarchy
    logger.info("Phase 4: Caching world transforms")
    _cache_world_transforms(hierarchy)
    
    logger.info(f"Hierarchical optimization completed in {(time.time() - overall_start_time):.3f}s")
    return hierarchy


def _create_combined_arrangement(nodes):
    """Create a combined arrangement from multiple motif nodes using super-objects."""
    from hsm_core.scene_motif import Arrangement
    if not nodes:
        return None

    combined_objs = []

    for i, node in enumerate(nodes):
        if not node.arrangement or not node.arrangement.objs:
            continue

        # Create a super-object for this motif
        motif_meshes = []
        individual_objects_before = []

        for obj in node.arrangement.objs:
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                obj_mesh = deepcopy(obj.mesh)
                if hasattr(obj, 'bounding_box') and obj.bounding_box:
                    obj_mesh.apply_transform(obj.bounding_box.no_scale_matrix)
                motif_meshes.append(obj_mesh)

            # Store individual object info for gravity redistribution
            if hasattr(obj, 'label'):
                individual_objects_before.append((obj.label, deepcopy(obj.bounding_box.centroid)))

        if not motif_meshes:
            continue

        # Ensure we get a single Trimesh object, not a Scene object
        if len(motif_meshes) == 1:
            combined_mesh = motif_meshes[0]
        else:
            combined_mesh = trimesh.util.concatenate(motif_meshes)
            # If concatenation results in a Scene, extract the first geometry
            if hasattr(combined_mesh, 'geometry') and combined_mesh.geometry:
                logger.debug(f"Concatenation resulted in Scene object with {len(combined_mesh.geometry)} geometries")
                first_geom = next(iter(combined_mesh.geometry.values()))
                if hasattr(first_geom, 'is_convex'):  # It's a Trimesh object
                    combined_mesh = first_geom
                    logger.debug("Extracted first Trimesh geometry from Scene")
                else:
                    logger.warning(f"First geometry is not a Trimesh object: {type(first_geom)}")

        bounds = combined_mesh.bounds
        centroid = combined_mesh.centroid
        half_size = (bounds[1] - bounds[0]) / 2

        super_obj = Obj(
            label=f"motif_{node.motif_type}_{i}",
            mesh=combined_mesh,
            bounding_box=BoundingBox(
                centroid=centroid,
                half_size=half_size,
                coord_axes=np.eye(3)
            )
        )

        # Store mapping for gravity redistribution
        setattr(super_obj, '_motif_node', node)
        setattr(super_obj, '_individual_objects_before', individual_objects_before)
        combined_objs.append(super_obj)

    if not combined_objs:
        return None

    return Arrangement(
        combined_objs,
        f"Combined arrangement of {len(nodes)} motifs",
        "combined_motifs()"
    )


def _create_parent_combined_arrangement(node):
    """Create a combined arrangement for a parent node by:
    - creating super-objects for child motifs (preserving rigid body relationships)
    - including parent-only objects (objects not contained in any child motif)
    """
    from hsm_core.scene_motif import Arrangement
    from hsm_core.scene_motif.core.obj import Obj
    from hsm_core.scene_motif.core.bounding_box import BoundingBox

    if not node or not node.arrangement:
        return None

    # 1) Build super-objects for each child motif (preserving internal structure)
    child_super_objs = []
    child_labels: set[str] = set()
    for i, child in enumerate(node.children):
        if not child.arrangement or not child.arrangement.objs:
            continue

        # Store individual object positions before creating super-object
        individual_objects_before = []
        for obj in child.arrangement.objs:
            if hasattr(obj, 'label'):
                child_labels.add(obj.label)
            individual_objects_before.append((obj.label, deepcopy(obj.bounding_box.centroid)))

        motif_meshes = []
        for obj in child.arrangement.objs:
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                obj_mesh = deepcopy(obj.mesh)
                if hasattr(obj, 'bounding_box') and obj.bounding_box:
                    obj_mesh.apply_transform(obj.bounding_box.no_scale_matrix)
                motif_meshes.append(obj_mesh)

        if not motif_meshes:
            continue

        # Ensure we get a single Trimesh object, not a Scene object
        if len(motif_meshes) == 1:
            combined_mesh = motif_meshes[0]
        else:
            combined_mesh = trimesh.util.concatenate(motif_meshes)
            # If concatenation results in a Scene, extract the first geometry
            if hasattr(combined_mesh, 'geometry') and combined_mesh.geometry:
                logger.debug(f"Parent concatenation resulted in Scene object with {len(combined_mesh.geometry)} geometries")
                first_geom = next(iter(combined_mesh.geometry.values()))
                if hasattr(first_geom, 'is_convex'):  # It's a Trimesh object
                    combined_mesh = first_geom
                    logger.debug("Extracted first Trimesh geometry from parent Scene")
                else:
                    logger.warning(f"Parent first geometry is not a Trimesh object: {type(first_geom)}")

        bounds = combined_mesh.bounds
        centroid = combined_mesh.centroid
        half_size = (bounds[1] - bounds[0]) / 2

        super_obj = Obj(
            label=f"motif_{child.motif_type}_{i}",
            mesh=combined_mesh,
            bounding_box=BoundingBox(
                centroid=centroid,
                half_size=half_size,
                coord_axes=np.eye(3)
            )
        )

        # Store mapping between super-object and individual objects for gravity redistribution
        setattr(super_obj, '_motif_node', child)
        setattr(super_obj, '_individual_objects_before', individual_objects_before)
        child_super_objs.append(super_obj)

    # 2) Collect parent-only objects (those not belonging to any child motif)
    parent_extra_objs = []
    for obj in node.arrangement.objs:
        try:
            label = getattr(obj, 'label', None)
            # Heuristics: skip placeholders referencing sub-arrangements; include only real meshes
            if label and (label in child_labels or label.startswith('sub_arrangements[') or label.startswith('execute_results[')):
                continue
            if hasattr(obj, 'mesh') and obj.mesh is not None:
                obj_copy = deepcopy(obj)
                parent_extra_objs.append(obj_copy)
        except Exception:
            # Be defensive; skip any problematic object
            continue

    combined_objs = child_super_objs + parent_extra_objs
    if not combined_objs:
        return None
    return Arrangement(combined_objs, f"Parent combined for {node.motif_type}", f"combined_parent({node.motif_type})")


def _apply_parent_transforms(node, original_combined, optimized_combined) -> None:
    """Distribute transforms from an optimized combined arrangement back to:
    - child motif objects (translation applied to each child's objects)
    - parent-only objects (matched by label)
    """
    if not node or not node.arrangement:
        return
    if len(original_combined.objs) != len(optimized_combined.objs):
        logger.warning("Mismatch in object count between original and optimized parent-combined arrangements")
        # Fall back to just updating the node arrangement with no distribution
        return

    # Build lookup tables for matching objects between original and optimized arrangements
    orig_objs_by_label = {}
    opt_objs_by_label = {}

    for obj in original_combined.objs:
        lbl = getattr(obj, 'label', None)
        if lbl:
            orig_objs_by_label[lbl] = obj

    for obj in optimized_combined.objs:
        lbl = getattr(obj, 'label', None)
        if lbl:
            opt_objs_by_label[lbl] = obj

    # Build a lookup for all objects in the node arrangement by label
    node_objs_by_label = {}
    for obj in node.arrangement.objs:
        lbl = getattr(obj, 'label', None)
        if lbl:
            node_objs_by_label[lbl] = obj

    # Apply transforms by matching labels between original and optimized objects
    for label, orig_obj in orig_objs_by_label.items():
        if label not in opt_objs_by_label:
            continue

        opt_obj = opt_objs_by_label[label]

        # Safely fetch centroids
        orig_bb = getattr(orig_obj, 'bounding_box', None)
        opt_bb = getattr(opt_obj, 'bounding_box', None)
        orig_centroid = orig_bb.centroid if (orig_bb is not None and hasattr(orig_bb, 'centroid')) else None
        opt_centroid = opt_bb.centroid if (opt_bb is not None and hasattr(opt_bb, 'centroid')) else None
        if orig_centroid is None or opt_centroid is None:
            continue

        translation = opt_centroid - orig_centroid

        # super-object representing a child motif
        if hasattr(orig_obj, '_motif_node') and orig_obj._motif_node:
            motif_node = orig_obj._motif_node
            individual_objects_before = getattr(orig_obj, '_individual_objects_before', [])

            logger.debug(f"Redistributing gravity transform for super-object {label} (motif: {motif_node.motif_type})")
            _redistribute_super_object_transform(motif_node, translation, individual_objects_before)

        # parent-level object
        else:
            logger.debug(f"Applying translation {translation} to parent object {label}")
            if label in node_objs_by_label:
                target_obj = node_objs_by_label[label]
                if target_obj.bounding_box is not None:
                    target_obj.bounding_box.centroid += translation
                    logger.debug(f"  Moved parent object {label} by {translation}")
            else:
                logger.warning(f"Could not find parent object {label} in node arrangement for transform application")

    # No need to replace node.arrangement; we updated objects in place
    # Ensure hierarchy remains consistent
    return


def _redistribute_super_object_transform(motif_node, super_object_translation, individual_objects_before):
    """Redistribute super-object movement back to individual objects within the motif.

    Args:
        motif_node: The motif node containing the individual objects
        super_object_translation: The translation vector applied to the super-object
        individual_objects_before: List of (label, centroid) tuples for objects before super-object creation
    """
    if not motif_node or not motif_node.arrangement:
        return

    # Create lookup of objects by label in the motif
    motif_objs_by_label = {}
    for obj in motif_node.arrangement.objs:
        lbl = getattr(obj, 'label', None)
        if lbl:
            motif_objs_by_label[lbl] = obj

    # For each individual object, calculate its movement based on relative position
    for label, original_centroid in individual_objects_before:
        if label not in motif_objs_by_label:
            logger.warning(f"Could not find individual object {label} in motif for gravity redistribution")
            continue

        target_obj = motif_objs_by_label[label]

        # Apply the same translation as the super-object (simplified approach)
        # This preserves rigid body relationships while allowing gravity settling
        if hasattr(target_obj, 'bounding_box') and target_obj.bounding_box is not None:
            target_obj.bounding_box.centroid += super_object_translation
            logger.debug(f"  Redistributed gravity transform to {label}: {super_object_translation}")


def _distribute_transforms_to_motifs(
    nodes,
    original_combined,
    optimized_combined
) -> None:
    """Distribute the transforms from optimized combined arrangement back to individual motifs."""
    if len(original_combined.objs) != len(optimized_combined.objs):
        logger.warning("Mismatch in object count between original and optimized combined arrangements")
        return

    for orig_obj, opt_obj in zip(original_combined.objs, optimized_combined.objs):
        if not hasattr(orig_obj, '_motif_node'):
            continue

        motif_node = orig_obj._motif_node
        individual_objects_before = getattr(orig_obj, '_individual_objects_before', [])

        orig_centroid = orig_obj.bounding_box.centroid
        opt_centroid = opt_obj.bounding_box.centroid
        translation = opt_centroid - orig_centroid

        if motif_node and individual_objects_before:
            logger.debug(f"Redistributing inter-motif transform for motif {motif_node.motif_type}")
            _redistribute_super_object_transform(motif_node, translation, individual_objects_before)


def _cache_world_transforms(hierarchy) -> None:
    """Cache world-space positions for all objects in hierarchy."""
    def _compute_world_transforms(node, parent_transform=np.eye(4)):
        """Recursively compute and cache world transforms for all objects in the hierarchy."""
        if not node.arrangement or not node.arrangement.objs:
            return
        
        # For each object in this node's arrangement
        for obj in node.arrangement.objs:
            # Compute world transform: parent_transform @ local_transform
            local_tf = obj.bounding_box.no_scale_matrix if (obj.bounding_box and obj.bounding_box.no_scale_matrix is not None) else np.eye(4)
            world_tf = parent_transform @ local_tf
            
            # Cache on object
            setattr(obj, '_world_transform', world_tf)
            setattr(obj, '_world_position', world_tf[:3, 3])
        
        # Recurse to children with updated parent transform
        for child in node.children:
            _compute_world_transforms(child, parent_transform)
    
    if hierarchy.root:
        _compute_world_transforms(hierarchy.root)
        logger.debug("Cached world transforms for all objects in hierarchy")

