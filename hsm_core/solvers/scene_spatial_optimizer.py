"""
Scene Spatial Optimizer

This optimizer uses actual meshes and room geometry for collision detection
and support validation.
"""

import copy
import time
from typing import List, Dict, Optional, Tuple, Set, TYPE_CHECKING, Union
import numpy as np
import trimesh
import trimesh.transformations

from hsm_core.scene.geometry.placer import SceneObjectPlacer
from hsm_core.scene.core.objects import SceneObject
from hsm_core.scene.core.objecttype import ObjectType
from hsm_core.solvers.config import SceneSpatialOptimizerConfig
from hsm_core.utils import get_logger

if TYPE_CHECKING:
    from hsm_core.scene.core.manager import Scene

logger = get_logger('solvers.scene_spatial_optimizer')

class SceneSpatialOptimizer:
    """
    Scene-level spatial optimizer that refines DFS solver placements.

    For each scene motif:
    1. Evaluate if optimization needed (detect collisions/unsupported objects)
    2. If optimization needed:
       - Collision resolution: vertical lift → horizontal displacement
       - Use adaptive step size based on penetration depth
       - Apply movement constraints (floor/wall/ceiling attachment)
       - Support validation: raycast from center + corners with 0.01m threshold
       - Reposition to nearest support with minimal adjustment
    """
    
    def __init__(self, scene: 'Scene', config: Optional[SceneSpatialOptimizerConfig] = None):
        """
        Initialize the mesh-based spatial optimizer.

        Args:
            scene: Scene object containing motifs and room geometry.
            config: Configuration for spatial optimization.
        """
        self.scene = scene
        self.config = config or SceneSpatialOptimizerConfig()

        # Use shared mesh cache from scene if available, otherwise create own cache
        if hasattr(scene, '_mesh_cache') and scene._mesh_cache is not None:
            self._object_meshes = scene._mesh_cache
            self._use_shared_cache = True
        else:
            self._object_meshes: Dict[str, trimesh.Trimesh] = {}
            self._use_shared_cache = False

        self._room_scene: Optional[trimesh.Scene] = None
        self._floor_mesh: Optional[trimesh.Trimesh] = None
        self._wall_meshes: List[trimesh.Trimesh] = []
        self._ceiling_mesh: Optional[trimesh.Trimesh] = None
        
        self._collision_cache: Dict[Tuple[str, str], Tuple[bool, float]] = {}  # (obj_name, other_name) -> (has_collision, timestamp)
        self._position_cache: Dict[str, Tuple[float, float, float]] = {}  # obj_name -> last_known_position
        
        self.stats = {
            'objects_processed': 0,
            'collisions_resolved': 0,
            'support_fixes_applied': 0,
            'processing_time': 0.0,
        }

    def _initialize_room_geometry(self) -> None:
        """Initialize room geometry from the scene for mesh-based operations."""
        temp_scene_placer = SceneObjectPlacer(room_height=self.scene.room_height)
        
        try:
            room_geometry = temp_scene_placer.create_room_geom(self.scene.room_polygon, self.scene.door_location, self.scene.window_location)
            floor_data = room_geometry['floor']
            self._wall_meshes = []
            self._floor_mesh = floor_data[0] if isinstance(floor_data, tuple) else floor_data
            
            for name in room_geometry.keys():
                if name.startswith('wall'):
                    wall_data = room_geometry[name]
                    wall_mesh = wall_data[0] if isinstance(wall_data, tuple) else wall_data
                    self._wall_meshes.append(wall_mesh)
            logger.info("Room geometry initialized")

        except Exception as e:
            logger.error("Failed to initialize room geometry: %s", e)
            self._room_scene = None

    def _load_object_meshes(self, objects: List[SceneObject]) -> None:
        """Load trimesh objects for current stage objects only """
        for obj in objects:
            try:
                if obj.name not in self._object_meshes:
                    mesh = self._load_mesh_for_object(obj)
                    if mesh is not None:
                        # Store working mesh
                        self._object_meshes[obj.name] = mesh.copy()
            except Exception as e:
                logger.warning("Failed to load mesh for %s: %s", obj.name, e)

    def _ensure_mesh_loaded(self, obj: SceneObject) -> Optional[trimesh.Trimesh]:
        """Ensure mesh is loaded for an object (lazy loading)."""
        if obj.name not in self._object_meshes:
            mesh = self._load_mesh_for_object(obj)
            if mesh is not None:
                self._object_meshes[obj.name] = mesh.copy()
                return mesh
            return None
        return self._object_meshes[obj.name]

    def _load_mesh_for_object(self, obj: SceneObject) -> Optional[trimesh.Trimesh]:
        """Load mesh for an object using shared cache or direct loading."""
        # Try to use scene's shared mesh cache first
        if self._use_shared_cache and hasattr(self.scene, 'get_or_load_mesh'):
            return self.scene.get_or_load_mesh(obj)
        else:
            # Fallback to direct loading
            return self._load_single_object_mesh(obj)

    def _load_single_object_mesh(self, obj: SceneObject) -> Optional[trimesh.Trimesh]:
        """Load and preprocess mesh for a single object."""
        try:
            from hsm_core.scene.utils.mesh_utils import preprocess_object_mesh
            # Check if mesh_path is set
            if not obj.mesh_path or obj.mesh_path.strip() == "":
                logger.warning("Object '%s' has no mesh_path set - cannot load mesh for spatial optimization", obj.name)
                return None
            
            # Load and preprocess the mesh
            mesh = preprocess_object_mesh(obj)
            if mesh is None:
                logger.warning("Failed to load mesh for '%s' from path '%s' - mesh loading returned None", obj.name, obj.mesh_path)
                return None
            
            # Apply object's rotation
            if obj.rotation != 0:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    angle=np.radians(obj.rotation),
                    direction=[0, 1, 0])
                mesh.apply_transform(rotation_matrix)
                
            # Apply object's position
            translation_matrix = trimesh.transformations.translation_matrix(obj.position)
            mesh.apply_transform(translation_matrix)
            
            return mesh
            
        except Exception as e:
            logger.warning("Exception loading mesh for '%s': %s", obj.name, e)
            return None

    def _optimize_motif_as_unit(self, motif_objects: List[SceneObject], context_objects: List[SceneObject]) -> List[SceneObject]:
        """
        Optimize an entire motif as a single unit, preserving internal object relationships.
        
        First evaluates if optimization is needed by detecting mesh intersections and 
        unsupported objects. If the motif is already well-placed, preserves its position
        to maintain the DFS solver's valid placements.
        """
        if not motif_objects:
            return []

        start_time = time.time()

        # Create combined representative for the motif first
        combined_mesh = self._create_combined_motif_mesh(motif_objects)
        motif_representative = self._create_motif_representative(motif_objects, combined_mesh)
        self._object_meshes[motif_representative.name] = combined_mesh.copy()

        # Check if the motif representative needs optimization
        needs_optimisation = (
            self._find_collisions(motif_representative, context_objects)
            or not self._is_properly_supported_mesh(motif_representative, context_objects)
        )

        if not needs_optimisation:
            motif_id = motif_objects[0].motif_id if hasattr(motif_objects[0], 'motif_id') else 'unknown'
            logger.info("Motif %s is already well-placed", motif_id)
            
            # Return the objects with optimized_world_pos set to their current positions
            preserved_motif_objects: List[SceneObject] = []
            for obj in motif_objects:
                optimized_obj = copy.deepcopy(obj)
                self._set_optimized_position(optimized_obj, (float(obj.position[0]), float(obj.position[1]), float(obj.position[2])))
                preserved_motif_objects.append(optimized_obj)
            return preserved_motif_objects
        
        # Optimize the motif representative
        optimized_representative = self._optimize_object(motif_representative, context_objects)
        
        # Calculate the transformation applied to the motif
        original_bottom_center = np.array(motif_representative.position, dtype=float)
        new_bottom_center = np.array(optimized_representative.position, dtype=float)
        translation = new_bottom_center - original_bottom_center
        # logger.debug("Motif %s translation: %s", motif_objects[0].motif_id, translation)
        
        # Apply the same transformation to all objects in the motif
        optimized_motif_objects: List[SceneObject] = []
        for obj in motif_objects:
            optimized_obj = copy.deepcopy(obj)
            world_pos_arr = np.array(obj.position, dtype=float) + np.array(translation, dtype=float)
            world_pos = (float(world_pos_arr[0]), float(world_pos_arr[1]), float(world_pos_arr[2]))
            self._set_optimized_position(optimized_obj, world_pos)
            optimized_motif_objects.append(optimized_obj)

        elapsed_time = time.time() - start_time
        logger.info("Optimized motif %s with %d objects in %.3fs", motif_objects[0].motif_id, len(optimized_motif_objects), elapsed_time)
        
        # Log position updates for each object
        position_changes = 0
        for i, (original_obj, optimized_obj) in enumerate(zip(motif_objects, optimized_motif_objects)):
            original_pos = tuple("%.3f" % p for p in original_obj.position)
            new_pos = tuple("%.3f" % p for p in optimized_obj.position)
            if original_pos != new_pos:
                logger.info("Object %d - %s: %s -> %s", i, optimized_obj.name, original_pos, new_pos)
                position_changes += 1

        if position_changes == 0:
            logger.debug("No position changes for motif %s", motif_objects[0].motif_id)

        return optimized_motif_objects

    def _create_combined_motif_mesh(self, motif_objects: List[SceneObject]) -> Optional[trimesh.Trimesh]:
        """Create a combined mesh representing the entire motif."""
        try:
            motif_meshes = []
            
            for obj in motif_objects:
                if obj.name in self._object_meshes:
                    mesh = self._get_mesh_at_current_position(obj)
                    if mesh is not None:
                        motif_meshes.append(mesh)
            
            if not motif_meshes:
                return None
            
            if len(motif_meshes) == 1:
                return motif_meshes[0].copy()
            else:
                return trimesh.util.concatenate(motif_meshes)
                
        except Exception as e:
            logger.warning("Failed to create combined motif mesh: %s", e)
            return None

    def _create_motif_representative(self, motif_objects: List[SceneObject], combined_mesh: trimesh.Trimesh) -> SceneObject:
        """Create a representative SceneObject for the entire motif."""
        # Calculate combined properties (bottom-centered)
        bounds = combined_mesh.bounds  # shape (2, 3): [min_xyz, max_xyz]
        dimensions = bounds[1] - bounds[0]
        
        # Calculate the center position of the motif based on actual object positions
        positions = np.array([obj.position for obj in motif_objects])
        motif_center = np.mean(positions, axis=0)
        
        # Use the lowest Y position as the bottom
        bottom_y = float(np.min(positions[:, 1]))
        bottom_center = (
            float(motif_center[0]),
            float(bottom_y),
            float(motif_center[2]),
        )

        # Use the first object as a template
        template_obj = motif_objects[0]
        
        representative = SceneObject(
            name=f"motif_{template_obj.motif_id}_combined",
            position=bottom_center,
            dimensions=tuple(dimensions),
            rotation=template_obj.rotation,
            mesh_path=template_obj.mesh_path,
            obj_type=template_obj.obj_type,
            motif_id=template_obj.motif_id,
            parent_name=getattr(template_obj, "parent_name", None)
        )
        
        return representative

    def create_motif_representative(self, motif_objects: List[SceneObject]) -> Optional[SceneObject]:
        """Build a motif representative from a list of SceneObjects.
        """
        try:
            combined_mesh = self._create_combined_motif_mesh(motif_objects)
            if combined_mesh is None:
                return None
            return self._create_motif_representative(motif_objects, combined_mesh)
        except Exception:
            return None

    def _optimize_object(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Optimize single object with collision resolution and support validation."""
        if obj.name.startswith("motif_") or "_combined" in obj.name:
            logger.debug("Optimizing motif: %s (%s)", obj.name, obj.obj_type.name)
        else:
            logger.debug("Optimizing single object: %s (%s)", obj.name, obj.obj_type.name)
        
        # Step 1: Resolve collisions
        collision_resolved_obj = self._resolve_collisions(obj, context_objects)
        
        # Step 2: Ensure proper support
        support_fixed_obj = self._ensure_mesh_support(collision_resolved_obj, context_objects)
        
        # Update statistics
        if collision_resolved_obj.position != obj.position:
            self.stats['collisions_resolved'] += 1
        if support_fixed_obj.position != collision_resolved_obj.position:
            self.stats['support_fixes_applied'] += 1
        
        self.stats['objects_processed'] += 1
        
        # --------------------------------------------------------------
        # Persist world-space coordinate so downstream code can consume
        # the optimiser output without relying on in-place mutation.
        # --------------------------------------------------------------
        support_fixed_obj.optimized_world_pos = (
            float(support_fixed_obj.position[0]),
            float(support_fixed_obj.position[1]),
            float(support_fixed_obj.position[2])
        )
        
        return support_fixed_obj

    def _resolve_collisions(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """
        Resolve collisions using two-step strategy: vertical lift first, then horizontal displacement.
        Uses adaptive step size based on penetration depth.
        """
        logger.debug("Resolving collisions for %s", obj.name)
        max_iterations = self.config.max_collision_iterations
        current_obj = copy.deepcopy(obj)
        
        obj_mesh = self._get_mesh_at_current_position(obj)
        if obj_mesh is None:
            logger.warning("No mesh found for %s - skipping collision resolution", obj.name)
            return obj
        
        # Check for collisions with context objects
        for iteration in range(max_iterations):
            colliding_objects = self._find_collisions(current_obj, context_objects)
            
            if not colliding_objects:
                if iteration > 0:
                    logger.debug("Collision resolved for %s after %d iteration(s)", obj.name, iteration + 1)
                break
            
            # Calculate penetration depth for adaptive step size
            penetration_depth = self._get_penetration_depth(current_obj, colliding_objects)
            adaptive_step_size = penetration_depth * self.config.adaptive_step_factor
            min_step_size = 0.01  # 1cm minimum step
            adaptive_step_size = max(adaptive_step_size, min_step_size)
            adaptive_step_size = min(adaptive_step_size, self.config.max_step_size)

            # Move away from collisions
            old_position = current_obj.position
            current_obj = self._resolve_single_mesh_collision(current_obj, colliding_objects, adaptive_step_size)
            
            # Update mesh position if object moved
            if current_obj.position != old_position:
                self._transform_mesh_to_position(current_obj, current_obj.position)
        
        return current_obj

    def _find_collisions(self, obj: SceneObject, context_objects: List[SceneObject]) -> List[SceneObject]:
        """Find collisions with caching for performance optimization"""
        
        colliding = []
        obj_mesh = self._get_mesh_at_current_position(obj)
        
        if obj_mesh is None:
            return []

        relevant_context = self._get_relevant_collision_context(obj, context_objects)
        logger.debug("Checking collisions for %s against %d context objects", obj.name, len(relevant_context))
        logger.debug("Context objects: %s", [f"{o.name}({o.obj_type.name})" for o in relevant_context[:5]])  # Show first 5

        for other_obj in relevant_context:
            # Check cache first
            cache_key = (obj.name, other_obj.name)
            obj_pos = tuple(obj.position)
            other_pos = tuple(other_obj.position)
            
            # Check if both positions are unchanged since last check
            if (cache_key in self._collision_cache and
                self._position_cache.get(obj.name) == obj_pos and
                self._position_cache.get(other_obj.name) == other_pos):
                
                is_collision, _ = self._collision_cache[cache_key]
                if is_collision:
                    colliding.append(other_obj)
                continue  # Use cached result
            
            # Cache miss or positions changed - do full check
            if not self._bbox_collision_check(obj, other_obj):
                # Update cache: no bbox collision = no collision
                self._collision_cache[cache_key] = (False, time.time())
                self._position_cache[obj.name] = obj_pos
                self._position_cache[other_obj.name] = other_pos
                continue
                
            other_mesh = self._get_mesh_at_current_position(other_obj)
            if other_mesh is not None:
                is_collision, penetration = self._check_mesh_collision(
                    obj_mesh, other_mesh, return_penetration=True
                )
                
                # Update cache with result
                self._collision_cache[cache_key] = (is_collision, time.time())
                self._position_cache[obj.name] = obj_pos
                self._position_cache[other_obj.name] = other_pos

                if is_collision:
                    colliding.append(other_obj)
                    logger.debug("Collision detected: %s <-> %s (penetration: %.3f)", 
                               obj.name, other_obj.name, penetration)
        
        logger.debug("Found %d collisions for %s", len(colliding), obj.name)
        return colliding

    def _bbox_collision_check(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Quick bounding box check to filter out obviously non-colliding objects."""
        # Calculate distance between object centers
        pos1 = np.array(obj1.position)
        pos2 = np.array(obj2.position)
        distance = np.linalg.norm(pos1 - pos2)
        
        # Calculate combined bounding box diagonal
        dim1 = np.array(obj1.dimensions)
        dim2 = np.array(obj2.dimensions)
        combined_diagonal = np.linalg.norm(dim1 + dim2) / 2
        
        # Add some tolerance for safety
        return distance < combined_diagonal + 0.1

    def _check_mesh_collision(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, 
                             return_penetration: bool = False) -> Union[bool, Tuple[bool, float]]:
        """Check if two meshes are colliding, optionally return penetration depth."""
        # try:
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object('obj1', mesh1)
        
        if return_penetration:
            is_collision, contacts = collision_manager.in_collision_single(mesh2, return_data=True)
        else:
            is_collision = collision_manager.in_collision_single(mesh2)
            contacts = None
        
        if not return_penetration:
            return is_collision
        
        # Calculate penetration depth if requested
        penetration_depth = 0.0
        if is_collision and contacts:
            for contact in contacts:
                if hasattr(contact, 'depth') and contact.depth > 0:
                    penetration_depth = max(penetration_depth, contact.depth)
        
        return is_collision, penetration_depth

    def _resolve_single_mesh_collision(self, obj: SceneObject, colliding_objects: List[Union[SceneObject, str]], step_size: float) -> SceneObject:
        """
        Resolve collision using two-step strategy: vertical lift first, then horizontal displacement.
        Skip vertical lift for floor-bound objects (LARGE type).
        """
        resolved_obj = copy.deepcopy(obj)
        
        if not colliding_objects:
            return resolved_obj

        # Skip vertical lift for floor-bound objects (LARGE type) - they must stay on floor
        should_try_vertical = obj.obj_type not in [ObjectType.LARGE, ObjectType.CEILING]
        
        if should_try_vertical:
            # Step 1: Try vertical movement first
            vertical_step = step_size * self.config.vertical_step_factor
            
            # Check if moving up resolves collisions
            test_obj = copy.deepcopy(resolved_obj)
            new_position = (test_obj.position[0], test_obj.position[1] + vertical_step, test_obj.position[2])
            test_obj.position = new_position
            
            # Create a test mesh for collision detection
            if test_obj.name in self._object_meshes:
                test_mesh = self._object_meshes[test_obj.name].copy()
                old_position = resolved_obj.position
                translation = np.array(new_position) - np.array(old_position)
                if np.any(translation != 0):
                    test_mesh.apply_translation(translation)
                # Store test mesh temporarily for collision detection
                original_mesh = self._object_meshes[test_obj.name]
                self._object_meshes[test_obj.name] = test_mesh
            
            # Test if vertical movement resolves collisions
            scene_objects_only = [c for c in colliding_objects if isinstance(c, SceneObject)]
            remaining_collisions = self._find_collisions(test_obj, scene_objects_only)
            
            # Restore original mesh
            if test_obj.name in self._object_meshes:
                self._object_meshes[test_obj.name] = original_mesh
            
            if not remaining_collisions:
                resolved_obj = test_obj
                
                # Invalidate cache for this object since position changed
                self._position_cache[obj.name] = tuple(resolved_obj.position)
                # Invalidate all cache entries involving this object
                keys_to_remove = [k for k in self._collision_cache.keys() if obj.name in k]
                for k in keys_to_remove:
                    del self._collision_cache[k]
                
                logger.debug("Resolved collision for %s by moving up %.3fm from %s to %s", 
                            obj.name, vertical_step, obj.position, resolved_obj.position)
                return resolved_obj
        
        # Step 2: Horizontal displacement if vertical movement fails or was skipped
        movement_direction = np.array([0.0, 0.0, 0.0])
        
        # Calculate wall-aligned movement for wall objects
        if obj.obj_type == ObjectType.WALL:
            angle_rad = np.radians(obj.rotation)
            wall_tangent = np.array([np.cos(angle_rad), 0, -np.sin(angle_rad)])
            wall_tangent_norm = np.linalg.norm(wall_tangent)
            if wall_tangent_norm > 1e-6:
                wall_tangent = wall_tangent / wall_tangent_norm
        
        for colliding in colliding_objects:
            if isinstance(colliding, str) and colliding == "room_boundary":
                # Move toward room center
                if hasattr(self.scene, 'room_polygon') and self.scene.room_polygon:
                    room_centroid = self.scene.room_polygon.centroid
                    room_center = np.array([room_centroid.x, obj.position[1], room_centroid.y])
                    direction = room_center - np.array(obj.position)
                    direction[1] = 0  # Keep horizontal only
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                        if obj.obj_type == ObjectType.WALL:
                            direction = np.dot(direction, wall_tangent) * wall_tangent
                        movement_direction += direction
            elif isinstance(colliding, SceneObject):
                # Use AABB-based separation direction
                direction = self._get_aabb_separation_direction(obj, colliding)
                if obj.obj_type == ObjectType.WALL:
                    direction = np.dot(direction, wall_tangent) * wall_tangent
                movement_direction += direction
        
        if np.linalg.norm(movement_direction) > 1e-6:
            movement_direction = movement_direction / np.linalg.norm(movement_direction)
        else:
            movement_direction = np.array([1.0, 0.0, 0.0])
        
        # Apply movement constraints and calculate new position
        movement_direction = self._apply_movement_constraints(movement_direction, obj.obj_type)
        horizontal_step = step_size * self.config.horizontal_step_factor
        new_position = np.array(obj.position) + movement_direction * horizontal_step
        
        # Keep within room bounds
        margin = self.config.room_bounds_margin
        x_min, y_min, x_max, y_max = self.scene.room_polygon.bounds
        new_position = (
            max(x_min + margin, min(new_position[0], x_max - margin)),
            new_position[1],
            max(y_min + margin, min(new_position[2], y_max - margin))
        )
        
        resolved_obj.position = tuple(new_position)
        
        # Invalidate cache for this object since position changed
        self._position_cache[obj.name] = tuple(new_position)
        # Invalidate all cache entries involving this object
        keys_to_remove = [k for k in self._collision_cache.keys() if obj.name in k]
        for k in keys_to_remove:
            del self._collision_cache[k]
        
        logger.debug("Adjusted %s horizontally by %.3fm from %s to %s", 
                    obj.name, horizontal_step, obj.position, resolved_obj.position)
        return resolved_obj

    def _ensure_mesh_support(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Ensure object is properly supported using mesh-based validation and fixes."""
        # Skip support validation for motif representatives
        is_motif_representative = obj.name.startswith("motif_") and obj.name.endswith("_combined")
        if not is_motif_representative:
            return obj
            
        if not self._is_properly_supported_mesh(obj, context_objects):
            return self._fix_mesh_support(obj, context_objects)
        return obj

    def _is_properly_supported_mesh(self, obj: SceneObject, context_objects: List[SceneObject]) -> bool:
        """Check if an object is properly supported using mesh-based raycasting."""
        if obj.obj_type in [ObjectType.LARGE, ObjectType.SMALL]:
            return self._check_surface_support_mesh(obj, context_objects)
        elif obj.obj_type == ObjectType.WALL:
            return self._check_wall_attachment_mesh(obj)
        elif obj.obj_type == ObjectType.CEILING:
            return self._check_ceiling_attachment_mesh(obj)
        return True # Default to supported

    def _check_surface_support_mesh(self, obj: SceneObject, context_objects: List[SceneObject]) -> bool:
        """
        Check for surface support by raycasting down from center and corner vertices.
        """
        logger.debug("Surface support check for %s (%s)", obj.name, obj.obj_type.name)
        logger.debug("Position: %s, Dimensions: %s", obj.position, obj.dimensions)
        
        # Check if we can get mesh at current position
        obj_mesh = self._get_mesh_at_current_position(obj)
        if obj_mesh is None:
            logger.debug("No mesh loaded for %s, assuming supported", obj.name)
            return True
        
        # Bottom of object in world coordinates (position is bottom-centered)
        object_bottom_y: float = obj.position[1]
        support_tolerance = self.config.support_tolerance
        logger.debug("Object bottom Y: %.3fm, Support tolerance: %.3fm", object_bottom_y, support_tolerance)
        
        # Check parent first (for small objects, but not motif representatives)
        is_motif_representative = obj.name.startswith("motif_") and obj.name.endswith("_combined")
        parent_name = getattr(obj, "parent_name", None)
        # logger.debug("Object %s - parent_name=%s, obj_type=%s, is_motif_representative=%s", 
                    # obj.name, parent_name, obj.obj_type, is_motif_representative)
        
        if parent_name and obj.obj_type == ObjectType.SMALL:
            logger.debug("Checking parent support for %s (parent: %s)", obj.name, parent_name)
            parent_objects = [o for o in context_objects if o.name == parent_name]
            if parent_objects:
                parent_obj = parent_objects[0]
                supported, support_y = self._compute_support_from_object(
                    obj, parent_obj, object_bottom_y, support_tolerance
                )
                if supported:
                    logger.debug("%s is supported by parent %s at Y=%.3f", obj.name, parent_obj.name, support_y)
                    return True
                else:
                    logger.debug("%s is not supported by parent %s at Y=%s", obj.name, parent_obj.name, support_y)
                    return False
            else:
                logger.debug("Parent object '%s' not found in context for %s", parent_name, obj.name)
        else:
            logger.debug("No parent object found for %s (parent_name=%s, obj_type=%s)", 
                        obj.name, parent_name, obj.obj_type)
        
        # Check other objects on the same parent (for small objects, but not motif representatives)
        if parent_name and obj.obj_type == ObjectType.SMALL and not is_motif_representative:
            logger.debug("Checking other objects on parent %s for %s", parent_name, obj.name)
            sibling_objects = [o for o in context_objects 
                             if getattr(o, "parent_name", None) == parent_name and o.name != obj.name]
            logger.debug("Found %d sibling objects on parent %s", len(sibling_objects), parent_name)
            
            for sibling_obj in sibling_objects:
                supported, support_y = self._compute_support_from_object(
                    obj, sibling_obj, object_bottom_y, support_tolerance
                )
                if supported:
                    logger.debug("%s is supported by sibling %s at Y=%.3f", obj.name, sibling_obj.name, support_y)
                    return True
                else:
                    logger.debug("%s is not supported by sibling %s at Y=%s", obj.name, sibling_obj.name, support_y)
        elif is_motif_representative:
            logger.debug("Skipping sibling check for motif representative %s", obj.name)

       # Check floor support
        logger.debug("Checking floor support for %s", obj.name)
        if self._floor_mesh:
            # Create raycast from object's bottom center
            ray_origin = np.array([obj.position[0], object_bottom_y + 0.001, obj.position[2]])
            ray_direction = np.array([0, -1, 0])
            
            try:
                locations, _, _ = self._floor_mesh.ray.intersects_location([ray_origin], [ray_direction])
                if len(locations) > 0:
                    # Check if floor is below or at the object bottom (within tolerance)
                    loc_arr = np.asarray(locations)
                    floor_y = float(loc_arr[0, 1])
                    floor_distance = abs(floor_y - object_bottom_y)
                    logger.debug("Floor raycast hit at distance: %.3fm", floor_distance)
                    if floor_y <= object_bottom_y + support_tolerance:
                        logger.debug("%s is supported by floor", obj.name)
                        return True
                    else:
                        logger.debug("Floor at same level as object (%.3fm), not supporting", floor_y)
                else: 
                    logger.debug("Floor raycast found no hits")
            except Exception as e:
                logger.debug("Floor raycast failed: %s", e)
        else:
            logger.debug("No floor mesh available for support check")

        logger.debug("%s is NOT supported by any surface", obj.name)
        return False

    def _check_wall_attachment_mesh(self, obj: SceneObject) -> bool:
        """Check if a wall object is attached to a wall using mesh raycasting."""
        obj_mesh = self._get_mesh_at_current_position(obj)
        if obj_mesh is None or not self._wall_meshes:
            logger.debug("%s: No wall meshes available, assuming proper attachment", obj.name)
            return True  # Cannot check, assume it's fine

        # Get the "front" direction vector based on object's rotation
        angle_rad = np.radians(obj.rotation)
        front_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
        back_vector = -front_vector

        # Check the back face center and edges for wall attachment
        check_points = []
        back_face_center = np.array(obj.position) - front_vector * (obj.dimensions[2] / 2)
        check_points.append(back_face_center)

        obj_half_width = obj.dimensions[0] / 2
        check_points.extend([
            back_face_center + np.array([-obj_half_width, 0, 0]),  # Left edge
            back_face_center + np.array([obj_half_width, 0, 0]),   # Right edge
        ])
        
        tolerance = self.config.support_tolerance + 0.02
        
        # Try target wall first if available
        target_wall_id = getattr(obj, "wall_id", None)
        wall_candidates = []
        
        if target_wall_id is not None and isinstance(self._wall_meshes, dict) and target_wall_id in self._wall_meshes:
            wall_candidates = [self._wall_meshes[target_wall_id]]
        else:
            if isinstance(self._wall_meshes, dict):
                wall_candidates = list(self._wall_meshes.values())
            else:
                wall_candidates = self._wall_meshes
        
        for wall_mesh in wall_candidates:
            for check_point in check_points:
                # Raycast from check point towards the wall
                ray_origin = check_point - back_vector * 0.02
                hit, _, _ = wall_mesh.ray.intersects_location([ray_origin], [back_vector], multiple_hits=False)
                if len(hit) > 0:
                    hit_distance = np.linalg.norm(hit[0] - ray_origin)
                    # Account for the 2cm offset when checking tolerance
                    actual_distance = hit_distance + 0.02
                    if actual_distance < tolerance:
                        logger.debug("%s is attached to wall (actual distance: %.3fm)", obj.name, actual_distance)
                        return True

        
        logger.debug("%s is NOT attached to any wall", obj.name)
        return False

    def _check_ceiling_attachment_mesh(self, obj: SceneObject) -> bool:
        """Check ceiling attachment using upward raycast to ceiling mesh."""
        obj_mesh = self._get_mesh_at_current_position(obj)
        if obj_mesh is None or self._ceiling_mesh is None:
            return True  # Cannot check, assume it's fine
        
        # Cast rays from the top surface of the object upwards
        # With bottom-centered positions, the top is at position_y + height
        top_center = np.array([obj.position[0], obj.position[1] + obj.dimensions[1], obj.position[2]])
        
        # Raycast from just above the object upwards
        ray_origin = top_center - np.array([0, 0.01, 0])
        ray_direction = np.array([0, 1, 0])
        
        # Check for hit within tolerance
        hit, _, _ = self._ceiling_mesh.ray.intersects_location([ray_origin], [ray_direction], multiple_hits=False)
        if len(hit) > 0:
            distance = np.linalg.norm(hit[0] - ray_origin)
            if distance < self.config.support_tolerance:
                return True
        
        logger.debug("%s is not attached to the ceiling", obj.name)
        return False

    def _fix_mesh_support(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Fix support for an object by finding the best support surface below it."""
        if obj.obj_type in [ObjectType.LARGE, ObjectType.SMALL]:
            return self._fix_surface_support_mesh(obj, context_objects)
        elif obj.obj_type == ObjectType.WALL:
            return self._fix_wall_support_mesh(obj)
        elif obj.obj_type == ObjectType.CEILING:
            return self._fix_ceiling_support_mesh(obj)
        return obj

    def _fix_surface_support_mesh(self, obj: SceneObject, context_objects: List[SceneObject]) -> SceneObject:
        """Fix surface support by moving object to proper supported position."""
        if obj.name not in self._object_meshes:
            return obj
        obj_mesh = self._object_meshes[obj.name]
        object_bottom_y: float = obj.position[1]
        support_tolerance: float = self.config.support_tolerance

        max_support_y: float = -np.inf  # initialise to negative infinity so that we can detect *any* valid support
        hits_found: bool = False

        # --- Support from other objects first -------------------------------------
        support_context = self._get_relevant_support_context(obj, context_objects)
        logger.debug("Support context: %s (from %d total)", [s.name for s in support_context], len(context_objects))

        for sup_obj in support_context:
            # First, try the precise ray-cast routine if a mesh is available.
            if sup_obj.name in self._object_meshes:
                supported, support_y = self._compute_support_from_object(
                    obj,
                    sup_obj,
                    object_bottom_y,
                    support_tolerance,
                )

                # get the highest support surface below the object
                if support_y is not None and support_y <= object_bottom_y:
                    max_support_y = max(max_support_y, support_y)
                    hits_found = True
                continue

            # use bounding box top face as fallback (position is bottom)
            sup_top_y: float = sup_obj.position[1] + sup_obj.dimensions[1]
            if sup_top_y <= object_bottom_y + support_tolerance:
                max_support_y = max(max_support_y, sup_top_y)
                hits_found = True

        # --- Floor support as fallback ---------------------------------------------------
        if not hits_found and self._floor_mesh:
            ray_origin = np.array([obj.position[0], object_bottom_y + 1e-4, obj.position[2]])
            try:
                locations, _, _ = self._floor_mesh.ray.intersects_location([ray_origin], [[0, -1, 0]])
            except Exception:
                locations = []

            if len(locations) > 0:
                loc_arr = np.asarray(locations)
                floor_y: float = float(np.max(loc_arr[:, 1]))
                if floor_y <= object_bottom_y:  # Only count hits below current bottom
                    max_support_y = max(max_support_y, floor_y)
                    hits_found = True

        if not hits_found:
            logger.debug("%s is NOT supported by any surface", obj.name)
            logger.debug("Checked %d support objects", len(support_context))
            # Nothing to do – leave object unchanged.
            return obj

        target_position_y: float = max_support_y + self.config.support_stability_offset
        current_y: float = float(obj.position[1])

        if abs(current_y - target_position_y) < self.config.support_tolerance:
            # Adjustment smaller than threshold – keep current placement.
            return obj

        translation_y: float = target_position_y - current_y
        new_position = list(obj.position)
        new_position[1] = float(target_position_y)
        obj.position = (float(new_position[0]), float(new_position[1]), float(new_position[2]))

        # Update internal scene mesh position
        obj_mesh.apply_translation([0, translation_y, 0])
        logger.debug("Fixed surface support for %s: %.3fm -> %.3fm (Δ=%.3fm)", obj.name, current_y, target_position_y, translation_y)

        self.stats["support_fixes_applied"] += 1

        return obj

    def _fix_wall_support_mesh(self, obj: SceneObject) -> SceneObject:
        """Fix wall object attachment by moving it to the target wall or nearest wall."""
        if obj.name not in self._object_meshes or not self._wall_meshes:
            return obj
        obj_mesh = self._object_meshes[obj.name]
        
        # Try to use target wall if available, otherwise use nearest wall
        target_wall_id = getattr(obj, "wall_id", None)
        candidate_walls = []
        
        if target_wall_id is not None and isinstance(self._wall_meshes, dict) and target_wall_id in self._wall_meshes:
            # Use the specific target wall
            candidate_walls = [self._wall_meshes[target_wall_id]]
            logger.debug("Using target wall %s for %s", target_wall_id, obj.name)
        else:
            # Fallback to all walls
            if isinstance(self._wall_meshes, dict):
                candidate_walls = list(self._wall_meshes.values())
            else:
                candidate_walls = self._wall_meshes
            logger.debug("Using nearest wall for %s (no target wall specified)", obj.name)
        
        # Find the nearest point on candidate walls
        closest_point = None
        min_dist = float('inf')
        
        for wall_mesh in candidate_walls:
            point, dist, _ = wall_mesh.nearest.on_surface([obj_mesh.center_mass])
            if dist[0] < min_dist:
                min_dist = dist[0]
                closest_point = point[0]

        if closest_point is None:
            return obj

        # Get the "front" direction vector of the object
        angle_rad = np.radians(obj.rotation)
        front_vector = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
        back_vector = -front_vector

        # Add a small gap (0.5mm) to prevent z-fighting
        gap = 0.0005

        offset_back_origin   = front_vector * gap  # back-origin variant
        offset_center_origin = front_vector * (obj.dimensions[2] / 2 + gap)  # centre-origin variant

        candidate_pos_back   = closest_point + offset_back_origin
        candidate_pos_center = closest_point + offset_center_origin

        # Select the candidate with the shorter displacement from the current
        # position (in the XZ-plane).  Y is preserved regardless.
        disp_back   = np.linalg.norm((candidate_pos_back - np.array(obj.position))[[0, 2]])
        disp_center = np.linalg.norm((candidate_pos_center - np.array(obj.position))[[0, 2]])

        # Select the candidate with the shorter displacement and preserve original height
        chosen_pos = candidate_pos_back if disp_back < disp_center else candidate_pos_center
        chosen_pos[1] = obj.position[1]  # Preserve original Y height

        new_position = chosen_pos

        # Ensure position is within room bounds
        if hasattr(self.scene, 'room_polygon') and self.scene.room_polygon:
            margin = self.config.room_bounds_margin
            x_min, y_min, x_max, y_max = self.scene.room_polygon.bounds
            new_position = (
                max(x_min + margin, min(new_position[0], x_max - margin)),
                new_position[1],
                max(y_min + margin, min(new_position[2], y_max - margin))
            )

        # Move the object
        translation = np.array(new_position, dtype=float) - np.array(obj.position, dtype=float)
        obj.position = (float(new_position[0]), float(new_position[1]), float(new_position[2]))
        obj_mesh.apply_translation(translation)
        
        logger.debug("Fixed wall support for %s: moved to %s", obj.name, new_position)
        self.stats['support_fixes_applied'] += 1
        return obj

    def _fix_ceiling_support_mesh(self, obj: SceneObject) -> SceneObject:
        """Fix ceiling support by moving object to be flush with ceiling."""
        if obj.name not in self._object_meshes or self._ceiling_mesh is None:
            return obj
        obj_mesh = self._object_meshes[obj.name]
        
        # Position is centroid. To attach top to ceiling:
        # top_y = ceiling_height - offset
        # centroid_y = top_y - height/2
        ceiling_height = self.scene.room_height
        ceiling_offset = self.config.support_stability_offset
        object_height = obj.dimensions[1]
        new_y = ceiling_height - ceiling_offset - object_height / 2
        
        new_position = (obj.position[0], new_y, obj.position[2])
        translation = np.array(new_position) - np.array(obj.position)
        obj.position = new_position
        obj_mesh.apply_translation(translation)
        
        logger.debug("Fixed ceiling support for %s, moved to Y=%.3f", obj.name, new_y)
        self.stats['support_fixes_applied'] += 1
        return obj

    def _get_mesh_at_current_position(self, obj: SceneObject) -> Optional[trimesh.Trimesh]:
        """Get mesh at the object's current position."""
        return self._load_single_object_mesh(obj)

    def _reload_mesh_with_current_position(self, obj: SceneObject) -> None:
        """Reload mesh with the object's current position and update cache."""
        if obj.name not in self._object_meshes:
            return
        
        mesh = self._get_mesh_at_current_position(obj)
        if mesh is not None:
            self._object_meshes[obj.name] = mesh

    def _transform_mesh_to_position(self, obj: SceneObject, new_position: tuple) -> None:
        """Transform existing mesh to new position without reloading."""
        if obj.name not in self._object_meshes:
            return
            
        mesh = self._object_meshes[obj.name]
        old_position = obj.position
        translation = np.array(new_position) - np.array(old_position)
        
        # Apply translation to mesh
        if np.any(translation != 0):
            mesh.apply_translation(translation)

    def _set_optimized_position(self, obj: SceneObject, position: tuple) -> None:
        """Set both position and optimized_world_pos to the same value."""
        obj.position = position
        obj.optimized_world_pos = position

    def _get_distance_between_objects(self, obj1: SceneObject, obj2: SceneObject) -> float:
        """Calculate distance between two objects."""
        pos1 = np.array(obj1.position)
        pos2 = np.array(obj2.position)
        return np.linalg.norm(pos1 - pos2)

    def _get_relevant_collision_context(self, obj: SceneObject, all_context: List[SceneObject]) -> List[SceneObject]:
        """Get relevant objects for collision detection - parent always included for small objects.

        Non-small objects: use all provided context (except self) to ensure full same-type checks.
        Small objects: include parent and nearby objects within a threshold.
        """
        # Exclude self
        all_context = [o for o in all_context if o.name != obj.name]

        if obj.obj_type != ObjectType.SMALL:
            return all_context

        filtered: List[SceneObject] = []
        for other in all_context:
            if self._is_parent_child_relationship(obj, other):
                filtered.append(other)
            else:
                if self._get_distance_between_objects(obj, other) < 1.0:
                    filtered.append(other)

        return filtered

    def _get_relevant_support_context(self, obj: SceneObject, all_context: List[SceneObject]) -> List[SceneObject]:
        """Get relevant objects for support validation - optimized for performance."""
        if obj.obj_type == ObjectType.SMALL:
            # Small objects: check parent first, then nearby objects
            parent_name = getattr(obj, "parent_name", None)
            if parent_name:
                logger.debug("Small object %s looking for parent '%s' in context (%d objects)", 
                           obj.name, parent_name, len(all_context))
                context_names = [o.name for o in all_context]
                logger.debug("Available context objects: %s", context_names)
                parent_objects = [o for o in all_context if o.name == parent_name]
                if parent_objects:
                    logger.debug("Found parent object %s for small object %s", parent_name, obj.name)
                    return parent_objects
                else:
                    logger.debug("Parent object '%s' not found in context for small object %s", parent_name, obj.name)
        
        # Other types: check nearby objects only
        nearby = []
        for other in all_context:
            distance = self._get_distance_between_objects(obj, other)
            if distance < 2.0:
                nearby.append(other)
        return nearby

    def _is_parent_child_relationship(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Check if two objects have a parent-child relationship."""
        return ((hasattr(obj1, 'parent_name') and obj1.parent_name == obj2.name) or
                (hasattr(obj2, 'parent_name') and obj2.parent_name == obj1.name))

    def _apply_movement_constraints(self, direction: np.ndarray, obj_type: ObjectType) -> np.ndarray:
        """Apply movement constraints based on object type."""
        constrained_direction = direction.copy()
        
        if obj_type == ObjectType.LARGE:
            constrained_direction[1] = 0.0
        elif obj_type == ObjectType.CEILING:
            constrained_direction[1] = 0.0
        elif obj_type == ObjectType.WALL:
            constrained_direction[1] *= self.config.wall_y_movement_factor
        # SMALL objects can move in all directions (no constraints)
        
        norm = np.linalg.norm(constrained_direction)
        if norm > 1e-6:
            return constrained_direction / norm
        # Return default direction if constrained direction is near-zero
        return np.array([1.0, 0.0, 0.0])

    def _group_objects_by_motif(self, objects: List[SceneObject]) -> Dict[str, List[SceneObject]]:
        """Group objects by their motif ID."""
        groups = {}
        for obj in objects:
            if hasattr(obj, 'motif_id') and obj.motif_id:
                if obj.motif_id not in groups:
                    groups[obj.motif_id] = []
                groups[obj.motif_id].append(obj)
        return groups

    def _get_objects_not_in_motifs(self, all_objects: List[SceneObject], motif_groups: Dict[str, List[SceneObject]]) -> List[SceneObject]:
        """Get objects that are not part of any motif."""
        motif_object_names = set()
        for objects in motif_groups.values():
            motif_object_names.update(obj.name for obj in objects)
        
        return [obj for obj in all_objects if obj.name not in motif_object_names]

    def _print_summary(self):
        """Print optimization summary."""
        logger.info("="*60)
        logger.info("Scene Spatial Optimizer Summary")
        logger.info("="*60)
        logger.info("Objects processed: %d", self.stats['objects_processed'])
        logger.info("Collisions resolved: %d", self.stats['collisions_resolved'])
        logger.info("Support fixes applied: %d", self.stats['support_fixes_applied'])
        logger.info("Processing time: %.2fs", self.stats['processing_time'])
        logger.info("="*60)


    def _get_penetration_depth(self, obj: SceneObject, colliding_objects: List[SceneObject]) -> float:
        """Get the maximum penetration depth for adaptive step size calculation."""
        if not colliding_objects:
            return 0.0
        
        obj_mesh = self._get_mesh_at_current_position(obj)
        if obj_mesh is None:
            return 0.0
            
        max_penetration = 0.0
        for other_obj in colliding_objects:
            other_mesh = self._get_mesh_at_current_position(other_obj)
            if other_mesh is not None:
                collision_result = self._check_mesh_collision(
                    obj_mesh, other_mesh, return_penetration=True
                )
                
                if isinstance(collision_result, tuple):
                    is_collision, penetration = collision_result
                    if is_collision:
                        max_penetration = max(max_penetration, penetration)
        
        return max_penetration

    def _get_aabb_separation_direction(self, obj: SceneObject, colliding: SceneObject) -> np.ndarray:
        """
        Calculate separation direction using AABB overlap analysis.
        This is more reliable than center-to-center for complex shapes like shelves.
        Returns the direction to move obj to separate from colliding.
        """
        obj_mesh = self._get_mesh_at_current_position(obj)
        other_mesh = self._get_mesh_at_current_position(colliding)
        
        if obj_mesh is None or other_mesh is None:
            # Fallback to center-to-center
            direction = np.array(obj.position) - np.array(colliding.position)
            direction[1] = 0  # Keep horizontal
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        
        # Get AABBs
        obj_bounds = obj_mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        other_bounds = other_mesh.bounds
        
        # Calculate overlap in each axis (horizontal only: x and z)
        overlap_x = min(obj_bounds[1][0], other_bounds[1][0]) - max(obj_bounds[0][0], other_bounds[0][0])
        overlap_z = min(obj_bounds[1][2], other_bounds[1][2]) - max(obj_bounds[0][2], other_bounds[0][2])
        
        # If no overlap, use center-to-center
        if overlap_x <= 0 or overlap_z <= 0:
            direction = np.array(obj.position) - np.array(colliding.position)
            direction[1] = 0
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        
        # Move in the direction of minimum overlap (minimum translation vector)
        direction = np.array([0.0, 0.0, 0.0])
        
        if overlap_x <= overlap_z:
            # Separate along X axis
            obj_center_x = (obj_bounds[0][0] + obj_bounds[1][0]) / 2
            other_center_x = (other_bounds[0][0] + other_bounds[1][0]) / 2
            direction[0] = 1.0 if obj_center_x > other_center_x else -1.0
        else:
            # Separate along Z axis
            obj_center_z = (obj_bounds[0][2] + obj_bounds[1][2]) / 2
            other_center_z = (other_bounds[0][2] + other_bounds[1][2]) / 2
            direction[2] = 1.0 if obj_center_z > other_center_z else -1.0
        
        return direction


    def _compute_support_from_object(
        self,
        obj: SceneObject,
        sup_obj: SceneObject,
        object_bottom_y: float,
        support_tolerance: float,
    ) -> Tuple[bool, Optional[float]]:
        """Check if support object provides vertical support using raycast from center + corners."""
        hit_any = False
        support_y = None
        
        sup_mesh = self._get_mesh_at_current_position(sup_obj)
        if sup_mesh is None:
            return False, None

        # Build raycast sample points (center + four corners)
        ray_height = object_bottom_y + 0.001
        check_points = [np.array([obj.position[0], ray_height, obj.position[2]])]
        half_width = float(obj.dimensions[0] * 0.5 * 0.95)
        half_depth = float(obj.dimensions[2] * 0.5 * 0.95)
        for dx, dz in [(half_width, half_depth), (-half_width, half_depth), 
                      (half_width, -half_depth), (-half_width, -half_depth)]:
            check_points.append(np.array([obj.position[0] + dx, ray_height, obj.position[2] + dz]))

        for ray_origin in check_points:
            hits, _, _ = sup_mesh.ray.intersects_location([ray_origin], [[0, -1, 0]])

            if len(hits) == 0:
                continue

            hits_arr = np.asarray(hits)
            candidate_y = float(hits_arr[0, 1])
            distance = abs(candidate_y - ray_origin[1])

            # logger.debug("Support raycast: origin_y=%.3f, hit_y=%.3f, distance=%.3f, tolerance=%.3f", 
            #             ray_origin[1], candidate_y, distance, support_tolerance)

            # Record support surface if it's below or at the object bottom
            if candidate_y <= object_bottom_y + support_tolerance:
                support_y = candidate_y if support_y is None else max(support_y, candidate_y)
                if distance < support_tolerance:
                    hit_any = True

        return hit_any, support_y