from dataclasses import dataclass

@dataclass
class SceneSpatialOptimizerConfig:
    """Configuration for scene level spatial optimization."""
    use_motif_level_optimization: bool = True
    
    # Collision detection settings
    bbox_collision_tolerance: float = 0.001
    max_collision_iterations: int = 20
    wall_y_movement_factor: float = 0.1
    adaptive_step_factor: float = 0.5
    vertical_step_factor: float = 0.3 
    horizontal_step_factor: float = 0.4
    max_step_size: float = 0.15
    room_bounds_margin: float = 0.01
    
    # Support Detection
    support_tolerance: float = 0.01  # threshold consider object supported
    support_stability_offset: float = 0.001  # 1mm offset for stability

@dataclass
class DFSSolverConfig:
    """Configuration for the DFS solver."""
    grid_size: float = 0.1
    max_duration: float = 10.0
    max_candidates_per_motif: int = 10
    # wall_proximity_threshold: float = 0.1 # now use grid size
    alignment_threshold: float = -0.7
    epsilon: float = 1e-6
    
    # soft constraints
    initial_placement_range: float = 5.0 # the distance (in meters) over which initial placement score decays
    initial_placement_weight: float = 5.0 # weight for initial placement score
    wall_alignment_weight: float = 2.5  # weight for wall alignment score
    wall_alignment_range: float = 0.5   # the distance (in meters) over which wall score decays

