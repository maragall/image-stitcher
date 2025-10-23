"""Global optimization module for microscope tile registration.

This module provides functionality for globally optimizing tile positions using a
maximum spanning tree approach. It ensures consistent and optimal positioning of
all tiles while respecting their pairwise relationships.

The module includes:
- Maximum spanning tree computation based on correlation values
- Global position optimization from local translations
- Robust handling of disconnected components
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ._typing_utils import Int

# Configure logger
logger = logging.getLogger(__name__)

# Constants for weight computation
VALID_TRANSLATION_WEIGHT_BONUS = 10.0
MIN_VALID_WEIGHT = -1.0
MAX_VALID_WEIGHT = 1.0 + VALID_TRANSLATION_WEIGHT_BONUS


@dataclass
class TileGrid:
    """Validated tile grid with type-enforced structure.
    
    Replaces 6 validation functions with simple construction validation.
    """
    df: pd.DataFrame
    column_scheme: Dict[str, str]
    
    @classmethod
    def from_dataframe(cls, grid: pd.DataFrame) -> 'TileGrid':
        """Create validated TileGrid from DataFrame.
        
        Args:
            grid: DataFrame with tile data
            
        Returns:
            Validated TileGrid instance
            
        Raises:
            TypeError: If input is not a DataFrame
            ValueError: If structure is invalid
        """
        if not isinstance(grid, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(grid).__name__}")
        
        if grid.empty:
            raise ValueError("Empty grid DataFrame")
        
        # Detect column scheme
        scheme = cls._detect_columns(grid)
        
        # Validate at least one direction exists
        has_left = all(scheme.get(f'left_{c}') in grid.columns for c in ['x', 'y', 'ncc'])
        has_top = all(scheme.get(f'top_{c}') in grid.columns for c in ['x', 'y', 'ncc'])
        
        if not (has_left or has_top):
            raise ValueError("No complete direction column sets found")
        
        # Basic range validation
        for direction in ['left', 'top']:
            ncc_col = scheme.get(f'{direction}_ncc')
            if ncc_col in grid.columns:
                ncc_vals = grid[ncc_col].dropna()
                if len(ncc_vals) > 0 and not ncc_vals.between(-1.0, 1.0).all():
                    raise ValueError(f"Invalid NCC values in {ncc_col}")
        
        return cls(df=grid, column_scheme=scheme)
    
    @staticmethod
    def _detect_columns(grid: pd.DataFrame) -> Dict[str, str]:
        """Detect column naming convention."""
        # Check for _first suffix (newer)
        first_suffix = {'left_x': 'left_x_first', 'left_y': 'left_y_first', 
                       'left_ncc': 'left_ncc_first', 'top_x': 'top_x_first',
                       'top_y': 'top_y_first', 'top_ncc': 'top_ncc_first'}
        
        # Check for no suffix (older)
        no_suffix = {'left_x': 'left_x', 'left_y': 'left_y', 'left_ncc': 'left_ncc',
                    'top_x': 'top_x', 'top_y': 'top_y', 'top_ncc': 'top_ncc'}
        
        first_matches = sum(1 for col in first_suffix.values() if col in grid.columns)
        no_matches = sum(1 for col in no_suffix.values() if col in grid.columns)
        
        scheme = first_suffix if first_matches >= no_matches else no_suffix
        scheme.update({'left_valid3': 'left_valid3', 'top_valid3': 'top_valid3'})
        return scheme


def validate_grid_dataframe(grid: pd.DataFrame) -> None:
    """[DEPRECATED] Use TileGrid.from_dataframe() instead.
    
    Legacy validation function - replaced by TileGrid dataclass.
    
    Args:
        grid: DataFrame to validate
        
    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame structure or content is invalid
    """
    # Delegate to TileGrid which does proper validation
    TileGrid.from_dataframe(grid)


def compute_maximum_spanning_tree(grid: pd.DataFrame) -> nx.Graph:
    """Compute maximum spanning tree for global tile position optimization.

    Creates a graph where edges represent tile overlaps, weighted by their
    normalized cross-correlation values. Valid translations get a bonus weight
    to prefer them in the spanning tree.

    Args:
        grid: DataFrame with tile positions and translations

    Returns:
        NetworkX Graph representing the maximum spanning tree

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If MST computation fails
    """
    validate_grid_dataframe(grid)
    
    # Create graph
    connection_graph = nx.Graph()
    
    # Add edges with weights
    for i, g in grid.iterrows():
        for direction in ["left", "top"]:
            if not pd.isna(g[direction]):
                # Resolve column names with proper validation
                try:
                    ncc_col, y_col, x_col = _resolve_column_names(grid, direction)
                except ValueError as e:
                    logger.warning(f"Skipping {direction} connection for tile {i}: {e}")
                    continue
                
                # Validate and compute weight
                try:
                    weight = _compute_edge_weight(g, ncc_col, direction)
                    if weight is None:
                        logger.debug(f"Skipping {direction} connection for tile {i}: NaN NCC value")
                        continue
                except ValueError as e:
                    logger.warning(f"Skipping {direction} connection for tile {i}: {e}")
                    continue
                
                # Convert neighbor index to integer
                try:
                    neighbor_idx = int(g[direction])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid neighbor index for tile {i} {direction}: {g[direction]}")
                    continue
                
                # Validate translation values (inline validation)
                try:
                    y_val, x_val = g[y_col], g[x_col]
                    
                    # Validate Y translation
                    if pd.isna(y_val):
                        raise ValueError(f"NaN translation value for {direction}_y")
                    if not isinstance(y_val, (int, float, np.number)):
                        raise ValueError(f"Invalid translation value type for {direction}_y: {type(y_val)}")
                    if not np.isfinite(y_val):
                        raise ValueError(f"Non-finite translation value for {direction}_y: {y_val}")
                    if abs(y_val) > 10000:  # pixels
                        raise ValueError(f"Suspiciously large translation value for {direction}_y: {abs(y_val)} > 10000")
                    
                    # Validate X translation
                    if pd.isna(x_val):
                        raise ValueError(f"NaN translation value for {direction}_x")
                    if not isinstance(x_val, (int, float, np.number)):
                        raise ValueError(f"Invalid translation value type for {direction}_x: {type(x_val)}")
                    if not np.isfinite(x_val):
                        raise ValueError(f"Non-finite translation value for {direction}_x: {x_val}")
                    if abs(x_val) > 10000:
                        raise ValueError(f"Suspiciously large translation value for {direction}_x: {abs(x_val)} > 10000")
                    
                    y_translation = float(y_val)
                    x_translation = float(x_val)
                except ValueError as e:
                    logger.warning(f"Skipping {direction} connection for tile {i}: {e}")
                    continue
                    
                # Store translation with proper semantics
                # Translation is from neighbor_idx -> i (neighbor to current)
                # Edge will be created as i -> neighbor_idx (current to neighbor)
                # So translation direction is opposite to edge f->t direction
                
                # Add edge with proper translation semantics
                connection_graph.add_edge(
                    i,
                    neighbor_idx,
                    weight=weight,
                    direction=direction,
                    f=i,  # current tile
                    t=neighbor_idx,  # neighbor tile  
                    y=y_translation,  # translation from neighbor -> current
                    x=x_translation,  # translation from neighbor -> current
                )
    
    if not connection_graph.edges:
        raise ValueError("No valid connections between tiles")
    
    # Check minimum edge count for viable tree
    num_nodes = len(connection_graph.nodes)
    num_edges = len(connection_graph.edges)
    min_edges_required = num_nodes - 1
    
    if num_edges < min_edges_required:
        raise ValueError(
            f"Insufficient edges for spanning tree: {num_edges} edges for {num_nodes} nodes. "
            f"Minimum required: {min_edges_required}. Check NCC threshold or image quality."
        )
    
    # Check if graph has multiple disconnected components
    num_components = nx.number_connected_components(connection_graph)
    if num_components > 1:
        # Get component sizes for informative error message
        components = list(nx.connected_components(connection_graph))
        component_sizes = [len(c) for c in components]
        logger.error(
            f"Graph has {num_components} disconnected components with sizes: {component_sizes}. "
            f"Cannot form single spanning tree."
        )
        raise ValueError(
            f"Tile graph has {num_components} disconnected components (sizes: {component_sizes}). "
            f"Check NCC threshold, image quality, or tile overlap. "
            f"Tiles may be too far apart or have insufficient overlap for registration."
        )
    
    # Validate all edge weights before MST computation (inline validation)
    invalid_edges = []
    for u, v, data in connection_graph.edges(data=True):
        weight = data.get('weight')
        if weight is None:
            invalid_edges.append((u, v, "missing weight"))
        elif pd.isna(weight):
            invalid_edges.append((u, v, "NaN weight"))
        elif not isinstance(weight, (int, float, np.number)):
            invalid_edges.append((u, v, f"invalid weight type: {type(weight)}"))
        elif not np.isfinite(weight):
            invalid_edges.append((u, v, f"non-finite weight: {weight}"))
    
    if invalid_edges:
        error_summary = "; ".join([f"Edge ({u},{v}): {error}" for u, v, error in invalid_edges[:5]])
        if len(invalid_edges) > 5:
            error_summary += f"; ... and {len(invalid_edges) - 5} more"
        raise ValueError(f"Invalid edge weights found: {error_summary}")
        
    # Compute maximum spanning tree
    try:
        tree = nx.maximum_spanning_tree(connection_graph)
        logger.info(f"Created spanning tree with {len(tree.edges)} edges from {len(connection_graph.edges)} total edges")
        
        # Check if tree is connected
        if not nx.is_connected(tree):
            num_components = nx.number_connected_components(tree)
            logger.warning(f"Tree has {num_components} disconnected components")
        
        return tree
    except Exception as e:
        raise RuntimeError(f"Failed to compute maximum spanning tree: {e}")


def _resolve_column_names(grid: pd.DataFrame, direction: str) -> Tuple[str, str, str]:
    """Resolve column names for a given direction with proper validation.
    
    Args:
        grid: DataFrame to check for columns
        direction: Direction ('left' or 'top')
        
    Returns:
        Tuple of (ncc_col, y_col, x_col) names
        
    Raises:
        ValueError: If required columns are missing
    """
    # Try _first suffix convention first
    ncc_col_first = f"{direction}_ncc_first"
    y_col_first = f"{direction}_y_first"
    x_col_first = f"{direction}_x_first"
    
    # Try non-suffix convention
    ncc_col_nosuffix = f"{direction}_ncc"
    y_col_nosuffix = f"{direction}_y"
    x_col_nosuffix = f"{direction}_x"
    
    # Check if all _first columns exist
    if all(col in grid.columns for col in [ncc_col_first, y_col_first, x_col_first]):
        return ncc_col_first, y_col_first, x_col_first
    
    # Check if all non-suffix columns exist
    if all(col in grid.columns for col in [ncc_col_nosuffix, y_col_nosuffix, x_col_nosuffix]):
        return ncc_col_nosuffix, y_col_nosuffix, x_col_nosuffix
    
    # Neither complete set exists - provide detailed error
    available_cols = [col for col in grid.columns if direction in col]
    raise ValueError(
        f"Missing required {direction} columns. Need either "
        f"[{ncc_col_first}, {y_col_first}, {x_col_first}] or "
        f"[{ncc_col_nosuffix}, {y_col_nosuffix}, {x_col_nosuffix}]. "
        f"Available {direction}-related columns: {available_cols}"
    )


def _compute_edge_weight(row: pd.Series, ncc_col: str, direction: str) -> Optional[float]:
    """Compute edge weight with proper validation and NaN handling.
    
    Args:
        row: DataFrame row containing tile data
        ncc_col: Column name for NCC values
        direction: Direction ('left' or 'top')
        
    Returns:
        Computed weight value, or None if NaN NCC
        
    Raises:
        ValueError: If weight computation fails
    """
    # Get base weight (NCC value)
    try:
        base_weight = row[ncc_col]
    except KeyError:
        raise ValueError(f"Column {ncc_col} not found in data")
    
    # Handle NaN NCC values - return None to signal skip
    if pd.isna(base_weight):
        return None
    
    # Validate NCC range
    if not isinstance(base_weight, (int, float, np.number)):
        raise ValueError(f"Invalid NCC value type: {type(base_weight)}")
    
    if not (MIN_VALID_WEIGHT <= base_weight <= MAX_VALID_WEIGHT):
        raise ValueError(
            f"NCC value {base_weight} outside valid range [{MIN_VALID_WEIGHT}, {MAX_VALID_WEIGHT}]"
        )
    
    weight = float(base_weight)
    
    # Add bonus for validated translations
    valid3_col = f"{direction}_valid3"
    try:
        is_valid = row[valid3_col]
        # Handle NaN in valid3 column (treat as False)
        if pd.isna(is_valid):
            is_valid = False
        # Convert to boolean if needed
        if isinstance(is_valid, (int, np.integer)):
            is_valid = bool(is_valid)
        
        if is_valid:
            weight += VALID_TRANSLATION_WEIGHT_BONUS
            
    except KeyError:
        # valid3 column missing - log warning but continue without bonus
        logger.warning(f"Column {valid3_col} not found, skipping validation bonus")
    
    return weight


def compute_final_position(
    grid: pd.DataFrame,
    tree: nx.Graph,
    source_index: Int = 0,
    pixel_size_um: Optional[float] = None
) -> pd.DataFrame:
    """Compute final tile positions using the maximum spanning tree.

    Traverses the spanning tree to compute consistent global positions for all
    tiles based on their pairwise translations. Handles multiple connected
    components by positioning each relative to its own reference tile.

    Args:
        grid: DataFrame with tile translations
        tree: Maximum spanning tree from compute_maximum_spanning_tree
        source_index: Index of the reference tile (default: 0)

    Returns:
        DataFrame with added x_pos and y_pos columns

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If position computation fails
    """
    if not isinstance(tree, nx.Graph):
        raise ValueError("tree must be a NetworkX Graph")
    if not tree.edges:
        raise ValueError("Empty spanning tree")
    if source_index not in grid.index:
        raise ValueError(f"source_index {source_index} not found in grid DataFrame")
        
    # Initialize position columns
    grid['y_pos'] = np.nan
    grid['x_pos'] = np.nan
    
    # Track which tiles are in the tree
    tiles_in_tree = set(tree.nodes())
    tiles_not_in_tree = set(grid.index) - tiles_in_tree
    
    if tiles_not_in_tree:
        logger.warning(
            f"Found {len(tiles_not_in_tree)} tiles not in spanning tree (featureless/isolated). "
            f"These will keep their original stage positions."
        )

    # Find connected components
    components = list(nx.connected_components(tree))
    logger.info(f"Found {len(components)} connected components")
    
    try:
        # Process each component
        for component in components:
            # Choose reference tile for this component
            component_source = source_index if source_index in component else min(component)
        
            # Set reference position
            grid.loc[component_source, "y_pos"] = 0
            grid.loc[component_source, "x_pos"] = 0
        
            # Process nodes in breadth-first order
            nodes_to_process = [component_source]
            processed_nodes: Set[int] = set()
            
            while nodes_to_process:
                current_node = nodes_to_process.pop(0)
                processed_nodes.add(current_node)
                
                # Process neighbors
                for neighbor in tree.neighbors(current_node):
                    if neighbor not in processed_nodes:
                        nodes_to_process.append(neighbor)
                        
                        # Get edge properties with validation
                        edge_data = tree.edges[current_node, neighbor]
                        required_keys = ["f", "t", "y", "x"]
                        for key in required_keys:
                            if key not in edge_data:
                                raise RuntimeError(f"Edge {current_node}-{neighbor} missing required key: {key}")
                        
                        current_pos = (
                            grid.loc[current_node, "y_pos"],
                            grid.loc[current_node, "x_pos"]
                        )
                        
                        # Validate translation values
                        try:
                            y_trans, x_trans = edge_data["y"], edge_data["x"]
                            if not (isinstance(y_trans, (int, float)) and isinstance(x_trans, (int, float))):
                                raise ValueError("Translation values must be numeric")
                        except (KeyError, ValueError, TypeError) as e:
                            raise RuntimeError(f"Invalid translation values in edge {current_node}-{neighbor}: {e}")
                        
                        # Apply the correct logic based on edge direction
                        # Translation is from neighbor -> current (stored as f->t edge)
                        # If current_node == t: we're traversing f->t (neighbor->current), so ADD translation
                        # If current_node == f: we're traversing t->f (current->neighbor), so SUBTRACT translation
                        if current_node == edge_data["t"]:
                            # Traversing neighbor -> current: ADD translation
                            grid.loc[neighbor, "y_pos"] = current_pos[0] + y_trans
                            grid.loc[neighbor, "x_pos"] = current_pos[1] + x_trans
                        else:
                            # Traversing current -> neighbor: SUBTRACT translation  
                            grid.loc[neighbor, "y_pos"] = current_pos[0] - y_trans
                            grid.loc[neighbor, "x_pos"] = current_pos[1] - x_trans

        # For tiles not in tree, use original stage positions if available
        if tiles_not_in_tree:
            if 'stage_x' in grid.columns and 'stage_y' in grid.columns:
                # Get pixel size from connected tiles (stage mm per pixel)
                # Use first connected tile as reference to calculate scale
                ref_idx = source_index
                ref_stage_x = grid.loc[ref_idx, 'stage_x']
                ref_stage_y = grid.loc[ref_idx, 'stage_y']
                ref_px_x = grid.loc[ref_idx, 'x_pos']
                ref_px_y = grid.loc[ref_idx, 'y_pos']
                
                for tile_idx in tiles_not_in_tree:
                    stage_x = grid.loc[tile_idx, 'stage_x']
                    stage_y = grid.loc[tile_idx, 'stage_y']
                    
                    # Convert mm to pixels using provided pixel size, or compute from neighbors
                    if pixel_size_um is not None:
                        px_per_mm = 1000.0 / pixel_size_um
                    else:
                        # Fallback: estimate from connected tile spacing if possible
                        px_per_mm = 1333.0  # ~0.75 um/pixel at 10x
                    
                    grid.loc[tile_idx, "x_pos"] = ref_px_x + (stage_x - ref_stage_x) * px_per_mm
                    grid.loc[tile_idx, "y_pos"] = ref_px_y + (stage_y - ref_stage_y) * px_per_mm
                    
                logger.info(f"Set {len(tiles_not_in_tree)} isolated tiles using stage coordinates")
            else:
                raise RuntimeError(
                    f"Cannot position {len(tiles_not_in_tree)} isolated tiles: no stage coordinates available"
                )
        
        # Validate that all positions are now set
        if grid[["y_pos", "x_pos"]].isna().any().any():
            raise RuntimeError("Failed to compute positions for all tiles")
            
        # Normalize positions to start from (0,0)
        for dim in "yx":
            col = f"{dim}_pos"
            grid[col] = grid[col] - grid[col].min()
            grid[col] = pd.array(np.round(grid[col]), dtype='Int32')

        logger.info("Successfully computed final positions for all tiles")
        return grid
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute final positions: {e}")
