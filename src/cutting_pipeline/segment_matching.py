"""
Label-based matching functions for video segment alignment.

Focuses on the critical first 4 labels during matching:
1. Motion direction
2. Action phase
3. Silhouette
4. Camera motion
"""

from __future__ import annotations

from typing import Any


def extract_labels(segment_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Extract labels from a segment dictionary."""
    return segment_dict.get("labels")


def _motion_direction_match_score(
    source_labels: dict[str, Any],
    target_labels: dict[str, Any],
) -> float:
    """Score motion direction match: 0-1."""
    if not source_labels or not target_labels:
        return 0.0
    
    source_dir = source_labels.get("motion_direction", {})
    target_dir = target_labels.get("motion_direction", {})
    
    source_primary = str(source_dir.get("primary_direction", "")).lower()
    target_primary = str(target_dir.get("primary_direction", "")).lower()
    
    if not source_primary or not target_primary:
        return 0.0
    
    # Exact match gets full score
    if source_primary == target_primary:
        return 1.0
    
    # Extract category from direction (e.g., "charge_right" -> "charge")
    source_category = source_primary.split("_")[0] if "_" in source_primary else source_primary
    target_category = target_primary.split("_")[0] if "_" in target_primary else target_primary
    
    # Category match gets partial score
    if source_category == target_category:
        return 0.6
    
    # Check for related directions
    directional_groups = {
        "push": ["charge", "push", "forward", "drive"],
        "pull": ["retreat", "pull", "backward", "back"],
        "vertical": ["jump", "leap", "ascend", "up"],
        "rotate": ["spin", "rotate", "twist", "turn"],
    }
    
    for group, directions in directional_groups.items():
        if source_category in directions and target_category in directions:
            return 0.4
    
    return 0.0


def _action_phase_match_score(
    source_labels: dict[str, Any],
    target_labels: dict[str, Any],
) -> float:
    """Score action phase match: 0-1."""
    if not source_labels or not target_labels:
        return 0.0
    
    source_phase = source_labels.get("action_phase", {})
    target_phase = target_labels.get("action_phase", {})
    
    source_primary = str(source_phase.get("primary_phase", "")).lower()
    target_primary = str(target_phase.get("primary_phase", "")).lower()
    
    if not source_primary or not target_primary:
        return 0.0
    
    # Exact match
    if source_primary == target_primary:
        return 1.0
    
    # Check phase similarity
    # Group related phases
    phase_groups = {
        "startup": ["charge", "startup", "prepare", "setup"],
        "strike": ["swing", "strike", "hit", "impact"],
        "flight": ["flight", "airborne", "jump", "leap", "apex"],
        "recovery": ["recovery", "landing", "fall", "land", "aftermath"],
    }
    
    for group, phases in phase_groups.items():
        if source_primary in phases and target_primary in phases:
            return 0.7
    
    return 0.0


def _silhouette_match_score(
    source_labels: dict[str, Any],
    target_labels: dict[str, Any],
) -> float:
    """Score silhouette match: 0-1."""
    if not source_labels or not target_labels:
        return 0.0
    
    source_silhouette = source_labels.get("silhouette", {})
    target_silhouette = target_labels.get("silhouette", {})
    
    # Use the similarity_score directly
    source_score = float(source_silhouette.get("similarity_score", 0.0))
    target_score = float(target_silhouette.get("similarity_score", 0.0))
    
    if source_score == 0.0 and target_score == 0.0:
        return 0.0
    
    # Average the similarity scores and apply a threshold
    avg_score = (source_score + target_score) / 2.0
    
    # Shape type match bonus
    source_shape = str(source_silhouette.get("shape_type", "")).lower()
    target_shape = str(target_silhouette.get("shape_type", "")).lower()
    
    shape_match = 0.0 if source_shape != target_shape else 0.15
    
    return min(1.0, avg_score + shape_match)


def _camera_motion_match_score(
    source_labels: dict[str, Any],
    target_labels: dict[str, Any],
) -> float:
    """Score camera motion match: 0-1."""
    if not source_labels or not target_labels:
        return 0.0
    
    source_camera = source_labels.get("camera_motion", {})
    target_camera = target_labels.get("camera_motion", {})
    
    source_type = str(source_camera.get("motion_type", "")).lower()
    target_type = str(target_camera.get("motion_type", "")).lower()
    
    if not source_type or not target_type:
        return 0.0
    
    # Exact match
    if source_type == target_type:
        motion_type_score = 1.0
    else:
        # Extract base type (e.g., "pan_left" -> "pan")
        source_base = source_type.split("_")[0]
        target_base = target_type.split("_")[0]
        
        if source_base == target_base:
            motion_type_score = 0.6
        else:
            motion_type_score = 0.0
    
    # Check speed match
    source_speed = str(source_camera.get("speed_class", "")).lower()
    target_speed = str(target_camera.get("speed_class", "")).lower()
    
    speed_score = 1.0 if source_speed == target_speed else 0.3
    
    return (motion_type_score * 0.7) + (speed_score * 0.3)


def calculate_label_match_score(
    source_segment: dict[str, Any],
    target_segment: dict[str, Any],
    focus_top_4: bool = True,
) -> float:
    """
    Calculate overall label match score between two segments: 0-1.
    
    Args:
        source_segment: Source segment dict with labels
        target_segment: Target segment dict with labels
        focus_top_4: If True, only use first 4 critical labels (for matching phase)
    
    Returns:
        Match score from 0.0 to 1.0
    """
    source_labels = extract_labels(source_segment)
    target_labels = extract_labels(target_segment)
    
    # If either segment has no labels, return neutral score
    if not source_labels or not target_labels:
        return 0.5  # Neutral score
    
    # Calculate individual scores for the 4 critical labels
    motion_dir_score = _motion_direction_match_score(source_labels, target_labels)
    action_phase_score = _action_phase_match_score(source_labels, target_labels)
    silhouette_score = _silhouette_match_score(source_labels, target_labels)
    camera_score = _camera_motion_match_score(source_labels, target_labels)
    
    # Weights for the 4 critical labels
    weights = {
        "motion_direction": 0.30,  # Most important for cut continuity
        "action_phase": 0.30,      # Important for narrative continuity
        "silhouette": 0.25,        # Important for visual continuity
        "camera_motion": 0.15,     # Supporting factor
    }
    
    weighted_score = (
        motion_dir_score * weights["motion_direction"]
        + action_phase_score * weights["action_phase"]
        + silhouette_score * weights["silhouette"]
        + camera_score * weights["camera_motion"]
    )
    
    return weighted_score


def apply_label_match_bonus(
    base_score: float,
    source_segment: dict[str, Any],
    target_segment: dict[str, Any],
    bonus_weight: float = 0.15,
) -> float:
    """
    Apply label matching as a bonus to an existing score.
    
    Args:
        base_score: Base matching score
        source_segment: Source segment dict
        target_segment: Target segment dict
        bonus_weight: Weight of the label bonus (0-1)
    
    Returns:
        Adjusted score
    """
    if bonus_weight <= 0:
        return base_score
    
    label_score = calculate_label_match_score(source_segment, target_segment)
    
    # Boost only if labels match well (above 0.6)
    if label_score > 0.6:
        bonus = (label_score - 0.6) * bonus_weight
        return base_score + bonus
    
    return base_score
