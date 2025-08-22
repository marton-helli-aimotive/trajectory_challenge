"""Safety-critical evaluation metrics for trajectory prediction."""

from typing import Any

import numpy as np


def minimum_distance(
    predicted_points: np.ndarray, ground_truth_points: np.ndarray
) -> float:
    """Calculate minimum distance between predicted and ground truth trajectories."""
    if predicted_points.size == 0 or ground_truth_points.size == 0:
        return float("inf")

    min_dist = float("inf")
    for pred_point in predicted_points:
        for gt_point in ground_truth_points:
            dist = np.linalg.norm(pred_point - gt_point)
            min_dist = min(min_dist, float(dist))

    return min_dist


def time_to_collision(
    predicted_points: list[tuple[float, float]],
    ground_truth_points: list[tuple[float, float]],
    time_steps: list[float] | None = None,
    collision_threshold: float = 2.0,
) -> float | None:
    """Calculate time to collision between predicted and actual trajectories.

    Args:
        predicted_points: List of (x, y) predicted coordinates
        ground_truth_points: List of (x, y) ground truth coordinates
        time_steps: Time stamps for each point (if None, assumes uniform spacing)
        collision_threshold: Distance threshold for collision detection (meters)

    Returns:
        Time to collision in seconds, or None if no collision predicted
    """
    if not predicted_points or not ground_truth_points:
        return None

    min_len = min(len(predicted_points), len(ground_truth_points))

    time_steps = list(range(min_len)) if time_steps is None else time_steps[:min_len]

    for i in range(min_len):
        pred_x, pred_y = predicted_points[i]
        true_x, true_y = ground_truth_points[i]
        dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

        if dist <= collision_threshold:
            return time_steps[i] if i < len(time_steps) else float(i)

    return None


def lateral_error(
    predicted_points: list[tuple[float, float]],
    ground_truth_points: list[tuple[float, float]],
) -> list[float]:
    """Calculate lateral (cross-track) error between trajectories.

    Computes perpendicular distance from predicted points to the line segments
    of the ground truth trajectory.

    Args:
        predicted_points: List of (x, y) predicted coordinates
        ground_truth_points: List of (x, y) ground truth coordinates

    Returns:
        List of lateral errors for each predicted point
    """
    if not predicted_points or not ground_truth_points or len(ground_truth_points) < 2:
        return [float("inf")] * len(predicted_points)

    lateral_errors = []

    for pred_x, pred_y in predicted_points:
        min_lateral_error = float("inf")

        # Find minimum distance to any line segment in ground truth
        for i in range(len(ground_truth_points) - 1):
            x1, y1 = ground_truth_points[i]
            x2, y2 = ground_truth_points[i + 1]

            # Calculate perpendicular distance to line segment
            lateral_err = _point_to_line_segment_distance(
                pred_x, pred_y, x1, y1, x2, y2
            )
            min_lateral_error = min(min_lateral_error, lateral_err)

        lateral_errors.append(min_lateral_error)

    return lateral_errors


def _point_to_line_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculate perpendicular distance from point to line segment."""
    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Degenerate line segment
        return float(np.sqrt((px - x1) ** 2 + (py - y1) ** 2))

    # Parameter t for closest point on line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # Clamp t to line segment
    t = max(0.0, min(1.0, t))

    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return float(np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2))


def collision_probability(
    predicted_points: list[tuple[float, float]],
    ground_truth_points: list[tuple[float, float]],
    uncertainty_bounds: list[tuple[float, float]] | None = None,
    collision_threshold: float = 2.0,
) -> float:
    """Calculate collision probability based on trajectory predictions.

    Args:
        predicted_points: List of (x, y) predicted coordinates
        ground_truth_points: List of (x, y) ground truth coordinates
        uncertainty_bounds: Optional list of (std_x, std_y) uncertainty bounds
        collision_threshold: Distance threshold for collision detection

    Returns:
        Probability of collision (0.0 to 1.0)
    """
    if not predicted_points or not ground_truth_points:
        return 0.0

    min_len = min(len(predicted_points), len(ground_truth_points))
    collision_events = 0

    for i in range(min_len):
        pred_x, pred_y = predicted_points[i]
        true_x, true_y = ground_truth_points[i]

        # Base distance
        base_dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

        if uncertainty_bounds and i < len(uncertainty_bounds):
            # Consider uncertainty in collision probability
            std_x, std_y = uncertainty_bounds[i]

            # Monte Carlo approximation of collision probability
            n_samples = 1000
            collision_count = 0

            for _ in range(n_samples):
                # Sample from uncertainty distribution
                sampled_pred_x = np.random.normal(pred_x, std_x)
                sampled_pred_y = np.random.normal(pred_y, std_y)

                sampled_dist = np.sqrt(
                    (sampled_pred_x - true_x) ** 2 + (sampled_pred_y - true_y) ** 2
                )

                if sampled_dist <= collision_threshold:
                    collision_count += 1

            if collision_count > 0:
                collision_events += 1
        else:
            # Deterministic collision check
            if base_dist <= collision_threshold:
                collision_events += 1

    return collision_events / min_len if min_len > 0 else 0.0


def safety_critical_scenarios(
    predicted_points: list[tuple[float, float]],
    ground_truth_points: list[tuple[float, float]],
    ttc_threshold: float = 5.0,
    min_distance_threshold: float = 5.0,
) -> dict[str, Any]:
    """Identify safety-critical scenarios based on trajectory analysis.

    Args:
        predicted_points: List of (x, y) predicted coordinates
        ground_truth_points: List of (x, y) ground truth coordinates
        ttc_threshold: Time-to-collision threshold for critical scenarios
        min_distance_threshold: Minimum distance threshold for critical scenarios

    Returns:
        Dictionary with safety scenario analysis
    """
    results: dict[str, Any] = {
        "is_critical": False,
        "critical_factors": [],
        "severity_score": 0.0,
        "risk_level": "low",
    }

    if not predicted_points or not ground_truth_points:
        return results

    # Calculate safety metrics
    # Convert to numpy arrays for minimum distance calculation
    pred_array = np.array(predicted_points)
    gt_array = np.array(ground_truth_points)
    min_dist = minimum_distance(pred_array, gt_array)
    ttc = time_to_collision(predicted_points, ground_truth_points)
    lateral_errors = lateral_error(predicted_points, ground_truth_points)

    severity_factors: list[float] = []

    # Check minimum distance
    if min_dist < min_distance_threshold:
        results["critical_factors"].append("minimum_distance")
        severity_factors.append(1.0 - (min_dist / min_distance_threshold))

    # Check time to collision
    if ttc is not None and ttc < ttc_threshold:
        results["critical_factors"].append("time_to_collision")
        severity_factors.append(1.0 - (ttc / ttc_threshold))

    # Check lateral errors
    max_lateral_error = max(lateral_errors) if lateral_errors else 0.0
    if max_lateral_error > 3.0:  # 3 meter threshold
        results["critical_factors"].append("lateral_deviation")
        severity_factors.append(min(max_lateral_error / 10.0, 1.0))

    # Calculate severity score
    if severity_factors:
        results["is_critical"] = True
        results["severity_score"] = max(severity_factors)

        if results["severity_score"] > 0.8:
            results["risk_level"] = "high"
        elif results["severity_score"] > 0.5:
            results["risk_level"] = "medium"
        else:
            results["risk_level"] = "low"

    return results
