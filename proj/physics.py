# physics.py
import numpy as np
from scipy.interpolate import CubicSpline

def resample_points(points, dx):
    """
    Interpoluje punkty rzeki, aby były rozmieszczone w równych odstępach dx.
    """
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumulative_dist[-1]
    
    if total_length < dx or len(points) < 2:
        return points

    num_intervals = int(np.ceil(total_length / dx))
    num_points = max(2, num_intervals + 1)
    
    new_dists = np.linspace(0, total_length, num_points)
    
    try:
        if len(points) < 4:
            new_x = np.interp(new_dists, cumulative_dist, points[:, 0])
            new_y = np.interp(new_dists, cumulative_dist, points[:, 1])
            return np.column_stack((new_x, new_y))
        else:
            cs_x = CubicSpline(cumulative_dist, points[:, 0])
            cs_y = CubicSpline(cumulative_dist, points[:, 1])
            new_points = np.column_stack((cs_x(new_dists), cs_y(new_dists)))
            return new_points
            
    except Exception as e:
        print(f"Resample error: {e}")
        return points

def compute_curvature(points):
    """
    Oblicza lokalną krzywiznę w każdym punkcie.
    """
    if len(points) < 3:
        return np.zeros(len(points))

    x, y = points[:, 0], points[:, 1]
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    numerator = dx * ddy - dy * ddx
    denominator = np.power(dx**2 + dy**2, 1.5) + 1e-8
    
    return numerator / denominator

def compute_weighted_curvature(curvature, friction):
    """
    Krzywizna ważona (wpływ upstream).
    """
    weighted_curv = np.zeros_like(curvature)
    if len(curvature) == 0:
        return weighted_curv
        
    decay_factor = np.exp(-friction)
    
    current_weight = 0.0
    for i in range(len(curvature)):
        current_weight = current_weight * decay_factor + curvature[i]
        weighted_curv[i] = current_weight
        
    return weighted_curv

def calculate_migration_vectors(points):
    """
    Oblicza wektory normalne.
    """
    if len(points) < 2:
        return np.zeros_like(points), np.zeros_like(points)

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    
    norms = np.sqrt(dx**2 + dy**2) + 1e-8
    
    nx = -dy / norms
    ny = dx / norms
    
    return nx, ny

def smooth_signal(x, k=7):
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")
