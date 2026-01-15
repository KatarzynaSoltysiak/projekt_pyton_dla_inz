import numpy as np
from scipy.interpolate import CubicSpline

def resample_points(points, dx):
    """
    Interpoluje punkty rzeki, aby były rozmieszczone w równych odstępach dx.
    Używa Cubic Spline dla zachowania gładkości.
    """
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumulative_dist[-1]
    
    if total_length < dx or len(points) < 3:
        return points

    num_points = int(total_length / dx)
    new_dists = np.linspace(0, total_length, num_points)
    
    try:
        cs_x = CubicSpline(cumulative_dist, points[:, 0])
        cs_y = CubicSpline(cumulative_dist, points[:, 1])
        new_points = np.column_stack((cs_x(new_dists), cs_y(new_dists)))
        return new_points
    except Exception as e:
        return points

def compute_curvature(points):
    """
    Oblicza lokalną krzywiznę w każdym punkcie metodą różnic skończonych.
    Wzór (2) z artykułu Paris et al. 2023.
    """
    x, y = points[:, 0], points[:, 1]
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Wzór na krzywiznę k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
    numerator = dx * ddy - dy * ddx
    denominator = np.power(dx**2 + dy**2, 1.5) + 1e-8
    
    return numerator / denominator

def compute_weighted_curvature(curvature, friction):
    """
    Oblicza krzywiznę ważoną uwzględniającą wpływ górnego biegu rzeki (upstream).
    Implementacja modelu Sylvestera (2019).
    """
    weighted_curv = np.zeros_like(curvature)
    decay_factor = np.exp(-friction)
    
    current_weight = 0.0
    for i in range(len(curvature)):
        current_weight = current_weight * decay_factor + curvature[i]
        weighted_curv[i] = current_weight
        
    return weighted_curv

def calculate_migration_vectors(points):
    """
    Oblicza wektory normalne (kierunek migracji) dla każdego punktu.
    """
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    
    norms = np.sqrt(dx**2 + dy**2) + 1e-8
    
    nx = -dy / norms
    ny = dx / norms
    
    return nx, ny