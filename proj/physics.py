# physics.py
import numpy as np
from scipy.interpolate import CubicSpline

def resample_points(points, dx):
    """
    Interpoluje punkty rzeki, aby były rozmieszczone w równych odstępach dx.
    """
    # Zabezpieczenie przed pustymi tablicami
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    
    # Cumulative distance musi startować od 0
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumulative_dist[-1]
    
    # Jeśli rzeka jest krótsza niż krok próbkowania lub ma za mało punktów, zwracamy ją bez zmian
    if total_length < dx or len(points) < 2:
        return points

    # --- NAPRAWA BŁĘDU ---
    # Wcześniej: int(total_length / dx) mogło dać 0 lub 1 dla krótkich odcinków.
    # Teraz: Zawsze przynajmniej 2 punkty (start i koniec).
    num_intervals = int(np.ceil(total_length / dx))
    num_points = max(2, num_intervals + 1)
    
    new_dists = np.linspace(0, total_length, num_points)
    
    try:
        # CubicSpline wymaga co najmniej 2 punktów (w nowszych scipy, w starszych >3)
        # Dla bezpieczeństwa przy bardzo małej liczbie punktów używamy interpolacji liniowej
        if len(points) < 4:
            # Interpolacja liniowa dla krótkich odcinków (zabezpiecza przed błędem CubicSpline)
            new_x = np.interp(new_dists, cumulative_dist, points[:, 0])
            new_y = np.interp(new_dists, cumulative_dist, points[:, 1])
            return np.column_stack((new_x, new_y))
        else:
            # Cubic Spline dla dłuższych, ładniejszych krzywych
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