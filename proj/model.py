import numpy as np
import physics

class OxbowLake:
    def __init__(self, points, creation_time):
        self.points = points
        self.creation_time = creation_time

class RiverChannel:
    def __init__(self, x, y, width=60.0, dt=1.0, dx=40.0):
        self.points = np.column_stack((x, y))
        self.width = width
        self.dt = dt
        self.dx = dx
        
        self.k_mig = 0.5
        self.friction = 0.05
        
        self.oxbows = []
        self.current_time = 0.0

    def migrate(self):
        """Wykonuje jeden krok migracji fizycznej."""
        self.current_time += self.dt
        
        curvature = physics.compute_curvature(self.points)
        w_curv = physics.compute_weighted_curvature(curvature, self.friction)
        nx, ny = physics.calculate_migration_vectors(self.points)
        
        migration_rate = self.width * self.k_mig * w_curv
        
        dampening = np.ones_like(migration_rate)
        buffer = 10
        if len(dampening) > 2 * buffer:
            dampening[:buffer] = np.linspace(0, 1, buffer)
            dampening[-buffer:] = np.linspace(1, 0, buffer)
        migration_rate *= dampening

        self.points[:, 0] += nx * migration_rate * self.dt
        self.points[:, 1] += ny * migration_rate * self.dt
        
        self.points = physics.resample_points(self.points, self.dx)
        
        self.check_cutoffs()

    def check_cutoffs(self):
        """Wykrywa samoprzecięcia i tworzy starorzecza."""
        points = self.points
        n = len(points)
        if n < 10: return

        min_separation = int(self.width / self.dx) * 2 + 2 
        cutoff_found = False
        
        for i in range(n - min_separation):
            segment_check = points[i + min_separation:]
            dists = np.linalg.norm(segment_check - points[i], axis=1)
            
            close_indices = np.where(dists < self.width)[0]
            
            if len(close_indices) > 0:
                j = (i + min_separation) + close_indices[0]
                
                oxbow_loop = points[i:j+1].copy()
                self.oxbows.append(OxbowLake(oxbow_loop, self.current_time))
                
                mid_point = (points[i] + points[j]) * 0.5
                self.points = np.vstack([points[:i], mid_point, points[j+1:]])
                
                print(f"[{self.current_time:.1f}s] Odcięcie zakola! (Rozmiar: {len(oxbow_loop)} pkt)")
                cutoff_found = True
                break 
        
        if cutoff_found:
            self.points = physics.resample_points(self.points, self.dx)