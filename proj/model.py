# model.py
import numpy as np
import physics

class OxbowLake:
    def __init__(self, points, creation_time):
        self.points = points
        self.creation_time = creation_time

class RiverChannel:
    def __init__(self, x, y=None, width=60.0, dt=1.0, dx=30.0, parent_id=None):
        if y is None:
            self.points = np.array(x)
        else:
            self.points = np.column_stack((x, y))
            
        self.width = width
        self.dt = dt
        self.dx = dx
        self.parent_id = parent_id
        
        self.is_active = True
        self.children = []
        self.oxbows = []
        
        # PARAMETRY
        self.k_mig = 1.0
        self.friction = 0.05
        self.current_time = 0.0
        
        # Pamięć generalnego kierunku rzeki (kluczowe dla kształtu wachlarza)
        self.general_direction = np.array([1.0, 0.0])

    def grow_downstream(self, growth_speed=3.0):
        """Wydłuża rzekę zgodnie z jej unikalnym kierunkiem (wachlarz)."""
        if not self.is_active or len(self.points) < 1:
            return

        # 1. Obliczamy lokalny wektor na końcu rzeki
        lookback = min(len(self.points), 5)
        if lookback < 2:
            local_dir = self.general_direction.copy()
        else:
            local_dir = self.points[-1] - self.points[-lookback]
            dist = np.linalg.norm(local_dir)
            if dist > 0:
                local_dir /= dist
            else:
                local_dir = self.general_direction.copy()

        # 2. Szum (lekkie meandrowanie podczas wzrostu)
        angle_change = np.random.normal(0, 0.08) 
        c, s = np.cos(angle_change), np.sin(angle_change)
        
        noisy_dir_x = local_dir[0] * c - local_dir[1] * s
        noisy_dir_y = local_dir[0] * s + local_dir[1] * c
        noisy_dir = np.array([noisy_dir_x, noisy_dir_y])
        
        # 3. Bias ciągnie w stronę GENERALNEGO KIERUNKU tej konkretnej gałęzi
        bias_strength = 0.2
        
        final_dir = noisy_dir * (1.0 - bias_strength) + self.general_direction * bias_strength
        final_dir /= (np.linalg.norm(final_dir) + 1e-6)

        new_point = self.points[-1] + final_dir * growth_speed
        self.points = np.vstack([self.points, new_point])

    def migrate(self):
        """Fizyczna migracja meandrów (zgodnie z artykułem Paris et al.)."""
        if not self.is_active or len(self.points) < 10:
            return

        self.current_time += self.dt
        
        curvature = physics.compute_curvature(self.points)
        w_curv = physics.compute_weighted_curvature(curvature, self.friction)
        nx, ny = physics.calculate_migration_vectors(self.points)
        
        migration_rate = self.width * self.k_mig * w_curv
        
        # Tłumienie migracji na końcach (zakotwiczenie)
        dampening = np.ones_like(migration_rate)
        buffer = 15
        
        # Zawsze zerujemy początek - rzeka nie może oderwać się od rodzica
        dampening[0] = 0.0 
        
        if len(dampening) > 2 * buffer:
            dampening[:buffer] = 0.0 
            dampening[-5:] = 0.0     
            dampening[buffer:buffer+10] = np.linspace(0, 1, 10)
        else:
            safe_idx = min(len(dampening)-1, 5)
            dampening[:safe_idx] = 0.0

        migration_rate *= dampening

        self.points[:, 0] += nx * migration_rate * self.dt
        self.points[:, 1] += ny * migration_rate * self.dt
        
        self.points = physics.resample_points(self.points, self.dx)
        self.check_cutoffs()

    def check_cutoffs(self):
        """Odcinanie pętli (starorzecza)."""
        points = self.points
        n = len(points)
        if n < 20: return
        min_sep = int(self.width / self.dx) * 2 + 5
        if n <= min_sep: return

        start_search = 10 
        for i in range(start_search, n - min_sep, 2):
            segment = points[i + min_sep:]
            dists = np.linalg.norm(segment - points[i], axis=1)
            close = np.where(dists < self.width * 0.7)[0]
            
            if len(close) > 0:
                j = (i + min_sep) + close[0]
                if j - i > 15: 
                    oxbow_loop = points[i:j+1].copy()
                    self.oxbows.append(OxbowLake(oxbow_loop, self.current_time))
                    mid = (points[i] + points[j]) * 0.5
                    self.points = np.vstack([points[:i], mid, points[j+1:]])
                    self.points = physics.resample_points(self.points, self.dx)
                    break

    def branch(self):
        """Tworzenie nowych gałęzi (bifurkacja)."""
        if len(self.points) < 5: return []
        
        tip = self.points[-1].copy()
        prev = self.points[-5]
        
        # Aktualny kierunek przepływu na końcu rzeki
        current_dir = tip - prev
        current_dir /= (np.linalg.norm(current_dir) + 1e-6)
        
        # ASYMETRYCZNY PODZIAŁ - WACHLARZ
        # Kąt rozwarcia między gałęziami (30-60 stopni)
        spread = np.radians(np.random.uniform(30, 60))
        # Losowy przechył (żeby nie zawsze było symetrycznie względem środka)
        tilt = np.radians(np.random.uniform(-15, 15))
        
        theta_left = tilt + spread / 2.0
        theta_right = tilt - spread / 2.0
        
        # Proporcje szerokości (np. 70% wody w lewo, 30% w prawo)
        ratio = np.random.uniform(0.3, 0.7)
        w_left = self.width * np.sqrt(ratio)
        w_right = self.width * np.sqrt(1.0 - ratio)

        def rotate_vector(vec, angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c])
        
        # Nowe kierunki generalne dla dzieci
        dir_left = rotate_vector(current_dir, theta_left)
        dir_right = rotate_vector(current_dir, theta_right)
        
        # Punkty startowe (lekkie rozsunięcie)
        start_step = self.width * 0.2
        p2_l = tip + dir_left * start_step
        p2_r = tip + dir_right * start_step
        
        pts_left = np.vstack([tip, p2_l])
        pts_right = np.vstack([tip, p2_r])
        
        ch_left = RiverChannel(pts_left, width=w_left, dt=self.dt, dx=self.dx, parent_id=self)
        ch_right = RiverChannel(pts_right, width=w_right, dt=self.dt, dx=self.dx, parent_id=self)
        
        # PRZEKAZANIE KIERUNKÓW (kluczowe!)
        ch_left.general_direction = dir_left
        ch_right.general_direction = dir_right
        
        # Mutacja parametrów (różnorodność)
        ch_left.k_mig = self.k_mig * np.random.uniform(0.8, 1.2)
        ch_right.k_mig = self.k_mig * np.random.uniform(0.8, 1.2)
        
        self.is_active = False
        self.children = [ch_left, ch_right]
        return [ch_left, ch_right]