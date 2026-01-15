import matplotlib.pyplot as plt
import numpy as np

def init_plot():
    """Inicjalizuje okno wykresu."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    return fig, ax

def update_plot(ax, river):
    """
    Rysuje aktualny stan rzeki na przekazanym obiekcie Axes.
    """
    ax.clear()
    ax.set_title(f"Symulacja Meandrowania (Czas: {river.current_time:.0f} lat)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for lake in river.oxbows:
        age = river.current_time - lake.creation_time
        alpha = max(0.1, 1.0 - (age / 400.0))
        ax.plot(lake.points[:, 0], lake.points[:, 1], 
                color='green', alpha=alpha, linewidth=1.5, linestyle='--')

    ax.plot(river.points[:, 0], river.points[:, 1], 
            color='blue', linewidth=2.5, label='Koryto rzeki')
    
    if len(river.points) > 0:
        ax.scatter(river.points[-1, 0], river.points[-1, 1], color='red', s=20, zorder=5)

    ax.legend(loc='upper right')