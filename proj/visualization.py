# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def init_plot():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_facecolor('#f4f4f8')
    ax.grid(True, alpha=0.3, color='white')
    return fig, ax

def update_plot_network(ax, all_channels, current_time):
    ax.clear()
    ax.set_title(f"Symulacja Delty Rzecznej (Czas: {current_time:.0f} lat)")
    ax.set_facecolor('#f4f4f8')
    ax.grid(True, alpha=0.3, color='white')

    all_x = []
    all_y = []

    for channel in all_channels:
        if len(channel.points) > 0:
            all_x.extend(channel.points[:, 0])
            all_y.extend(channel.points[:, 1])

        for lake in channel.oxbows:
            age = current_time - lake.creation_time
            alpha = max(0.05, 0.4 - (age / 300.0))
            ax.plot(lake.points[:, 0], lake.points[:, 1], 
                    color='#556B2F', alpha=alpha, linewidth=1.0)

        lw = 1.5 + np.sqrt(channel.width) * 0.8
        
        color = '#1E90FF' if channel.is_active else '#4682B4'
        
        ax.plot(channel.points[:, 0], channel.points[:, 1], 
                color=color, linewidth=lw, alpha=0.85, zorder=2,
                solid_capstyle='round') 

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        margin_x = max(200, (max_x - min_x) * 0.1)
        margin_y = max(200, (max_y - min_y) * 0.2)
        
        ax.set_xlim(min_x - 100, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
        ax.set_aspect('equal', adjustable='datalim')