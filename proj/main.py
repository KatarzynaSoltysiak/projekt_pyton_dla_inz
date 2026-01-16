# main.py
import numpy as np
import matplotlib.pyplot as plt
from model import RiverChannel
import visualization
import random
import physics


def main():
    length = 300
    num_points = 30
    x = np.linspace(0, length, num_points)
    y = np.sin(x / 150.0) * 5.0 
    
    initial_channel = RiverChannel(x, y, width=100.0, dt=0.5, dx=20.0)
    
    initial_channel.k_mig = 1.2 
    
    all_channels = [initial_channel]
    
    plt.ion()
    fig, ax = visualization.init_plot()
    
    delta_start_x = 400 
    map_limit_x = 3500    
    
    max_branches = 25     
    
    total_steps = 3000
    current_time = 0
    
    print("Start symulacji DELTY (Wersja: SZEROKA I GĘSTA)...")
    
    try:
        for step in range(total_steps):
            current_time += 0.5
            
            active_channels = [ch for ch in all_channels if ch.is_active]
            
            if not active_channels:
                print("Koniec - brak aktywnych kanałów.")
                break

            for channel in active_channels:
                # Zanik słabych odnóg
                if channel.width < 20 and len(channel.points) > 80:
                    if random.random() < 0.01:
                        channel.is_active = False
                        continue

                #WZROST
                head_x = channel.points[-1, 0]
                
                if head_x < map_limit_x:
                    channel.grow_downstream(growth_speed=4.0)
                else:
                    channel.is_active = False
                    continue 
                
                #MIGRACJA
                channel.migrate()
                
                #ROZGAŁĘZIANIE
                if (head_x > delta_start_x and len(all_channels) < max_branches):
                    
                    curv = np.abs(
                        np.mean(
                            physics.compute_curvature(channel.points[-30:])
                        )
                    )

                    length_factor = min(1.0, len(channel.points) / 80)

                    p_branch = (
                        0.008 + 
                        curv * 10 + 
                        0.02 * length_factor
                    )

                    p_branch = min(p_branch, 0.035)

                    if random.random() < p_branch:


                        if len(channel.points) > 20: 
                            new_branches = channel.branch()
                            all_channels.extend(new_branches)
                            print(f"[{current_time:.1f}] BIFURKACJA! (Kanałów: {len(all_channels)})")
            
            if step % 5 == 0:
                visualization.update_plot_network(ax, all_channels, current_time)
                plt.pause(0.001)
                
    except KeyboardInterrupt:
        print("Zatrzymano ręcznie.")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()