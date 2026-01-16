# main.py
import numpy as np
import matplotlib.pyplot as plt
from model import RiverChannel
import visualization
import random

def main():
    # 1. Start - Płaska linia na start, żeby nie robić pętli od razu
    length = 300
    num_points = 30
    x = np.linspace(0, length, num_points)
    y = np.sin(x / 150.0) * 5.0 
    
    # 2. Inicjalizacja
    # ZMIANA: width=100.0 (Było 50.0) -> Rzeka jest teraz 2x szersza
    initial_channel = RiverChannel(x, y, width=100.0, dt=0.5, dx=20.0)
    
    # Lekko zwiększamy meandrowanie, bo szersza rzeka potrzebuje więcej energii
    initial_channel.k_mig = 1.2 
    
    all_channels = [initial_channel]
    
    plt.ion()
    fig, ax = visualization.init_plot()
    
    # Parametry sterujące
    delta_start_x = 400   # Rozgałęzienia zaczynają się wcześniej
    map_limit_x = 3500    
    
    # ZMIANA: max_branches=40 (Było 15) -> Więcej odnóg
    max_branches = 40     
    
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
                # A. WZROST
                head_x = channel.points[-1, 0]
                
                if head_x < map_limit_x:
                    channel.grow_downstream(growth_speed=4.0)
                else:
                    channel.is_active = False
                    continue 
                
                # B. MIGRACJA
                channel.migrate()
                
                # C. ROZGAŁĘZIANIE (AGRESYWNIEJSZE)
                if (head_x > delta_start_x and len(all_channels) < max_branches):
                    
                    # ZMIANA: Szansa 4% (Było 1.5%) -> Częstsze bifurkacje
                    # ZMIANA: Wymagana długość > 20 (Było 40) -> Szybsze podziały
                    if random.random() < 0.04:
                        if len(channel.points) > 20: 
                            new_branches = channel.branch()
                            all_channels.extend(new_branches)
                            print(f"[{current_time:.1f}] BIFURKACJA! (Kanałów: {len(all_channels)})")
            
            # Wizualizacja co 5 kroków (płynniej widać zmiany)
            if step % 5 == 0:
                visualization.update_plot_network(ax, all_channels, current_time)
                plt.pause(0.001)
                
    except KeyboardInterrupt:
        print("Zatrzymano ręcznie.")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()