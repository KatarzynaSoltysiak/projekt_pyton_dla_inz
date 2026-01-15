import numpy as np
import matplotlib.pyplot as plt
from model import RiverChannel
import visualization

def main():
    length = 2500
    num_points = 250
    
    x = np.linspace(0, length, num_points)
    y = np.sin(x / 200.0) * 60.0 + np.random.normal(0, 3, num_points)
    
    river = RiverChannel(x, y, width=50.0, dt=0.5, dx=30.0)
    
    plt.ion()
    fig, ax = visualization.init_plot()
    
    total_steps = 1000
    
    print("Rozpoczynanie symulacji...")
    try:
        for i in range(total_steps):
            river.migrate()
            
            if i % 20 == 0:
                visualization.update_plot(ax, river)
                plt.pause(0.001)
                
    except KeyboardInterrupt:
        print("Zatrzymano symulację ręcznie.")
    
    print("Koniec symulacji. Zamykanie...")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()