import numpy as np
import matplotlib.pyplot as plt 

def main():
    time_points = np.arange(31)
    
    signal = np.zeros(31) - 1.0
    signal[10:20] = 1.5
    
    kernel = np.array([-1, 1])
    
    edge_detected = np.zeros(31)
    
    for i in range(len(signal) - 1):
        segment = signal[i:i+2]
        
        result = np.dot(kernel, segment)
        
        edge_detected[i+1] = result
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot([14, 15], [-1, 1], '-sk', markersize=4)
    axes[0].set_xlim(0, 30)
    axes[0].set_ylim(-1.2, 1.7)
    axes[0].set_title("A)\nKernel")
    
    axes[1].plot(time_points, signal, '-sk', markersize=4)
    axes[1].set_xlim(0, 30)
    axes[1].set_ylim(-1.2, 1.7)
    axes[1].set_title("B)\nTime Series signal")
    
    axes[2].plot(time_points, signal, '-sk', label='Signal', markersize=4)
    axes[2].plot(time_points, edge_detected, ':o', color='darkgrey', label='Edge detection', markersize=4)
    axes[2].set_xlim(0, 30)
    axes[2].set_ylim(-3, 3) 
    axes[2].set_title("C)")
    axes[2].legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
