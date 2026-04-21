import matplotlib.pyplot as plt
import os

def save_plot(data, filename, ylabel, title, rolling=True):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(data, alpha=0.4, color='steelblue', label='raw')
    if rolling and len(data) >= 10:
        roll = [sum(data[max(0,i-9):i+1])/min(i+1,10) for i in range(len(data))]
        plt.plot(roll, color='steelblue', linewidth=2, label='10-ep avg')
        plt.legend()
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=150)
    plt.close()
    print(f"Saved plots/{filename}.png")