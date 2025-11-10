import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

"""
t (time) → Tiempo transcurrido en segundos desde el inicio del entrenamiento hasta el final del episodio.
r (reward) → Recompensa total obtenida en ese episodio.
l (length) → Cantidad de pasos (timesteps) en el episodio.
"""

# Get the file path from the command line arguments
monitor_csv_path = sys.argv[1]

df = pd.read_csv(monitor_csv_path, comment="#", encoding="latin1")  # Ignora la línea de metadata
print(df.head())  # Muestra las primeras filas

# Rolling average window size
window_size = 10
df["r_smooth"] = df["r"].rolling(window=window_size).mean()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Reward over Time
axes[0, 0].plot(df["t"], df["r"], marker="o", linestyle="-", color="b")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Total Reward")
axes[0, 0].set_title("Episode Reward over Time")
axes[0, 0].grid()

# Plot 2: Episode Length over Time
axes[0, 1].plot(df["t"], df["l"], marker="o", linestyle="-", color="g")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Episode Length (# steps)")
axes[0, 1].set_title("Episode Length over Time")
axes[0, 1].grid()

# Plot 3: Reward vs. Episode Length (Scatter)
axes[1, 0].scatter(df["l"], df["r"], color="r", alpha=0.7)
axes[1, 0].set_xlabel("Episode Length (# steps)")
axes[1, 0].set_ylabel("Total Reward")
axes[1, 0].set_title("Reward vs. Episode Length")
axes[1, 0].grid()

# Plot 4: Smoothed Reward over Time
axes[1, 1].plot(df["t"], df["r_smooth"], linestyle="-", color="purple")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Smoothed Reward")
axes[1, 1].set_title(f"Smoothed Reward over Time (Window={window_size})")
axes[1, 1].grid()

# Adjust layout and show
plt.tight_layout()



plt.show()