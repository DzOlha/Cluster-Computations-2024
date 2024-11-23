import os
import matplotlib.pyplot as plt
import json
import numpy as np


def plot_scaling(results):
    # Ensure num_processes is sorted as integers
    num_processes = sorted(results.keys(), key=lambda x: int(x))

    times = [results[p]["time"] for p in num_processes]
    speedup = [results[p]["speedup"] for p in num_processes]

    num_processes_int = list(map(int, num_processes))

    plt.figure(figsize=(12, 6))

    # Plot execution time
    plt.subplot(1, 2, 1)
    plt.plot(num_processes_int, times, marker='o', label="Execution Time")
    plt.xlabel("Number of Processes")
    plt.ylabel("Time (s)")
    plt.title("Execution Time vs Processes")
    plt.grid()
    plt.legend()

    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.plot(num_processes_int, speedup, marker='x', label="Speedup (from file)")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Processes")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # Save the plot to a file
    plot_file_path = os.path.join(os.path.dirname(__file__), "scaling_results.png")
    plt.savefig(plot_file_path)
    print(f"Scaling plot saved to {plot_file_path}")
    plt.show()


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")

if __name__ == "__main__":
    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
        with open(RESULTS_FILE, "r") as f:
            try:
                _results = json.load(f)
                plot_scaling(_results)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    else:
        print(f"No valid results file found at {RESULTS_FILE}.")
