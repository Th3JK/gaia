import csv
import numpy as np
import os

def csv_export(results, file_path):
    """
    Save experimental results to a CSV file, including mean and standard deviation.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "Experiment", "DE", "PSO", "SOMA", "FA", "TLBO"])

        for func_name, experiments in results.items():
            for row in experiments:
                writer.writerow([func_name] + row)

            # Compute mean and std per algorithm (columns 2–6)
            data = np.array([row[1:] for row in experiments], dtype=float)
            mean_row = ["", "Mean"] + np.mean(data, axis=0).tolist()
            std_row = ["", "Std. Dev."] + np.std(data, axis=0).tolist()

            writer.writerow(mean_row)
            writer.writerow(std_row)
            writer.writerow([])  # spacing