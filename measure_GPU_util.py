import threading
import time
import subprocess
import re
from collections import defaultdict
import pandas as pd


class GPUUtilizationMonitor:
    def __init__(self, gpu_ids=None):
        self.gpu_ids = set(gpu_ids) if gpu_ids is not None else None
        self._stop_event = threading.Event()
        self._data = defaultdict(list)
        self._thread = None

    def _parse_nvidia_smi_output(self, output):
        gpu_stats = {}
        for idx, line in enumerate(output.splitlines()):
            # print(f"Parsing line {idx}: {line}")  # Debug log
            match = re.search(r'(\d+),\s*(\d+),\s*(\d+)', line)
            if match:
                utilization = int(match.group(1))
                memory_used = int(match.group(2))
                memory_total = int(match.group(3))
                memory_util = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                gpu_stats[idx] = (utilization, memory_util)
        return gpu_stats

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                     "--format=csv,nounits,noheader"],
                    capture_output=True, text=True, check=True
                )
                # print(f"Raw nvidia-smi output:\n{result.stdout}")  # Debug log
                stats = self._parse_nvidia_smi_output(result.stdout)
                for gpu_id, (util, mem_util) in stats.items():
                    if self.gpu_ids is None or gpu_id in self.gpu_ids:
                        self._data[gpu_id].append((util, mem_util))
            except subprocess.CalledProcessError as e:
                print(f"Error querying nvidia-smi: {e}")
            time.sleep(1)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            print("Monitoring already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def get_stats(self):
        stats_summary = {}
        for gpu_id, data in self._data.items():
            if data:
                utilization_values = [x[0] for x in data]
                memory_values = [x[1] for x in data]
                stats_summary[gpu_id] = {
                    "utilization": {
                        "min": min(utilization_values),
                        "max": max(utilization_values),
                        "avg": sum(utilization_values) / len(utilization_values),
                    },
                    "memory": {
                        "min": min(memory_values),
                        "max": max(memory_values),
                        "avg": sum(memory_values) / len(memory_values),
                    }
                }
        return stats_summary
    
    def get_combined_stats(self):
        """
        Calculate combined averages across all GPUs.

        Returns:
            dict: A single dictionary with overall min, max, and avg stats for utilization and memory.
        """
        combined_stats = {
            "utilization": {"min": float("inf"), "max": float("-inf"), "avg": 0},
            "memory": {"min": float("inf"), "max": float("-inf"), "avg": 0},
        }
        total_samples = 0
        total_memory_samples = 0

        for gpu_id, stats in self._data.items():
            if stats:
                utilization_values = [x[0] for x in stats]
                memory_values = [x[1] for x in stats]

                # Update utilization stats
                combined_stats["utilization"]["min"] = min(combined_stats["utilization"]["min"], min(utilization_values))
                combined_stats["utilization"]["max"] = max(combined_stats["utilization"]["max"], max(utilization_values))
                combined_stats["utilization"]["avg"] += sum(utilization_values)

                # Update memory stats
                combined_stats["memory"]["min"] = min(combined_stats["memory"]["min"], min(memory_values))
                combined_stats["memory"]["max"] = max(combined_stats["memory"]["max"], max(memory_values))
                combined_stats["memory"]["avg"] += sum(memory_values)

                total_samples += len(utilization_values)
                total_memory_samples += len(memory_values)

        # Finalize average calculations
        if total_samples > 0:
            combined_stats["utilization"]["avg"] /= total_samples
        if total_memory_samples > 0:
            combined_stats["memory"]["avg"] /= total_memory_samples

        return combined_stats
    

    def add_gpu_utilization_to_csv(self,file_name):
        """
        Add GPU utilization stats to an existing CSV file using pandas.

        Args:
            file_name (str): Path to the CSV file to update.
            gpu_stats (dict): Dictionary containing utilization and memory statistics.
        """
        utils_matrics =  open(file_name.replace(".csv", "_util.csv"),"w")
        # Flatten the GPU stats dictionary into columns
        gpu_stats = self.get_combined_stats()
        print("######### gpu_stats", gpu_stats)
        flat_gpu_stats = {
            "util_min": gpu_stats["utilization"]["min"],
            "util_max": gpu_stats["utilization"]["max"],
            "util_avg": gpu_stats["utilization"]["avg"],
            "mem_min": gpu_stats["memory"]["min"],
            "mem_max": gpu_stats["memory"]["max"],
            "mem_avg": gpu_stats["memory"]["avg"],
        }
        print("######### flat_gpu_stats", flat_gpu_stats)
        utils_matrics.write(str(flat_gpu_stats ))
        utils_matrics.flush()
        utils_matrics.close()

        # Load the existing CSV file into a DataFrame
        df = pd.read_csv(file_name)
        df =  df.iloc[:, :-1]

        # Add the GPU stats as new columns
        for col_name, value in flat_gpu_stats.items():
            df[col_name] = value

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_name, index=False)
        print(f"Updated file written to: {file_name}")

if __name__ == "__main__":
    gpu_ids_to_monitor = [0, 1]
    monitor = GPUUtilizationMonitor(gpu_ids=gpu_ids_to_monitor)
    monitor.start()
    print(f"Monitoring GPU(s): {gpu_ids_to_monitor}...")

    try:
        time.sleep(1)
    finally:
        monitor.stop()
        print("Monitoring stopped.")
        stats = monitor.get_stats()
        print("GPU Stats Summary:")
        
        for gpu_id, summary in stats.items():
            print(f"GPU {gpu_id}: {summary}")

        combined_stats = monitor.get_combined_stats()
        print("\nCombined GPU Stats Summary:")
        print(combined_stats)
        monitor.add_gpu_utilization_to_csv("/workspace_perf/result_fp16/meta-llama/Meta-Llama-3-70B/130+2048+32+2+2.csv")
