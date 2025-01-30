import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(nvidia_path, dell_path):
    # Load Excel sheets
    nvidia_df = pd.read_excel(nvidia_path)
    dell_df = pd.read_excel(dell_path)

    # Standardize column names for easier matching
    nvidia_df = nvidia_df.rename(columns={
        'token_throughput(token/sec)': 'Token Throughput',
        '#GPUs': 'GPUs',
        'avg_time_to_first_token(ms)': 'Prefill Latency',
        'ISL': 'Input Length',
        'OSL': 'Output Length',
        'Concurrency': 'Batch Size',
        'util_avg': 'Compute Utilization',
        'avg_inter_token_latency(ms)': 'Inter-Token Latency'
    })
    
    dell_df = dell_df.rename(columns={
        'Total token throughput (tok/sec)': 'Token Throughput',
        'No. H200 GPU on single server': 'GPUs',
        'Prefill Latency (ms)': 'Prefill Latency',
        'Input Length': 'Input Length',
        'Output Length': 'Output Length',
        'Batch size': 'Batch Size',
        'Calculated Compute Utilization': 'Compute Utilization',
        'ITL (ms)': 'Inter-Token Latency'
    })

    # Subtract 2 from 'Input Length' in NVIDIA data before merging
    nvidia_df['Input Length'] = nvidia_df['Input Length'] - 2

    return nvidia_df, dell_df

def merge_data(nvidia_df, dell_df):
    """Merge NVIDIA and Dell datasets on matching columns."""
    merged_df = pd.merge(nvidia_df, dell_df, on=['Input Length', 'Output Length', 'GPUs', 'Batch Size'], suffixes=('_NVIDIA', '_DELL'))
    return merged_df

def compare_and_plot(merged_df):
    """Plot bar comparisons between NVIDIA and Dell."""
    if merged_df.empty:
        print("No matching rows found. Check if datasets are aligned correctly.")
        return
    metrics = ['Token Throughput', 'Prefill Latency', 'Inter-Token Latency']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        x_labels = [f"ISL {row['Input Length']} | OSL {row['Output Length']} | GPUs {row['GPUs']} | Batch {row['Batch Size']}" for _, row in merged_df.iterrows()]
        x = np.arange(len(x_labels))

        width = 0.35  # Bar width

        plt.bar(x - width/2, merged_df[f'{metric}_NVIDIA'], width, label='NVIDIA', color='blue', alpha=0.7)
        plt.bar(x + width/2, merged_df[f'{metric}_DELL'], width, label='DELL', color='orange', alpha=0.7)

        plt.xlabel("Matching Configurations")
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} between NVIDIA and Dell')
        plt.xticks(rotation=45, ha="right", ticks=x, labels=x_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        output_filename = f'comparison_{metric.replace(" ", "_").lower()}.png'
        plt.savefig(output_filename, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Usage
    nvidia_file = "/workspace_perf/result_fp16/meta-llama/Meta-Llama-3-70B.xlsx"  # Path to Nvidia Excel file
    dell_file = "/workspace_perf/plotting/VLLM_Dell- H200.xlsx"  # Path to Dell Excel file
    output_filename = "VLMvsTRT.png" 
    
    nvidia_df, dell_df = load_data(nvidia_file, dell_file)
    merged_df = merge_data(nvidia_df, dell_df)
    compare_and_plot(merged_df)
