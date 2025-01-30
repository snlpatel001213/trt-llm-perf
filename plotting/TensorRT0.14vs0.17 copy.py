import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(trt014_path, trt017_path):
    # Load Excel sheets for both TRT versions
    trt014_df = pd.read_excel(trt014_path)
    trt017_df = pd.read_excel(trt017_path)

    # Standardize column names for easier matching
    trt014_df = trt014_df.rename(columns={
        'token_throughput(token/sec)': 'Token Throughput',
        '#GPUs': 'GPUs',
        'avg_time_to_first_token(ms)': 'Prefill Latency',
        'ISL': 'Input Length',
        'OSL': 'Output Length',
        'Concurrency': 'Batch Size',
        'util_avg': 'Compute Utilization',
        'avg_inter_token_latency(ms)': 'Inter-Token Latency'
    })
    
    trt017_df = trt017_df.rename(columns={
        'token_throughput(token/sec)': 'Token Throughput',
        '#GPUs': 'GPUs',
        'avg_time_to_first_token(ms)': 'Prefill Latency',
        'ISL': 'Input Length',
        'OSL': 'Output Length',
        'Concurrency': 'Batch Size',
        'util_avg': 'Compute Utilization',
        'avg_inter_token_latency(ms)': 'Inter-Token Latency'
    })

    # Subtract 2 from 'Input Length' in TRT0.14 data before merging
    trt014_df['Input Length'] = trt014_df['Input Length'] - 2
    trt017_df['Input Length'] = trt017_df['Input Length'] - 2

    return trt014_df, trt017_df

def merge_data(trt014_df, trt017_df):
    """Merge two datasets (TRT0.14 and TRT0.17) on matching columns."""
    merged_df = pd.merge(trt014_df, trt017_df, on=['Input Length', 'Output Length', 'GPUs', 'Batch Size'], suffixes=('_TRT0.14', '_TRT0.17'))
    merged_df.to_markdown("merged.csv")
    return merged_df

def compare_and_plot(merged_df):
    """Plot bar comparisons between TRT0.14 and TRT0.17."""
    if merged_df.empty:
        print("No matching rows found. Check if datasets are aligned correctly.")
        return
    merged_df.to_markdown("merged_trt_comparison.csv")
    metrics = ['Token Throughput', 'Prefill Latency', 'Inter-Token Latency']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        x_labels = [f"ISL {row['Input Length']} | OSL {row['Output Length']} | GPUs {row['GPUs']} | Batch {row['Batch Size']}" for _, row in merged_df.iterrows()]
        x = np.arange(len(x_labels))

        width = 0.35  # Bar width

        plt.bar(x - width/2, merged_df[f'{metric}_TRT0.14'], width, label='TRT0.14', color='blue', alpha=0.7)
        plt.bar(x + width/2, merged_df[f'{metric}_TRT0.17'], width, label='TRT0.17', color='orange', alpha=0.7)

        plt.xlabel("Matching Configurations")
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} between TRT0.14 and TRT0.17')
        plt.xticks(rotation=45, ha="right", ticks=x, labels=x_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        output_filename = f'comparison_{metric.replace(" ", "_").lower()}_trt.png'
        plt.savefig(output_filename, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Usage
    trt014_file = "/workspace_perf/result_fp16/meta-llama/Meta-Llama-3-70B-0.14-old.xlsx"  # Path to TRT0.14 Excel file
    trt017_file = "/workspace_perf/result_fp16/meta-llama/Meta-Llama-3-70B-0.17.xlsx"  # Path to TRT0.17 Excel file
    output_filename = "TRT0.14_vs_TRT0.17.png" 
    
    trt014_df, trt017_df = load_data(trt014_file, trt017_file)
    merged_df = merge_data(trt014_df, trt017_df)
    compare_and_plot(merged_df)
