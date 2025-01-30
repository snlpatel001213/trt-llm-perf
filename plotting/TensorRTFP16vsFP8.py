import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(fp8_path, fp16_path):
    # Load Excel sheets for FP8 and FP16 data
    fp8_df = pd.read_excel(fp8_path)
    fp16_df = pd.read_excel(fp16_path)

    # Standardize column names for easier matching
    fp8_df = fp8_df.rename(columns={
        'token_throughput(token/sec)': 'Token Throughput',
        '#GPUs': 'GPUs',
        'avg_time_to_first_token(ms)': 'Prefill Latency',
        'ISL': 'Input Length',
        'OSL': 'Output Length',
        'Concurrency': 'Batch Size',
        'util_avg': 'Compute Utilization',
        'avg_inter_token_latency(ms)': 'Inter-Token Latency'
    })
    
    fp16_df = fp16_df.rename(columns={
        'token_throughput(token/sec)': 'Token Throughput',
        '#GPUs': 'GPUs',
        'avg_time_to_first_token(ms)': 'Prefill Latency',
        'ISL': 'Input Length',
        'OSL': 'Output Length',
        'Concurrency': 'Batch Size',
        'util_avg': 'Compute Utilization',
        'avg_inter_token_latency(ms)': 'Inter-Token Latency'
    })

    # Subtract 2 from 'Input Length' in FP8 data before merging
    fp8_df['Input Length'] = fp8_df['Input Length'] - 2
    fp16_df['Input Length'] = fp16_df['Input Length'] - 2

    return fp8_df, fp16_df

def merge_data(fp8_df, fp16_df):
    """Merge FP8 and FP16 datasets on matching columns."""
    merged_df = pd.merge(fp8_df, fp16_df, on=['Input Length', 'Output Length', 'GPUs', 'Batch Size'], suffixes=('_FP8', '_FP16'))
    return merged_df

def compare_and_plot(merged_df):
    """Plot bar comparisons between FP8 and FP16."""
    if merged_df.empty:
        print("No matching rows found. Check if datasets are aligned correctly.")
        return
    merged_df.to_markdown("merged_fp8_fp16_comparison.csv")
    metrics = ['Token Throughput', 'Prefill Latency', 'Inter-Token Latency']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        x_labels = [f"ISL {row['Input Length']} | OSL {row['Output Length']} | GPUs {row['GPUs']} | Batch {row['Batch Size']}" for _, row in merged_df.iterrows()]
        x = np.arange(len(x_labels))

        width = 0.35  # Bar width

        plt.bar(x - width/2, merged_df[f'{metric}_FP8'], width, label='FP8', color='blue', alpha=0.7)
        plt.bar(x + width/2, merged_df[f'{metric}_FP16'], width, label='FP16', color='orange', alpha=0.7)

        plt.xlabel("Matching Configurations")
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} between FP8 and FP16')
        plt.xticks(rotation=45, ha="right", ticks=x, labels=x_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        output_filename = f'comparison_{metric.replace(" ", "_").lower()}_fp8_vs_fp16.png'
        plt.savefig(output_filename, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Usage
    fp8_file = "/workspace_perf/result_fp8/meta-llama/Meta-Llama-3-70B.xlsx"  # Path to FP8 Excel file
    fp16_file = "/workspace_perf/result_fp16/meta-llama/Meta-Llama-3-70B-0.14-old.xlsx"  # Path to FP16 Excel file
    output_filename = "FP8_vs_FP16.png" 
    
    fp8_df, fp16_df = load_data(fp8_file, fp16_file)
    merged_df = merge_data(fp8_df, fp16_df)
    compare_and_plot(merged_df)
