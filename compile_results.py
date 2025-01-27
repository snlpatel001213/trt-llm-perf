import os
import pandas as pd
import argparse

def extract_details_from_filename(filename):
    """Extract details from the filename."""
    try:
        parts = filename.replace('.csv', '').split('+')
        isl, osl, num_requests, concurrency, num_gpus = parts
        return int(isl), int(osl), int(num_requests), int(concurrency), int(num_gpus)
    except ValueError:
        print(f"Error parsing filename: {filename}")
        return None

def main(input_folder):
    # Prepare the final dataframe
    final_data = []
    columns = [
        "ISL", "OSL", "#Requests", "Concurrency", "#GPUs",  # Extracted from filename
        "num_samples", "num_error_samples", "total_latency(ms)", "seq_throughput(seq/sec)", "token_throughput(token/sec)",
        "avg_sequence_latency(ms)", "max_sequence_latency(ms)", "min_sequence_latency(ms)", "p99_sequence_latency(ms)",
        "p90_sequence_latency(ms)", "p50_sequence_latency(ms)", "avg_time_to_first_token(ms)", "max_time_to_first_token(ms)",
        "min_time_to_first_token(ms)", "p99_time_to_first_token(ms)", "p90_time_to_first_token(ms)", "p50_time_to_first_token(ms)",
        "avg_inter_token_latency(ms)", "max_inter_token_latency(ms)", "min_inter_token_latency(ms)", "p99_inter_token_latency(ms)",
        "p90_inter_token_latency(ms)", "p50_inter_token_latency(ms)", "util_min","util_max","util_avg","mem_min","mem_max","mem_avg"
    ]

    # Walk through the folder and process each CSV file
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(file_path)
                details = extract_details_from_filename(file)
                if details:
                    isl, osl, num_requests, concurrency, num_gpus = details

                    # Read the CSV content
                    try:
                        df = pd.read_csv(file_path)
                        # Verify if the data structure matches the expected format
                        if not df.empty :
                            for _, row in df.iterrows():
                                final_data.append([
                                    isl, osl, num_requests, concurrency, num_gpus, *row.values
                                ])                                
                    except Exception as e:
                        print(f"Error reading {file}: {e}")

    # Create the final DataFrame and save to Excel
    final_df = pd.DataFrame(final_data, columns=columns)
    output_excel = str(input_folder)+".xlsx"
    final_df.to_excel(output_excel, index=False)

    print(f"Consolidated data written to {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files and consolidate into an Excel file.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing CSV files.")

    args = parser.parse_args()
    main(args.input_folder)