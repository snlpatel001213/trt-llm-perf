import os
import subprocess
from tqdm import tqdm
from colorama import Fore, Style
from measure_GPU_util import GPUUtilizationMonitor
import time 
import traceback
import sys
import http.client, urllib

#define test name
TEST_NAME = "LONG_CONTEXT"
# Set up logging
log_file = f"run_progress_FP16-{TEST_NAME}.log"


def log_message(message):
    print(Fore.GREEN + str(message) + Style.RESET_ALL)
    with open(log_file, "a") as log:
        log.write(f"{message}\n")
        log.flush()
        
def read_log_progress():
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            return set(line.strip() for line in log if "Completed:" in line)
    return set()

def save_progress(tp_size, isl, osl, concurrency):
    log_message(f"Completed: tp_size={tp_size}, isl={isl}, osl={osl}, concurrency={concurrency}")
        
def gpus_to_ids(tp_size):
    return [i for i in range(0,tp_size)]

log_message(f"Run started at {os.popen('date').read().strip()}")

def concurrency2request(concurrency):
    concurrency2reqmap = {
        "2":1024,
        "4":1024,
        "8":1024,
        "16":1024,
        "32":1024,
        "64":1024,
        "128":4096,
        "256":4096
    }
    return str(concurrency2reqmap[str(concurrency)])

# Define parameters
model_name = "meta-llama/Meta-Llama-3-8B"
tp_sizes = [1,2]
isl_osl_combinations = [
    [20000, 1000]
]
concurrency_values = [2, 8, 16, 32, 64] #
# tp_sizes = [8]
# isl_osl_combinations = [[3100, 200],[12125, 500],[128,128]]
# concurrency_values = [8, 16, 32] #[2, 4, 8, 16, 32, 64, 128, 256] # 
# batch_size = 8
# num_requests = 300
padding_token = 2

dataset_dir = f"/workspace_perf/dataset_fp16-{TEST_NAME}/{model_name}"
result_dir = f"/workspace_perf/result_fp16-{TEST_NAME}/{model_name}"
converted_checkpoint_dir = f"/workspace_perf/converted-checkpoint-dir_fp16-{TEST_NAME}/{model_name}"

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Download model
print(Fore.GREEN + "Downloading model..." + Style.RESET_ALL)
log_message("Downloading model...")

# Read progress from log
completed_steps = read_log_progress()

try:
    # output = subprocess.check_output(["huggingface-cli", "download", model_name], text=True)
    # model_path = output.strip().split("\n")[-1]
    model_path = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    log_message(f"Model is stored at: {model_path}")
except subprocess.CalledProcessError as e:
    log_message("Model download failed.")
    print(Fore.RED + "Model download failed." + Style.RESET_ALL)
    raise e

# Count total combinations
total_combinations = len(tp_sizes) * len(isl_osl_combinations) * len(concurrency_values)
remaining_combinations = total_combinations
log_message(f"Total combinations to run: {total_combinations}")
print(Fore.YELLOW + f"Total combinations to run: {total_combinations}" + Style.RESET_ALL)

# Iterate over combinations
progress = tqdm(total=total_combinations, desc="Running combinations", colour="blue")

for tp_size in tp_sizes:
    print(Fore.CYAN + f"GPUs: {tp_size}" + Style.RESET_ALL)
    log_message(f"GPUs: {tp_size}")

    os.makedirs(converted_checkpoint_dir, exist_ok=True)
    log_message(f"Storing converted checkpoint at: {converted_checkpoint_dir}")
    checkpoint_param = [
        "python3", "/workspace_perf/TensorRT-LLM/examples/llama/convert_checkpoint.py",
        "--model_dir", model_path,
        "--output_dir", converted_checkpoint_dir,
        "--dtype", "bfloat16",
        "--tp_size", str(tp_size),
        "--pp_size","1"
    ]
    print("checkpoint_param :", checkpoint_param)
    subprocess.run(checkpoint_param)
    log_message("Checkpoint Conversion done successfully")

    for isl_osl in isl_osl_combinations:
        isl, osl = isl_osl
        isl += padding_token
        
        # Build TensorRT engine for ISL/OSL and max concurrency
        engine_dir = f"/workspace_perf/engine-dir_fp16-{TEST_NAME}/{model_name}+{isl}+{osl}+{tp_size}"
        engine_param = [
                    "trtllm-build",
                    "--checkpoint_dir",converted_checkpoint_dir,
                    "--use_fused_mlp","enable",
                    "--gpt_attention_plugin","bfloat16",
                    "--output_dir", engine_dir,
                    "--max_batch_size",str(concurrency_values[-1]),  # Changed from batch_size to max concurrency, this is the max parallel requests the engine is allowed to process, since we have a explicit concurrency target, just set it as concurrency, if it is smaller than concurrency, it will bottleneck the performance
                    "--max_num_tokens","8192",  # IFB need a larger window to make sure multiple requests can be processed simultaneously, if this is smaller than the max_input_len, it actually always handles 1 request at a time, if OOM occurs, change it to 4096
                    "--max_input_len",str(isl),
                    "--max_seq_len",str(isl + osl),
                    "--reduce_fusion","enable",
                    "--workers", str(tp_size),
                    "--use_paged_context_fmha","enable",
                    "--multiple_profiles","enable",
                    ]
        if  os.path.exists(engine_dir):
            is_empty_engine = os.listdir(engine_dir) == [] # retuens true if dir is empty
            if (is_empty_engine):
                print("building the engine as engine dir exists but emptry, engine param :", engine_param)
                subprocess.run(engine_param)
            else:
                print(f"engine exists, {engine_dir} skipping as dir is is_empty : {str(is_empty_engine)}")
        else:
            print("Building the engine engine does not exist , engine param :", engine_param)
            subprocess.run(engine_param)

        log_message(f"Engine  creation complete for tp_size={tp_size}, isl={isl}, osl={osl}")

        for concurrency in concurrency_values:
            try:
                if f"Completed: tp_size={tp_size}, isl={isl}, osl={osl}, concurrency={concurrency}" in completed_steps:
                    print(f'>>> Already completed: tp_size={tp_size}, isl={isl}, osl={osl}, concurrency={concurrency}" in completed_steps')
                    continue  # Skip already completed steps
                dataset = f"{dataset_dir}/{isl}+{osl}+{concurrency2request(concurrency)}.json"
                result_csv = f"{result_dir}/{isl}+{osl}+{concurrency2request(concurrency)}+{concurrency}+{tp_size}.csv"
                log_message(f"Processing tp_size={tp_size}, isl={isl}, osl={osl}, concurrency={concurrency}")
                os.makedirs(engine_dir, exist_ok=True)
                # Prepare dataset
                dataset_param = [
                    "python", "/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py",
                    "--output", str(dataset),
                    "--tokenizer", str(model_name),
                    "token-norm-dist",
                    "--num-requests", str(concurrency2request(concurrency)),
                    "--input-mean", str(isl),
                    "--output-mean", str(osl),
                    "--input-stdev", "0",
                    "--output-stdev", "0"
                ]
                print("dataset param :", dataset_param)
                subprocess.run(dataset_param)
                log_message("Dataset preparation complete")


                # Run benchmark
                monitor = GPUUtilizationMonitor(gpu_ids=gpus_to_ids(tp_size=tp_size))
                monitor.start()
                benchmark_param = [
                        "mpirun","--allow-run-as-root",
                        "-np", str(tp_size),"--oversubscribe",
                        "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark",
                        "--engine_dir",engine_dir,
                        "--type","IFB",
                        "--api","executor",
                        "--dataset",dataset,
                        "--eos_id","-1",
                        "--scheduler_policy","guaranteed_no_evict",
                        "--kv_cache_free_gpu_mem_fraction","0.95",
                        "--output_csv",result_csv,
                        "--warm_up","5",
                        "--log_level","info",
                        # "--max_num_tokens",   # We don't need to set this at runtime, it will use engine's max batch size and max num tokens
                        # str(isl),
                        # "--max_batch_size",
                        # str(batch_size),
                        "--concurrency",str(concurrency),
                        "--enable_chunked_context",
                        "--streaming",
                    ]
                print("benchmark param :", benchmark_param)
                subprocess.run(benchmark_param)
                monitor.stop()
                monitor.add_gpu_utilization_to_csv(result_csv)

                log_message(f"Processed for tp_size={tp_size}, isl={isl}, osl={osl}, concurrency={concurrency}")
                log_message(f"Written result to: {result_csv}")
                save_progress(tp_size, isl, osl, concurrency)
                progress.update(1)

                # Clean up specific dataset after each run
                subprocess.run(["rm", "-rf", dataset])
                subprocess.run(["python3", "compile_results.py", "--input_folder", result_dir])
               
                print(f"Completed for tp_size={tp_size}, isl={isl}, osl={osl}, concurrency={concurrency}, remianing combinaions : {remaining_combinations}")
                remaining_combinations = remaining_combinations-1
        
            except Exception as  e:
                print(f"Error: {str(traceback.format_exc())}")
                sys.exit(1)

        # remove and refresh engine after each ISL/OSL and Concurrency change
        # subprocess.run(["rm", "-rf", engine_dir])
    # Remove converted checkpoint and engine dir after each tp
    # subprocess.run(["rm", "-rf", converted_checkpoint_dir])

progress.close()
log_message(f"Run processed at {os.popen('date').read().strip()}")
print(Fore.GREEN + "All combinations processed successfully!" + Style.RESET_ALL)
