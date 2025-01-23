# LLM Performance Benchmarking Procedure

This document summarizes performance measurements of Large language models using TensorRT-LLM.

**Hardware Requirement**

Below procedures are applicable to H200, H100, GH200, L40S and A100 GPUs for a few popular models. Performance benchmark can help estimate throughput and latency on different generation of GPUs, custom input and output length, concurrency, quantization and advance feature such as medusa/speculative decoding. Some of such performance measurements are given here – \[<sup>[\[1\]](#footnote-1)</sup>\]. As per the size of model, GPU required for the benchmarking also differs. Larger models (=>40B) may require to be split to multiple GPUs, in this case NVlink connectivity is must between GPUs. It is advised to perform benchmarking with docker on BareMetal machine**.** A server having sudo user access with ubuntu 22.04 and latest GPU driver is desired. The test setup also required to be connected to Internet for building the binaries or downloading docker image at once.

Entire procedure of the LLM benchmarking is open sourced at TensorRT-LLM benchmarking guidelines \[<sup>[\[2\]](#footnote-2)</sup>\]

**Accessing Performance Benchmark Utility**

TensorRT-LLM provides the trtllm-bench CLI, a packaged benchmarking utility.

trtllm-bench CLI can be built from Tensorrt-LLM Github Repository\[<sup>[\[3\]](#footnote-3)</sup>\] or downloaded precompiled.

1. To build the CLI, following steps to be performed \[<sup>[\[4\]](#footnote-4)</sup>\]

- git clone <https://github.com/NVIDIA/TensorRT-LLM.git>
- make -C docker release_run LOCAL_USER=1

1. download precompiled docker

- sudo docker run -it --gpus all --privileged -v /raid/supatel/cache:~/.cache snlpatel/trtllm-prebuilt:v0.14_release bash

The workflow for trtllm-bench is composed of the following steps:

1. Prepare a dataset to drive the inflight batching benchmark.
2. Build a benchmark engine using trtllm-bench build subcommand.
3. Run the max throughput benchmark using the trtllm-bench throughput subcommand or low latency benchmark using the trtllm-bench latency subcommand.

**Max Throughput benchmarking \[<sup>[\[5\]](#footnote-5)</sup>\]**

Max throughput bencharking estimates the max through irrespective of latency. Following 3 commands are required for the throughput benchmarking \[<sup>[\[6\]](#footnote-6)</sup>\]

1. To generate a synthetic dataset of 1000 requests with a uniform ISL/OSL of 128/128 for [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), simply run:

- `python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 3000 > /tmp/synthetic_128_128.txt`

1. To build an engine for benchmarking, you can specify the dataset generated with prepare_dataset.py through --dataset option. The trtllm-bench’s tuning heuristic uses the high-level statistics of the dataset (average ISL/OSL, max sequence length) to optimize engine build settings. The following command builds an FP8 quantized engine optimized using the dataset’s ISL/OSL.

- `trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8`

1. The trtllm-bench command line tool provides a max throughput benchmark that is accessible via the throughput subcommand. This benchmark tests a TensorRT-LLM engine under maximum load to provide an upper bound throughput number.

- `trtllm-bench --model meta-llama/Llama-2-7b-hf throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1`

**Output from the throughput benchmark looks like below :**

```
TP Size: 1
PP Size: 1
Max Runtime Batch Size: 4096
Max Runtime Tokens: 8192
Scheduling Policy: Guaranteed No Evict
KV Memory Percentage: 90.0%
Issue Rate (req/sec): 2.0827970096792666e+19
\===========================================================
\= STATISTICS
\===========================================================
Number of requests: 3000
Average Input Length (tokens): 128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec): 18886.813971319196
Request Throughput (req/sec): 147.55323415093122
Total Latency (seconds): 20.331645167
```

**Low Latency Benchmark\[<sup>[\[7\]](#footnote-7)</sup>\]**

The low latency benchmark follows a similar workflow to the [throughput benchmark](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#running-a-max-throughput-benchmark) but requires building the engine separately from trtllm-bench. Low latency benchmarks support advance features such as Medusa enabled speculative decoding.

1. Quantize the checkpoint:

- `cd tensorrt_llm/examples/llama`
- `python ../quantization/quantize.py --model_dir $checkpoint_dir --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 --output_dir /tmp/meta-llama/Meta-Llama-3-70B/checkpoint --calib_size 512 --tp_size $tp_size`

1. build the optimized engine

- `trtllm-build --checkpoint_dir /tmp/meta-llama/Meta-Llama-3-70B/checkpoint --use_fused_mlp enable --gpt_attention_plugin bfloat16 --output_dir /tmp/meta-llama/Meta-Llama-3-70B/engine --max_batch_size 1 --max_seq_len $(($isl+$osl)) --reduce_fusion enable --gemm_plugin fp8 --workers $tp_size --use_fp8_context_fmha enable --max_num_tokens $isl --use_paged_context_fmha disable --multiple_profiles enable`

1. Run the low-latency benchmark:
- `trtllm-bench --model meta-llama/Meta-Llama-3-70B latency --dataset $DATASET_PATH --engine_dir /tmp/meta-llama/Meta-Llama-3-70B/engine`

For advance feature such as medusa and speculative decoding Please refer to \[<sup>[\[8\]](#footnote-8)</sup>\]

**Sample Output from low latency benchmark**

```
1732901285,\[BENCHMARK\] num_samples 500
1732901285,\[BENCHMARK\] num_error_samples 0
1732901285,\[BENCHMARK\] num_samples 500
1732901285,\[BENCHMARK\] total_latency(ms) 1145372.75
1732901285,\[BENCHMARK\] seq_throughput(seq/sec) 0.44
1732901285,\[BENCHMARK\] token_throughput(token/sec) 218.27
1732901285,\[BENCHMARK\] avg_acceptance_rate(tokens/decoding steps) 1.00
1732901285,\[BENCHMARK\] avg_sequence_latency(ms) 267253.75
1732901285,\[BENCHMARK\] max_sequence_latency(ms) 344978.41
1732901285,\[BENCHMARK\] min_sequence_latency(ms) 74460.22
1732901285,\[BENCHMARK\] p99_sequence_latency(ms) 336154.41
1732901285,\[BENCHMARK\] p90_sequence_latency(ms) 293960.56
1732901285,\[BENCHMARK\] p50_sequence_latency(ms) 292345.81
1732901285,\[BENCHMARK\] avg_time_to_first_token(ms) 196802.52
1732901285,\[BENCHMARK\] max_time_to_first_token(ms) 273543.03
1732901285,\[BENCHMARK\] min_time_to_first_token(ms) 2153.45
1732901285,\[BENCHMARK\] p99_time_to_first_token(ms) 264721.28
1732901285,\[BENCHMARK\] p90_time_to_first_token(ms) 220965.77
1732901285,\[BENCHMARK\] p50_time_to_first_token(ms) 219359.58
1732901285,\[BENCHMARK\] avg_inter_token_latency(ms) 141.18
1732901285,\[BENCHMARK\] max_inter_token_latency(ms) 146.66
1732901285,\[BENCHMARK\] min_inter_token_latency(ms) 32.34
1732901285,\[BENCHMARK\] p99_inter_token_latency(ms) 146.64
1732901285,\[BENCHMARK\] p90_inter_token_latency(ms) 146.54
1732901285,\[BENCHMARK\] p50_inter_token_latency(ms) 143.23
1. <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md> [↑](#footnote-ref-1)
2. <https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#tensorrt-llm-benchmarking> [↑](#footnote-ref-2)
3. <https://github.com/NVIDIA/TensorRT-LLM/tree/main?tab=readme-ov-file> [↑](#footnote-ref-3)
4. <https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#compile-the-model-into-a-tensorrt-engine> [↑](#footnote-ref-4)
5. <https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#max-throughput-benchmark> [↑](#footnote-ref-5)
6. <https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#quickstart> [↑](#footnote-ref-6)
7. <https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#low-latency-benchmark> [↑](#footnote-ref-7)
8. <https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html#building-a-medusa-low-latency-engine> [↑](#footnote-ref-8)
```


# LLM PERF Instructions

## start a container 
1. This is a long running job so start a terminal session with tmux or screen.
4. start container with precompiled perf libraries `sudo docker run -it --gpus all --privileged -v ${PWD}/:/workspace_perf -v /raid/supatel/cache:/root/.cache -w /workspace_perf  snlpatel/trtllm-prebuilt:v0.14_release bash`
2. Additioanlly install `pip install tqdm colorama openpyxl pandas`
3. git clone git clone -b release/0.14 https://github.com/NVIDIA/TensorRT-LLM.git
4. log in to huggingface `huggingface-cli login --token <your token>`
  
  `${PWD}` is the location where all the perf will be stored.
  `/raid/supatel/cache` is the location where downloaded hf LLM models will be stored. PLEASE CHANGE THIS LOCATION AS PER YOUR SYSTEM. Both `mpi_run_loop_FP16.sh` and `mpi_run_loop_FP8.sh` must be located at this path. 

# run the test - FP16

1. run `python mpi_run_loop_FP16.py` inside the container.
2. As it compeletes the perf testing each permutation will create its own log and result file.


# run the test - FP8

1. run `python mpi_run_loop_FP8.bash` inside the container.
2. As it compeletes the perf testing each permutation will create its own log and result file.




