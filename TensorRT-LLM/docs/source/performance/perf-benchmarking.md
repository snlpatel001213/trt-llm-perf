(perf-benchmarking)=

# TensorRT-LLM Benchmarking

```{important}
This benchmarking suite is a work in progress.
Expect breaking API changes.
```

TensorRT-LLM provides the `trtllm-bench` CLI, a packaged benchmarking utility.

#### Supported Networks for Benchmarking

- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [tiiuae/falcon-180B](https://huggingface.co/tiiuae/falcon-180B)
- [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
- [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)
- [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
- [mistralai/Mixtral-8x7B-v0.1-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1-Instruct)


#### Support Quantization Modes

TensorRT-LLM supports a number of quantization modes:

- None (no quantization applied)
- W8A16
- W4A16
- W4A16_AWQ
- W4A8_AWQ
- W4A16_GPTQ
- FP8
- INT8

For more information about quantization, refer to [](../reference/precision.md) and
the [support matrix](../reference/precision.md#support-matrix) of the supported quantization methods for each network.


## Inflight Benchmarking with a Dataset

This section covers how to benchmark TensorRT-LLM using inflight batching.


### Quickstart

This quick start focuses on running a short max throughput benchmark on
`meta-llama/Llama-2-7b-hf` on a synthetic dataset with a uniform distribution of prompts with ISL:OSL
of 128:128.
To run the benchmark from start to finish, run the following commands:

```shell
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 3000 > /tmp/synthetic_128_128.txt
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
trtllm-bench --model meta-llama/Llama-2-7b-hf throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
```

And that's it!
After the benchmark completes, `trtllm-bench` prints a summary with summary metrics.

```shell
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0
Dtype:                  float16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       2048
Max Sequence Length:    4098

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   99.0%
Issue Rate (req/sec):   3.680275266452667e+18
===========================================================
= STATISTICS
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec):  23405.927228471104
Request Throughput (req/sec):   182.8588064724305
Total Latency (seconds):        16.406100739
===========================================================
```

### Workflow

The workflow for `trtllm-bench` is composed of the following steps:

1. Prepare a dataset to drive the inflight batching benchmark.
2. Build a benchmark engine using `trtllm-bench build` subcommand.
3. Run the max throughput benchmark using the `trtllm-bench throughput` subcommand.

#### Preparing a Dataset

The inflight benchmark utilizes a fixed JSON schema so that it is simple and
straightforward to specify requests. The schema is defined as follows:

| Key             | Required |     Type      | Description                                     |
| :-------------- | :------: | :-----------: | :---------------------------------------------- |
| `task_id`       |    Y     |    String     | Unique identifier for the request.              |
| `prompt`        |    N*    |    String     | Input text for a generation request.            |
| `logits`        |    N*    | List[Integer] | List of logits that make up the request prompt. |
| `output_tokens` |    Y     |    Integer    | Number of generated tokens for this request.    |

Prompt and logits are mutually exclusive, but one of `prompt` or `logits` is required.
If you specify `logits`, the `prompt` entry is ignored for request generation.

Refer to the following examples of valid entries for the inflight benchmark:

- Entries with a human-readable prompt and no logits.

  ```json
  {"task_id": 1, "prompt": "Generate an infinite response to the following: This is the song that never ends, it goes on and on my friend.", "output_tokens": 1000}
  {"task_id": 2, "prompt": "Generate an infinite response to the following: Na, na, na, na", "output_tokens": 1000}
  ```

- Entries which contain logits.

  ```json
  {"task_id":0,"logits":[863,22056,25603,11943,8932,13195,3132,25032,21747,22213],"output_tokens":128}
  {"task_id":1,"logits":[14480,13598,15585,6591,1252,8259,30990,26778,7063,30065,21764,11023,1418],"output_tokens":128}
  ```

```{tip}
Specify each entry on one line.
To simplify passing the data, a complete JSON entry is on each line so that the benchmarker
can simply read a line and assume a complete entry. When creating a dataset, be sure that a complete
JSON entry is on every line.
```

#### Using `prepare_dataset` to Create Synthetic Datasets

In order to prepare a synthetic dataset, you can use the provided script in the `benchmarks/cpp`
directory. For example, to generate a synthetic dataset of 1000 requests with a uniform ISL/OSL of
128/128 for [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), simply run:

```shell
benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000 > /tmp/synthetic_128_128.txt
```

You can pipe the above command to a file to reuse the same dataset, or simply pipe its output to the
benchmark script (example below).

### Building a Benchmark Engine

The second thing you'll need once you have a dataset is an engine to benchmark against. In order to
build a pre-configured engine for one of the supported ISL:OSL combinations, you can run the following
using the dataset you generated with `prepare_dataset.py` to build an FP8 quantized engine:

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
```

or manually set a max sequence length that you plan to run with specifically:

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf build --max_seq_len 256 --quantization FP8
```

> [!NOTE] `trtllm-bench build` reproduces benchmark engines for performance study. These engine
configurations are not guaranteed to be optimal for all cases and should be viewed as reproducers
for the benchmark data we provide on our [Performance Overview](./perf-overview.md).

Looking a little closer, the `build` sub-command
will perform a lookup and build an engine using those reference settings. The
look up table directly corresponds to the performance table found in our
[Performance Overview](./perf-overview.md#throughput-measurements). The
output of the `build` sub-command looks similar to the snippet below (for `meta-llama/Llama-2-7b-hf`):

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
[TensorRT-LLM] TensorRT-LLM version: 0.12.0
[08/12/2024-19:13:06] [TRT-LLM] [I] Found dataset.
[08/12/2024-19:13:07] [TRT-LLM] [I]
===========================================================
= DATASET DETAILS
===========================================================
Max Input Sequence Length:      128
Max Output Sequence Length:     128
Max Sequence Length:    256
Number of Sequences:    3000
===========================================================


[08/12/2024-19:13:07] [TRT-LLM] [I] Set multiple_profiles to True.
[08/12/2024-19:13:07] [TRT-LLM] [I] Set use_paged_context_fmha to True.
[08/12/2024-19:13:07] [TRT-LLM] [I] Set use_fp8_context_fmha to True.
[08/12/2024-19:13:07] [TRT-LLM] [I]
===========================================================
= ENGINE BUILD INFO
===========================================================
Model Name:             meta-llama/Llama-2-7b-hf
Workspace Directory:    /tmp
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1

===========================================================
= ENGINE CONFIGURATION DETAILS
===========================================================
Max Sequence Length:            256
Max Batch Size:                 4096
Max Num Tokens:                 8192
Quantization:                   FP8
===========================================================

Loading Model: [1/3]    Downloading HF model
Downloaded model to /data/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
Time: 0.115s
Loading Model: [2/3]    Loading HF model to memory
current rank: 0, tp rank: 0, pp rank: 0
Time: 60.786s
Loading Model: [3/3]    Building TRT-LLM engine
Time: 163.331s
Loading model done.
Total latency: 224.232s
[TensorRT-LLM][INFO] Engine version 0.12.0 found in the config file, assuming engine(s) built by new builder API.

<snip verbose logging>

[08/12/2024-19:17:09] [TRT-LLM] [I]

===========================================================
ENGINE SAVED: /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
===========================================================
```

The engine in this case will be written to `/tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1` (the end of the log).

### Running a Max Throughput Benchmark

The `trtllm-bench` command line tool provides a max throughput benchmark that is accessible via the
`throughput` subcommand. This benchmark tests a TensorRT-LLM engine under maximum load to provide an
upper bound throughput number.

#### How the Benchmarker Works

The benchmarker reads a data file where a single line contains
a complete JSON request entry as specified in [](#preparing-a-dataset).
The process that the benchmarker is as follows:

1. Iterate over all input requests. If `logits` is specified, construct the request using the specified
list of logits. Otherwise, tokenize the `prompt` with as specified by `--model $HF_MODEL_NAME`.
1. Submit the dataset to the TensorRT-LLM `Executor` API as fast as possible (offline mode).
1. Wait for all requests to return, compute statistics, and then report results.

To run the benchmarker, run the following commands with the [engine](#building-a-benchmark-engine) and
[dataset](#preparing-a-dataset) generated from previous steps:

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
[TensorRT-LLM] TensorRT-LLM version: 0.12.0
[08/12/2024-19:36:48] [TRT-LLM] [I] Preparing to run throughput benchmark...
[08/12/2024-19:36:49] [TRT-LLM] [I] Setting up benchmarker and infrastructure.
[08/12/2024-19:36:49] [TRT-LLM] [I] Ready to start benchmark.
[08/12/2024-19:36:49] [TRT-LLM] [I] Initializing Executor.
[TensorRT-LLM][INFO] Engine version 0.12.0 found in the config file, assuming engine(s) built by new builder API.

<snip verbose logging>

[TensorRT-LLM][INFO] Executor instance created by worker
[08/12/2024-19:36:58] [TRT-LLM] [I] Starting response daemon...
[08/12/2024-19:36:58] [TRT-LLM] [I] Executor started.
[08/12/2024-19:36:58] [TRT-LLM] [I] Request serving started.
[08/12/2024-19:36:58] [TRT-LLM] [I] Starting statistics collection.
[08/12/2024-19:36:58] [TRT-LLM] [I] Benchmark started.
[08/12/2024-19:36:58] [TRT-LLM] [I] Collecting live stats...
[08/12/2024-19:36:59] [TRT-LLM] [I] Request serving stopped.
[08/12/2024-19:37:19] [TRT-LLM] [I] Collecting last stats...
[08/12/2024-19:37:19] [TRT-LLM] [I] Ending statistics collection.
[08/12/2024-19:37:19] [TRT-LLM] [I] Stop received.
[08/12/2024-19:37:19] [TRT-LLM] [I] Stopping response parsing.
[08/12/2024-19:37:19] [TRT-LLM] [I] Collecting last responses before shutdown.
[08/12/2024-19:37:19] [TRT-LLM] [I] Completed request parsing.
[08/12/2024-19:37:19] [TRT-LLM] [I] Parsing stopped.
[08/12/2024-19:37:19] [TRT-LLM] [I] Request generator successfully joined.
[08/12/2024-19:37:19] [TRT-LLM] [I] Statistics process successfully joined.
[08/12/2024-19:37:19] [TRT-LLM] [I]
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0
Dtype:                  float16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       256
Max Sequence Length:    256

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   90.0%
Issue Rate (req/sec):   2.0827970096792666e+19
===========================================================
= STATISTICS
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec):  18886.813971319196
Request Throughput (req/sec):   147.55323415093122
Total Latency (seconds):        20.331645167
===========================================================

[TensorRT-LLM][INFO] Orchestrator sendReq thread exiting
[TensorRT-LLM][INFO] Orchestrator recv thread exiting
[TensorRT-LLM][INFO] Leader sendThread exiting
[TensorRT-LLM][INFO] Leader recvReq thread exiting
[TensorRT-LLM][INFO] Refreshed the MPI local session
```

## Low Latency Benchmark

The low latency benchmark follows a similar workflow to the [throughput benchmark](#running-a-max-throughput-benchmark)
but requires building the engine separately from `trtllm-bench`. Low latency benchmarks has the following modes:

- A single-request low-latency engine
- A Medusa-enabled speculative-decoding engine

### Low Latency TensorRT-LLM Engine for Llama-3 70B

To build a low-latency engine for the latency benchmark, run the following quantize and build commands.
The `$checkpoint_dir` is the path to the [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) Hugging Face checkpoint in your cache or downloaded to a specific location with the [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
To prepare a dataset, follow the same process as specified in [](#preparing-a-dataset).

#### Benchmarking a non-Medusa Low Latency Engine

To quantize the checkpoint:

```shell
cd tensorrt_llm/examples/llama
python ../quantization/quantize.py \
    --model_dir $checkpoint_dir \
    --dtype bfloat16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir /tmp/meta-llama/Meta-Llama-3-70B/checkpoint \
    --calib_size 512 \
    --tp_size $tp_size
```

then build,

```shell
trtllm-build \
    --checkpoint_dir /tmp/meta-llama/Meta-Llama-3-70B/checkpoint \
    --use_fused_mlp enable \
    --gpt_attention_plugin bfloat16 \
    --output_dir /tmp/meta-llama/Meta-Llama-3-70B/engine \
    --max_batch_size 1 \
    --max_seq_len $(($isl+$osl)) \
    --reduce_fusion enable \
    --gemm_plugin fp8 \
    --workers $tp_size \
    --use_fp8_context_fmha enable \
    --max_num_tokens $isl \
    --use_paged_context_fmha disable \
    --multiple_profiles enable
```

After the engine is built, run the low-latency benchmark:

```shell
env TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG=1 \
  TRTLLM_MMHA_KERNEL_BLOCK_SIZE=256 \
  TRTLLM_MMHA_BLOCKS_PER_SEQUENCE=32 \
  FORCE_MULTI_BLOCK_MODE=ON \
  TRTLLM_ENABLE_FDL=1 \
  trtllm-bench --model meta-llama/Meta-Llama-3-70B \
  latency \
  --dataset $DATASET_PATH \
  --engine_dir /tmp/meta-llama/Meta-Llama-3-70B/engine
```

#### Building a Medusa Low-Latency Engine

To build a Medusa-enabled engine requires checkpoints that contain Medusa heads.
NVIDIA provides TensorRT-LLM checkpoints on the [NVIDIA](https://huggingface.co/nvidia) page on Hugging Face.
The checkpoints are pre-quantized and can be directly built after downloading them with the
[huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
After you download the checkpoints, run the following command. Make sure to
specify the `$tp_size` supported by your Medusa checkpoint and the path to its stored location `$checkpoint_dir`.
Additionally, `$max_seq_len` should be set to the model's maximum position embedding.

Using Llama-3.1 70B as an example, for a tensor parallel 8 and bfloat16 dtype:

```shell
tp_size=8
max_seq_len=131072
trtllm-build --checkpoint_dir $checkpoint_dir \
    --speculative_decoding_mode medusa \
    --max_batch_size 1 \
    --gpt_attention_plugin bfloat16 \
    --max_seq_len $max_seq_len \
    --output_dir /tmp/meta-llama/Meta-Llama-3.1-70B/medusa/engine \
    --use_fused_mlp enable \
    --paged_kv_cache enable \
    --use_paged_context_fmha disable \
    --multiple_profiles enable \
    --reduce_fusion enable \
    --use_fp8_context_fmha enable \
    --workers $tp_size \
    --low_latency_gemm_plugin fp8
```

After the engine is built, you need to define the Medusa choices.
The choices are specified with a YAML file like the following example (`medusa.yaml`):

```yaml
- [0]
- [0, 0]
- [1]
- [0, 1]
- [2]
- [0, 0, 0]
- [1, 0]
- [0, 2]
- [3]
- [0, 3]
- [4]
- [0, 4]
- [2, 0]
- [0, 5]
- [0, 0, 1]
```

To run the Medusa-enabled engine, run the following command:

```shell
env TRTLLM_ENABLE_PDL=1 \
  UB_ONESHOT=1 \
  UB_TP_SIZE=$tp_size \
  TRTLLM_ENABLE_PDL=1 \
  TRTLLM_PDL_OVERLAP_RATIO=0.15 \
  TRTLLM_PREFETCH_RATIO=-1 \
  trtllm-bench --model meta-llama/Meta-Llama-3-70B \
  latency \
  --dataset $DATASET_PATH \
  --engine_dir /tmp/meta-llama/Meta-Llama-3-70B/medusa/engine \
  --medusa_choices medusa.yml
```

## Summary

The following table summarizes the commands needed for running benchmarks:

| Scenario | Phase | Command |
| - | - | - |
| Dataset | Preparation | `python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer $HF_MODEL token-norm-dist --input-mean $ISL --output-mean $OSL --input-stdev 0 --output-stdev 0 --num-requests $NUM_REQUESTS > $DATASET_PATH` |
| Throughput | Build | `trtllm-bench --model $HF_MODEL build --dataset $DATASET_PATH` |
| Throughput | Benchmark | `trtllm-bench --model $HF_MODEL throughput --dataset $DATASET_PATH --engine_dir $ENGINE_DIR` |
| Latency | Build | See [section about building low latency engines](#low-latency-tensorrt-llm-engine-for-llama-3-70b) |
| Non-Medusa Latency | Benchmark | `trtllm-bench --model $HF_MODEL latency --dataset $DATASET_PATH --engine_dir $ENGINE_DIR` |
| Medusa Latency | Benchmark | `trtllm-bench --model $HF_MODEL latency --dataset $DATASET_PATH --engine_dir $ENGINE_DIR --medusa_choices $MEDUSA_CHOICES` |

where,

`$HF_MODEL`
: The Hugging Face name of a model.

`$NUM_REQUESTS`
: The number of requests to generate.

`$DATASET_PATH`
: The path where the dataset was written when preparing the dataset.

`$ENGINE_DIR`
: The engine directory as printed by `trtllm-bench build`.

`$MEDUSA_CHOICES`
: A YAML config representing the Medusa tree for the benchmark.
