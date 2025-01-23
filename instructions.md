# LLM PERF Instructions

## start a container 
1. This is a long running job so start a terminal session with tmux or screen.
2. Additioanlly install `pip install tqdm coloram openpyxl`
3. log in to huggingface `huggingface-cli login --token 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`
4. start container with precompiled perf libraries `sudo docker run -it --gpus all --privileged -v ${PWD}/:/workspace_perf -v /raid/supatel/cache:/root/.cache -w /workspace_perf  snlpatel/trtllm-prebuilt:v0.14_release bash`
  
  `${PWD}` is the location where all the perf will be stored.
  `/raid/supatel/cache` is the location where downloaded hf LLM models will be stored. PLEASE CHANGE THIS LOCATION AS PER YOUR SYSTEM. Both `mpi_run_loop_FP16.sh` and `mpi_run_loop_FP8.sh` must be located at this path. 

# run the test - FP16

1. run `bash mpi_run_loop_FP16.sh` inside the container.
2. As it compeletes the perf testing each permutation will create its own log and result file.


# run the test - FP8

1. run `bash mpi_run_loop_FP8.sh` inside the container.
2. As it compeletes the perf testing each permutation will create its own log and result file.


# collect all the results in a sheet

1. To compile all the results in form of a sheet you need to run `python compile_results.py --folder_path` where `--folder_path` is a path to `result_dir` used in the script. 
2. For fp8 and fp16, python file need to be executed seperately.
3. This step can be run at any point even if many experiments are still running.




