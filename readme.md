

# LLM PERF Instructions

## start a container 
1. This is a long running job so start a terminal session with tmux or screen.
4. start container with precompiled perf libraries `sudo docker run -it --gpus all --privileged -v ${PWD}/:/workspace_perf -v /home/<username>/cache:/root/.cache -w /workspace_perf snlpatel/trtllm-prebuilt:v0.17_release bash`
2. Additioanlly install `pip install tqdm colorama openpyxl pandas pandas openpyxl`
3. git clone git clone  https://github.com/NVIDIA/TensorRT-LLM.git
4. log in to huggingface `huggingface-cli login --token <your token>`
  
  `${PWD}` is the location where all the perf will be stored.
  `/raid/supatel/cache` is the location where downloaded hf LLM models will be stored. PLEASE CHANGE THIS LOCATION AS PER YOUR SYSTEM. Both `mpi_run_loop_FP16.py` and `mpi_run_loop_FP8.py` must be located at this path. 

# run the test - FP16

1. Define test name, TP size, ISL, OSL, an concurrency
2. run `python mpi_run_loop_FP16.py` inside the container.
3. As it compeletes the perf testing each permutation will create its own log and result file.


# run the test - FP8

1. Define test name, TP size, ISL, OSL, an concurrency
2. run `python mpi_run_loop_FP8.py` inside the container.
3. As it compeletes the perf testing each permutation will create its own log and result file.




