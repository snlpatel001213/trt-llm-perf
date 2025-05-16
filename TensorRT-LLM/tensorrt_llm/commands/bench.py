from pathlib import Path

import click

from tensorrt_llm.bench.benchmark.low_latency import latency_command
from tensorrt_llm.bench.benchmark.throughput import throughput_command
from tensorrt_llm.bench.build.build import build_command
from tensorrt_llm.bench.dataclasses import BenchmarkEnvironment


@click.group(name="trtllm-bench", context_settings={'show_default': True})
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    help="The Huggingface name of the model to benchmark.",
)
@click.option(
    "--workspace",
    "-w",
    required=False,
    type=click.Path(writable=True, readable=True),
    default="/tmp",  # nosec B108
    help="The directory to store benchmarking intermediate files.",
)
@click.pass_context
def main(
    ctx,
    model: str,
    workspace: Path,
) -> None:
    ctx.obj = BenchmarkEnvironment(model=model, workspace=workspace)

    # Create the workspace where we plan to store intermediate files.
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


main.add_command(build_command)
main.add_command(throughput_command)
main.add_command(latency_command)

if __name__ == "__main__":
    main()
