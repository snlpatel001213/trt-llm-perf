import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .. import bindings as tllm
from ..bindings import executor as tllm
from ..builder import EngineConfig
from ..executor import (GenerationExecutor, GenerationResult, LoRARequest,
                        PromptAdapterRequest)
from ..logger import logger
from .llm_utils import (LLMARGS_DOCSTRING, CachedModelLoader, LlmArgs,
                        LlmBuildStats, ModelLoader, _ModelRuntimeContext)
from .mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                          external_mpi_comm_available)
from .tokenizer import TokenizerBase
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import (SamplingParams, append_docstring, exception_handler,
                    get_device_count)


class RequestOutput(GenerationResult):
    """The output data of a completion request to the LLM.

    Fields:
        request_id (int): The unique ID of the request.
        prompt (str, optional): The prompt string of the request.
        prompt_token_ids (List[int]): The token ids of the prompt.
        outputs (List[CompletionOutput]): The output sequences of the request.
        context_logits (torch.Tensor, optional): The logits on the prompt token ids.
        finished (bool): Whether the whole request is finished.
    """

    def __init__(self,
                 generation_result: GenerationResult,
                 prompt: Optional[str] = None,
                 tokenizer: Optional[TokenizerBase] = None) -> None:
        self.__dict__.update(generation_result.__dict__)
        self.prompt = prompt
        self.tokenizer = tokenizer

    def handle_response(self, response):
        super().handle_response(response)

        sampling_params = self._generation_request.sampling_params
        kwargs = {
            'skip_special_tokens':
            sampling_params.skip_special_tokens,
            'spaces_between_special_tokens':
            sampling_params.spaces_between_special_tokens
        }
        if sampling_params.detokenize and self.tokenizer is not None:
            for beam_output in self.outputs:
                beam_output._last_text_len = len(beam_output.text)
                if hasattr(self.tokenizer, 'decode_incrementally'):
                    if self.streaming and not sampling_params.use_beam_search:
                        beam_output.text, beam_output._incremental_states = self.tokenizer.decode_incrementally(
                            beam_output.token_ids_diff,
                            prev_text=beam_output.text,
                            states=beam_output._incremental_states,
                            flush=self.finished,
                            **kwargs)
                    else:
                        beam_output.text, _ = self.tokenizer.decode_incrementally(
                            beam_output.token_ids,
                            flush=self.finished,
                            **kwargs)
                else:
                    beam_output.text = self.tokenizer.decode(
                        beam_output.token_ids, **kwargs)

    def _repr_fields(self):
        return [
            'request_id', 'prompt', 'prompt_token_ids', 'outputs', 'finished'
        ]


PromptInputs = Union[str, List[int]]


@append_docstring(LLMARGS_DOCSTRING)
class LLM:
    """LLM class is the main class for running a LLM model.

    Args:
    """

    def __init__(self,
                 model: str,
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 tokenizer_mode: Literal['auto', 'slow'] = 'auto',
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any):

        self._executor_cls = kwargs.pop("executor_cls", GenerationExecutor)
        self.mpi_session: Optional[MpiSession] = None

        try:
            self.args = LlmArgs.from_kwargs(
                model=model,
                tokenizer=tokenizer,
                tokenizer_mode=tokenizer_mode,
                skip_tokenizer_init=skip_tokenizer_init,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                revision=revision,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        except Exception as e:
            logger.error(
                f"Failed to parse the arguments for the LLM constructor: {e}")
            raise e

        if self.args.parallel_config.is_multi_gpu:
            if get_device_count() < self.args.parallel_config.world_size:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.args.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.args.parallel_config.world_size} workers'
            )
            if not external_mpi_comm_available(
                    self.args.parallel_config.world_size):
                self.mpi_session = MpiPoolSession(
                    n_workers=self.args.parallel_config.world_size)
            else:
                self.mpi_session = MpiCommSession(
                    n_workers=self.args.parallel_config.world_size)

        try:
            # Due to the Executor can only accept a engine path, we need to save the engine to a directory
            self._engine_dir: Optional[Path] = None
            self._executor: Optional[GenerationExecutor] = None

            self._workspace = tempfile.TemporaryDirectory(
                suffix="-llm-workspace", dir=self.args.workspace)

            self.runtime_context: Optional[_ModelRuntimeContext] = None
            self.llm_build_stats = LlmBuildStats()

            self._build_model()
            self._tokenizer = self._try_load_tokenizer()
        except Exception as e:
            if self.mpi_session is not None:
                self.mpi_session.shutdown()
            raise e

        exception_handler.register(self, '_shutdown')

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name)

    def generate(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[LoRARequest,
                                     Sequence[LoRARequest]]] = None,
        prompt_adapter_request: Optional[Union[
            PromptAdapterRequest, Sequence[PromptAdapterRequest]]] = None,
    ) -> Union[RequestOutput, List[RequestOutput]]:
        ''' Generate output for the given prompts in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.

        Args:
            inputs (PromptInputs or Sequence[PromptInputs]): The prompt text or token ids.
                it can be single prompt or batched prompts.
            sampling_params (SamplingParams, List[SamplingParams], optional): The sampling params for the
                generation, a default one will be used if not provided. Defaults to None.
            use_tqdm (bool): Whether to use tqdm to display the progress bar. Defaults to True.
            lora_request (LoRARequest, Sequence[LoRARequest], optional): LoRA request to use for generation,
                if any. Defaults to None.
            prompt_adapter_request (PromptAdapterRequest, Sequence[PromptAdapterRequest], optional):
                Prompt Adapter request to use for generation, if any. Defaults to None.

        Returns:
            Union[RequestOutput, List[RequestOutput]]: The output data of the completion request to the LLM.
        '''
        if isinstance(inputs, str) or isinstance(inputs[0], str):
            unbatched = isinstance(inputs, str)
        else:
            unbatched = isinstance(inputs[0], int)

        if unbatched:
            inputs = [inputs]

        futures = []
        for i, request_inputs in enumerate(inputs):
            if isinstance(sampling_params, list):
                sp = sampling_params[i]
            else:
                sp = sampling_params
            if isinstance(lora_request, list):
                lora_req = lora_request[i]
            else:
                lora_req = lora_request
            if isinstance(prompt_adapter_request, list):
                pa_req = prompt_adapter_request[i]
            else:
                pa_req = prompt_adapter_request
            future = self.generate_async(request_inputs,
                                         sampling_params=sp,
                                         lora_request=lora_req,
                                         prompt_adapter_request=pa_req,
                                         streaming=False)
            futures.append(future)

        for future in tqdm(futures,
                           desc="Processed requests",
                           dynamic_ncols=True,
                           disable=not use_tqdm):
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    def generate_async(
        self,
        inputs: PromptInputs,
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
    ) -> RequestOutput:
        ''' Generate output for the given prompt in the asynchronous mode.
        Asynchronous generation accepts single prompt only.

        Args:
            inputs (PromptInputs): The prompt text or token ids; it must be single prompt.
            sampling_params (SamplingParams, optional): The sampling params for the generation,
                a default one will be used if not provided. Defaults to None.
            lora_request (LoRARequest, optional): LoRA request to use for generation, if any.
                Defaults to None.
            prompt_adapter_request (PromptAdapterRequest, optional): Prompt Adapter request to
                use for generation, if any. Defaults to None.
            streaming (bool): Whether to use the streaming mode for the generation. Defaults to
                False.

        Returns:
            RequestOutput: The output data of the completion request to the LLM.
        '''
        sampling_params = self._prepare_sampling_params(sampling_params)

        if isinstance(inputs, str):
            prompt_token_ids = self._prepare_prompt_token_ids(
                inputs, sampling_params)
            prompt = inputs
        elif isinstance(inputs, list) and isinstance(inputs[0], int):
            prompt_token_ids = inputs
            prompt = None
        else:
            raise TypeError(
                f"The inputs must be type str or list of int, but got {type(inputs)}"
            )

        self._check_arguments(prompt_token_ids, sampling_params)
        result = self._executor.generate_async(
            prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            streaming=streaming,
        )
        return RequestOutput(result, prompt, self.tokenizer)

    def _get_stats(self, timeout=None) -> str:
        ''' Get the stats from the runtime.

        Exceptions:
            NoStatsAvailable: If the stats are not available.

        Returns:
            str: The stats in JSON format.

        Known issue:
            The `_get_stats` cannot mix with `_get_stats_async` in the same LLM instance.
        '''
        return self._executor.get_stats(timeout=timeout)

    async def _get_stats_async(self, timeout=None) -> str:
        ''' Get the stats from the runtime.

        Exceptions:
            NoStatsAvailable: If the stats are not available.

        Returns:
            str: The stats in JSON format.

        Known issue:
            The `_get_stats_async` cannot mix with `_get_stats` in the same LLM instance.
        '''
        return await self._executor.aget_stats(timeout=timeout)

    def _prepare_prompt_token_ids(self, prompt: str,
                                  sampling_params: SamplingParams) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to tokenize string prompt")

        if sampling_params.truncate_prompt_tokens is None:
            return self.tokenizer.encode(
                prompt, add_special_tokens=sampling_params.add_special_tokens)
        else:
            return self.tokenizer.encode(
                prompt,
                add_special_tokens=sampling_params.add_special_tokens,
                truncation=True,
                max_length=sampling_params.truncate_prompt_tokens)

    def _prepare_sampling_params(
            self,
            sampling_params: Optional[SamplingParams] = None) -> SamplingParams:
        if sampling_params is None:
            if self.tokenizer is None:
                raise ValueError(
                    "tokenizer is required to initialize a default sampling_params, or you can explicitly specify a sampling_params"
                )
            return SamplingParams(end_id=self.tokenizer.eos_token_id,
                                  pad_id=self.tokenizer.pad_token_id)
        elif isinstance(sampling_params, SamplingParams):
            if sampling_params.end_id is None:
                if self.tokenizer is None:
                    raise ValueError(
                        "tokenizer is required to reset end_id if it is None, or you can explicitly specify the end_id for sampling_params"
                    )
            return sampling_params.setup(self.tokenizer)
        else:
            raise TypeError(
                f"The sampling_params must be type SamplingParams or None, but got {type(sampling_params)}"
            )

    def _check_arguments(self, prompt_token_ids: List[int],
                         sampling_params: SamplingParams) -> None:

        build_config = self.args.build_config

        built_enging_cfg_file = self.args.model / 'config.json'
        with open(built_enging_cfg_file) as f:
            built_enging_cfg = json.load(f)

        prompt_len = len(prompt_token_ids)

        if prompt_len + sampling_params.max_tokens > built_enging_cfg[
                'build_config']['max_seq_len']:
            raise ValueError(
                f"The sum of prompt length ({prompt_len}) and max_tokens ({sampling_params.max_tokens}) should not exceed "
                f"max_seq_len ({build_config.max_seq_len})")
        if sampling_params.beam_width > build_config.max_beam_width:
            raise ValueError(
                f"sampling_params's beam_width ({sampling_params.beam_width}) should not exceed max_beam_width ({build_config.max_beam_width})"
            )

    def _build_model(self):
        model_loader = CachedModelLoader(self.args,
                                         mpi_session=self.mpi_session,
                                         workspace=self.workspace,
                                         llm_build_stats=self.llm_build_stats)
        self._engine_dir = model_loader()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        self.args.model = self._engine_dir
        assert self.args.is_local_model

        executor_config = tllm.ExecutorConfig(
            max_beam_width=self.args.build_config.max_beam_width,
            scheduler_config=self.args.scheduler_config,
            batching_type=tllm.BatchingType.INFLIGHT)
        if self.args.kv_cache_config is not None:
            executor_config.kv_cache_config = self.args.kv_cache_config
        if self.args.peft_cache_config is not None:
            executor_config.peft_cache_config = self.args.peft_cache_config
        elif self.args.build_config.plugin_config.lora_plugin:
            engine_config = EngineConfig.from_json_file(self._engine_dir /
                                                        "config.json")
            lora_config = engine_config.build_config.lora_config
            max_lora_rank = lora_config.max_lora_rank
            num_lora_modules = engine_config.pretrained_config.num_hidden_layers * \
                len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)
            executor_config.peft_cache_config = tllm.PeftCacheConfig(
                num_device_module_layer=max_lora_rank * num_lora_modules *
                self.args.max_loras,
                num_host_module_layer=max_lora_rank * num_lora_modules *
                self.args.max_cpu_loras,
            )
        if self.args.decoding_config is not None:
            executor_config.decoding_config = self.args.decoding_config
        if self.args.logits_post_processor_map:
            executor_config.logits_post_processor_config = tllm.LogitsPostProcessorConfig(
                processor_map=self.args.logits_post_processor_map)
        executor_config.normalize_log_probs = self.args.normalize_log_probs
        executor_config.enable_chunked_context = self.args.enable_chunked_context
        executor_config.max_beam_width = self.args.build_config.max_beam_width

        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=executor_config,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size))

    def _try_load_tokenizer(self) -> Optional[TokenizerBase]:
        if self.args.skip_tokenizer_init:
            return None

        if self.args.tokenizer is not None:
            assert isinstance(self.args.tokenizer, TokenizerBase)
            return self.args.tokenizer

        if self.runtime_context is not None:
            return self.runtime_context.tokenizer

        return ModelLoader.load_hf_tokenizer(
            self.args.model_dir,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.tokenizer_mode != 'slow')

    @property
    def tokenizer(self) -> Optional[TokenizerBase]:
        return self._tokenizer

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path.

        Args:
            engine_dir (str): The path to save the engine.

        Returns:
            None
        '''
        logger.info(f"Save model to {engine_dir}")
        if self._engine_dir is None:
            raise RuntimeError("The engine is not built yet.")
        if self._engine_dir.absolute() != os.path.abspath(engine_dir):
            shutil.copytree(self._engine_dir, engine_dir, dirs_exist_ok=True)

    def _shutdown(self):
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()

        if self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self._shutdown()
        return False  # propagate exceptions

    def __getstate__(self):
        raise RuntimeError("LLM object can not be pickled.")

    def __del__(self):
        self._shutdown()
