# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import numpy as np
import tensorrt as trt

from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.models.llama.model import LLaMAForCausalLM, LLaMAModel

from ..._common import default_net, default_trtnet
from ..._utils import pad_vocab_size
from ...bindings import KVCacheType
from ...functional import (Tensor, _create_tensor, concat,
                           gather_last_token_logits, index_select, shape)
from ...layers import AttentionParams, ColumnLinear, SpecDecodingParams
from ...module import Module, ModuleList
from ...plugin import TRT_LLM_PLUGIN_NAMESPACE
from .config import EagleConfig


class TreeParams(object):

    def __init__(self, paths: Tensor = None):
        self.paths = paths


def eagle_sample_and_accept_draft_plugin(lm_logits: Tensor = None,
                                         draft_tokens: Tensor = None,
                                         draft_lens: Tensor = None,
                                         eagle_temperature: Tensor = None,
                                         rand_data_validation: Tensor = None,
                                         tree_params: TreeParams = None,
                                         greedy_sampling: bool = True):
    '''
    Takes input logits and samples golden token + predictions from draft tokens.
    Runs acceptance algorithm to accept draft tokens.
    When greedy_sampling is True, all decoding is done using Top1 and token equality is used
    for acceptance. Otherwise, speculative decoding multi-round sampling and multinomial
    samplings are used.

    Non-greedy sampling is not supported yet.

    Visit tests/model/eagle/test_sample_accept_draft_tokens.py for input/output examples.

    Parameters:
        lm_logits : Tensor
            [num_tokens, vocab_size]
            Logits produced by the base model.

        draft_tokens : Tensor
            [batch_size, max_decoding_draft_tokens]
            Input draft tokens. Only the first draft_lens[bi] tokens are relevant for bi'th row.

        draft_lens : Tensor
            [batch_size]
            Lengths of the draft_tokens. 0 for context request. Actual draft length for generation requests.

        eagle_temperature : Tensor
            [batch_size]
            Temperature of the decoding.

        rand_data_validation : Tensor
            [batch_size]
            Random data for multinomial sampling.

        tree_params : TreeParams
            Tree params of the input draft tokens.

        greedy_sampling : bool
            Whether to do greedy or non-greedy sampling.

    Return:
        accepted_tokens : Tensor
            [batch_size, max_path_len]
            Accepted token ids. Only the first num_accepted_tokens[bi] tokens are relevant for bi'th row,

        num_accepted_tokens : Tensor
            [batch_size]
            Number of accepted tokens per request. Each entry is >= 1.

        accepted_paths : Tensor
            [batch_size]
            Indices of the accepted path per request of the input paths in tree_params.paths.

        next_draft_tokens : Tensor
            [batch_size, max_decoding_draft_tokens]
            Empty tensor used to allocate space for the next draft tokens.

        next_draft_lens : Tensor
            [batch_size]
            Empty tensor used to allocate space for lens of the next draft tokens.

        hidden_size_batch_level_starts : Tensor
            [max_draft_path_len * batch_size + 1]
            Empty tensor used to allocate space for eagle_prepare_drafter_inputs_plugin.
    '''

    assert greedy_sampling, "Non-greedy sampling is not supported yet"

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EagleSampleAndAcceptDraftTokens', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id",
                              np.array([int(lm_logits.dtype)], np.int32),
                              trt.PluginFieldType.INT32)

    greedy_sampling = 1 if greedy_sampling else 0
    greedy_sampling = trt.PluginField("greedy_sampling",
                                      np.array(greedy_sampling, dtype=np.int32),
                                      trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, greedy_sampling])
    plugin = plg_creator.create_plugin("eagle_sample_and_accept_draft_plugin",
                                       pfc)

    plug_inputs = [
        lm_logits, draft_tokens, draft_lens, eagle_temperature,
        rand_data_validation, tree_params.paths
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    accepted_tokens = _create_tensor(layer.get_output(0), layer)
    num_accepted_tokens = _create_tensor(layer.get_output(1), layer)
    accepted_paths = _create_tensor(layer.get_output(2), layer)
    next_draft_tokens = _create_tensor(layer.get_output(3), layer)
    next_draft_lens = _create_tensor(layer.get_output(4), layer)
    hidden_size_batch_level_starts = _create_tensor(layer.get_output(5), layer)
    return tuple([
        accepted_tokens, num_accepted_tokens, accepted_paths, next_draft_tokens,
        next_draft_lens, hidden_size_batch_level_starts
    ])


def eagle_draft_decoder_plugin(layer_idx: int, top_k_sampling: bool,
                               logits: Tensor, rand_sample: Tensor,
                               tree_params: TreeParams,
                               input_draft_token_ids: Tensor,
                               input_draft_lens: Tensor):
    '''
    Parameters:
        layer_idx : int
            The index of the EagleNet.

        top_k_sampling: bool
            Whether to use top K sampling. Otherwise, use multinomial sampling.

        logits : Tensor
            [num_input_logits, vocab_size]
            Input logits.

        rand_sample : Tensor
            [num_input_logits]
            Used by multinomial sampling.

        tree_params : TreeParams
            Tree params of the input draft tokens.

        input_draft_token_ids: Tensor
            [batch_size, max_decoding_draft_tokens]
            Draft tokens generated by previous EagleNets.

        input_draft_lens: Tensor
            [batch_size]
            Number of draft tokens for each request.

    Return:
        output_draft_token_ids: Tensor
            [batch_size, max_decoding_draft_tokens]
            Draft tokens generated by this EagleNets, also include the previous draft tokens.

        output_draft_draft_lens: Tensor
            [batch_size]
            Number of draft tokens for each request.

    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EagleDecodeDraftTokens', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id", np.array([int(logits.dtype)],
                                                  np.int32),
                              trt.PluginFieldType.INT32)

    layer_idx_t = trt.PluginField("layer_idx",
                                  np.array(layer_idx, dtype=np.int32),
                                  trt.PluginFieldType.INT32)

    top_k_sampling_t = 1 if top_k_sampling else 0
    top_k_sampling_t = trt.PluginField(
        "top_k_sampling", np.array(top_k_sampling_t, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, layer_idx_t, top_k_sampling_t])
    plugin = plg_creator.create_plugin("eagle_draft_decoder_plugin", pfc)

    plug_inputs = [
        logits, rand_sample, tree_params.paths, input_draft_token_ids,
        input_draft_lens
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    output_draft_token_ids = _create_tensor(layer.get_output(0), layer)
    output_draft_lens = _create_tensor(layer.get_output(1), layer)
    return tuple([output_draft_token_ids, output_draft_lens])


def eagle_prepare_drafter_inputs_plugin(
        layer_idx: int, attention_params: AttentionParams, input_ids: Tensor,
        accepted_token_ids: Tensor, accepted_lens: Tensor,
        accepted_path_ids: Tensor, next_draft_tokens: Tensor,
        next_draft_lens: Tensor, next_draft_paths: Tensor,
        prev_draft_lens: Tensor, prev_draft_paths: Tensor,
        hidden_size_batch_level_starts: Tensor):
    '''
    Prepares inputs for the EagleNet inference.

    Visit tests/model/eagle/test_prepare_drafter_inputs.py for input/output examples.

    Parameters:
        layer_idx : int
            Index of the EagleNet. 0 means context phase EagleNet or EagleNet0,
            > 0 means EagleNetX or generation phase of EagleNet

        attention_params : AttentionParams

        input_ids : Tensor
            [num_tokens]
            Tokens ids, inputs to the base model.

        accepted_token_ids : Tensor
            [batch_size, max_path_len]
            Accepted tokens ids.

        accepted_lens : Tensor
            [batch_size]
            Number of accepted tokens.

        accepted_path_ids : Tensor
            [batch_size]
            Idx of the accepted path in prev_draft_paths.

        next_draft_tokens : Tensor
            [batch_size, max_decoding_draft_tokens]
            Tokens ids of the draft tokens being drafted by EagleNet

        next_draft_lens : Tensor
            [batch_size]
            Number of drafted tokens in next_draft_tokens

        next_draft_paths : Tensor
            [batch_size, max_decoding_tokens, max_path_len]
            Paths of the draft tokens for the next iteration. In EAGLE-1 is the same as prev_draft_paths

        prev_draft_lens : Tensor
            [batch_size]
            Number of draft tokens, inputs to the base model.
            0 for ctx requests and actual draft len for gen requests.

        prev_draft_paths : Tensor
            [batch_size, max_decoding_tokens, max_path_len]
            Paths of the draft tokens inputs to the base model.

        hidden_size_batch_level_starts : Tensor
            [max_draft_path_len * batch_size + 1]
            Exclusive sum of the starts of the segments of the hidden states in the concatenated array.
            Hidden states shape is (flattened and w/o padding)
            [max_draft_path_len, batch_size, num_output_tokens_i_j], where num_output_tokens_i_j
            depends on the path of request j at level i.

    Return:
        sequence_length : Tensor
            [batch_size]
            Sequence length of the next EagleNet iteration.
            For EagleNet0 equals to the (prompt_len + num_generated_tokens + accepted_lens).
            For EagleNetX (X > 0) (seq_len_eagle_net_0 + spec_decoding_generation_lengths).

        context_length : Tensor
            [batch_size]
            Context length of the next EagleNet iteration.
            For EagleNet0 it is either the actual context length of the request (for ctx requests)
            or the number of accepted tokens in this iteration. EagleNet0's attn does chunked context attn.
            For EagleNetX (X > 0), context length equals to the sequence length of the EagleNet0.

        spec_decoding_generation_lengths : Tensor
            [batch_size]
            Only relevant for EagleNetX (X > 0).
            Number of draft tokens.

        spec_decoding_position_offsets : Tensor
            [batch_size, max_decoding_tokens]
            Only relevant for EagleNetX (X > 0).
            Position offsets of the selected tokens from output_ids.

        spec_decoding_packed_mask : Tensor
            [batch_size, max_decoding_tokens, ceil(max_decoding_tokens / 32)]
            Only relevant for EagleNetX (X > 0).
            uint32_t packed masks.

        output_ids : Tensor
            [num_output_tokens]
            Token ids selected for the EagleNet iteration.
            Tensor's actual size is larger than num_output_tokens. Tensor has to be sliced.

        position_ids : Tensor
            [num_output_tokens]
            Position ids of the tokens selected for the EagleNet iteration.
            Tensor's actual size is larger than num_output_tokens. Tensor has to be sliced.

        hidden_states_indices : Tensor
            [num_output_tokens]
            Indices of the hidden states to be selected from aggregated hidden states for the next iteration.
            Tensor's actual size is larger than num_output_tokens. Tensor has to be sliced.

        last_token_indices : Tensor
            [num_last_token_indices]
            Indices of the hidden states to be converted to logits after the next EagleNet iteration.
            Tensor's actual size is larger than num_output_tokens. Tensor has to be sliced.

        num_output_tokens : Tensor
            [1]
            Number of selected tokens for the next iteration.

        num_last_token_indices : Tensor
            [1]
            Number of logits selected after the next EagleNet iteration.

        out_hidden_size_batch_level_starts : Tensor
            [max_draft_path_len * batch_size + 1]
            Same as hidden_size_batch_level_starts, but with updated path lens for the next level.
    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EaglePrepareDrafterInputs', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    layer_idx = trt.PluginField("layer_idx", np.array(layer_idx,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([layer_idx])
    plugin = plg_creator.create_plugin("eagle_prepare_drafter_inputs_plugin",
                                       pfc)

    plug_inputs = [
        attention_params.sequence_length, attention_params.context_lengths,
        input_ids, accepted_token_ids, accepted_lens, accepted_path_ids,
        next_draft_tokens, next_draft_lens, next_draft_paths, prev_draft_lens,
        prev_draft_paths, hidden_size_batch_level_starts
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    sequence_length = _create_tensor(layer.get_output(0), layer)
    context_length = _create_tensor(layer.get_output(1), layer)
    spec_decoding_generation_lengths = _create_tensor(layer.get_output(2),
                                                      layer)
    spec_decoding_position_offsets = _create_tensor(layer.get_output(3), layer)
    spec_decoding_packed_mask = _create_tensor(layer.get_output(4), layer)
    output_ids = _create_tensor(layer.get_output(5), layer)
    position_ids = _create_tensor(layer.get_output(6), layer)
    hidden_states_indices = _create_tensor(layer.get_output(7), layer)
    last_token_indices = _create_tensor(layer.get_output(8), layer)
    # TODO we can slice output_ids, position_ids and hidden_states_indices directly inside of the plugin:
    # Similarly to https://github.com/NVIDIA/TensorRT/tree/release/10.5/samples/sampleNonZeroPlugin.
    num_output_tokens = _create_tensor(layer.get_output(9), layer)
    num_last_token_indices = _create_tensor(layer.get_output(10), layer)
    out_hidden_size_batch_level_starts = _create_tensor(layer.get_output(11),
                                                        layer)
    return tuple([
        sequence_length, context_length, spec_decoding_generation_lengths,
        spec_decoding_position_offsets, spec_decoding_packed_mask, output_ids,
        position_ids, hidden_states_indices, last_token_indices,
        num_output_tokens, num_last_token_indices,
        out_hidden_size_batch_level_starts
    ])


class EagleNet(Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.drafter = LLaMAModel(config)
        self.config = config

        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(config.hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=config.dtype,
                                        tp_group=config.mapping.tp_group,
                                        tp_size=config.mapping.tp_size,
                                        gather_output=True)
        else:
            self.lm_head = None

    def forward(self, hidden_states, input_ids, position_ids,
                last_token_indices, spec_decoding_params, kv_cache_params,
                attention_params):
        hidden_states, cache = self.drafter(
            input_ids,
            position_ids=position_ids,
            use_cache=True,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            hidden_states_for_embed=hidden_states)

        if self.config.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_indices,
                default_net().plugin_config.remove_input_padding)
            return self.lm_head(hidden_states), hidden_states, cache

        return None, hidden_states, cache


class EagleForCausalLM(LLaMAForCausalLM):
    config_class = EagleConfig

    def __init__(self, config: EagleConfig):

        super().__init__(config)
        self.num_eagle_layers = config.num_eagle_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        vocab_size_padded = pad_vocab_size(self.vocab_size,
                                           config.mapping.tp_size)
        eagle_net_config = config.eagle_net_config
        eagle_net_config.fc_after_embed = True
        eagle_net_config.use_input_layernorm_in_first_layer = False
        self.eagle_nets = ModuleList([
            EagleNet(config=eagle_net_config)
            for _ in range(self.num_eagle_layers)
        ])
        self.max_draft_len = config.max_draft_len

    def _prepare_drafter_inputs(
            self, layer_idx, input_ids, accepted_token_ids, accepted_lens,
            accepted_path_ids, next_draft_tokens, next_draft_lens,
            next_draft_paths, prev_draft_lens, prev_draft_paths,
            input_attention_params, input_kv_cache_params, hidden_states,
            host_ctx_eagle_net_request_types,
            host_ctx_eagle_net_context_lengths,
            host_ctx_eagle_net_past_key_value_lengths,
            host_gen_eagle_net_request_types,
            host_gen_eagle_net_context_lengths,
            host_gen_eagle_net_past_key_value_lengths,
            hidden_size_batch_level_starts):

        drafter_inputs = eagle_prepare_drafter_inputs_plugin(
            layer_idx, input_attention_params, input_ids, accepted_token_ids,
            accepted_lens, accepted_path_ids, next_draft_tokens,
            next_draft_lens, next_draft_paths, prev_draft_lens,
            prev_draft_paths, hidden_size_batch_level_starts)

        sequence_length, context_lengths, \
            spec_decoding_generation_lengths, spec_decoding_position_offsets, \
            spec_decoding_packed_mask, output_ids, position_ids, hidden_states_indices, \
            last_token_indices, num_output_tokens, num_last_token_indices, out_hidden_size_batch_level_starts \
            = drafter_inputs

        attention_params = input_attention_params
        kv_cache_params = input_kv_cache_params
        attention_params.sequence_length = sequence_length
        attention_params.context_lengths = context_lengths

        if layer_idx == 0:
            attention_params.host_request_types = host_ctx_eagle_net_request_types
            attention_params.host_context_lengths = host_ctx_eagle_net_context_lengths
            kv_cache_params.host_past_key_value_lengths = host_ctx_eagle_net_past_key_value_lengths
        else:
            attention_params.host_request_types = host_gen_eagle_net_request_types
            attention_params.host_context_lengths = host_gen_eagle_net_context_lengths
            kv_cache_params.host_past_key_value_lengths = host_gen_eagle_net_past_key_value_lengths

        spec_decoding_params = None
        if layer_idx > 0:
            spec_decoding_params = SpecDecodingParams(
                True, self.max_draft_len, spec_decoding_generation_lengths,
                spec_decoding_position_offsets, spec_decoding_packed_mask)

        # TODO uncomment, when the issue with shape inference is resolved.
        # output_ids = slice(output_ids, starts=[0], sizes=num_output_tokens)
        # position_ids = slice(position_ids, starts=[0], sizes=num_output_tokens)
        # last_token_indices = slice(last_token_indices, starts=[0], sizes=num_last_token_indices)

        # Get hidden states for accepted ids
        hidden_states = self._slice_hidden_states(hidden_states,
                                                  hidden_states_indices,
                                                  num_output_tokens)

        eagle_net_inputs = {}
        eagle_net_inputs["input_ids"] = output_ids
        eagle_net_inputs["position_ids"] = position_ids
        eagle_net_inputs["last_token_indices"] = last_token_indices
        eagle_net_inputs["attention_params"] = attention_params
        eagle_net_inputs["kv_cache_params"] = kv_cache_params
        eagle_net_inputs["spec_decoding_params"] = spec_decoding_params
        eagle_net_inputs["hidden_states"] = hidden_states
        return eagle_net_inputs, out_hidden_size_batch_level_starts

    def _slice_hidden_states(self, hidden_states, indices, num_indices):
        # TODO uncomment, when the issue with shape inference is resolved.
        # indices = slice(indices, starts=[0], sizes=num_indices)
        hidden_states = index_select(hidden_states, 0, indices)

        hidden_states = hidden_states.view(
            concat([shape(indices, 0),
                    shape(hidden_states, 1)]))
        return hidden_states

    def _eagle_fwd_helper(self, lm_logits, hidden_states, *args, **kwargs):
        '''
        EAGLE inference can be viewed as
        TRT_Engine(Target -> Draft0 -> Draft1 -> .. -> DraftK-1) -> Runtime -> TRT_Engine(..) -> ..
        Target is Base model and Draft is EagleNet.

        Each EagleNet call can be viewed as call to Draft LLM in TensorRT-LLM.
        We have to
            1. prepare input tensors before the EagleNet call (like in the the runtime),
            2. call EagleNet,
            3. decode draft tokens after the EagleNet.
        The only difference with normal execution of the Draft model is that in EAGLE,
        all these 3 things happen inside of the TensorRT engine execution.
        We do 1 and 3 inside of the plugins.
        For 1. We call eagle_prepare_drafter_inputs_plugin and for 3. eagle_draft_decoder_plugin.

        The first call to the EagleNet (Draft0 == EagleNet0) is the context phase.
        For context request we populate the KV cache of the EagleNet.
        For generation request that have accepted tokens we emulate KV cache reuse by doing chunked attention,
        where chunk is the newly accepted tokens -- all previous tokens are already in the KV cache.

        The following calls to the EagleNet (EagleNetX (X > 0)) are generation phase.
        For each EagleNetX we select tokens based on the current path which are going to be used for the generation.

        Let's consider an example: prompt ABCD. EAGLE-1, i.e tree is fixed for the iteration.
        Tree:
                ┌───┐
                │ 0 │
                └─┬─┘
            ┌─────┴─────┐
          ┌─┴─┐ ┌─┴─┐ ┌─┴─┐
          │ 1 │ │ 2 │ │ 3 │
          └─┬─┘ └─┬─┘ └───┘
          ┌─┴─┐ ┌─┴─┐
          │ 4 │ │ 5 │
          └─┬─┘ └─┬─┘
          ┌─┴─┐ ┌─┴─┐
          │ 6 │ │ 7 │
          └───┘ └───┘

        First iteration of the TRT engine. Request is context request:
        1. Base model is called for [ABCD] tokens produces token E.
        2. Draft0 is called for tokens [BCDE] and produces
           three possibilities F, G and H for positions 1, 2 and 3 respectively.
        3. Since H (position 3) is a leaf, it is not chosen as the input to Draft1 inference.
        4. Draft1 is called for tokens [FG] with appropriate mask of:
             |F|G
            F|1|0
            G|0|1
            It produces tokens I and J for positions 4 and 5.
        6. Draft2 is called for inputs [FGIJ] with mask of
             |F|G|I|J
            F|1|0|0|0
            G|0|1|0|0
            I|1|0|1|0
            J|0|1|0|1
            Note that we could've stored FG in KV cache and provide only IJ tokens here
            with mask for past KV cache, but it is not supported in TensorRT-LLM attention at the moment.

            Draft2 produces tokens K and L at positions 6 and 7.
        7. Resulting outputs are:
            7.1 accepted_ids [E]
            7.2 next_draft_tokens [FGHIJKL]

        Second iteration of the TRT engine. Request is the generation request.
        1. Base model is called for [EFGHIJKL]. Let's assume that it accepts [FIKM], i.e. the left-most path in the tree.
        2. Draft0 is called as context phase for [FIKM] -- to append to kv cache of the existing [BCDE].
           It produces tokens N, O and P for positions 1, 2 and 3.
        3. Draft1 is called as generation phase for [NO] tokens.
        etc.

        '''
        input_tree_params = kwargs["tree_params"]

        draft_tokens = kwargs['draft_tokens']
        draft_lens = kwargs['draft_lens']
        eagle_temperature = kwargs['eagle_temperature']
        rand_data_validation = kwargs['rand_data_validation']
        rand_data_sample = kwargs['rand_data_sample']
        input_ids = kwargs['input_ids']
        host_ctx_eagle_net_request_types = kwargs[
            'host_ctx_eagle_net_request_types']
        host_ctx_eagle_net_context_lengths = kwargs[
            'host_ctx_eagle_net_context_lengths']
        host_ctx_eagle_net_past_key_value_lengths = kwargs[
            'host_ctx_eagle_net_past_key_value_lengths']
        host_gen_eagle_net_request_types = kwargs[
            'host_gen_eagle_net_request_types']
        host_gen_eagle_net_context_lengths = kwargs[
            'host_gen_eagle_net_context_lengths']
        host_gen_eagle_net_past_key_value_lengths = kwargs[
            'host_gen_eagle_net_past_key_value_lengths']

        # Sample target tokens and accept them
        # next_draft_tokens, next_draft_lens, hidden_size_batch_level_starts are outputted here just to
        # reserve the tensor with max size, which eagle_draft_decoder_plugin and
        # eagle_prepare_drafter_inputs_plugin are going to directly write to
        output = eagle_sample_and_accept_draft_plugin(lm_logits, draft_tokens,
                                                      draft_lens,
                                                      eagle_temperature,
                                                      rand_data_validation,
                                                      input_tree_params)
        accepted_tokens, num_accepted_tokens, accepted_paths, next_draft_tokens, \
            next_draft_lens, hidden_size_batch_level_starts = output

        attention_params = kwargs["attention_params"]
        kv_cache_params = kwargs["kv_cache_params"]

        input_hidden_states = hidden_states

        # NOTE EAGLE-1 output paths are the same as input path.
        next_draft_paths = input_tree_params.paths

        # Run EAGLE nets
        for li in range(self.num_eagle_layers):
            # Prepare EAGLE Net inputs.
            eagle_net_inputs, hidden_size_batch_level_starts = self._prepare_drafter_inputs(
                layer_idx=li,
                input_ids=input_ids,
                accepted_token_ids=accepted_tokens,
                accepted_lens=num_accepted_tokens,
                accepted_path_ids=accepted_paths,
                next_draft_tokens=next_draft_tokens,
                next_draft_lens=next_draft_lens,
                next_draft_paths=next_draft_paths,
                prev_draft_lens=draft_lens,
                prev_draft_paths=input_tree_params.paths,
                input_attention_params=attention_params,
                input_kv_cache_params=kv_cache_params,
                hidden_states=input_hidden_states,
                host_ctx_eagle_net_request_types=
                host_ctx_eagle_net_request_types,
                host_ctx_eagle_net_context_lengths=
                host_ctx_eagle_net_context_lengths,
                host_ctx_eagle_net_past_key_value_lengths=
                host_ctx_eagle_net_past_key_value_lengths,
                host_gen_eagle_net_request_types=
                host_gen_eagle_net_request_types,
                host_gen_eagle_net_context_lengths=
                host_gen_eagle_net_context_lengths,
                host_gen_eagle_net_past_key_value_lengths=
                host_gen_eagle_net_past_key_value_lengths,
                hidden_size_batch_level_starts=hidden_size_batch_level_starts)

            # Run EAGLE Net
            # TODO: handle base net kv cache and eagle net kv cache in the same tensors, but treat the differently here.
            logits, hidden_states, _ = self.eagle_nets[li](**eagle_net_inputs)

            # Decode draft tokens
            # FIXME We need to take top_k_sampling as an input
            top_k_sampling = True
            next_draft_tokens, next_draft_lens = eagle_draft_decoder_plugin(
                li, top_k_sampling, logits, rand_data_sample, input_tree_params,
                next_draft_tokens, next_draft_lens)

            # Update params
            if li == 0:
                eagle_net_0_attention_params = eagle_net_inputs[
                    "attention_params"]
                input_hidden_states = hidden_states
            else:
                attention_params = eagle_net_inputs["attention_params"]
                attention_params.context_lengths = eagle_net_0_attention_params.sequence_length
                attention_params.sequence_length = eagle_net_0_attention_params.sequence_length
                kv_cache_params = eagle_net_inputs["kv_cache_params"]
                input_hidden_states = concat(
                    [input_hidden_states, hidden_states])

        # Mark tensors as output
        accepted_tokens.mark_output('accepted_tokens')
        num_accepted_tokens.mark_output('num_accepted_tokens')
        accepted_paths.mark_output('accepted_paths')
        next_draft_tokens.mark_output('next_draft_tokens')
        next_draft_lens.mark_output('next_draft_lens')

        return next_draft_tokens

    def forward(self, *args, **kwargs):
        extra_args = [
            "draft_tokens", "draft_lens", "eagle_temperature",
            "rand_data_validation", "rand_data_sample", "tree_params",
            "host_ctx_eagle_net_request_types",
            "host_ctx_eagle_net_context_lengths",
            "host_ctx_eagle_net_past_key_value_lengths",
            "host_gen_eagle_net_request_types",
            "host_gen_eagle_net_context_lengths",
            "host_gen_eagle_net_past_key_value_lengths"
        ]

        base_kwargs = {k: v for k, v in kwargs.items() if k not in extra_args}

        # Base model forward
        hidden_states = super().forward(*args, **base_kwargs)

        assert kwargs['use_cache'] and default_net(
        ).plugin_config.paged_kv_cache

        lm_logits, hidden_states = hidden_states

        if self.mapping.is_last_pp_rank():
            # Call eagle logic to accept prev draft tokens and predict next draft tokens
            next_draft_tokens = self._eagle_fwd_helper(lm_logits, hidden_states,
                                                       *args, **kwargs)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        if self.mapping.is_last_pp_rank():
            return next_draft_tokens
        return hidden_states

    def prepare_inputs(self, *args, **kwargs):
        """
        Inputs needed:
            device_request_types: [bs]
            draft_tokens: [bs, max_draft_len]
            draft_lens: [bs]
            spec_decoding_generation_lengths: [bs]
            spec_decoding_position_offsets: [bs, max_gen_tokens]
            spec_decoding_packed_mask: [bs, max_draft_len, packed_length] **
            eagle_temperature: [bs]
            rand_data_sample: [bs]
            rand_data_validation: [bs, max_draft_tokens]

            ** The mask is tricky since the boolean mask will need to be
               packed in runtime. So, the last dim will be:
                    packed_length = ceil((max_draft_tokens+1)/32)
        """
        default_range = GenerationMixin.default_range
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        max_batch_size = kwargs['max_batch_size']
        assert max_batch_size is not None
        bb_range = default_range(max_batch_size)
        bb0_range = default_range(max_batch_size, min_range=0, opt_offset=1)

        kwargs['speculative_decoding_draft_tokens_external'] = False
        kwargs['max_draft_len'] = self.max_draft_len
        kwargs['spec_decoding_is_generation_length_variable'] = True

        # Call base class prepare inputs
        inputs = super().prepare_inputs(*args, **kwargs)

        assert inputs['spec_decoding_params'] is not None

        enable_two_optimization_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=KVCacheType.PAGED
            if paged_kv_cache else KVCacheType.CONTINUOUS)
        if enable_two_optimization_profiles:
            bb_range = [bb_range, bb_range]
            bb0_range = [bb0_range, bb0_range]
            draft_len_range = [self.max_draft_len]
            path_len_range = [self.num_eagle_layers + 1]
        else:
            bb_range = [bb_range]
            bb0_range = [bb0_range]
            draft_len_range = [self.max_draft_len]
            path_len_range = [self.num_eagle_layers + 1]

        draft_tokens = Tensor(name='draft_tokens',
                              dtype=trt.int32,
                              shape=[-1, self.max_draft_len],
                              dim_range=OrderedDict([
                                  ('batch_size', bb_range),
                                  ('draft_len', draft_len_range),
                              ]))
        draft_lens = Tensor(name='draft_lens',
                            dtype=trt.int32,
                            shape=[-1],
                            dim_range=OrderedDict([
                                ('batch_size', bb_range),
                            ]))
        eagle_temperature = Tensor(name='eagle_temperature',
                                   dtype=trt.float32,
                                   shape=[-1],
                                   dim_range=OrderedDict([
                                       ("batch_size", bb_range),
                                   ]))
        rand_data_validation = Tensor(name='rand_data_validation',
                                      dtype=trt.float32,
                                      shape=[-1, self.max_draft_len],
                                      dim_range=OrderedDict([
                                          ('batch_size', bb_range),
                                          ('draft_len', draft_len_range),
                                      ]))
        rand_data_sample = Tensor(name='rand_data_sample',
                                  dtype=trt.float32,
                                  shape=[-1],
                                  dim_range=OrderedDict([
                                      ('batch_size', bb_range),
                                  ]))
        tree_paths = Tensor(
            name='tree_paths',
            dtype=trt.int32,
            # FIXME max_accepted len is not necessary self.num_eagle_layers + 1. Only True for EAGLE-1
            shape=[-1, self.max_draft_len, self.num_eagle_layers + 1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('draft_len', draft_len_range),
                ('path_len', path_len_range),
            ]))

        host_ctx_eagle_net_request_types = Tensor(
            name='host_ctx_eagle_net_request_types',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_ctx_eagle_net_context_lengths = Tensor(
            name='host_ctx_eagle_net_context_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_ctx_eagle_net_past_key_value_lengths = Tensor(
            name='host_ctx_eagle_net_past_key_value_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_gen_eagle_net_request_types = Tensor(
            name='host_gen_eagle_net_request_types',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_gen_eagle_net_context_lengths = Tensor(
            name='host_gen_eagle_net_context_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_gen_eagle_net_past_key_value_lengths = Tensor(
            name='host_gen_eagle_net_past_key_value_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))

        tree_params = TreeParams(paths=tree_paths)

        inputs['draft_tokens'] = draft_tokens
        inputs['draft_lens'] = draft_lens
        inputs['eagle_temperature'] = eagle_temperature
        inputs['rand_data_validation'] = rand_data_validation
        inputs['rand_data_sample'] = rand_data_sample
        inputs['tree_params'] = tree_params
        inputs[
            'host_ctx_eagle_net_request_types'] = host_ctx_eagle_net_request_types
        inputs[
            'host_ctx_eagle_net_context_lengths'] = host_ctx_eagle_net_context_lengths
        inputs[
            'host_ctx_eagle_net_past_key_value_lengths'] = host_ctx_eagle_net_past_key_value_lengths
        inputs[
            'host_gen_eagle_net_request_types'] = host_gen_eagle_net_request_types
        inputs[
            'host_gen_eagle_net_context_lengths'] = host_gen_eagle_net_context_lengths
        inputs[
            'host_gen_eagle_net_past_key_value_lengths'] = host_gen_eagle_net_past_key_value_lengths
        return inputs
