/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/pybind/common/bindTypes.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::pybind::batch_manager
{

void initBindings(pybind11::module_& m)
{
    using GenLlmReq = tb::GenericLlmRequest<runtime::ITensor::SharedPtr>;

    PybindUtils::bindSet<tb::ReqIdsSet>(m, "ReqIdsSet");

    py::enum_<tb::LlmRequestType>(m, "LlmRequestType")
        .value("LLMREQUEST_TYPE_CONTEXT_AND_GENERATION", tb::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("LLMREQUEST_TYPE_CONTEXT_ONLY", tb::LLMREQUEST_TYPE_CONTEXT_ONLY)
        .value("LLMREQUEST_TYPE_GENERATION_ONLY", tb::LLMREQUEST_TYPE_GENERATION_ONLY)
        .export_values();

    py::class_<tb::batch_scheduler::ContextChunkingConfig>(m, "ContextChunkingConfig")
        .def(py::init<tle::ContextChunkingPolicy, tensorrt_llm::runtime::SizeType32>(), py::arg("chunking_policy"),
            py::arg("chunk_unit_size"))
        .def_readwrite("chunking_policy", &tb::batch_scheduler::ContextChunkingConfig::chunkingPolicy)
        .def_readwrite("chunk_unit_size", &tb::batch_scheduler::ContextChunkingConfig::chunkUnitSize);

    py::classh<GenLlmReq>(m, "GenericLlmRequest")
        .def("validate", &GenLlmReq::validate, py::arg("max_input_len"), py::arg("max_seq_len"),
            py::arg("max_draft_len"), py::arg("max_endocer_input_len") = std::nullopt,
            py::arg("enable_kv_cache_reuse") = false)
        .def("set_exclude_input_from_output", &GenLlmReq::setExcludeInputFromOutput, py::arg("exclude"))
        .def("get_num_tokens", &GenLlmReq::getNumTokens, py::arg("beam"))
        .def_property_readonly("max_beam_num_tokens", &GenLlmReq::getMaxBeamNumTokens)
        .def("get_token", &GenLlmReq::getToken, py::arg("beam"), py::arg("pos"))
        .def("get_tokens", py::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getTokens, py::const_), py::arg("beam"))
        .def("get_tokens", py::overload_cast<>(&GenLlmReq::getTokens, py::const_))
        .def_property_readonly("max_num_generated_tokens", &GenLlmReq::getMaxNumGeneratedTokens)
        .def("add_new_token", &GenLlmReq::addNewToken, py::arg("token"), py::arg("beam"))
        .def("add_new_tokens", &GenLlmReq::addNewTokens, py::arg("beam_tokens"))
        .def_property_readonly("num_draft_tokens", &GenLlmReq::getNumDraftTokens)
        .def("set_generated_tokens", &GenLlmReq::setGeneratedTokens, py::arg("generated_beam_tokens"))
        .def("pause", &GenLlmReq::pause, py::arg("max_input_len"))
        .def_property("max_sent_token_len", &GenLlmReq::getMaxSentTokenLen, &GenLlmReq::setMaxSentTokenLen)
        .def("prompt_embedding_table",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getPromptEmbeddingTable();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            })
        .def("bad_words_list",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getBadWordsList();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            })
        .def_property(
            "draft_logits",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getDraftLogits();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            },
            [](GenLlmReq& self, at::Tensor& logits)
            { self.setDraftLogits(std::make_optional<GenLlmReq::TensorPtr>(tr::TorchView::of(logits))); })
        .def("embedding_bias",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getEmbeddingBias();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            })
        .def("lora_config",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getLoraConfig();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            })
        .def("lora_weights",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getLoraWeights();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            })
        .def("stop_words_list",
            [](GenLlmReq& self)
            {
                std::optional<at::Tensor> value{std::nullopt};
                auto tensor = self.getStopWordsList();
                if (tensor)
                {
                    value = tr::Torch::tensor(*tensor);
                }
                return value;
            })
        .def_property_readonly("prompt_vocab_size", &GenLlmReq::getPromptVocabSize)
        .def_property_readonly("lora_task_id", &GenLlmReq::getLoraTaskId)
        .def_property_readonly("lookahead_config", &GenLlmReq::getLookaheadConfig)
        .def_property_readonly(
            "context_current_position", py::overload_cast<>(&GenLlmReq::getContextCurrentPosition, py::const_))
        .def_property("context_chunk_size", &GenLlmReq::getContextChunkSize, &GenLlmReq::setContextChunkSize)
        .def_readwrite("request_id", &GenLlmReq::mRequestId)
        .def_readwrite("prompt_len", &GenLlmReq::mPromptLen)
        .def_readwrite("max_new_tokens", &GenLlmReq::mMaxNewTokens)
        .def_readwrite("sampling_config", &GenLlmReq::mSamplingConfig)
        .def_property(
            "state", [](GenLlmReq& self) { return self.mState; },
            [](GenLlmReq& self, tb::LlmRequestState state) { self.mState = state; })
        .def_property("streaming", &GenLlmReq::isStreaming, &GenLlmReq::setStreaming)
        .def_readwrite("end_id", &GenLlmReq::mEndId)
        .def_readwrite("pad_id", &GenLlmReq::mPadId)
        .def_readwrite("seq_slot", &GenLlmReq::mSeqSlot)
        .def_property_readonly("return_log_probs", &GenLlmReq::returnLogProbs)
        .def_property_readonly("return_context_logits", &GenLlmReq::setReturnContextLogits)
        .def_property_readonly("return_generation_logits", &GenLlmReq::setReturnGenerationLogits)
        .def_property_readonly("log_probs", py::overload_cast<>(&GenLlmReq::getLogProbs, py::const_))
        .def("get_log_probs", py::overload_cast<GenLlmReq::SizeType32>(&GenLlmReq::getLogProbs, py::const_))
        .def("set_log_probs", &GenLlmReq::setLogProbs, py::arg("log_probs"), py::arg("beam"))
        .def("set_return_encoder_output", &GenLlmReq::setReturnEncoderOutput, py::arg("return_encoder_output"))
        .def("get_return_encoder_output", &GenLlmReq::getReturnEncoderOutput)
        .def("priority", py::overload_cast<>(&GenLlmReq::priority, py::const_))
        .def("set_priority", py::overload_cast<tle::PriorityType>(&GenLlmReq::setPriority))
        .def_property_readonly("cum_log_probs", &GenLlmReq::getCumLogProbs)
        .def("set_cum_log_prob", &GenLlmReq::setCumLogProb, py::arg("cum_log_prob"), py::arg("beam"))
        .def_property_readonly("orig_prompt_len", &GenLlmReq::getOrigPromptLen)
        .def("has_draft_tokens", &GenLlmReq::hasDraftTokens)
        .def("move_to_next_context_chunk", &GenLlmReq::moveToNextContextChunk)
        .def("is_full_context_request", py::overload_cast<>(&GenLlmReq::isFullContextRequest, py::const_))
        .def("is_last_context_chunk", py::overload_cast<>(&GenLlmReq::isLastContextChunk, py::const_))
        .def("is_first_context_chunk", py::overload_cast<>(&GenLlmReq::isFirstContextChunk, py::const_))
        .def("get_context_remaining_length", py::overload_cast<>(&GenLlmReq::getContextRemainingLength, py::const_))
        .def("create_response", &GenLlmReq::createResponse)
        .def_property(
            "draft_tokens", [](GenLlmReq& self) { return *self.getDraftTokens(); },
            [](GenLlmReq& self, GenLlmReq::VecTokens& draftTokens)
            { self.setDraftTokens(std::make_shared<GenLlmReq::VecTokens>(std::move(draftTokens))); })
        .def_property_readonly("position_ids",
            [](GenLlmReq& self)
            {
                std::optional<std::vector<GenLlmReq::SizeType32>> positionIds = std::nullopt;
                if (self.getPositionIds())
                {
                    positionIds = *self.getPositionIds().value();
                }
                return positionIds;
            })
        .def_property(
            "draft_tokens",
            [](GenLlmReq& self)
            {
                std::optional<GenLlmReq::VecTokens> draftTokens = std::nullopt;
                if (self.getDraftTokens())
                {
                    draftTokens = *self.getDraftTokens();
                }
                return draftTokens;
            },
            [](GenLlmReq& self, std::optional<GenLlmReq::VecTokens> const& draftTokens)
            {
                if (draftTokens)
                {
                    self.setDraftTokens(std::make_shared<GenLlmReq::VecTokens>(draftTokens.value()));
                }
            });

    py::classh<tb::LlmRequest, GenLlmReq>(m, "LlmRequest", pybind11::dynamic_attr())
        .def(py::init(
                 [](tb::LlmRequest::RequestIdType request_id, tb::LlmRequest::SizeType32 max_new_tokens,
                     std::vector<tb::LlmRequest::TokenIdType> input_tokens, runtime::SamplingConfig sampling_config,
                     bool is_streaming, std::optional<tb::LlmRequest::SizeType32> end_id,
                     std::optional<tb::LlmRequest::SizeType32> pad_id, std::optional<at::Tensor> embedding_bias,
                     std::optional<at::Tensor> bad_words_list, std::optional<at::Tensor> stop_words_list,
                     std::optional<std::vector<tb::LlmRequest::SizeType32>> position_ids,
                     std::optional<at::Tensor> prompt_embedding_table,
                     std::optional<tb::LlmRequest::SizeType32> prompt_vocab_size,
                     std::optional<LoraTaskIdType> lora_task_id, std::optional<at::Tensor> lora_weights,
                     std::optional<at::Tensor> lora_config,
                     std::optional<executor::LookaheadDecodingConfig> lookahead_config,
                     std::optional<executor::KvCacheRetentionConfig> kv_cache_retention_config, bool return_log_probs,
                     bool return_context_logits, bool return_generation_logits,
                     std::optional<tb::LlmRequest::VecTokens> draft_tokens, std::optional<at::Tensor> draft_logits,
                     bool exclude_input_from_output,
                     std::optional<tb::LlmRequest::LogitsPostProcessor> logits_post_processor,
                     bool apply_logits_post_processor_batched,
                     std::optional<tb::LlmRequest::VecTokens> encoder_input_tokens, bool return_encoder_output,
                     std::optional<tb::LlmRequest::RequestIdType> client_id, executor::PriorityType priority,
                     std::optional<at::Tensor> encoder_input_features,
                     std::optional<tb::LlmRequest::SizeType32> encoder_output_length,
                     std::optional<at::Tensor> cross_attention_mask, tb::LlmRequestType llm_request_type,
                     std::optional<tb::LlmRequest::VecTokenExtraIds> input_token_extra_ids,
                     tb::LlmRequest::SizeType32 num_return_sequences)
                 {
                     auto makeOptionalTensor = [](std::optional<at::Tensor> const& atTensor)
                     {
                         std::optional<tb::LlmRequest::TensorPtr> tensorPtr = std::nullopt;
                         if (atTensor)
                         {
                             tensorPtr = tr::TorchView::of(atTensor.value());
                         }
                         return tensorPtr;
                     };

                     auto embedding_bias_tensor_ptr = makeOptionalTensor(embedding_bias);
                     auto bad_words_list_tensor_ptr = makeOptionalTensor(bad_words_list);
                     auto stop_words_list_tensor_ptr = makeOptionalTensor(stop_words_list);
                     auto prompt_embedding_table_tensor_ptr = makeOptionalTensor(prompt_embedding_table);
                     auto lora_weights_tensor_ptr = makeOptionalTensor(lora_weights);
                     auto lora_config_tensor_ptr = makeOptionalTensor(lora_config);
                     auto draft_logits_tensor_ptr = makeOptionalTensor(draft_logits);
                     auto encoder_input_features_tensor_ptr = makeOptionalTensor(encoder_input_features);
                     auto cross_attention_mask_tensor_ptr = makeOptionalTensor(cross_attention_mask);

                     return tb::LlmRequest{request_id, max_new_tokens, input_tokens, sampling_config, is_streaming,
                         end_id, pad_id, embedding_bias_tensor_ptr, bad_words_list_tensor_ptr,
                         stop_words_list_tensor_ptr, position_ids, prompt_embedding_table_tensor_ptr, prompt_vocab_size,
                         lora_task_id, lora_weights_tensor_ptr, lora_config_tensor_ptr, lookahead_config,
                         kv_cache_retention_config, return_log_probs, return_context_logits, return_generation_logits,
                         draft_tokens, draft_logits_tensor_ptr, exclude_input_from_output, logits_post_processor,
                         apply_logits_post_processor_batched, encoder_input_tokens, return_encoder_output, client_id,
                         priority, encoder_input_features_tensor_ptr, encoder_output_length,
                         cross_attention_mask_tensor_ptr, llm_request_type, input_token_extra_ids,
                         num_return_sequences};
                 }),
            py::arg("request_id"), py::arg("max_new_tokens"), py::arg("input_tokens"), py::arg("sampling_config"),
            py::arg("is_streaming"), py::arg("end_id") = std::nullopt, py::arg("pad_id") = std::nullopt,
            py::arg("embedding_bias") = std::nullopt, py::arg("bad_words_list") = std::nullopt,
            py::arg("stop_words_list") = std::nullopt, py::arg("position_ids") = std::nullopt,
            py::arg("prompt_embedding_table") = std::nullopt, py::arg("prompt_vocab_size") = std::nullopt,
            py::arg("lora_task_id") = std::nullopt, py::arg("lora_weights") = std::nullopt,
            py::arg("lora_config") = std::nullopt, py::arg("lookahead_config") = std::nullopt,
            py::arg("kv_cache_retention_config") = std::nullopt, py::arg("return_log_probs") = false,
            py::arg("return_context_logits") = false, py::arg("return_generation_logits") = false,
            py::arg("draft_tokens") = std::nullopt, py::arg("draft_logits") = std::nullopt,
            py::arg("exclude_input_from_output") = false, py::arg("logits_post_processor") = std::nullopt,
            py::arg("apply_logits_post_processor_batched") = false, py::arg("encoder_input_tokens") = std::nullopt,
            py::arg("return_encoder_output") = false, py::arg("client_id") = std::nullopt,
            py::arg("priority") = executor::Request::kDefaultPriority, py::arg("encoder_input_features") = std::nullopt,
            py::arg("encoder_output_len") = std::nullopt, py::arg("cross_attention_mask") = std::nullopt,
            py::arg_v("llm_request_type", tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
                "LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION"),
            py::arg("input_token_extra_ids") = std::nullopt, py::arg("num_return_sequences") = 1)
        .def("move_prompt_embedding_table_to_gpu", &tb::LlmRequest::movePromptEmbeddingTableToGpu, py::arg("manager"))
        .def("move_lora_weights_to_gpu", &tb::LlmRequest::moveLoraWeightsToGpu, py::arg("manager"));

    py::bind_vector<tb::RequestVector>(m, "RequestVector");
    // Note: Making an opaque binding out of RequestList would impact any std::vector<unsigned> conversion
    // PybindUtils::bindList<tb::RequestList>(m, "RequestList");

    py::classh<tb::SequenceSlotManager>(m, "SequenceSlotManager")
        .def(py::init<tb::SequenceSlotManager::SlotIdType, uint64_t>(), py::arg("max_num_slots"),
            py::arg("max_sequence_idle_microseconds"))
        .def("get_sequence_slot", &tb::SequenceSlotManager::getSequenceSlot, py::arg("start_flag"),
            py::arg("sequence_id"))
        .def("free_sequence_slot", &tb::SequenceSlotManager::freeSequenceSlot, py::arg("sequence_id"))
        .def("free_idle_sequence_slots", &tb::SequenceSlotManager::freeIdleSequenceSlots);

    py::classh<tb::rnn_state_manager::RnnStateManager>(m, "RnnStateManager")
        .def(py::init<tr::SizeType32, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            py::arg("max_num_sequences"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"));

    py::class_<tb::DecoderBuffers::DraftBuffers>(m, "DraftBuffers")
        .def(py::init())
        .def_readwrite("next_draft_tokens_device", &tb::DecoderBuffers::DraftBuffers::nextDraftTokensDevice)
        .def_readwrite("next_draft_tokens_host", &tb::DecoderBuffers::DraftBuffers::nextDraftTokensHost)
        .def_readwrite(
            "prev_draft_tokens_lengths_device", &tb::DecoderBuffers::DraftBuffers::prevDraftTokensLengthsDevice)
        .def_readwrite("prev_draft_tokens_lengths_host", &tb::DecoderBuffers::DraftBuffers::prevDraftTokensLengthsHost)
        .def_readwrite(
            "next_draft_tokens_lengths_device", &tb::DecoderBuffers::DraftBuffers::nextDraftTokensLengthsDevice)
        .def_readwrite("next_draft_tokens_lengths_host", &tb::DecoderBuffers::DraftBuffers::nextDraftTokensLengthsHost)
        .def_readwrite(
            "accepted_lengths_cum_sum_device", &tb::DecoderBuffers::DraftBuffers::acceptedLengthsCumSumDevice)
        .def_readwrite("accepted_packed_paths_device", &tb::DecoderBuffers::DraftBuffers::acceptedPackedPathsDevice)
        .def_readwrite("predicted_draft_logits", &tb::DecoderBuffers::DraftBuffers::predictedDraftLogits)
        .def("create", &tb::DecoderBuffers::DraftBuffers::create, py::arg("max_num_sequences"),
            py::arg("max_tokens_per_step"), py::arg("runtime"), py::arg("model_config"));

    py::classh<tb::DecoderBuffers>(m, "DecoderBuffers")
        .def(py::init<runtime::SizeType32, runtime::SizeType32, runtime::SizeType32, runtime::SizeType32,
                 runtime::SizeType32, runtime::TllmRuntime const&, runtime::ModelConfig const&,
                 runtime::WorldConfig const&>(),
            py::arg("max_num_sequences"), py::arg("max_beam_width"), py::arg("max_attention_window"),
            py::arg("max_seq_len"), py::arg("max_tokens_per_step"), py::arg("runtime"), py::arg("model_config"),
            py::arg("world_config"))
        .def_readwrite("logits", &tb::DecoderBuffers::logits)
        .def_readwrite("slot_output_ids", &tb::DecoderBuffers::slotOutputIds)
        .def_readwrite("slot_output_ids_host", &tb::DecoderBuffers::slotOutputIdsHost)
        .def_readwrite("cache_indirection_input", &tb::DecoderBuffers::cacheIndirectionInput)
        .def_readwrite("cache_indirection_output", &tb::DecoderBuffers::cacheIndirectionOutput)
        .def_readwrite("sequence_lengths", &tb::DecoderBuffers::sequenceLengths)
        .def_readwrite("sequence_lengths_host", &tb::DecoderBuffers::sequenceLengthsHost)
        .def_readwrite("finished", &tb::DecoderBuffers::finished)
        .def_readwrite("new_output_tokens", &tb::DecoderBuffers::newOutputTokens)
        .def_readwrite("new_output_tokens_host", &tb::DecoderBuffers::newOutputTokensHost)
        .def_readwrite("cum_log_probs", &tb::DecoderBuffers::cumLogProbs)
        .def_readwrite("cum_log_probs_host", &tb::DecoderBuffers::cumLogProbsHost)
        .def_readwrite("log_probs", &tb::DecoderBuffers::logProbs)
        .def_readwrite("log_probs_host", &tb::DecoderBuffers::logProbsHost)
        .def_readwrite("finish_reasons_host", &tb::DecoderBuffers::finishReasonsHost)
        .def_readwrite("draft_buffers", &tb::DecoderBuffers::draftBuffers);

    py::class_<tb::SlotDecoderBuffers>(m, "SlotDecoderBuffers")
        .def(py::init<runtime::SizeType32, runtime::SizeType32, runtime::TllmRuntime const&>(),
            py::arg("max_beam_width"), py::arg("max_seq_len"), py::arg("runtime"))
        .def_readwrite("output_ids", &tb::SlotDecoderBuffers::outputIds)
        .def_readwrite("output_ids_host", &tb::SlotDecoderBuffers::outputIdsHost)
        .def_readwrite("sequence_lengths_host", &tb::SlotDecoderBuffers::sequenceLengthsHost)
        .def_readwrite("cum_log_probs", &tb::SlotDecoderBuffers::cumLogProbs)
        .def_readwrite("cum_log_probs_host", &tb::SlotDecoderBuffers::cumLogProbsHost)
        .def_readwrite("log_probs", &tb::SlotDecoderBuffers::logProbs)
        .def_readwrite("log_probs_host", &tb::SlotDecoderBuffers::logProbsHost)
        .def_readwrite("finish_reasons_host", &tb::SlotDecoderBuffers::finishReasonsHost);
}

} // namespace tensorrt_llm::pybind::batch_manager
