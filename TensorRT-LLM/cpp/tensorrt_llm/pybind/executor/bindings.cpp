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

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.h"
#include "executor.h"
#include "streamCaster.h"
#include "tensorCaster.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"

#include <optional>
#include <vector>

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;
using Tensor = tle::Tensor;
using SizeType32 = tle::SizeType32;
using FloatType = tle::FloatType;
using VecTokens = tle::VecTokens;
using IdType = tle::IdType;
using VecTokenExtraIds = tle::VecTokenExtraIds;

namespace tensorrt_llm::pybind::executor
{

void InitBindings(pybind11::module_& m)
{
    m.attr("__version__") = tle::version();
    py::enum_<tle::ModelType>(m, "ModelType")
        .value("DECODER_ONLY", tle::ModelType::kDECODER_ONLY)
        .value("ENCODER_ONLY", tle::ModelType::kENCODER_ONLY)
        .value("ENCODER_DECODER", tle::ModelType::kENCODER_DECODER);

    py::enum_<tle::BatchingType>(m, "BatchingType")
        .value("STATIC", tle::BatchingType::kSTATIC)
        .value("INFLIGHT", tle::BatchingType::kINFLIGHT);

    auto decodingModeGetstate = [](tle::DecodingMode const& self) { return py::make_tuple(self.getState()); };
    auto decodingModeSetstate = [](py::tuple state)
    {
        if (state.size() != 1)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DecodingMode(state[0].cast<tle::DecodingMode::UnderlyingType>());
    };
    py::class_<tle::DecodingMode>(m, "DecodingMode")
        .def("Auto", &tle::DecodingMode::Auto)
        .def("TopK", &tle::DecodingMode::TopK)
        .def("TopP", &tle::DecodingMode::TopP)
        .def("TopKTopP", &tle::DecodingMode::TopKTopP)
        .def("BeamSearch", &tle::DecodingMode::BeamSearch)
        .def("Medusa", &tle::DecodingMode::Medusa)
        .def("Lookahead", &tle::DecodingMode::Lookahead)
        .def("Eagle", &tle::DecodingMode::Eagle)
        .def("isAuto", &tle::DecodingMode::isAuto)
        .def("isTopK", &tle::DecodingMode::isTopK)
        .def("isTopP", &tle::DecodingMode::isTopP)
        .def("isTopKorTopP", &tle::DecodingMode::isTopKorTopP)
        .def("isTopKandTopP", &tle::DecodingMode::isTopKandTopP)
        .def("isBeamSearch", &tle::DecodingMode::isBeamSearch)
        .def("isMedusa", &tle::DecodingMode::isMedusa)
        .def("isLookahead", &tle::DecodingMode::isLookahead)
        .def("isEagle", &tle::DecodingMode::isEagle)
        .def(py::pickle(decodingModeGetstate, decodingModeSetstate));

    py::enum_<tle::RequestType>(m, "RequestType")
        .value("REQUEST_TYPE_CONTEXT_AND_GENERATION", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION)
        .value("REQUEST_TYPE_CONTEXT_ONLY", tle::RequestType::REQUEST_TYPE_CONTEXT_ONLY)
        .value("REQUEST_TYPE_GENERATION_ONLY", tle::RequestType::REQUEST_TYPE_GENERATION_ONLY);

    py::enum_<tle::CapacitySchedulerPolicy>(m, "CapacitySchedulerPolicy")
        .value("MAX_UTILIZATION", tle::CapacitySchedulerPolicy::kMAX_UTILIZATION)
        .value("GUARANTEED_NO_EVICT", tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
        .value("STATIC_BATCH", tle::CapacitySchedulerPolicy::kSTATIC_BATCH);

    py::enum_<tle::ContextChunkingPolicy>(m, "ContextChunkingPolicy")
        .value("EQUAL_PROGRESS", tle::ContextChunkingPolicy::kEQUAL_PROGRESS)
        .value("FIRST_COME_FIRST_SERVED", tle::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED);

    py::enum_<tle::CommunicationType>(m, "CommunicationType").value("MPI", tle::CommunicationType::kMPI);

    py::enum_<tle::CommunicationMode>(m, "CommunicationMode")
        .value("LEADER", tle::CommunicationMode::kLEADER)
        .value("ORCHESTRATOR", tle::CommunicationMode::kORCHESTRATOR);

    py::class_<tle::KvCacheStats>(m, "KvCacheStats")
        .def(py::init<>())
        .def_readwrite("max_num_blocks", &tle::KvCacheStats::maxNumBlocks)
        .def_readwrite("free_num_blocks", &tle::KvCacheStats::freeNumBlocks)
        .def_readwrite("used_num_blocks", &tle::KvCacheStats::usedNumBlocks)
        .def_readwrite("tokens_per_block", &tle::KvCacheStats::tokensPerBlock)
        .def_readwrite("alloc_total_blocks", &tle::KvCacheStats::allocTotalBlocks)
        .def_readwrite("alloc_new_blocks", &tle::KvCacheStats::allocNewBlocks)
        .def_readwrite("reused_blocks", &tle::KvCacheStats::reusedBlocks)
        .def_readwrite("missed_blocks", &tle::KvCacheStats::missedBlocks)
        .def_readwrite("cache_hit_rate", &tle::KvCacheStats::cacheHitRate);

    py::class_<tle::StaticBatchingStats>(m, "StaticBatchingStats")
        .def(py::init<>())
        .def_readwrite("num_scheduled_requests", &tle::StaticBatchingStats::numScheduledRequests)
        .def_readwrite("num_context_requests", &tle::StaticBatchingStats::numContextRequests)
        .def_readwrite("num_ctx_tokens", &tle::StaticBatchingStats::numCtxTokens)
        .def_readwrite("num_gen_tokens", &tle::StaticBatchingStats::numGenTokens)
        .def_readwrite("empty_gen_slots", &tle::StaticBatchingStats::emptyGenSlots);

    py::class_<tle::InflightBatchingStats>(m, "InflightBatchingStats")
        .def(py::init<>())
        .def_readwrite("num_scheduled_requests", &tle::InflightBatchingStats::numScheduledRequests)
        .def_readwrite("num_context_requests", &tle::InflightBatchingStats::numContextRequests)
        .def_readwrite("num_gen_requests", &tle::InflightBatchingStats::numGenRequests)
        .def_readwrite("num_paused_requests", &tle::InflightBatchingStats::numPausedRequests)
        .def_readwrite("num_ctx_tokens", &tle::InflightBatchingStats::numCtxTokens)
        .def_readwrite("micro_batch_id", &tle::InflightBatchingStats::microBatchId)
        .def_readwrite("avg_num_decoded_tokens_per_iter", &tle::InflightBatchingStats::avgNumDecodedTokensPerIter);

    py::class_<tle::IterationStats>(m, "IterationStats")
        .def(py::init<>())
        .def_readwrite("timestamp", &tle::IterationStats::timestamp)
        .def_readwrite("iter", &tle::IterationStats::iter)
        .def_readwrite("iter_latency_ms", &tle::IterationStats::iterLatencyMS)
        .def_readwrite("new_active_requests_queue_latency_ms", &tle::IterationStats::newActiveRequestsQueueLatencyMS)
        .def_readwrite("num_new_active_requests", &tle::IterationStats::numNewActiveRequests)
        .def_readwrite("num_active_requests", &tle::IterationStats::numActiveRequests)
        .def_readwrite("num_queued_requests", &tle::IterationStats::numQueuedRequests)
        .def_readwrite("num_completed_requests", &tle::IterationStats::numCompletedRequests)
        .def_readwrite("max_num_active_requests", &tle::IterationStats::maxNumActiveRequests)
        .def_readwrite("gpu_mem_usage", &tle::IterationStats::gpuMemUsage)
        .def_readwrite("cpu_mem_usage", &tle::IterationStats::cpuMemUsage)
        .def_readwrite("pinned_mem_usage", &tle::IterationStats::pinnedMemUsage)
        .def_readwrite("kv_cache_stats", &tle::IterationStats::kvCacheStats)
        .def_readwrite("cross_kv_cache_stats", &tle::IterationStats::crossKvCacheStats)
        .def_readwrite("static_batching_stats", &tle::IterationStats::staticBatchingStats)
        .def_readwrite("inflight_batching_stats", &tle::IterationStats::inflightBatchingStats)
        .def("to_json_str",
            [](tle::IterationStats const& iterationStats)
            { return tle::JsonSerialization::toJsonStr(iterationStats); });

    py::class_<tle::DebugTensorsPerIteration>(m, "DebugTensorsPerIteration")
        .def(py::init<>())
        .def_readwrite("iter", &tle::DebugTensorsPerIteration::iter)
        .def_readwrite("debug_tensors", &tle::DebugTensorsPerIteration::debugTensors);

    py::enum_<tle::RequestStage>(m, "RequestStage")
        .value("QUEUED", tle::RequestStage::kQUEUED)
        .value("ENCODER_IN_PROGRESS", tle::RequestStage::kENCODER_IN_PROGRESS)
        .value("CONTEXT_IN_PROGRESS", tle::RequestStage::kCONTEXT_IN_PROGRESS)
        .value("GENERATION_IN_PROGRESS", tle::RequestStage::kGENERATION_IN_PROGRESS)
        .value("GENERATION_COMPLETE", tle::RequestStage::kGENERATION_COMPLETE);

    py::class_<tle::DisServingRequestStats>(m, "DisServingRequestStats")
        .def(py::init<>())
        .def_readwrite("kv_cache_transfer_ms", &tle::DisServingRequestStats::kvCacheTransferMS);

    py::class_<tle::RequestStats>(m, "RequestStats")
        .def(py::init<>())
        .def_readwrite("id", &tle::RequestStats::id)
        .def_readwrite("stage", &tle::RequestStats::stage)
        .def_readwrite("context_prefill_position", &tle::RequestStats::contextPrefillPosition)
        .def_readwrite("num_generated_tokens", &tle::RequestStats::numGeneratedTokens)
        .def_readwrite("avg_num_decoded_tokens_per_iter", &tle::RequestStats::avgNumDecodedTokensPerIter)
        .def_readwrite("scheduled", &tle::RequestStats::scheduled)
        .def_readwrite("paused", &tle::RequestStats::paused)
        .def_readwrite("dis_serving_stats", &tle::RequestStats::disServingStats)
        .def_readwrite("alloc_total_blocks_per_request", &tle::RequestStats::allocTotalBlocksPerRequest)
        .def_readwrite("alloc_new_blocks_per_request", &tle::RequestStats::allocNewBlocksPerRequest)
        .def_readwrite("reused_blocks_per_request", &tle::RequestStats::reusedBlocksPerRequest)
        .def_readwrite("missed_blocks_per_request", &tle::RequestStats::missedBlocksPerRequest)
        .def_readwrite("kv_cache_hit_rate_per_request", &tle::RequestStats::kvCacheHitRatePerRequest)
        .def("to_json_str",
            [](tle::RequestStats const& iterationStats) { return tle::JsonSerialization::toJsonStr(iterationStats); });

    py::class_<tle::RequestStatsPerIteration>(m, "RequestStatsPerIteration")
        .def(py::init<>())
        .def_readwrite("iter", &tle::RequestStatsPerIteration::iter)
        .def_readwrite("request_stats", &tle::RequestStatsPerIteration::requestStats)
        .def("to_json_str",
            [](tle::RequestStatsPerIteration const& iterationStats)
            { return tle::JsonSerialization::toJsonStr(iterationStats); });

    py::class_<tle::SamplingConfig>(m, "SamplingConfig")
        // A modified version of constructor to accpect deprecated args randomSeed and minLength
        // TODO(enweiz): use the original constructor after the deprecated args are removed
        .def(
            py::init(
                [](tle::SizeType32 beamWidth, std::optional<tle::SizeType32> const& topK,
                    std::optional<tle::FloatType> const& topP, std::optional<tle::FloatType> const& topPMin,
                    std::optional<tle::TokenIdType> const& topPResetIds, std::optional<tle::FloatType> const& topPDecay,
                    std::optional<tle::RandomSeedType> seed, std::optional<tle::RandomSeedType> const& randomSeed,
                    std::optional<tle::FloatType> const& temperature, std::optional<tle::SizeType32> minTokens,
                    std::optional<tle::SizeType32> const& minLength,
                    std::optional<tle::FloatType> const& beamSearchDiversityRate,
                    std::optional<tle::FloatType> const& repetitionPenalty,
                    std::optional<tle::FloatType> const& presencePenalty,
                    std::optional<tle::FloatType> const& frequencyPenalty,
                    std::optional<tle::FloatType> const& lengthPenalty,
                    std::optional<tle::SizeType32> const& earlyStopping,
                    std::optional<tle::SizeType32> const& noRepeatNgramSize,
                    std::optional<tle::SizeType32> const& numReturnSequences)
                {
                    if (randomSeed.has_value())
                    {
                        TLLM_LOG_WARNING("random_seed is being deprecated; please use seed instead.");
                        if (!seed.has_value())
                        {
                            seed = randomSeed;
                        }
                    }
                    if (minLength.has_value())
                    {
                        TLLM_LOG_WARNING("min_length is being deprecated; please use min_tokens instead.");
                        if (!minTokens.has_value())
                        {
                            minTokens = minLength;
                        }
                    }
                    return std::make_unique<tle::SamplingConfig>(beamWidth, topK, topP, topPMin, topPResetIds,
                        topPDecay, seed, temperature, minTokens, beamSearchDiversityRate, repetitionPenalty,
                        presencePenalty, frequencyPenalty, lengthPenalty, earlyStopping, noRepeatNgramSize,
                        numReturnSequences);
                }),
            py::arg("beam_width") = 1, py::kw_only(), py::arg("top_k") = py::none(), py::arg("top_p") = py::none(),
            py::arg("top_p_min") = py::none(), py::arg("top_p_reset_ids") = py::none(),
            py::arg("top_p_decay") = py::none(), py::arg("seed") = py::none(), py::arg("random_seed") = py::none(),
            py::arg("temperature") = py::none(), py::arg("min_tokens") = py::none(), py::arg("min_length") = py::none(),
            py::arg("beam_search_diversity_rate") = py::none(), py::arg("repetition_penalty") = py::none(),
            py::arg("presence_penalty") = py::none(), py::arg("frequency_penalty") = py::none(),
            py::arg("length_penalty") = py::none(), py::arg("early_stopping") = py::none(),
            py::arg("no_repeat_ngram_size") = py::none(), py::arg("num_return_sequences") = py::none())
        .def_property("beam_width", &tle::SamplingConfig::getBeamWidth, &tle::SamplingConfig::setBeamWidth)
        .def_property("top_k", &tle::SamplingConfig::getTopK, &tle::SamplingConfig::setTopK)
        .def_property("top_p", &tle::SamplingConfig::getTopP, &tle::SamplingConfig::setTopP)
        .def_property("top_p_min", &tle::SamplingConfig::getTopPMin, &tle::SamplingConfig::setTopPMin)
        .def_property("top_p_reset_ids", &tle::SamplingConfig::getTopPResetIds, &tle::SamplingConfig::setTopPResetIds)
        .def_property("top_p_decay", &tle::SamplingConfig::getTopPDecay, &tle::SamplingConfig::setTopPDecay)
        .def_property("seed", &tle::SamplingConfig::getSeed, &tle::SamplingConfig::setSeed)
        .def_property("random_seed", &tle::SamplingConfig::getRandomSeed, &tle::SamplingConfig::setRandomSeed)
        .def_property("temperature", &tle::SamplingConfig::getTemperature, &tle::SamplingConfig::setTemperature)
        .def_property("min_tokens", &tle::SamplingConfig::getMinTokens, &tle::SamplingConfig::setMinTokens)
        .def_property("min_length", &tle::SamplingConfig::getMinLength, &tle::SamplingConfig::setMinLength)
        .def_property("beam_search_diversity_rate", &tle::SamplingConfig::getBeamSearchDiversityRate,
            &tle::SamplingConfig::setBeamSearchDiversityRate)
        .def_property("repetition_penalty", &tle::SamplingConfig::getRepetitionPenalty,
            &tle::SamplingConfig::setRepetitionPenalty)
        .def_property("presence_penalty", &tle::SamplingConfig::getPresencePenalty,
            [](tle::SamplingConfig& self, std::optional<FloatType> v) { return self.setPresencePenalty(v); })
        .def_property(
            "frequency_penalty", &tle::SamplingConfig::getFrequencyPenalty, &tle::SamplingConfig::setFrequencyPenalty)
        .def_property("length_penalty", &tle::SamplingConfig::getLengthPenalty, &tle::SamplingConfig::setLengthPenalty)
        .def_property("early_stopping", &tle::SamplingConfig::getEarlyStopping, &tle::SamplingConfig::setEarlyStopping)
        .def_property("no_repeat_ngram_size", &tle::SamplingConfig::getNoRepeatNgramSize,
            &tle::SamplingConfig::setNoRepeatNgramSize)
        .def_property("num_return_sequences", &tle::SamplingConfig::getNumReturnSequences,
            &tle::SamplingConfig::setNumReturnSequences);

    py::class_<tle::OutputConfig>(m, "OutputConfig")
        .def(py::init<bool, bool, bool, bool, bool>(), py::arg("return_log_probs") = false,
            py::arg("return_context_logits") = false, py::arg("return_generation_logits") = false,
            py::arg("exclude_input_from_output") = false, py::arg("return_encoder_output") = false)
        .def_readwrite("return_log_probs", &tle::OutputConfig::returnLogProbs)
        .def_readwrite("return_context_logits", &tle::OutputConfig::returnContextLogits)
        .def_readwrite("return_generation_logits", &tle::OutputConfig::returnGenerationLogits)
        .def_readwrite("exclude_input_from_output", &tle::OutputConfig::excludeInputFromOutput)
        .def_readwrite("return_encoder_output", &tle::OutputConfig::returnEncoderOutput);

    py::class_<tle::ExternalDraftTokensConfig>(m, "ExternalDraftTokensConfig")
        .def(py::init<VecTokens, std::optional<Tensor>, std::optional<FloatType> const&>(), py::arg("tokens"),
            py::arg("logits") = py::none(), py::arg("acceptance_threshold") = py::none())
        .def_property_readonly("tokens", &tle::ExternalDraftTokensConfig::getTokens)
        .def_property_readonly("logits", &tle::ExternalDraftTokensConfig::getLogits)
        .def_property_readonly("acceptance_threshold", &tle::ExternalDraftTokensConfig::getAcceptanceThreshold);

    py::class_<tle::PromptTuningConfig>(m, "PromptTuningConfig")
        .def(py::init<Tensor, std::optional<VecTokenExtraIds>>(), py::arg("embedding_table"),
            py::arg("input_token_extra_ids") = py::none())
        .def_property_readonly("embedding_table", &tle::PromptTuningConfig::getEmbeddingTable)
        .def_property_readonly("input_token_extra_ids", &tle::PromptTuningConfig::getInputTokenExtraIds);

    py::class_<tle::LoraConfig>(m, "LoraConfig")
        .def(py::init<uint64_t, std::optional<Tensor>, std::optional<Tensor>>(), py::arg("task_id"),
            py::arg("weights") = py::none(), py::arg("config") = py::none())
        .def_property_readonly("task_id", &tle::LoraConfig::getTaskId)
        .def_property_readonly("weights", &tle::LoraConfig::getWeights)
        .def_property_readonly("config", &tle::LoraConfig::getConfig);

    py::class_<tle::LookaheadDecodingConfig>(m, "LookaheadDecodingConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32>(), py::arg("max_window_size"), py::arg("max_ngram_size"),
            py::arg("max_verification_set_size"))
        .def_property_readonly("max_window_size", &tle::LookaheadDecodingConfig::getWindowSize)
        .def_property_readonly("max_ngram_size", &tle::LookaheadDecodingConfig::getNgramSize)
        .def_property_readonly("max_verification_set_size", &tle::LookaheadDecodingConfig::getVerificationSetSize);

    py::class_<tle::EagleConfig>(m, "EagleConfig")
        .def(py::init<std::optional<tle::EagleChoices>>(), py::arg("eagle_choices"))
        .def_property_readonly("eagle_choices", &tle::EagleConfig::getEagleChoices);

    auto kvCacheRetentionConfig = py::class_<tle::KvCacheRetentionConfig>(m, "KvCacheRetentionConfig");

    py::class_<tle::KvCacheRetentionConfig::TokenRangeRetentionPriority>(
        kvCacheRetentionConfig, "TokenRangeRetentionPriority")
        .def(py::init<SizeType32, std::optional<SizeType32>, tle::RetentionPriority>(), py::arg("token_start"),
            py::arg("token_end"), py::arg("priority"))
        .def_readwrite("token_start", &tle::KvCacheRetentionConfig::TokenRangeRetentionPriority::tokenStart)
        .def_readwrite("token_end", &tle::KvCacheRetentionConfig::TokenRangeRetentionPriority::tokenEnd)
        .def_readwrite("priority", &tle::KvCacheRetentionConfig::TokenRangeRetentionPriority::priority);

    // There's a circular dependency between the declaration of the TokenRangeRetentionPriority and
    // KvCacheRetentionConfig bindings. Defer definition of the KvCacheRetentionConfig bindings until the
    // TokenRangeRetentionPriority bindings have been defined.
    kvCacheRetentionConfig
        .def(py::init<std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionPriority>, tle::RetentionPriority>(),
            py::arg("token_range_retention_priorities"), py::arg("decode_retention_priority"))
        .def_property_readonly(
            "token_range_retention_priorities", &tle::KvCacheRetentionConfig::getTokenRangeRetentionPriorities)
        .def_property_readonly("decode_retention_priority", &tle::KvCacheRetentionConfig::getDecodeRetentionPriority);

    py::class_<tle::ContextPhaseParams>(m, "ContextPhaseParams")
        .def(py::init<VecTokens, tle::ContextPhaseParams::RequestIdType>(), py::arg("first_gen_tokens"),
            py::arg("req_id"));

    py::class_<tle::Request> request(m, "Request");
    request
        // A modified version of constructor to accpect deprecated args maxNewTokens
        // TODO(enweiz): use the original constructor after the deprecated args are removed
        .def(py::init(
                 [](tle::VecTokens inputTokenIds, std::optional<tle::SizeType32> maxTokens,
                     std::optional<tle::SizeType32> maxNewTokens, bool streaming,
                     tle::SamplingConfig const& samplingConfig, tle::OutputConfig const& outputConfig,
                     std::optional<tle::SizeType32> const& endId, std::optional<tle::SizeType32> const& padId,
                     std::optional<std::vector<SizeType32>> positionIds,
                     std::optional<std::list<tle::VecTokens>> badWords,
                     std::optional<std::list<tle::VecTokens>> stopWords, std::optional<tle::Tensor> embeddingBias,
                     std::optional<tle::ExternalDraftTokensConfig> externalDraftTokensConfig,
                     std::optional<tle::PromptTuningConfig> pTuningConfig, std::optional<tle::LoraConfig> loraConfig,
                     std::optional<tle::LookaheadDecodingConfig> lookaheadConfig,
                     std::optional<tle::KvCacheRetentionConfig> kvCacheRetentionConfig,
                     std::optional<std::string> logitsPostProcessorName,
                     std::optional<tle::VecTokens> encoderInputTokenIds, std::optional<tle::IdType> clientId,
                     bool returnAllGeneratedTokens, tle::PriorityType priority, tle::RequestType type,
                     std::optional<tle::ContextPhaseParams> contextPhaseParams,
                     std::optional<tle::Tensor> encoderInputFeatures,
                     std::optional<tle::SizeType32> encoderOutputLength, std::optional<tle::Tensor> crossAttentionMask,
                     SizeType32 numReturnSequences, std::optional<tle::EagleConfig> eagleConfig)
                 {
                     if (maxNewTokens.has_value())
                     {
                         TLLM_LOG_WARNING("max_new_tokens is being deprecated; please use max_tokens instead.");
                         if (!maxTokens.has_value())
                         {
                             maxTokens = maxNewTokens;
                         }
                     }
                     TLLM_CHECK_WITH_INFO(maxTokens.has_value(), "missing required argument max_tokens");
                     return std::make_unique<tle::Request>(inputTokenIds, maxTokens.value(), streaming, samplingConfig,
                         outputConfig, endId, padId, positionIds, badWords, stopWords, embeddingBias,
                         externalDraftTokensConfig, pTuningConfig, loraConfig, lookaheadConfig, kvCacheRetentionConfig,
                         logitsPostProcessorName, encoderInputTokenIds, clientId, returnAllGeneratedTokens, priority,
                         type, contextPhaseParams, encoderInputFeatures, encoderOutputLength, crossAttentionMask,
                         numReturnSequences, eagleConfig);
                 }),
            py::arg("input_token_ids"), py::kw_only(), py::arg("max_tokens") = py::none(),
            py::arg("max_new_tokens") = py::none(), py::arg("streaming") = false,
            py::arg_v("sampling_config", tle::SamplingConfig(), "SamplingConfig()"),
            py::arg_v("output_config", tle::OutputConfig(), "OutputConfig()"), py::arg("end_id") = py::none(),
            py::arg("pad_id") = py::none(), py::arg("position_ids") = py::none(), py::arg("bad_words") = py::none(),
            py::arg("stop_words") = py::none(), py::arg("embedding_bias") = py::none(),
            py::arg("external_draft_tokens_config") = py::none(), py::arg("prompt_tuning_config") = py::none(),
            py::arg("lora_config") = py::none(), py::arg("lookahead_config") = py::none(),
            py::arg("kv_cache_retention_config") = py::none(), py::arg("logits_post_processor_name") = py::none(),
            py::arg("encoder_input_token_ids") = py::none(), py::arg("client_id") = py::none(),
            py::arg("return_all_generated_tokens") = false, py::arg("priority") = tle::Request::kDefaultPriority,
            py::arg_v("type", tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
                "RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION"),
            py::arg("context_phase_params") = py::none(), py::arg("encoder_input_features") = py::none(),
            py::arg("encoder_output_length") = py::none(), py::arg("cross_attention_mask") = py::none(),
            py::arg("num_return_sequences") = 1, py::arg("eagle_config") = py::none())
        .def_property_readonly("input_token_ids", &tle::Request::getInputTokenIds)
        .def_property_readonly("max_tokens", &tle::Request::getMaxTokens)
        .def_property_readonly("max_new_tokens", &tle::Request::getMaxNewTokens)
        .def_property("streaming", &tle::Request::getStreaming, &tle::Request::setStreaming)
        .def_property("sampling_config", &tle::Request::getSamplingConfig, &tle::Request::setSamplingConfig)
        .def_property("output_config", &tle::Request::getOutputConfig, &tle::Request::setOutputConfig)
        .def_property("end_id", &tle::Request::getEndId, &tle::Request::setEndId)
        .def_property("pad_id", &tle::Request::getPadId, &tle::Request::setPadId)
        .def_property("position_ids", &tle::Request::getPositionIds, &tle::Request::setPositionIds)
        .def_property("bad_words", &tle::Request::getBadWords, &tle::Request::setBadWords)
        .def_property("stop_words", &tle::Request::getStopWords, &tle::Request::setStopWords)
        .def_property("embedding_bias", &tle::Request::getEmbeddingBias, &tle::Request::setEmbeddingBias)
        .def_property("external_draft_tokens_config", &tle::Request::getExternalDraftTokensConfig,
            &tle::Request::setExternalDraftTokensConfig)
        .def_property(
            "prompt_tuning_config", &tle::Request::getPromptTuningConfig, &tle::Request::setPromptTuningConfig)
        .def_property("lora_config", &tle::Request::getLoraConfig, &tle::Request::setLoraConfig)
        .def_property("lookahead_config", &tle::Request::getLookaheadConfig, &tle::Request::setLookaheadConfig)
        .def_property("kv_cache_retention_config", &tle::Request::getKvCacheRetentionConfig,
            &tle::Request::setKvCacheRetentionConfig)
        .def_property("logits_post_processor_name", &tle::Request::getLogitsPostProcessorName,
            &tle::Request::setLogitsPostProcessorName)
        .def_property(
            "encoder_input_token_ids", &tle::Request::getEncoderInputTokenIds, &tle::Request::setEncoderInputTokenIds)
        .def_property("client_id", &tle::Request::getClientId, &tle::Request::setClientId)
        .def_property("return_all_generated_tokens", &tle::Request::getReturnAllGeneratedTokens,
            &tle::Request::setReturnAllGeneratedTokens)
        .def_property("request_type", &tle::Request::getRequestType, &tle::Request::setRequestType)
        .def_property(
            "encoder_input_features", &tle::Request::getEncoderInputFeatures, &tle::Request::setEncoderInputFeatures)
        .def_property(
            "cross_attention_mask", &tle::Request::getCrossAttentionMask, &tle::Request::setCrossAttentionMask)
        .def_property(
            "num_return_sequences", &tle::Request::getNumReturnSequences, &tle::Request::setNumReturnSequences)
        .def_property("eagle_config", &tle::Request::getEagleConfig, &tle::Request::setEagleConfig);
    request.attr("BATCHED_POST_PROCESSOR_NAME") = tle::Request::kBatchedPostProcessorName;

    py::enum_<tle::FinishReason>(m, "FinishReason")
        .value("NOT_FINISHED", tle::FinishReason::kNOT_FINISHED)
        .value("END_ID", tle::FinishReason::kEND_ID)
        .value("STOP_WORDS", tle::FinishReason::kSTOP_WORDS)
        .value("LENGTH", tle::FinishReason::kLENGTH);

    py::class_<tle::Result>(m, "Result")
        .def(py::init<>())
        .def_readwrite("is_final", &tle::Result::isFinal)
        .def_readwrite("output_token_ids", &tle::Result::outputTokenIds)
        .def_readwrite("cum_log_probs", &tle::Result::cumLogProbs)
        .def_readwrite("log_probs", &tle::Result::logProbs)
        .def_readwrite("context_logits", &tle::Result::contextLogits)
        .def_readwrite("generation_logits", &tle::Result::generationLogits)
        .def_readwrite("encoder_output", &tle::Result::encoderOutput)
        .def_readwrite("finish_reasons", &tle::Result::finishReasons)
        .def_readwrite("sequence_index", &tle::Result::sequenceIndex)
        .def_readwrite("is_sequence_final", &tle::Result::isSequenceFinal)
        .def_readwrite("decoding_iter", &tle::Result::decodingIter)
        .def_readwrite("context_phase_params", &tle::Result::contextPhaseParams)
        .def_readwrite("sequence_index", &tle::Result::sequenceIndex);

    py::class_<tle::Response>(m, "Response")
        .def(py::init<IdType, std::string, std::optional<IdType>>(), py::arg("request_id"), py::arg("error_msg"),
            py::arg("client_id") = std::nullopt)
        .def(py::init<IdType, tle::Result, std::optional<IdType>>(), py::arg("request_id"), py::arg("result"),
            py::arg("client_id") = std::nullopt)
        .def_property_readonly("request_id", &tle::Response::getRequestId)
        .def_property_readonly("client_id", &tle::Response::getClientId)
        .def("has_error", &tle::Response::hasError)
        .def_property_readonly("error_msg", &tle::Response::getErrorMsg)
        .def_property_readonly("result", &tle::Response::getResult);

    auto dynamicBatchConfigGetstate = [](tle::DynamicBatchConfig const& self)
    { return py::make_tuple(self.getEnableBatchSizeTuning(), self.getDynamicBatchMovingAverageWindow()); };

    auto dynamicBatchConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 2)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DynamicBatchConfig(state[0].cast<bool>(), state[1].cast<SizeType32>());
    };

    py::class_<tle::DynamicBatchConfig>(m, "DynamicBatchConfig")
        .def(py::init<bool, SizeType32>(), py::arg("enable_batch_size_tuning"),
            py::arg("dynamic_batch_moving_average_window"))
        .def_property_readonly("enable_batch_size_tuning", &tle::DynamicBatchConfig::getEnableBatchSizeTuning)
        .def_property_readonly(
            "dynamic_batch_moving_average_window", &tle::DynamicBatchConfig::getDynamicBatchMovingAverageWindow)
        .def(py::pickle(dynamicBatchConfigGetstate, dynamicBatchConfigSetstate));

    auto schedulerConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::SchedulerConfig(state[0].cast<tle::CapacitySchedulerPolicy>(),
            state[1].cast<std::optional<tle::ContextChunkingPolicy>>(),
            state[2].cast<std::optional<tle::DynamicBatchConfig>>());
    };

    auto schedulerConfigGetstate = [](tle::SchedulerConfig const& self)
    {
        return py::make_tuple(
            self.getCapacitySchedulerPolicy(), self.getContextChunkingPolicy(), self.getDynamicBatchConfig());
    };

    py::class_<tle::SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<tle::CapacitySchedulerPolicy, std::optional<tle::ContextChunkingPolicy>,
                 std::optional<tle::DynamicBatchConfig>>(),
            py::arg_v("capacity_scheduler_policy", tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
                "CapacitySchedulerPolicy.GUARANTEED_NO_EVICT"),
            py::arg("context_chunking_policy") = py::none(), py::arg("dynamic_batch_config") = py::none())
        .def_property_readonly("capacity_scheduler_policy", &tle::SchedulerConfig::getCapacitySchedulerPolicy)
        .def_property_readonly("context_chunking_policy", &tle::SchedulerConfig::getContextChunkingPolicy)
        .def_property_readonly("dynamic_batch_config", &tle::SchedulerConfig::getDynamicBatchConfig)
        .def(py::pickle(schedulerConfigGetstate, schedulerConfigSetstate));

    auto kvCacheConfigGetstate = [](tle::KvCacheConfig const& self)
    {
        return py::make_tuple(self.getEnableBlockReuse(), self.getMaxTokens(), self.getMaxAttentionWindowVec(),
            self.getSinkTokenLength(), self.getFreeGpuMemoryFraction(), self.getHostCacheSize(),
            self.getOnboardBlocks(), self.getCrossKvCacheFraction(), self.getSecondaryOffloadMinPriority());
    };
    auto kvCacheConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 9)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::KvCacheConfig(state[0].cast<bool>(), state[1].cast<std::optional<SizeType32>>(),
            state[2].cast<std::optional<std::vector<SizeType32>>>(), state[3].cast<std::optional<SizeType32>>(),
            state[4].cast<std::optional<float>>(), state[5].cast<std::optional<size_t>>(), state[6].cast<bool>(),
            state[7].cast<std::optional<float>>(), state[8].cast<std::optional<tle::RetentionPriority>>());
    };
    py::class_<tle::KvCacheConfig>(m, "KvCacheConfig")
        .def(py::init<bool, std::optional<SizeType32> const&, std::optional<std::vector<SizeType32>> const&,
                 std::optional<SizeType32> const&, std::optional<float> const&, std::optional<size_t> const&, bool,
                 std::optional<float> const&, std::optional<tle::RetentionPriority>>(),
            py::arg("enable_block_reuse") = false, py::arg("max_tokens") = py::none(),
            py::arg("max_attention_window") = py::none(), py::arg("sink_token_length") = py::none(),
            py::arg("free_gpu_memory_fraction") = py::none(), py::arg("host_cache_size") = py::none(),
            py::arg("onboard_blocks") = true, py::arg("cross_kv_cache_fraction") = py::none(),
            py::arg("secondary_offload_min_priority") = py::none())
        .def_property(
            "enable_block_reuse", &tle::KvCacheConfig::getEnableBlockReuse, &tle::KvCacheConfig::setEnableBlockReuse)
        .def_property("max_tokens", &tle::KvCacheConfig::getMaxTokens, &tle::KvCacheConfig::setMaxTokens)
        .def_property("max_attention_window", &tle::KvCacheConfig::getMaxAttentionWindowVec,
            &tle::KvCacheConfig::setMaxAttentionWindowVec)
        .def_property(
            "sink_token_length", &tle::KvCacheConfig::getSinkTokenLength, &tle::KvCacheConfig::setSinkTokenLength)
        .def_property("free_gpu_memory_fraction", &tle::KvCacheConfig::getFreeGpuMemoryFraction,
            &tle::KvCacheConfig::setFreeGpuMemoryFraction)
        .def_property("host_cache_size", &tle::KvCacheConfig::getHostCacheSize, &tle::KvCacheConfig::setHostCacheSize)
        .def_property("onboard_blocks", &tle::KvCacheConfig::getOnboardBlocks, &tle::KvCacheConfig::setOnboardBlocks)
        .def_property("cross_kv_cache_fraction", &tle::KvCacheConfig::getCrossKvCacheFraction,
            &tle::KvCacheConfig::setCrossKvCacheFraction)
        .def_property("secondary_offload_min_priority", &tle::KvCacheConfig::getSecondaryOffloadMinPriority,
            &tle::KvCacheConfig::setSecondaryOffloadMinPriority)
        .def(py::pickle(kvCacheConfigGetstate, kvCacheConfigSetstate));

    py::class_<tle::OrchestratorConfig>(m, "OrchestratorConfig")
        .def(py::init<bool, std::string, std::shared_ptr<mpi::MpiComm>, bool>(), py::arg("is_orchestrator") = true,
            py::arg("worker_executable_path") = "", py::arg("orch_leader_comm") = nullptr,
            py::arg("spawn_processes") = true)
        .def_property(
            "is_orchestrator", &tle::OrchestratorConfig::getIsOrchestrator, &tle::OrchestratorConfig::setIsOrchestrator)
        .def_property("worker_executable_path", &tle::OrchestratorConfig::getWorkerExecutablePath,
            &tle::OrchestratorConfig::setWorkerExecutablePath)
        .def_property("orch_leader_comm", &tle::OrchestratorConfig::getOrchLeaderComm,
            &tle::OrchestratorConfig::setOrchLeaderComm)
        .def_property("spawn_processes", &tle::OrchestratorConfig::getSpawnProcesses,
            &tle::OrchestratorConfig::setSpawnProcesses);

    auto parallelConfigGetstate = [](tle::ParallelConfig const& self)
    {
        return py::make_tuple(self.getCommunicationType(), self.getCommunicationMode(), self.getDeviceIds(),
            self.getParticipantIds(), self.getOrchestratorConfig());
    };
    auto parallelConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 5)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::ParallelConfig(state[0].cast<tle::CommunicationType>(), state[1].cast<tle::CommunicationMode>(),
            state[2].cast<std::optional<std::vector<SizeType32>>>(),
            state[3].cast<std::optional<std::vector<SizeType32>>>(),
            state[4].cast<std::optional<tle::OrchestratorConfig>>());
    };
    py::class_<tle::ParallelConfig>(m, "ParallelConfig")
        .def(py::init<tle::CommunicationType, tle::CommunicationMode, std::optional<std::vector<SizeType32>> const&,
                 std::optional<std::vector<SizeType32>> const&, std::optional<tle::OrchestratorConfig> const&>(),
            py::arg_v("communication_type", tle::CommunicationType::kMPI, "CommunicationType.MPI"),
            py::arg_v("communication_mode", tle::CommunicationMode::kLEADER, "CommunicationMode.LEADER"),
            py::arg("device_ids") = py::none(), py::arg("participant_ids") = py::none(),
            py::arg("orchestrator_config") = py::none())
        .def_property("communication_type", &tle::ParallelConfig::getCommunicationType,
            &tle::ParallelConfig::setCommunicationType)
        .def_property("communication_mode", &tle::ParallelConfig::getCommunicationMode,
            &tle::ParallelConfig::setCommunicationMode)
        .def_property("device_ids", &tle::ParallelConfig::getDeviceIds, &tle::ParallelConfig::setDeviceIds)
        .def_property(
            "participant_ids", &tle::ParallelConfig::getParticipantIds, &tle::ParallelConfig::setParticipantIds)
        .def_property("orchestrator_config", &tle::ParallelConfig::getOrchestratorConfig,
            &tle::ParallelConfig::setOrchestratorConfig)
        .def(py::pickle(parallelConfigGetstate, parallelConfigSetstate));

    auto peftCacheConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 11)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::PeftCacheConfig(state[0].cast<SizeType32>(), state[1].cast<SizeType32>(),
            state[2].cast<SizeType32>(), state[3].cast<SizeType32>(), state[4].cast<SizeType32>(),
            state[5].cast<SizeType32>(), state[6].cast<SizeType32>(), state[7].cast<SizeType32>(),
            state[8].cast<SizeType32>(), state[9].cast<std::optional<float>>(),
            state[10].cast<std::optional<size_t>>());
    };
    auto peftCacheConfigGetstate = [](tle::PeftCacheConfig const& self)
    {
        return py::make_tuple(self.getNumHostModuleLayer(), self.getNumDeviceModuleLayer(),
            self.getOptimalAdapterSize(), self.getMaxAdapterSize(), self.getNumPutWorkers(), self.getNumEnsureWorkers(),
            self.getNumCopyStreams(), self.getMaxPagesPerBlockHost(), self.getMaxPagesPerBlockDevice(),
            self.getDeviceCachePercent(), self.getHostCacheSize());
    };
    py::class_<tle::PeftCacheConfig>(m, "PeftCacheConfig")
        .def(py::init<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32, SizeType32,
                 SizeType32, std::optional<float> const&, std::optional<size_t> const&>(),
            py::arg("num_host_module_layer") = 0, py::arg("num_device_module_layer") = 0,
            py::arg("optimal_adapter_size") = 8, py::arg("max_adapter_size") = 64, py::arg("num_put_workers") = 1,
            py::arg("num_ensure_workers") = 1, py::arg("num_copy_streams") = 1,
            py::arg("max_pages_per_block_host") = 24, py::arg("max_pages_per_block_device") = 8,
            py::arg("device_cache_percent") = py::none(), py::arg("host_cache_size") = py::none())
        .def_property_readonly("num_host_module_layer", &tle::PeftCacheConfig::getNumHostModuleLayer)
        .def_property_readonly("num_device_module_layer", &tle::PeftCacheConfig::getNumDeviceModuleLayer)
        .def_property_readonly("optimal_adapter_size", &tle::PeftCacheConfig::getOptimalAdapterSize)
        .def_property_readonly("max_adapter_size", &tle::PeftCacheConfig::getMaxAdapterSize)
        .def_property_readonly("num_put_workers", &tle::PeftCacheConfig::getNumPutWorkers)
        .def_property_readonly("num_ensure_workers", &tle::PeftCacheConfig::getNumEnsureWorkers)
        .def_property_readonly("num_copy_streams", &tle::PeftCacheConfig::getNumCopyStreams)
        .def_property_readonly("max_pages_per_block_host", &tle::PeftCacheConfig::getMaxPagesPerBlockHost)
        .def_property_readonly("max_pages_per_block_device", &tle::PeftCacheConfig::getMaxPagesPerBlockDevice)
        .def_property_readonly("device_cache_percent", &tle::PeftCacheConfig::getDeviceCachePercent)
        .def_property_readonly("host_cache_size", &tle::PeftCacheConfig::getHostCacheSize)
        .def(py::pickle(peftCacheConfigGetstate, peftCacheConfigSetstate));

    auto decodingConfigGetstate = [](tle::DecodingConfig const& self)
    {
        return py::make_tuple(
            self.getDecodingMode(), self.getLookaheadDecodingConfig(), self.getMedusaChoices(), self.getEagleConfig());
    };
    auto decodingConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DecodingConfig(state[0].cast<std::optional<tle::DecodingMode>>(),
            state[1].cast<std::optional<tle::LookaheadDecodingConfig>>(),
            state[2].cast<std::optional<tle::MedusaChoices>>(), state[3].cast<std::optional<tle::EagleConfig>>());
    };
    py::class_<tle::DecodingConfig>(m, "DecodingConfig")
        .def(py::init<std::optional<tle::DecodingMode>, std::optional<tle::LookaheadDecodingConfig>,
                 std::optional<tle::MedusaChoices>, std::optional<tle::EagleConfig>>(),
            py::arg("decoding_mode") = py::none(), py::arg("lookahead_decoding_config") = py::none(),
            py::arg("medusa_choices") = py::none(), py::arg("eagle_config") = py::none())
        .def_property("decoding_mode", &tle::DecodingConfig::getDecodingMode, &tle::DecodingConfig::setDecodingMode)
        .def_property("lookahead_decoding_config", &tle::DecodingConfig::getLookaheadDecodingConfig,
            &tle::DecodingConfig::setLookaheadDecoding)
        .def_property("medusa_choices", &tle::DecodingConfig::getMedusaChoices, &tle::DecodingConfig::setMedusaChoices)
        .def_property("eagle_config", &tle::DecodingConfig::getEagleConfig, &tle::DecodingConfig::setEagleConfig)
        .def(py::pickle(decodingConfigGetstate, decodingConfigSetstate));

    auto debugConfigGetstate = [](tle::DebugConfig const& self)
    {
        return py::make_tuple(self.getDebugInputTensors(), self.getDebugOutputTensors(), self.getDebugTensorNames(),
            self.getDebugTensorsMaxIterations());
    };
    auto debugConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::DebugConfig(state[0].cast<bool>(), state[1].cast<bool>(), state[2].cast<std::vector<std::string>>(),
            state[3].cast<SizeType32>());
    };
    py::class_<tle::DebugConfig>(m, "DebugConfig")
        .def(py::init<bool, bool, std::vector<std::string>, SizeType32>(), py::arg("debug_input_tensors") = false,
            py::arg("debug_output_tensors") = false, py::arg("debug_tensor_names") = py::none(),
            py::arg("debug_tensors_max_iterations") = false)
        .def_property(
            "debug_input_tensors", &tle::DebugConfig::getDebugInputTensors, &tle::DebugConfig::setDebugInputTensors)
        .def_property(
            "debug_output_tensors", &tle::DebugConfig::getDebugOutputTensors, &tle::DebugConfig::setDebugOutputTensors)
        .def_property(
            "debug_tensor_names", &tle::DebugConfig::getDebugTensorNames, &tle::DebugConfig::setDebugTensorNames)
        .def_property("debug_tensors_max_iterations", &tle::DebugConfig::getDebugTensorsMaxIterations,
            &tle::DebugConfig::setDebugTensorsMaxIterations)
        .def(py::pickle(debugConfigGetstate, debugConfigSetstate));

    auto logitsPostProcessorConfigGetstate = [](tle::LogitsPostProcessorConfig const& self)
    { return py::make_tuple(self.getProcessorMap(), self.getProcessorBatched(), self.getReplicate()); };
    auto logitsPostProcessorConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 3)
        {
            throw std::runtime_error("Invalid LogitsPostProcessorConfig state!");
        }
        return tle::LogitsPostProcessorConfig(state[0].cast<std::optional<tle::LogitsPostProcessorMap>>(),
            state[1].cast<std::optional<tle::LogitsPostProcessorBatched>>(), state[2].cast<bool>());
    };

    py::class_<tle::LogitsPostProcessorConfig>(m, "LogitsPostProcessorConfig")
        .def(py::init<std::optional<tle::LogitsPostProcessorMap>, std::optional<tle::LogitsPostProcessorBatched>,
                 bool>(),
            py::arg("processor_map") = py::none(), py::arg("processor_batched") = py::none(),
            py::arg("replicate") = true)
        .def_property("processor_map", &tle::LogitsPostProcessorConfig::getProcessorMap,
            &tle::LogitsPostProcessorConfig::setProcessorMap)
        .def_property("processor_batched", &tle::LogitsPostProcessorConfig::getProcessorBatched,
            &tle::LogitsPostProcessorConfig::setProcessorBatched)
        .def_property(
            "replicate", &tle::LogitsPostProcessorConfig::getReplicate, &tle::LogitsPostProcessorConfig::setReplicate)
        .def(py::pickle(logitsPostProcessorConfigGetstate, logitsPostProcessorConfigSetstate));

    auto extendedRuntimePerfKnobConfigSetstate = [](py::tuple state)
    {
        if (state.size() != 4)
        {
            throw std::runtime_error("Invalid extendedRuntimePerfKnobConfig state!");
        }
        return tle::ExtendedRuntimePerfKnobConfig(
            state[0].cast<bool>(), state[1].cast<bool>(), state[2].cast<bool>(), state[2].cast<SizeType32>());
    };
    auto extendedRuntimePerfKnobConfigGetstate = [](tle::ExtendedRuntimePerfKnobConfig const& self)
    {
        return py::make_tuple(self.getMultiBlockMode(), self.getEnableContextFMHAFP32Acc(), self.getCudaGraphMode(),
            self.getCudaGraphCacheSize());
    };
    py::class_<tle::ExtendedRuntimePerfKnobConfig>(m, "ExtendedRuntimePerfKnobConfig")
        .def(
            py::init<bool, bool>(), py::arg("multi_block_mode") = true, py::arg("enable_context_fmha_fp32_acc") = false)
        .def_property("multi_block_mode", &tle::ExtendedRuntimePerfKnobConfig::getMultiBlockMode,
            &tle::ExtendedRuntimePerfKnobConfig::setMultiBlockMode)
        .def_property("enable_context_fmha_fp32_acc", &tle::ExtendedRuntimePerfKnobConfig::getEnableContextFMHAFP32Acc,
            &tle::ExtendedRuntimePerfKnobConfig::setEnableContextFMHAFP32Acc)
        .def_property("cuda_graph_mode", &tle::ExtendedRuntimePerfKnobConfig::getCudaGraphMode,
            &tle::ExtendedRuntimePerfKnobConfig::setCudaGraphMode)
        .def_property("cuda_graph_cache_size", &tle::ExtendedRuntimePerfKnobConfig::getCudaGraphCacheSize,
            &tle::ExtendedRuntimePerfKnobConfig::setCudaGraphCacheSize)
        .def(py::pickle(extendedRuntimePerfKnobConfigGetstate, extendedRuntimePerfKnobConfigSetstate));

    auto executorConfigGetState = [](tle::ExecutorConfig const& self)
    {
        return py::make_tuple(self.getMaxBeamWidth(), self.getSchedulerConfig(), self.getKvCacheConfig(),
            self.getEnableChunkedContext(), self.getNormalizeLogProbs(), self.getIterStatsMaxIterations(),
            self.getRequestStatsMaxIterations(), self.getBatchingType(), self.getMaxBatchSize(), self.getMaxNumTokens(),
            self.getParallelConfig(), self.getPeftCacheConfig(), self.getLogitsPostProcessorConfig(),
            self.getDecodingConfig(), self.getGpuWeightsPercent(), self.getMaxQueueSize(),
            self.getExtendedRuntimePerfKnobConfig(), self.getDebugConfig(), self.getRecvPollPeriodMs(),
            self.getMaxSeqIdleMicroseconds());
    };
    auto executorConfigSetState = [](py::tuple state)
    {
        if (state.size() != 20)
        {
            throw std::runtime_error("Invalid state!");
        }
        return tle::ExecutorConfig(state[0].cast<SizeType32>(), state[1].cast<tle::SchedulerConfig>(),
            state[2].cast<tle::KvCacheConfig>(), state[3].cast<bool>(), state[4].cast<bool>(),
            state[5].cast<SizeType32>(), state[6].cast<SizeType32>(), state[7].cast<tle::BatchingType>(),
            state[8].cast<std::optional<SizeType32>>(), state[9].cast<std::optional<SizeType32>>(),
            state[10].cast<std::optional<tle::ParallelConfig>>(), state[11].cast<std::optional<tle::PeftCacheConfig>>(),
            state[12].cast<std::optional<tle::LogitsPostProcessorConfig>>(),
            state[13].cast<std::optional<tle::DecodingConfig>>(), state[14].cast<float>(),
            state[15].cast<std::optional<SizeType32>>(), state[16].cast<tle::ExtendedRuntimePerfKnobConfig>(),
            state[17].cast<std::optional<tle::DebugConfig>>(), state[18].cast<SizeType32>(),
            state[19].cast<uint64_t>());
    };
    py::class_<tle::ExecutorConfig>(m, "ExecutorConfig")
        .def(py::init<SizeType32, tle::SchedulerConfig const&, tle::KvCacheConfig const&, bool, bool, SizeType32,
                 SizeType32, tle::BatchingType, std::optional<SizeType32>, std::optional<SizeType32>,
                 std::optional<tle::ParallelConfig>, tle::PeftCacheConfig const&,
                 std::optional<tle::LogitsPostProcessorConfig>, std::optional<tle::DecodingConfig>, float,
                 std::optional<SizeType32>, tle::ExtendedRuntimePerfKnobConfig const&, std::optional<tle::DebugConfig>,
                 SizeType32, uint64_t>(),
            py::arg("max_beam_width") = 1, py::arg_v("scheduler_config", tle::SchedulerConfig(), "SchedulerConfig()"),
            py::arg_v("kv_cache_config", tle::KvCacheConfig(), "KvCacheConfig()"),
            py::arg("enable_chunked_context") = false, py::arg("normalize_log_probs") = true,
            py::arg("iter_stats_max_iterations") = tle::kDefaultIterStatsMaxIterations,
            py::arg("request_stats_max_iterations") = tle::kDefaultRequestStatsMaxIterations,
            py::arg_v("batching_type", tle::BatchingType::kINFLIGHT, "BatchingType.INFLIGHT"),
            py::arg("max_batch_size") = py::none(), py::arg("max_num_tokens") = py::none(),
            py::arg("parallel_config") = py::none(),
            py::arg_v("peft_cache_config", tle::PeftCacheConfig(), "PeftCacheConfig()"),
            py::arg("logits_post_processor_config") = py::none(), py::arg("decoding_config") = py::none(),
            py::arg("gpu_weights_percent") = 1.0, py::arg("max_queue_size") = py::none(),
            py::arg_v("extended_runtime_perf_knob_config", tle::ExtendedRuntimePerfKnobConfig(),
                "ExtendedRuntimePerfKnobConfig()"),
            py::arg("debug_config") = py::none(), py::arg("recv_poll_period_ms") = 0,
            py::arg("max_seq_idle_microseconds") = 180000000)
        .def_property("max_beam_width", &tle::ExecutorConfig::getMaxBeamWidth, &tle::ExecutorConfig::setMaxBeamWidth)
        .def_property("max_batch_size", &tle::ExecutorConfig::getMaxBatchSize, &tle::ExecutorConfig::setMaxBatchSize)
        .def_property("max_num_tokens", &tle::ExecutorConfig::getMaxNumTokens, &tle::ExecutorConfig::setMaxNumTokens)
        .def_property(
            "scheduler_config", &tle::ExecutorConfig::getSchedulerConfig, &tle::ExecutorConfig::setSchedulerConfig)
        .def_property("kv_cache_config", &tle::ExecutorConfig::getKvCacheConfig, &tle::ExecutorConfig::setKvCacheConfig)
        .def_property("enable_chunked_context", &tle::ExecutorConfig::getEnableChunkedContext,
            &tle::ExecutorConfig::setEnableChunkedContext)
        .def_property("normalize_log_probs", &tle::ExecutorConfig::getNormalizeLogProbs,
            &tle::ExecutorConfig::setNormalizeLogProbs)
        .def_property("iter_stats_max_iterations", &tle::ExecutorConfig::getIterStatsMaxIterations,
            &tle::ExecutorConfig::setIterStatsMaxIterations)
        .def_property("request_stats_max_iterations", &tle::ExecutorConfig::getRequestStatsMaxIterations,
            &tle::ExecutorConfig::setRequestStatsMaxIterations)
        .def_property("batching_type", &tle::ExecutorConfig::getBatchingType, &tle::ExecutorConfig::setBatchingType)
        .def_property(
            "parallel_config", &tle::ExecutorConfig::getParallelConfig, &tle::ExecutorConfig::setParallelConfig)
        .def_property(
            "peft_cache_config", &tle::ExecutorConfig::getPeftCacheConfig, &tle::ExecutorConfig::setPeftCacheConfig)
        .def_property("logits_post_processor_config", &tle::ExecutorConfig::getLogitsPostProcessorConfig,
            &tle::ExecutorConfig::setLogitsPostProcessorConfig)
        .def_property(
            "decoding_config", &tle::ExecutorConfig::getDecodingConfig, &tle::ExecutorConfig::setDecodingConfig)
        .def_property("gpu_weights_percent", &tle::ExecutorConfig::getGpuWeightsPercent,
            &tle::ExecutorConfig::setGpuWeightsPercent)
        .def_property("max_queue_size", &tle::ExecutorConfig::getMaxQueueSize, &tle::ExecutorConfig::setMaxQueueSize)
        .def_property("extended_runtime_perf_knob_config", &tle::ExecutorConfig::getExtendedRuntimePerfKnobConfig,
            &tle::ExecutorConfig::setExtendedRuntimePerfKnobConfig)
        .def_property("debug_config", &tle::ExecutorConfig::getDebugConfig, &tle::ExecutorConfig::setDebugConfig)
        .def_property(
            "recv_poll_period_ms", &tle::ExecutorConfig::getRecvPollPeriodMs, &tle::ExecutorConfig::setRecvPollPeriodMs)
        .def_property("max_seq_idle_microseconds", &tle::ExecutorConfig::getMaxSeqIdleMicroseconds,
            &tle::ExecutorConfig::setMaxSeqIdleMicroseconds)
        .def(py::pickle(executorConfigGetState, executorConfigSetState));

    tensorrt_llm::pybind::executor::Executor::initBindings(m);
}

} // namespace tensorrt_llm::pybind::executor
