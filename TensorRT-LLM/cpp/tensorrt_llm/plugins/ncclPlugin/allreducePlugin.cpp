/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "allreducePlugin.h"

#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include <nccl.h>
#include <unordered_set>

using namespace nvinfer1;
using tensorrt_llm::plugins::AllreducePluginCreator;
using tensorrt_llm::plugins::AllreducePlugin;
using tensorrt_llm::kernels::AllReduceFusionOp;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::kernels::AllReduceStrategyConfig;

static char const* ALLREDUCE_PLUGIN_VERSION{"1"};
static char const* ALLREDUCE_PLUGIN_NAME{"AllReduce"};
PluginFieldCollection AllreducePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AllreducePluginCreator::mPluginAttributes;

AllreducePlugin::AllreducePlugin(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy,
    AllReduceStrategyConfig config, AllReduceFusionOp op, int32_t counter, float eps, int8_t affine, int8_t bias)
    : mGroup(std::move(group))
    , mType(type)
    , mStrategy(strategy)
    , mConfig(config)
    , mOp(op)
    , mEps(eps)
    , mAffine(affine)
    , mBias(bias)
{
}

// Parameterized constructor
AllreducePlugin::AllreducePlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mStrategy);
    read(d, mConfig);
    read(d, mOp);
    read(d, mEps);
    read(d, mAffine);
    read(d, mBias);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* AllreducePlugin::clone() const noexcept
{
    auto* plugin = new AllreducePlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AllreducePlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool AllreducePlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    int fusion_op_extra_inputs = 0;
    if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
    {
        ++fusion_op_extra_inputs;
        if (mAffine)
        {
            ++fusion_op_extra_inputs;
        }
        if (mBias)
        {
            ++fusion_op_extra_inputs;
        }
    }
    if (mStrategy == AllReduceStrategyType::NCCL)
    {
        TLLM_CHECK_WITH_INFO(nbInputs == (1 + fusion_op_extra_inputs), "NCCL strategy only accepts one input.");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            nbInputs == (2 + fusion_op_extra_inputs), "Non-NCCL strategies require a workspace tensor.");
    }

    if (mStrategy != AllReduceStrategyType::NCCL && pos == 1)
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT64) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void AllreducePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t AllreducePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

AllReduceStrategyType AllreducePlugin::selectImplementation(
    size_t messageSize, int worldSize, nvinfer1::DataType type) noexcept
{
    bool const isAuto = (mStrategy == AllReduceStrategyType::AUTO);

    if (!mIsP2PSupported)
    {
        if (!isAuto)
        {
            TLLM_LOG_INFO("Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL");
        }
        return AllReduceStrategyType::NCCL;
    }

    if (isAuto && !mIsNVLINKSupported)
    {
        return AllReduceStrategyType::NCCL;
    }

    auto const maxWorkspaceSize = utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(worldSize);

    AllReduceStrategyType strat = AllReduceStrategyType::NCCL;
    auto const messageSizeBytes = messageSize * common::getDTypeSize(type);

    if (messageSizeBytes <= maxWorkspaceSize)
    {
        // In some instances, the two-shot strategy has exhibited significant performance issues.
        // As a temporary measure, we have disabled the two-shot strategy.
        // TODO: remove this WAR after https://nvbugspro.nvidia.com/bug/4718747 is fixed.
        if (!isAuto)
        {
            strat = mStrategy;
        }
        else if (worldSize <= 2)
        {
            strat = AllReduceStrategyType::ONESHOT;
        }
        else if (worldSize <= 4)
        {
            if (messageSizeBytes < 1 * 1000 * 1000)
            {
                strat = AllReduceStrategyType::ONESHOT;
            }
            else
            {
                strat = AllReduceStrategyType::NCCL;
            }
        }
        else
        {
            if (messageSizeBytes < 500 * 1000)
            {
                strat = AllReduceStrategyType::ONESHOT;
            }
            else
            {
                strat = AllReduceStrategyType::NCCL;
            }
        }

        if (!kernels::configurationSupported(strat, messageSize, worldSize, type))
        {
            if (!isAuto)
            {
                TLLM_LOG_WARNING("Since not alignment, fallback to AllReduceStrategy: NCCL");
            }
            strat = AllReduceStrategyType::NCCL;
        }
    }
    else
    {
        if (!isAuto)
        {
            TLLM_LOG_WARNING("Since messageSize > maxWorkspace, fallback to AllReduceStrategy: NCCL");
        }
        strat = AllReduceStrategyType::NCCL;
    }

    return strat;
}

int AllreducePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    auto const sizePerElem = common::getDTypeSize(mType);

    kernels::AllReduceStrategyType runtimeStrategy;

    static char* forceNcclAllReduceStrategyChar = std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY");
    bool forceNcclAllReduceStrategy = (forceNcclAllReduceStrategyChar != nullptr);
    if (forceNcclAllReduceStrategy || mStrategy == AllReduceStrategyType::NCCL)
    {
        runtimeStrategy = AllReduceStrategyType::NCCL;
    }
    else
    {
        runtimeStrategy = selectImplementation(size, mGroup.size(), mType);
    }

    // Log runtime strategy
    auto const rank = COMM_SESSION.getRank();
    switch (runtimeStrategy)
    {
    case AllReduceStrategyType::NCCL:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: NCCL", rank);
        break;
    }
    case AllReduceStrategyType::ONESHOT:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: ONESHOT", rank);
        break;
    }
    case AllReduceStrategyType::TWOSHOT:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: TWOSHOT", rank);
        break;
    }
    default: break;
    }

    if (runtimeStrategy == AllReduceStrategyType::NCCL)
    {
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            NCCLCHECK(ncclAllReduce(inputs[0], outputs[1], size, (*getDtypeMap())[mType], ncclSum, *mNcclComm, stream));
            tensorrt_llm::kernels::AllReduceParams params;
            int fusion_ptr_idx = 0;
            if (mStrategy == AllReduceStrategyType::NCCL)
            {
                fusion_ptr_idx = 1;
            }
            else
            {
                fusion_ptr_idx = 2;
            }
            params.fusion_params.bias_buffer = mBias ? inputs[fusion_ptr_idx++] : nullptr;
            params.fusion_params.residual_buffer = inputs[fusion_ptr_idx++];
            params.fusion_params.weight_buffer = mAffine ? inputs[fusion_ptr_idx++] : nullptr;
            params.local_output_buffer_ptr = outputs[0];
            params.elts_total = size;
            params.fusion_params.hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
            params.fusion_params.eps = mEps;
            params.fusion_params.intermediate_buffer = outputs[1];
            tensorrt_llm::kernels::residualRmsNorm(params, mType, stream);
        }
        else
        {
            NCCLCHECK(ncclAllReduce(inputs[0], outputs[0], size, (*getDtypeMap())[mType], ncclSum, *mNcclComm, stream));
        }
    }
    else
    {
        auto const tpSize = mGroup.size();
        int tpRank = 0;
        for (auto const& currentRank : mGroup)
        {
            if (rank == currentRank)
                break;
            ++tpRank;
        }

        int token_num = size / inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

        auto params = tensorrt_llm::kernels::AllReduceParams::deserialize(
            reinterpret_cast<int64_t*>(const_cast<void*>(inputs[1])), tpSize, tpRank, mType, token_num, mOp);

        params.local_output_buffer_ptr = outputs[0];
        params.local_input_buffer_ptr = inputs[0];
        params.elts_total = size;
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {

            int fusion_ptr_idx = 2;
            params.fusion_params.bias_buffer = mBias ? inputs[fusion_ptr_idx++] : nullptr;
            params.fusion_params.residual_buffer = inputs[fusion_ptr_idx++];
            params.fusion_params.weight_buffer = mAffine ? inputs[fusion_ptr_idx++] : nullptr;
            params.fusion_params.hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
            params.fusion_params.eps = mEps;
            params.fusion_params.intermediate_buffer = outputs[1];
            for (int i = 0; i < tpSize; ++i)
            {
                params.fusion_params.lamport_peer_comm_buffer_ptrs[i]
                    = reinterpret_cast<void**>(const_cast<void*>(inputs[1]))[tpSize * 4 + i];
                params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tensorrt_llm::kernels::MAX_RANKS_PER_NODE]
                    = reinterpret_cast<void**>(const_cast<void*>(inputs[1]))[tpSize * 5 + i];
                params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tensorrt_llm::kernels::MAX_RANKS_PER_NODE * 2]
                    = reinterpret_cast<void**>(const_cast<void*>(inputs[1]))[tpSize * 6 + i];
            }
        }
        tensorrt_llm::kernels::customAllReduce(params, mType, runtimeStrategy, mConfig, mOp, stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType AllreducePlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    int fusion_op_extra_output = (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM ? 1 : 0);
    assert(index <= fusion_op_extra_output);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* AllreducePlugin::getPluginType() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

char const* AllreducePlugin::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

int AllreducePlugin::getNbOutputs() const noexcept
{
    return (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM ? 2 : 1);
}

bool AllreducePlugin::isCustomAllReduceSupported(int ranks_per_node) const noexcept
{
    constexpr bool isCudaVersionSupported =
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        true;
#else
        false;
#endif

    return isCudaVersionSupported && (ranks_per_node % 2 == 0) && (ranks_per_node <= kernels::MAX_RANKS_PER_NODE)
        && (ranks_per_node > 0);
}

class NvmlManager
{
public:
    NvmlManager()
    {
        NVML_CHECK(nvmlInit());
    }

    ~NvmlManager()
    {
        NVML_CHECK(nvmlShutdown());
    }
};

std::set<int> getLocalGroup(std::set<int> const& group)
{
    auto const myRank = COMM_SESSION.getRank();
    auto const myLocalRank = LOCAL_COMM_SESSION.getRank();
    auto const localSize = LOCAL_COMM_SESSION.getSize();

    std::vector<int32_t> ranks(localSize, 0);
    std::vector<int32_t> localRanks(localSize, 0);
    if (group.size() >= localSize)
    {
        LOCAL_COMM_SESSION.allgather(&myRank, ranks.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
        LOCAL_COMM_SESSION.allgather(&myLocalRank, localRanks.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
    }
    else
    {
        if (myRank == *group.begin())
        {
            ranks.clear();
            int rank;
            ranks.push_back(myRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, 0);
                ranks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it, 0);
            }

            localRanks.clear();
            localRanks.push_back(myLocalRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, 0);
                localRanks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it, 0);
            }
        }
        else
        {
            LOCAL_COMM_SESSION.sendValue(myRank, *group.begin(), 0);
            LOCAL_COMM_SESSION.recv(ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *group.begin(), 0);

            LOCAL_COMM_SESSION.sendValue(myLocalRank, *group.begin(), 0);
            LOCAL_COMM_SESSION.recv(
                localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *group.begin(), 0);
        }
    }

    std::set<int> localGroup;
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        auto rank = ranks[i];
        if (group.find(rank) != group.end())
        {
            localGroup.insert(localRanks[i]);
        }
    }
    return localGroup;
}

void AllreducePlugin::initGroupTopology() noexcept
{
    static std::map<std::set<int>, std::tuple<bool, bool>> cache;
    if (cache.find(mGroup) != cache.end())
    {
        auto [isNVLINKSupported, isP2PSupported] = cache[mGroup];
        mIsNVLINKSupported = isNVLINKSupported;
        mIsP2PSupported = isP2PSupported;
        return;
    }
    setGroupTopology();
    cache[mGroup] = {mIsNVLINKSupported, mIsP2PSupported};
}

void AllreducePlugin::setGroupTopology() noexcept
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
    std::set<int> localGroup = getLocalGroup(mGroup);
    if (mGroup.size() != localGroup.size())
    {
        mIsP2PSupported = false;
        mIsNVLINKSupported = false;
        TLLM_LOG_INFO("Found inter-node TP group for rank %d", rank);
        return;
    }
    TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);

    NvmlManager nvmlManager;
    std::unordered_set<int> visitedDevice;
    mIsP2PSupported = true;
    mIsNVLINKSupported = true;

    // Use cudaDeviceCanAccessPeer to determine whether p2p is supported,
    // and use nvml to determine whether there are nvlink links between ranks.
    for (int firstDeviceId : localGroup)
    {
        for (int secondDeviceId : localGroup)
        {
            if (firstDeviceId == secondDeviceId || visitedDevice.find(secondDeviceId) != visitedDevice.end())
            {
                continue;
            }

            int canAccessPeer = 0;
            TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, firstDeviceId, secondDeviceId));

            if (!canAccessPeer)
            {
                mIsP2PSupported = false;
                mIsNVLINKSupported = false;

                return;
            }

            nvmlDevice_t firstDevice;
            NVML_CHECK(nvmlDeviceGetHandleByIndex(firstDeviceId, &firstDevice));

            bool isNVLINK = false;

            for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++)
            {
                nvmlPciInfo_t remotePciInfo;
                if (nvmlDeviceGetNvLinkRemotePciInfo_v2(firstDevice, link, &remotePciInfo) != NVML_SUCCESS)
                {
                    continue;
                }

                nvmlDevice_t remoteDevice;
                auto const result = nvmlDeviceGetHandleByPciBusId_v2(remotePciInfo.busId, &remoteDevice);

                if (result == NVML_SUCCESS)
                {
                    // Two GPUs are connected directly through nvlink
                    unsigned int remoteDeviceId;
                    NVML_CHECK(nvmlDeviceGetIndex(remoteDevice, &remoteDeviceId));

                    if (remoteDeviceId == secondDeviceId)
                    {
                        isNVLINK = true;
                    }
                }
                else if (result == NVML_ERROR_NOT_FOUND)
                {
                    // Maybe Two GPUs are connected via nvswitch,
                    // now remotePciInfo represents the pci information of nvswitch,
                    // determine whether nvlink is supported by whether two GPUs are connected to the same nvswitch.
                    nvmlDevice_t secondDevice;
                    NVML_CHECK(nvmlDeviceGetHandleByIndex(secondDeviceId, &secondDevice));

                    for (unsigned int secondLink = 0; secondLink < NVML_NVLINK_MAX_LINKS; secondLink++)
                    {
                        nvmlPciInfo_t secondRemotePciInfo;
                        if (nvmlDeviceGetNvLinkRemotePciInfo_v2(secondDevice, secondLink, &secondRemotePciInfo)
                            != NVML_SUCCESS)
                        {
                            continue;
                        }

                        if (strcmp(remotePciInfo.busId, secondRemotePciInfo.busId) == 0)
                        {
                            isNVLINK = true;
                            break;
                        }
                    }
                }
                else
                {
                    NVML_CHECK(result);
                }

                if (isNVLINK)
                {
                    break;
                }
            }

            mIsNVLINKSupported &= isNVLINK;
        }
        visitedDevice.insert(firstDeviceId);
    }
}

int AllreducePlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    mNcclComm = getComm(mGroup);
    if (mStrategy != AllReduceStrategyType::NCCL)
    {
        initGroupTopology();
    }

    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    return 0;
}

void AllreducePlugin::terminate() noexcept {}

size_t AllreducePlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType) + sizeof(mStrategy) + sizeof(mConfig) + sizeof(mOp)
        + sizeof(mEps) + sizeof(mAffine) + sizeof(mBias);
}

void AllreducePlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mStrategy);
    write(d, mConfig);
    write(d, mOp);
    write(d, mEps);
    write(d, mAffine);
    write(d, mBias);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void AllreducePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

AllreducePluginCreator::AllreducePluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("strategy", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("config", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("fusion_op", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("counter", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("affine", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kINT8, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AllreducePluginCreator::getPluginName() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

char const* AllreducePluginCreator::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

PluginFieldCollection const* AllreducePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AllreducePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    AllReduceStrategyType strategy;
    AllReduceStrategyConfig config;
    AllReduceFusionOp fusion_op;
    int32_t counter;
    float eps;
    int8_t affine;
    int8_t bias;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            auto const* r = static_cast<int const*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "strategy"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            strategy = static_cast<AllReduceStrategyType>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "config"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            config = static_cast<AllReduceStrategyConfig>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "fusion_op"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            fusion_op = static_cast<AllReduceFusionOp>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "counter"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            counter = *static_cast<int32_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "eps"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = *static_cast<float const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "affine"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            affine = *static_cast<int8_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "bias"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            bias = *static_cast<int8_t const*>(fields[i].data);
        }
    }

    try
    {
        auto* obj = new AllreducePlugin(group, type, strategy, config, fusion_op, counter, eps, affine, bias);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AllreducePluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call AllreducePlugin::destroy()
    try
    {
        auto* obj = new AllreducePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
