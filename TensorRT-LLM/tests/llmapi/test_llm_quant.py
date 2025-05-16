import os
import sys

from tensorrt_llm.llmapi.llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_utils import CalibConfig, QuantAlgo, QuantConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_pre_ampere, skip_pre_hopper

try:
    from .test_llm import cnn_dailymail_path, llama_model_path
except ImportError:
    from test_llm import cnn_dailymail_path, llama_model_path


@skip_pre_ampere
def test_llm_int4_awq_quantization():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
    assert quant_config.quant_mode.has_any_quant()
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)

    llm = LLM(llama_model_path,
              quant_config=quant_config,
              calib_config=calib_config)

    sampling_params = SamplingParams(max_tokens=6)
    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I"


@skip_pre_hopper
def test_llm_fp8_quantization():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)

    llm = LLM(llama_model_path,
              quant_config=quant_config,
              calib_config=calib_config)
    sampling_params = SamplingParams(max_tokens=6)
    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I"


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization()
