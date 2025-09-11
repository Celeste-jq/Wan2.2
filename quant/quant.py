import contextlib
import json
import os
from types import SimpleNamespace
import torch
import torch.nn as nn
import json
import os
import torch
import torch_npu
import sys

torch_npu.npu.set_compile_mode(jit_compile=False)

from torch_npu.contrib import transfer_to_npu
from torch import distributed as dist


MODEL_TYPE = "wan2"     # msmodelslim中注册的模型名称


@contextlib.contextmanager
def generate_calib_dataset(quant_mode, model, quant_data_dir):
    """
    with generate_calib_dataset(model, save_path):
        images = pipe(
            prompts,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images
    """
    if quant_mode != 1 or int(os.getenv("RANK", "0")) != 0:
        yield
        return

    if os.path.islink(quant_data_dir):
        raise Exception(f"The directory {quant_data_dir} is link!")

    os.makedirs(quant_data_dir, exist_ok=True)

    save_path = os.path.join(quant_data_dir, "calib_data.pth")
    print(f"generate calib dataset: {save_path}")
    from quant.quant_utils import DumperManager
    dumper_mgr = DumperManager(model, capture_mode='args')
    try:
        yield
    finally:
        dumper_mgr.save(save_path)


def quantize_weight(model, weight_save_path, calib_dataset_path=None, calib_dataset_num=300):
    print(f"quantize weights function begin")
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
    from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
    from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import input_to_cpu

    # if os.path.islink(weight_save_path):
    #     raise Exception(f"The directory {weight_save_path} is link!")

    os.makedirs(weight_save_path, exist_ok=True)

    # 如果使用npu进行量化需开启二进制编译，避免在线编译算子
    torch.npu.set_compile_mode(jit_compile=False)

    torch.npu.empty_cache()
    # model = model.to("npu")
    print(f"model.dtype: {model.dtype}, dir(model): {dir(model)}")

    # 由于llm ptq接口限制，模型补充dtype属性
    if not hasattr(model, 'config'):
        model.config = SimpleNamespace()  # 使用轻量级命名空间
    if not hasattr(model.config, 'torch_dtype'):
        model.config.torch_dtype = torch.bfloat16
    if not hasattr(model, "dtype"):
        model.dtype = torch.bfloat16
    # if not hasattr(model.config, 'model_type'):
    model.config.model_type = MODEL_TYPE

    # # if calib_dataset_path is not None and os.path.exists(calib_dataset_path):
    # #     model = model.to("npu")
    # # if False:
    #     calib_dataset = torch.load(calib_dataset_path, map_location=model.device, weights_only=False)
    #     calib_dataset = calib_dataset[:calib_dataset_num]
        # calib_dataset = input_to_cpu(calib_dataset)
        # print(f"calib_dataset_path: {calib_dataset_path}, calib_dataset: {len(calib_dataset)}")

        # print(f"calib_dataset: {calib_dataset[0][0][0].device}, "
        #       f"calib_dataset: {calib_dataset[0][1][0:1].device}"
        #       f"calib_dataset: {calib_dataset[0][2][0].device}, "
        #       f"model_type: {model.config.model_type}")
        # anti_config = AntiOutlierConfig(w_bit=8, a_bit=8, anti_method="m6", dev_type="npu", dev_id=None, w_sym=True)
        # anti_outlier = AntiOutlier(model, calib_data=calib_dataset, cfg=anti_config)
        # anti_outlier.process()

    # disable_name_list = ['transformer.encoder.layers.0.self_attention.query_key_value',
    #                      'transformer.encoder.layers.0.self_attention.dense',
    #                      'transformer.encoder.layers.0.mlp.dense_h_to_4h']
    # 填写量化配置
    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        # disable_names=disable_name_list,
        dev_type='cpu',
        # dev_id=rank,
        act_method=3,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=True
    )

    print(f"weight_save_path: {weight_save_path}.")
    # 2.执行校准，不需要校准数据的场景不需要传calib_data
    calibrator = Calibrator(model, quant_config, disable_level='L0')  # disable_level: 自动回退n个linear
    calibrator.run()  # 执行PTQ量化校准
    # 3.多卡场景需要保存多个权重,按当前rank去区分名称
    if int(os.getenv("RANK", "0")) == 0:
        calibrator.save(weight_save_path, save_type=["safe_tensor"])
    print(f"quantize weights function end")


if __name__ == "__main__":
    with generate_calib_dataset(1, "1", "2") as f:
        print("data")
        print(f"result: {f}")