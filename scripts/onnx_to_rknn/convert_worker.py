#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX转RKNN转换工作脚本
在虚拟环境中运行，通过命令行参数接收配置

用法：
    python convert_worker.py --onnx <path> --output <path> --config <json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from rknn.api import RKNN
except ImportError:
    print("[ERROR] rknn-toolkit未安装")
    sys.exit(1)


CHIP_PLATFORMS = {
    "RK1808": {"target": "rk1808"},
    "RV1109": {"target": "rv1109"},
    "RV1126": {"target": "rv1126"},
    "RK3399Pro": {"target": "rk3399pro"},
    "RK3562": {"target": "rk3562"},
    "RK3566": {"target": "rk3566"},
    "RK3568": {"target": "rk3568"},
    "RK3576": {"target": "rk3576"},
    "RK3588": {"target": "rk3588"},
}


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    config: dict
) -> bool:
    """
    执行ONNX到RKNN转换

    Args:
        onnx_path: ONNX模型路径
        output_path: 输出RKNN路径
        config: 转换配置

    Returns:
        成功返回True
    """
    platform = config.get("platform", "")

    if platform not in CHIP_PLATFORMS:
        print(f"[ERROR] 不支持的平台: {platform}")
        return False

    target_platform = CHIP_PLATFORMS[platform]["target"]

    try:
        rknn = RKNN()

        # 构建配置参数
        rknn_config = {"target_platform": target_platform}

        # 预处理参数
        mean_values = config.get("mean_values")
        if mean_values:
            rknn_config["mean_values"] = mean_values
            print(f"[LOG] 均值归一化: {mean_values}")

        std_values = config.get("std_values")
        if std_values:
            rknn_config["std_values"] = std_values
            print(f"[LOG] 标准差归一化: {std_values}")

        # 量化参数 - RKNN Toolkit 2 支持的量化类型: w8a8, w8a16, w16a16i, w16a16i_dfp, w4a16
        do_quantization = config.get("do_quantization", False)
        if do_quantization:
            # 旧版量化类型映射到RKNN Toolkit 2格式
            old_to_new_dtype = {
                "asymmetric_quantized-u8": "w8a8",
                "asymmetric_quantized-i8": "w8a8",
                "quantized-u8": "w8a8",
                "quantized-i8": "w8a8",
                "w8a8": "w8a8",
                "w8a16": "w8a16",
                "w16a16i": "w16a16i",
                "w16a16i_dfp": "w16a16i_dfp",
                "w4a16": "w4a16",
            }
            quantized_dtype_input = config.get("quantized_dtype", "w8a8")
            quantized_dtype = old_to_new_dtype.get(quantized_dtype_input, "w8a8")
            quantized_algorithm = config.get("quantized_algorithm", "normal")
            rknn_config["quantized_dtype"] = quantized_dtype
            rknn_config["quantized_algorithm"] = quantized_algorithm
            print(f"[LOG] 量化类型: {quantized_dtype} (输入: {quantized_dtype_input})")
            print(f"[LOG] 量化算法: {quantized_algorithm}")

        # 优化参数
        optimization_level = config.get("optimization_level", 2)
        rknn_config["optimization_level"] = optimization_level
        print(f"[LOG] 优化级别: {optimization_level}")

        single_core_mode = config.get("single_core_mode", False)
        if single_core_mode:
            rknn_config["single_core_mode"] = True
            print("[LOG] 单核模式: 启用")

        model_data_size = config.get("model_data_size")
        if model_data_size:
            rknn_config["model_data_size"] = model_data_size
            print(f"[LOG] 模型数据大小限制: {model_data_size}")

        print(f"[LOG] 目标平台: {platform} ({target_platform})")
        print(f"[LOG] 配置参数: {rknn_config}")

        # 应用配置
        ret = rknn.config(**rknn_config)
        if ret != 0:
            print(f"[ERROR] 配置失败: ret={ret}")
            return False
        print("[LOG] 配置成功")

        # 加载ONNX模型
        load_args = {"model": onnx_path}

        input_name = config.get("input_name")
        if input_name:
            load_args["inputs"] = [input_name]
            print(f"[LOG] 输入节点: {input_name}")

        # 输入尺寸和数据类型 - 这些参数会真正生效
        input_size = config.get("input_size")
        input_dtype = config.get("input_dtype", "float32")
        if input_size:
            # input_size格式: [height, width]，需要添加channel
            # 根据数据类型推断channel数
            if input_dtype == "float32" or input_dtype == "float16":
                channel = 3  # RGB/BGR
            elif input_dtype == "uint8":
                channel = 3  # 通常也是3通道
            else:
                channel = 3
            
            # RKNN API要求input_size_list格式: [height, width, channel]
            full_size = [input_size[0], input_size[1], channel]
            load_args["input_size_list"] = [full_size]
            print(f"[LOG] 输入尺寸: {input_size[0]}x{input_size[1]}x{channel} (HxWxC)")
        
        # 输入数据类型 - 通过inputs参数指定
        if input_dtype and input_dtype != "float32":
            # RKNN支持的数据类型: float32, float16, uint8, int8
            # 需要通过inputs参数指定每个输入的类型
            input_name_default = config.get("input_name", "input")
            load_args["input_dtype"] = input_dtype
            print(f"[LOG] 输入数据类型: {input_dtype}")

        print(f"[LOG] 加载ONNX: {onnx_path}")
        print(f"[LOG] load_onnx参数: {load_args}")
        ret = rknn.load_onnx(**load_args)
        if ret != 0:
            print(f"[ERROR] 加载ONNX失败: ret={ret}")
            return False
        print("[LOG] 加载ONNX成功")

        # 构建
        build_args = {"do_quantization": do_quantization}

        dataset_path = config.get("dataset_path")
        if do_quantization and dataset_path:
            build_args["do_quantization"] = True
            build_args["dataset"] = dataset_path
            print(f"[LOG] 启用量化，数据集: {dataset_path}")
        else:
            print("[LOG] 不启用量化")

        batch_size = config.get("batch_size", 1)
        if batch_size and batch_size > 1:
            build_args["rknn_batch_size"] = batch_size
            print(f"[LOG] 批次大小: {batch_size}")

        print("[LOG] 开始构建RKNN模型...")
        ret = rknn.build(**build_args)
        if ret != 0:
            print(f"[ERROR] 构建失败: ret={ret}")
            return False
        print("[LOG] 构建成功")

        # 导出RKNN
        print(f"[LOG] 导出RKNN: {output_path}")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            print(f"[ERROR] 导出RKNN失败: ret={ret}")
            return False

        # 获取文件大小
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[SIZE] {output_size:.2f}")

        rknn.release()

        print(f"[SUCCESS] {output_path}")
        return True

    except Exception as e:
        import traceback
        print(f"[ERROR] 转换异常: {str(e)}")
        print(f"[ERROR] {traceback.format_exc()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ONNX转RKNN转换工作脚本")
    parser.add_argument("--onnx", required=True, help="ONNX模型路径")
    parser.add_argument("--output", required=True, help="输出RKNN路径")
    parser.add_argument("--config", required=True, help="配置JSON字符串")

    args = parser.parse_args()

    # 解析配置
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(f"[ERROR] 配置JSON解析失败: {e}")
        sys.exit(1)

    print(f"[LOG] ONNX路径: {args.onnx}")
    print(f"[LOG] 输出路径: {args.output}")

    # 执行转换
    success = convert_onnx_to_rknn(
        onnx_path=args.onnx,
        output_path=args.output,
        config=config
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()