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
    onnxPath: str,
    outputPath: str,
    config: dict
) -> bool:
    """
    执行ONNX到RKNN转换
    
    Args:
        onnxPath: ONNX模型路径
        outputPath: 输出RKNN路径
        config: 转换配置
        
    Returns:
        成功返回True
    """
    platform = config.get("platform", "")
    
    if platform not in CHIP_PLATFORMS:
        print(f"[ERROR] 不支持的平台: {platform}")
        return False
        
    targetPlatform = CHIP_PLATFORMS[platform]["target"]
    
    try:
        rknn = RKNN()
        
        # 构建配置参数
        rknnConfig = {"target_platform": targetPlatform}
        
        # 预处理参数
        meanValues = config.get("meanValues")
        if meanValues:
            rknnConfig["mean_values"] = meanValues
            print(f"[LOG] 均值归一化: {meanValues}")
            
        stdValues = config.get("stdValues")
        if stdValues:
            rknnConfig["std_values"] = stdValues
            print(f"[LOG] 标准差归一化: {stdValues}")
            
        # 量化参数
        doQuantization = config.get("doQuantization", False)
        if doQuantization:
            quantizedDtype = config.get("quantizedDtype", "asymmetric_quantized-u8")
            quantizedAlgorithm = config.get("quantizedAlgorithm", "normal")
            rknnConfig["quantized_dtype"] = quantizedDtype
            rknnConfig["quantized_algorithm"] = quantizedAlgorithm
            print(f"[LOG] 量化类型: {quantizedDtype}")
            print(f"[LOG] 量化算法: {quantizedAlgorithm}")
            
        # 优化参数
        optimizationLevel = config.get("optimizationLevel", 2)
        rknnConfig["optimization_level"] = optimizationLevel
        print(f"[LOG] 优化级别: {optimizationLevel}")
        
        singleCoreMode = config.get("singleCoreMode", False)
        if singleCoreMode:
            rknnConfig["single_core_mode"] = True
            print("[LOG] 单核模式: 启用")
            
        modelDataSize = config.get("modelDataSize")
        if modelDataSize:
            rknnConfig["model_data_size"] = modelDataSize
            print(f"[LOG] 模型数据大小限制: {modelDataSize}")
            
        print(f"[LOG] 目标平台: {platform} ({targetPlatform})")
        print(f"[LOG] 配置参数: {rknnConfig}")
        
        # 应用配置
        ret = rknn.config(**rknnConfig)
        if ret != 0:
            print(f"[ERROR] 配置失败: ret={ret}")
            return False
        print("[LOG] 配置成功")
        
        # 加载ONNX模型
        loadArgs = {"model": onnxPath}
        
        inputName = config.get("inputName")
        if inputName:
            loadArgs["inputs"] = [inputName]
            print(f"[LOG] 输入节点: {inputName}")
            
        inputSize = config.get("inputSize")
        if inputSize:
            loadArgs["input_size_list"] = [tuple(inputSize)]
            print(f"[LOG] 输入尺寸: {inputSize}")
            
        inputDtype = config.get("inputDtype", "float32")
        if inputDtype:
            loadArgs["input_dtype_list"] = [inputDtype]
            print(f"[LOG] 输入数据类型: {inputDtype}")
            
        print(f"[LOG] 加载ONNX: {onnxPath}")
        ret = rknn.load_onnx(**loadArgs)
        if ret != 0:
            print(f"[ERROR] 加载ONNX失败: ret={ret}")
            return False
        print("[LOG] 加载ONNX成功")
        
        # 构建
        buildArgs = {"do_quantization": False}
        
        datasetPath = config.get("datasetPath")
        if doQuantization and datasetPath:
            buildArgs["do_quantization"] = True
            buildArgs["dataset"] = datasetPath
            print(f"[LOG] 启用量化，数据集: {datasetPath}")
        else:
            print("[LOG] 不启用量化")
            
        batchSize = config.get("batchSize", 1)
        if batchSize and batchSize > 1:
            buildArgs["rknn_batch_size"] = batchSize
            print(f"[LOG] 批次大小: {batchSize}")
            
        print("[LOG] 开始构建RKNN模型...")
        ret = rknn.build(**buildArgs)
        if ret != 0:
            print(f"[ERROR] 构建失败: ret={ret}")
            return False
        print("[LOG] 构建成功")
        
        # 导出RKNN
        print(f"[LOG] 导出RKNN: {outputPath}")
        ret = rknn.export_rknn(outputPath)
        if ret != 0:
            print(f"[ERROR] 导出RKNN失败: ret={ret}")
            return False
            
        # 获取文件大小
        outputSize = Path(outputPath).stat().st_size / (1024 * 1024)
        print(f"[SIZE] {outputSize:.2f}")
        
        rknn.release()
        
        print(f"[SUCCESS] {outputPath}")
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
        onnxPath=args.onnx,
        outputPath=args.output,
        config=config
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()