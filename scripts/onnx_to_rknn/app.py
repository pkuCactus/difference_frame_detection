#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX转RKNN Web转换工具
支持上传ONNX模型并转换为瑞芯微不同芯片平台的RKNN模型

自动虚拟环境管理：
    - 根据选择的芯片平台自动创建对应的虚拟环境
    - rknn-toolkit: venv_toolkit (RK1808, RV1109, RV1126, RK3399Pro)
    - rknn-toolkit2: venv_toolkit2 (RK3562, RK3566, RK3568, RK3576, RK3588)
    - 通过 subprocess 在对应虚拟环境中执行转换

使用方法：
    pip install flask onnx
    python app.py
    
    访问 http://localhost:5000
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import venv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for

# 配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "datasets"
# 虚拟环境目录使用 TMPDIR
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
VENV_DIR = TMPDIR / "onnx_to_rknn_venvs"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
VENV_DIR.mkdir(parents=True, exist_ok=True)

import platform

# 检测系统架构和Python版本标签
def get_system_info():
    """获取系统架构信息"""
    machine = platform.machine().lower()
    system = platform.system().lower()
    
    # 架构映射
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "armv7l": "armv7l",
    }
    arch = arch_map.get(machine, machine)
    
    return {
        "system": system,
        "arch": arch,
        "machine": machine,
    }

SYSTEM_INFO = get_system_info()

# Python版本到cp标签映射
def get_python_cp_tag(py_version: str) -> str:
    """Python版本转cp标签，如 3.10 -> cp310"""
    major, minor = py_version.split(".")[:2]
    return f"cp{major}{minor}"

# Toolkit与虚拟环境映射
TOOLKIT_VENV_MAP = {
    "rknn-toolkit": {
        "venvName": "venv_toolkit",
        "packageName": "rknn-toolkit",
        "pythonVersion": "3.8",
        "installMode": "tar.gz",
        "tarUrl": "https://github.com/rockchip-linux/rknn-toolkit/releases/download/v{version}/rknn-toolkit-v{version}-packages.tar.gz",
        "versions": ["1.7.5", "1.7.3", "1.7.1", "1.7.0", "1.6.1", "1.6.0"],
        "note": "下载tar.gz解压后按Python版本和架构选择whl安装",
    },
    "rknn-toolkit2": {
        "venvName": "venv_toolkit2",
        "packageName": "rknn-toolkit2",
        "pythonVersion": "3.12",
        "installMode": "whl",
        "whlBaseUrl": "https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages",
        "versions": ["2.3.2", "2.2.0", "2.1.0", "2.0.0"],
        "note": "直接从GitHub下载对应版本whl",
    },
}

# RKNN Toolkit whl 文件目录（用户需要将下载的 whl 文件放到此目录）
WHL_DIR = BASE_DIR / "whl_files"
WHL_DIR.mkdir(parents=True, exist_ok=True)

# 支持的芯片平台
CHIP_PLATFORMS = {
    # rknn-toolkit (旧版) - Python 3.8
    "RK1808": {"toolkit": "rknn-toolkit", "target": "rk1808", "description": "RK1808 NPU"},
    "RV1109": {"toolkit": "rknn-toolkit", "target": "rv1109", "description": "RV1109 ISP"},
    "RV1126": {"toolkit": "rknn-toolkit", "target": "rv1126", "description": "RV1126 ISP"},
    "RK3399Pro": {"toolkit": "rknn-toolkit", "target": "rk3399pro", "description": "RK3399Pro NPU"},
    # rknn-toolkit2 (新版) - Python 3.10/3.11
    "RV1103": {"toolkit": "rknn-toolkit2", "target": "rv1103", "description": "RV1103 ISP"},
    "RV1106": {"toolkit": "rknn-toolkit2", "target": "rv1106", "description": "RV1106 ISP"},
    "RV1103B": {"toolkit": "rknn-toolkit2", "target": "rv1103b", "description": "RV1103B ISP"},
    "RV1106B": {"toolkit": "rknn-toolkit2", "target": "rv1106b", "description": "RV1106B ISP"},
    "RV1126B": {"toolkit": "rknn-toolkit2", "target": "rv1126b", "description": "RV1126B ISP"},
    "RK2118": {"toolkit": "rknn-toolkit2", "target": "rk2118", "description": "RK2118 NPU"},
    "RK3562": {"toolkit": "rknn-toolkit2", "target": "rk3562", "description": "RK3562"},
    "RK3566": {"toolkit": "rknn-toolkit2", "target": "rk3566", "description": "RK3566"},
    "RK3568": {"toolkit": "rknn-toolkit2", "target": "rk3568", "description": "RK3568"},
    "RK3576": {"toolkit": "rknn-toolkit2", "target": "rk3576", "description": "RK3576"},
    "RK3588": {"toolkit": "rknn-toolkit2", "target": "rk3588", "description": "RK3588"},
}

# 量化数据类型选项
QUANTIZED_DTYPES = {
    "asymmetric_quantized-u8": "非对称量化 uint8 (推荐)",
    "asymmetric_quantized-i8": "非对称量化 int8",
    "dynamic_fixed_point-8": "动态定点 8bit",
    "dynamic_fixed_point-16": "动态定点 16bit",
}

# 量化算法选项
QUANTIZED_ALGORITHMS = {
    "normal": "普通量化 (快速)",
    "mm": "MinMax量化",
    "kl_divergence": "KL散度量化 (精度高，速度慢)",
}

# 优化级别
OPTIMIZATION_LEVELS = {
    1: "基本优化",
    2: "标准优化 (推荐)",
    3: "激进优化 (可能影响精度)",
}


@dataclass
class ConversionConfig:
    """转换配置参数"""
    platform: str = ""
    inputSize: Optional[tuple[int, int]] = None
    inputName: Optional[str] = None
    meanValues: Optional[list[float]] = None
    stdValues: Optional[list[float]] = None
    inputDtype: str = "float32"
    doQuantization: bool = False
    quantizedDtype: str = "asymmetric_quantized-u8"
    quantizedAlgorithm: str = "normal"
    datasetPath: Optional[str] = None
    optimizationLevel: int = 2
    singleCoreMode: bool = False
    modelDataSize: Optional[int] = None
    batchSize: int = 1
    
    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "inputSize": list(self.inputSize) if self.inputSize else None,
            "inputName": self.inputName,
            "meanValues": self.meanValues,
            "stdValues": self.stdValues,
            "inputDtype": self.inputDtype,
            "doQuantization": self.doQuantization,
            "quantizedDtype": self.quantizedDtype,
            "quantizedAlgorithm": self.quantizedAlgorithm,
            "datasetPath": self.datasetPath,
            "optimizationLevel": self.optimizationLevel,
            "singleCoreMode": self.singleCoreMode,
            "modelDataSize": self.modelDataSize,
            "batchSize": self.batchSize,
        }


class ConversionTask:
    """转换任务"""

    def __init__(self, taskId: str, onnxPath: str, config: ConversionConfig):
        self.taskId = taskId
        self.onnxPath = onnxPath
        self.config = config
        self.outputPath: Optional[str] = None
        self.status = "pending"
        self.message = ""
        self.startTime = datetime.now()
        self.endTime: Optional[datetime] = None
        self.log: list[str] = []
        self.process: Optional[subprocess.Popen] = None

    def add_log(self, msg: str):
        """实时添加日志，支持并发安全"""
        self.log.append(msg)
        print(f"[{self.taskId}] {msg}")


class VirtualEnvManager:
    """虚拟环境管理器"""
    
    def __init__(self, venvBaseDir: Path = VENV_DIR):
        self.venvBaseDir = venvBaseDir
        self.envStatus: dict[str, dict] = {}
        self._check_all_envs()
        
    def _get_venv_path(self, toolkitType: str) -> Path:
        """获取虚拟环境路径"""
        venvName = TOOLKIT_VENV_MAP[toolkitType]["venvName"]
        return self.venvBaseDir / venvName
        
    def _get_python_path(self, toolkitType: str) -> Path:
        """获取虚拟环境中的Python路径"""
        venvPath = self._get_venv_path(toolkitType)
        if sys.platform == "win32":
            return venvPath / "Scripts" / "python.exe"
        return venvPath / "bin" / "python"
        
    def _get_pip_path(self, toolkitType: str) -> Path:
        """获取虚拟环境中的pip路径"""
        venvPath = self._get_venv_path(toolkitType)
        if sys.platform == "win32":
            return venvPath / "Scripts" / "pip.exe"
        return venvPath / "bin" / "pip"

    def _get_uv_path(self) -> Optional[str]:
        """查找 uv 可执行文件路径"""
        uvPath = shutil.which("uv")
        if uvPath:
            try:
                result = subprocess.run([uvPath, "--version"],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return uvPath
            except Exception:
                pass
        return None

    def _ensure_uv_installed(self) -> tuple[bool, str]:
        """确保 uv 已安装，未安装则自动安装"""
        uvPath = self._get_uv_path()
        if uvPath:
            return True, f"uv 已安装: {uvPath}"

        print("\n[ensure_uv] uv 未找到，开始自动安装...")

        # 方法1: 通过官方脚本安装
        try:
            print("  -> 尝试通过官方脚本安装 uv...")
            result = subprocess.run(
                ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print("  -> 下载安装脚本成功，执行安装...")
                installResult = subprocess.run(
                    ["sh"],
                    input=result.stdout,
                    capture_output=True, text=True, timeout=120
                )
                if installResult.returncode == 0:
                    print("  -> 脚本安装完成")
                    # 刷新 PATH 缓存
                    import importlib
                    importlib.reload(shutil)
                    uvPath = shutil.which("uv")
                    if uvPath:
                        print(f"  ✓ uv 安装成功: {uvPath}")
                        return True, f"uv 安装成功: {uvPath}"
                    else:
                        # uv 可能被安装到 ~/.local/bin，需要添加到 PATH
                        localBin = Path.home() / ".local" / "bin"
                        if localBin.exists() and (localBin / "uv").exists():
                            os.environ["PATH"] = str(localBin) + os.pathsep + os.environ.get("PATH", "")
                            uvPath = shutil.which("uv")
                            if uvPath:
                                print(f"  ✓ uv 安装成功 (添加到 PATH): {uvPath}")
                                return True, f"uv 安装成功: {uvPath}"
            else:
                print(f"  ✗ 下载脚本失败: {result.stderr[:200]}")
        except Exception as e:
            print(f"  ✗ 脚本安装异常: {e}")

        # 方法2: 通过 pip 安装
        try:
            print("  -> 尝试通过 pip 安装 uv...")
            pipCmds = ["pip", "pip3", sys.executable + " -m pip"]
            for pipCmd in pipCmds:
                parts = pipCmd.split() if " " in pipCmd else [pipCmd]
                result = subprocess.run(
                    parts + ["install", "uv"],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    uvPath = shutil.which("uv")
                    if uvPath:
                        print(f"  ✓ uv 通过 pip 安装成功: {uvPath}")
                        return True, f"uv 安装成功: {uvPath}"
        except Exception as e:
            print(f"  ✗ pip 安装异常: {e}")

        return False, "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"

    def _ensure_python_installed(self, pyVersion: str) -> tuple[bool, str]:
        """使用 uv 确保指定 Python 版本已安装"""
        uvPath = self._get_uv_path()
        if not uvPath:
            return False, "uv 未安装，无法安装 Python"

        pyCmdName = f"python{pyVersion}"
        print(f"\n[ensure_python] 确保 {pyCmdName} 已安装...")

        # 检查是否已存在
        pyCmd = shutil.which(pyCmdName)
        if pyCmd:
            try:
                result = subprocess.run([pyCmd, "--version"],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and pyVersion in result.stdout:
                    print(f"  ✓ {pyCmdName} 已存在: {pyCmd}")
                    return True, f"{pyCmdName} 已存在"
            except Exception:
                pass

        # 使用 uv 安装
        try:
            print(f"  -> 使用 uv 安装 {pyVersion}...")
            result = subprocess.run(
                [uvPath, "python", "install", pyVersion],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"  ✓ {pyVersion} 安装成功")
                return True, f"{pyVersion} 安装成功"
            else:
                err = result.stderr.strip()[:300] if result.stderr else "未知错误"
                print(f"  ✗ {pyVersion} 安装失败: {err}")
                return False, f"{pyVersion} 安装失败: {err}"
        except subprocess.TimeoutExpired:
            print(f"  ✗ {pyVersion} 安装超时")
            return False, f"{pyVersion} 安装超时"
        except Exception as e:
            print(f"  ✗ {pyVersion} 安装异常: {e}")
            return False, f"{pyVersion} 安装异常: {str(e)}"

    def _find_python_executable(self, pyVersion: str) -> Optional[str]:
        """查找指定版本的 Python 可执行文件"""
        """查找指定版本的 Python 可执行文件"""
        major, minor = pyVersion.split(".")[:2]
        candidates = [
            f"python{major}.{minor}",
            f"python{major}{minor}",
        ]
        if sys.platform == "win32":
            candidates.insert(0, f"py -{major}.{minor}")

        for cmd in candidates:
            pythonPath = shutil.which(cmd)
            if pythonPath:
                try:
                    result = subprocess.run(
                        [pythonPath, "--version"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and pyVersion in result.stdout:
                        return pythonPath
                except Exception:
                    pass

        # 回退到当前 Python（仅当版本匹配时）
        currentVersion = f"{sys.version_info.major}.{sys.version_info.minor}"
        if currentVersion == pyVersion:
            return sys.executable
        return None

    def _check_all_envs(self):
        """检查所有虚拟环境状态"""
        for toolkitType, info in TOOLKIT_VENV_MAP.items():
            venvPath = self._get_venv_path(toolkitType)
            pythonPath = self._get_python_path(toolkitType)
            
            exists = venvPath.exists()
            pythonExists = pythonPath.exists()
            
            # 检查rknn是否已安装
            rknnInstalled = False
            if pythonExists:
                try:
                    result = subprocess.run(
                        [str(pythonPath), "-c", "import rknn; print('ok')"],
                        capture_output=True,
                        timeout=5
                    )
                    rknnInstalled = result.returncode == 0
                except Exception:
                    rknnInstalled = False
                    
            self.envStatus[toolkitType] = {
                "venvPath": str(venvPath),
                "pythonPath": str(pythonPath),
                "exists": exists,
                "pythonReady": pythonExists,
                "rknnInstalled": rknnInstalled,
                "packageName": info["packageName"],
            }
            
    def get_status(self, toolkitType: str) -> dict:
        """获取指定toolkit的虚拟环境状态"""
        return self.envStatus.get(toolkitType, {})
        
    def get_all_status(self) -> dict:
        """获取所有虚拟环境状态"""
        return self.envStatus
        
    def create_venv(self, toolkitType: str) -> tuple[bool, str]:
        """创建虚拟环境（优先使用 uv，回退到标准 venv）"""
        venvPath = self._get_venv_path(toolkitType)
        toolkitInfo = TOOLKIT_VENV_MAP[toolkitType]
        requiredPyVersion = toolkitInfo["pythonVersion"]
        pyCmdName = f"python{requiredPyVersion}"

        print(f"\n[create_venv] toolkit={toolkitType}, requiredPy={requiredPyVersion}")

        if venvPath.exists() and self._get_python_path(toolkitType).exists():
            print(f"  -> 虚拟环境已存在且完整: {venvPath}")
            return True, f"虚拟环境已存在: {venvPath}"

        # 如果目录存在但环境不完整，删除重建
        if venvPath.exists():
            print(f"  -> 清理不完整的环境: {venvPath}")
            shutil.rmtree(venvPath)
            print(f"  -> 清理完成")

        # 优先尝试 uv
        uvPath = self._get_uv_path()
        if uvPath:
            print(f"  -> 使用 uv 创建虚拟环境: {uvPath}")
            try:
                cmd = [uvPath, "venv", str(venvPath), "--python", pyCmdName]
                print(f"  -> 执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                print(f"  -> uv venv 返回码: {result.returncode}")
                if result.stdout:
                    print(f"  -> stdout: {result.stdout.strip()[:200]}")
                if result.returncode != 0:
                    err = result.stderr.strip()[:300] if result.stderr else "无错误输出"
                    print(f"  ✗ uv venv 失败: {err}")
                    print(f"  -> 回退到标准 venv...")
                else:
                    print(f"  -> uv venv 创建成功")
                    pythonPath = self._get_python_path(toolkitType)
                    if pythonPath.exists():
                        print(f"  -> Python 可执行文件确认存在: {pythonPath}")
                        self._check_all_envs()
                        print(f"  ✓ 虚拟环境创建完成 (uv): {venvPath}")
                        return True, f"虚拟环境创建成功 (uv): {venvPath}"
                    else:
                        print(f"  ✗ uv 创建的 Python 不存在: {pythonPath}")
                        print(f"  -> 回退到标准 venv...")
            except subprocess.TimeoutExpired:
                print(f"  ✗ uv venv 超时")
                print(f"  -> 回退到标准 venv...")
            except Exception as e:
                print(f"  ✗ uv venv 异常: {e}")
                print(f"  -> 回退到标准 venv...")

        # 回退到标准 venv
        print(f"  -> 使用标准 venv 创建虚拟环境...")
        pythonCmd = self._find_python_executable(requiredPyVersion)
        if not pythonCmd:
            print(f"  ✗ 未找到 Python {requiredPyVersion}")
            return False, f"未找到 Python {requiredPyVersion}，请确保已安装"
        print(f"  -> 找到 Python: {pythonCmd}")

        try:
            result = subprocess.run(
                [pythonCmd, "-m", "venv", str(venvPath)],
                capture_output=True, text=True, timeout=60
            )
            print(f"  -> venv 命令返回码: {result.returncode}")
            if result.stdout:
                print(f"  -> venv stdout: {result.stdout.strip()[:200]}")
            if result.returncode != 0:
                err = result.stderr.strip()[:300] if result.stderr else "无错误输出"
                print(f"  ✗ venv 创建失败: {err}")
                return False, f"创建失败: {result.stderr}"
            print(f"  -> venv 创建成功")

            pythonPath = self._get_python_path(toolkitType)
            pipPath = self._get_pip_path(toolkitType)
            print(f"  -> 检查虚拟环境 Python: {pythonPath}")

            if not pythonPath.exists():
                print(f"  ✗ Python 可执行文件不存在: {pythonPath}")
                if venvPath.exists():
                    try:
                        subdirs = [p.name for p in venvPath.iterdir()]
                        print(f"  -> venv 目录内容: {subdirs}")
                        binDir = venvPath / "bin"
                        if binDir.exists():
                            bins = [p.name for p in binDir.iterdir()]
                            print(f"  -> bin 目录内容: {bins[:10]}")
                    except Exception as e:
                        print(f"  -> 列出目录失败: {e}")
                return False, f"Python创建失败: {pythonPath}"
            print(f"  -> Python 可执行文件确认存在")

            # 确保pip存在
            print(f"  -> 检查 pip: {pipPath}")
            if not pipPath.exists():
                print(f"  -> pip 不存在，使用 ensurepip 安装...")
                ensureResult = subprocess.run(
                    [str(pythonPath), "-m", "ensurepip", "--upgrade"],
                    capture_output=True, text=True, timeout=60
                )
                print(f"  -> ensurepip 返回码: {ensureResult.returncode}")
                if ensureResult.returncode != 0:
                    err = ensureResult.stderr.strip()[:300] if ensureResult.stderr else "无错误输出"
                    print(f"  ✗ ensurepip 失败: {err}")
                    return False, f"ensurepip失败: {ensureResult.stderr}"
                print(f"  -> ensurepip 成功")
            else:
                print(f"  -> pip 已存在")

            self._check_all_envs()
            print(f"  ✓ 虚拟环境创建完成: {venvPath}")
            return True, f"虚拟环境创建成功: {venvPath}"

        except subprocess.TimeoutExpired:
            print(f"  ✗ 创建超时 (>60s)")
            return False, "创建超时"
        except Exception as e:
            print(f"  ✗ 创建异常: {e}")
            return False, f"创建失败: {str(e)}"
            
    def _build_install_cmd(self, toolkitType: str) -> tuple[list[str], str]:
        """构建安装命令，优先使用 uv pip，回退到标准 pip"""
        uvPath = self._get_uv_path()
        pythonPath = self._get_python_path(toolkitType)
        pipPath = self._get_pip_path(toolkitType)

        if uvPath:
            return [uvPath, "pip", "install", "--python", str(pythonPath)], "uv"

        usePipModule = not pipPath.exists()
        if usePipModule:
            return [str(pythonPath), "-m", "pip", "install"], "pip"
        return [str(pipPath), "install"], "pip"

    def install_rknn(self, toolkitType: str) -> tuple[bool, str]:
        """在虚拟环境中安装rknn-toolkit（优先使用 uv pip）"""
        pythonPath = self._get_python_path(toolkitType)
        toolkitInfo = TOOLKIT_VENV_MAP[toolkitType]
        packageName = toolkitInfo["packageName"]
        pythonVersion = toolkitInfo["pythonVersion"]
        installMode = toolkitInfo.get("installMode", "whl")
        versions = toolkitInfo.get("versions", [])
        venvPath = self._get_venv_path(toolkitType)

        # 确保虚拟环境存在
        if not pythonPath.exists():
            print(f"[步骤1/2] 虚拟环境不存在，开始创建...")
            success, msg = self.create_venv(toolkitType)
            print(msg)
            if not success:
                return False, msg
        else:
            print(f"[步骤1/2] 虚拟环境已存在: {venvPath}")

        installCmd, toolName = self._build_install_cmd(toolkitType)
        print(f"  安装工具: {toolName}")
        print(f"  安装命令: {' '.join(installCmd)}")
        print(f"  包名: {packageName}, Python版本: {pythonVersion}, 安装模式: {installMode}")
        print(f"  系统: {SYSTEM_INFO['system']}, 架构: {SYSTEM_INFO['arch']}, cp标签: {get_python_cp_tag(pythonVersion)}")

        try:
            # 跳过 pip 源安装，直接下载 whl
            print(f"[步骤2/2] 跳过 pip 源，直接下载 whl 安装...")
            if installMode == "tar.gz":
                print(f"  模式: tar.gz（从GitHub下载大文件，可能较慢）")
                downloadedWhl = self._download_and_extract_toolkit_tar(toolkitInfo, pythonVersion)
            else:
                print(f"  模式: whl（从GitHub直接下载）")
                downloadedWhl = self._download_toolkit2_whl(toolkitInfo, pythonVersion)

            if not downloadedWhl:
                print(f"✗ 未找到可下载的 whl 文件")
                return False, "下载失败"

            print(f"安装本地 whl: {downloadedWhl.name}")
            result = subprocess.run(
                installCmd + [str(downloadedWhl)],
                capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
                if lines:
                    for line in lines[-3:]:
                        print(f"  {line[:200]}")
                print(f"✓ 安装成功")
                self._check_all_envs()
                return True, "安装成功"
            else:
                err = result.stderr.strip()[:400] if result.stderr else "未知错误"
                print(f"✗ 安装失败: {err}")
                return False, "安装失败"

        except subprocess.TimeoutExpired as e:
            print(f"✗ 安装超时: {e}")
            return False, "安装超时"
        except Exception as e:
            print(f"✗ 安装异常: {str(e)}")
            return False, f"安装异常: {str(e)}"
            
    def _build_whl_names(self, packageName: str, pythonVersion: str, versions: list[str]) -> list[str]:
        """
        根据系统架构和Python版本构建可能的whl文件名列表
        用于 rknn-toolkit2 直接从 GitHub raw 下载 whl
        """
        whlNames = []
        pkgWhlName = packageName.replace("-", "_")
        cpTag = get_python_cp_tag(pythonVersion)
        arch = SYSTEM_INFO["arch"]
        system = SYSTEM_INFO["system"]

        for version in versions:
            if system == "linux":
                if arch == "x86_64":
                    # 实际文件名同时包含两种 manylinux 标签（用点连接）
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-manylinux_2_17_x86_64.whl")
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-manylinux2014_x86_64.whl")
                elif arch == "aarch64":
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl")
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-manylinux_2_17_aarch64.whl")
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-manylinux2014_aarch64.whl")
                elif arch == "armv7l":
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-linux_armv7l.whl")
            elif system == "darwin":
                if arch == "x86_64":
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-macosx_10_9_x86_64.whl")
                elif arch == "aarch64":
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-macosx_11_0_arm64.whl")
            elif system == "windows":
                if arch == "x86_64":
                    whlNames.append(f"{pkgWhlName}-{version}-{cpTag}-{cpTag}-win_amd64.whl")

        return whlNames

    def _download_file(self, url: str, dest: Path, timeout: int = 60,
                        maxRetries: int = 3) -> bool:
        """带超时、速度检测、进度打印和重试的文件下载"""
        import urllib.request
        import urllib.error

        print(f"开始下载: {url}")
        connectTimeout = 120  # SSL 握手等连接阶段超时

        for attempt in range(maxRetries):
            if attempt > 0:
                delay = 2 ** (attempt - 1)  # 指数退避: 1, 2, 4 秒
                print(f"  第 {attempt + 1}/{maxRetries} 次尝试，等待 {delay}s...")
                time.sleep(delay)
                if dest.exists():
                    dest.unlink()

            startTime = time.time()
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.0'
                })
                response = urllib.request.urlopen(req, timeout=connectTimeout)

                totalSize = response.headers.get('Content-Length')
                totalSize = int(totalSize) if totalSize else None
                if totalSize and totalSize > 500 * 1024 * 1024:
                    print(f"  警告: 文件大小 {totalSize / (1024*1024):.0f}MB，下载可能需要较长时间")

                downloaded = 0
                lastReportTime = startTime
                lastReportBytes = 0
                chunkSize = 64 * 1024  # 64KB chunks

                with open(dest, 'wb') as f:
                    while True:
                        chunkStart = time.time()
                        chunk = response.read(chunkSize)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)
                        chunkTime = time.time() - chunkStart

                        # 检测是否卡死: 64KB 读取超过 60 秒认为卡住
                        if chunkTime > 60:
                            print(f"  下载速度过慢，单个块耗时 {chunkTime:.0f}s，中止")
                            raise TimeoutError("Download stalled")

                        # 检测总体超时
                        elapsed = time.time() - startTime
                        if elapsed > timeout:
                            print(f"  下载总体超时 ({timeout}s)，已下载 {downloaded / (1024*1024):.1f}MB")
                            raise TimeoutError(f"Download timeout after {timeout}s")

                        # 每 5 秒或每 10MB 报告一次
                        now = time.time()
                        if now - lastReportTime > 5 or downloaded - lastReportBytes > 10 * 1024 * 1024:
                            mb = downloaded / (1024 * 1024)
                            speed = (downloaded - lastReportBytes) / max(now - lastReportTime, 0.001)
                            if totalSize:
                                pct = min(100, downloaded * 100 // totalSize)
                                totalMb = totalSize / (1024 * 1024)
                                print(f"  进度: {pct}% ({mb:.1f}/{totalMb:.1f} MB, {speed/1024:.1f} KB/s)")
                            else:
                                print(f"  已下载: {mb:.1f} MB ({speed/1024:.1f} KB/s)")
                            lastReportTime = now
                            lastReportBytes = downloaded

                elapsed = time.time() - startTime
                fileSize = dest.stat().st_size / (1024 * 1024)
                print(f"下载完成: {dest.name} ({fileSize:.1f} MB, {elapsed:.1f}s)")
                return True

            except Exception as e:
                errMsg = str(e)[:120]
                print(f"  下载异常: {errMsg}")
                if dest.exists():
                    dest.unlink()
                if attempt == maxRetries - 1:
                    print(f"下载失败，已重试 {maxRetries} 次")
                    return False
                print(f"  即将重试...")

        return False

    def _download_and_extract_toolkit_tar(self, toolkitInfo: dict, pythonVersion: str) -> Optional[Path]:
        """下载 rknn-toolkit tar.gz 并找到匹配的 whl"""
        import tarfile

        versions = toolkitInfo["versions"]
        tarUrlTemplate = toolkitInfo["tarUrl"]
        cpTag = get_python_cp_tag(pythonVersion)
        arch = SYSTEM_INFO["arch"]

        for version in versions:
            extractDir = WHL_DIR / f"rknn-toolkit-v{version}-packages"
            tarName = f"rknn-toolkit-v{version}-packages.tar.gz"
            tarPath = WHL_DIR / tarName
            tarUrl = tarUrlTemplate.format(version=version)

            print(f"尝试版本 {version}...")

            # 先检查本地是否已有解压好的 whl
            if extractDir.exists():
                whlFiles = list(extractDir.rglob("*.whl"))
                print(f"  本地已解压，找到 {len(whlFiles)} 个 whl")
                matchingWhl = self._find_matching_whl(whlFiles, cpTag, arch)
                if matchingWhl:
                    return matchingWhl

            # 检查本地是否已有 tar.gz
            if tarPath.exists():
                print(f"  本地已有 tar.gz，尝试解压...")
            else:
                # 下载（文件可能很大，超时设长一些）
                print(f"  从 GitHub 下载 tar.gz（文件可能很大，请耐心等待或手动下载）...")
                if not self._download_file(tarUrl, tarPath, timeout=300):
                    continue

            # 解压
            if extractDir.exists():
                shutil.rmtree(extractDir)
            extractDir.mkdir(parents=True, exist_ok=True)

            try:
                with tarfile.open(tarPath, "r:gz") as tar:
                    tar.extractall(extractDir)
                print(f"  解压完成: {extractDir}")
            except Exception as e:
                print(f"  解压失败: {str(e)[:80]}")
                continue

            # 查找匹配的 whl
            whlFiles = list(extractDir.rglob("*.whl"))
            print(f"  找到 {len(whlFiles)} 个 whl 文件")

            matchingWhl = self._find_matching_whl(whlFiles, cpTag, arch)
            if matchingWhl:
                return matchingWhl

        return None

    def _find_matching_whl(self, whlFiles: list[Path], cpTag: str, arch: str) -> Optional[Path]:
        """从 whl 文件列表中找到匹配 Python 版本和架构的 whl"""
        archAliases = {
            "x86_64": ["x86_64", "amd64"],
            "aarch64": ["aarch64", "arm64"],
            "armv7l": ["armv7l"],
        }
        aliases = archAliases.get(arch, [arch])

        print(f"  查找匹配 whl (cpTag={cpTag}, arch={arch}, aliases={aliases})")
        print(f"  候选文件数: {len(whlFiles)}")
        for whlFile in whlFiles:
            name = whlFile.name
            nameLower = name.lower()
            if cpTag.lower() not in nameLower:
                continue
            for alias in aliases:
                if alias.lower() in nameLower:
                    print(f"  ✓ 匹配成功: {name} (alias={alias})")
                    return whlFile
        print(f"  ✗ 未找到匹配的 whl")
        return None

    def _download_toolkit2_whl(self, toolkitInfo: dict, pythonVersion: str) -> Optional[Path]:
        """下载 rknn-toolkit2 的 whl 文件（直接从 GitHub raw）"""
        versions = toolkitInfo["versions"]
        whlBaseUrl = toolkitInfo["whlBaseUrl"]
        packageName = toolkitInfo["packageName"]
        arch = SYSTEM_INFO["arch"]
        # GitHub 上的子目录名：x86_64 或 arm64
        subdir = "arm64" if arch == "aarch64" else arch

        print(f"  构建候选 whl 列表 (Python={pythonVersion}, arch={arch})...")
        for version in versions:
            whlNames = self._build_whl_names(packageName, pythonVersion, [version])
            print(f"  版本 {version}: {len(whlNames)} 个候选文件")
            for name in whlNames:
                print(f"    - {name}")

            for idx, whlName in enumerate(whlNames):
                whlUrl = f"{whlBaseUrl}/{subdir}/{whlName}"
                whlPath = WHL_DIR / whlName
                print(f"  尝试下载 {idx+1}/{len(whlNames)}: {whlName}")

                # 优先使用本地已下载的 whl
                if whlPath.exists() and whlPath.stat().st_size > 1000:
                    print(f"  ✓ 使用本地已下载的 whl: {whlPath.name} ({whlPath.stat().st_size / (1024*1024):.1f} MB)")
                    return whlPath

                if self._download_file(whlUrl, whlPath, timeout=120):
                    return whlPath
                else:
                    print(f"  该文件不存在或下载失败，继续下一个")

        print(f"  所有候选 whl 下载均失败")
        return None

    def prepare_env(self, toolkitType: str) -> tuple[bool, str]:
        """准备虚拟环境（创建并安装rknn）"""
        print("=" * 50)
        print(f"开始准备环境: {toolkitType}")
        print("=" * 50)

        status = self.get_status(toolkitType)
        print(f"当前状态: 存在={status.get('exists', False)}, "
              f"Python就绪={status.get('pythonReady', False)}, "
              f"rknn已安装={status.get('rknnInstalled', False)}")

        if status.get("rknnInstalled"):
            print("✓ 环境已就绪，无需操作")
            return True, "环境已就绪"

        # 检查虚拟环境是否存在
        if not status.get("exists"):
            print("--- 阶段1: 创建虚拟环境 ---")
            success, msg = self.create_venv(toolkitType)
            print(msg)
            if not success:
                print("✗ 环境准备失败: 虚拟环境创建失败")
                return False, msg
            print("✓ 虚拟环境创建完成")
        else:
            print("--- 阶段1: 虚拟环境已存在，跳过创建 ---")

        # 安装rknn
        print("--- 阶段2: 安装 RKNN Toolkit ---")
        success, msg = self.install_rknn(toolkitType)

        if success:
            print("✓ 环境准备完成")
        else:
            print("✗ 环境准备失败")

        print("=" * 50)
        return success, msg


# 全局虚拟环境管理器
venvManager = VirtualEnvManager()

# 存储转换任务
tasks: dict[str, ConversionTask] = {}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "onnx-to-rknn-secret-key-2024")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


def get_onnx_info(onnxPath: str) -> dict:
    """获取ONNX模型信息"""
    try:
        import onnx
        model = onnx.load(onnxPath)
        
        inputs = []
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param if dim.dim_param else "?")
            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": str(inp.type.tensor_type.elem_type)
            })
            
        outputs = []
        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param if dim.dim_param else "?")
            outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": str(out.type.tensor_type.elem_type)
            })
            
        fileSize = Path(onnxPath).stat().st_size / (1024 * 1024)
            
        return {
            "success": True,
            "inputs": inputs,
            "outputs": outputs,
            "fileSize": f"{fileSize:.2f} MB"
        }
    except ImportError:
        return {"success": False, "error": "onnx库未安装: pip install onnx"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_conversion_in_venv_async(task: ConversionTask):
    """异步执行转换（在后台线程中）"""
    success, message = run_conversion_in_venv(task)

    task.endTime = datetime.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.endTime - task.startTime).total_seconds()
        task.add_log(f"总耗时: {duration:.2f}秒")
    else:
        task.status = "failed"
        task.message = message


def _parse_worker_line(task: ConversionTask, line: str):
    """解析 convert_worker.py 的输出并实时写入日志"""
    if line.startswith("[LOG]"):
        task.add_log(line[5:].strip())
    elif line.startswith("[SUCCESS]"):
        task.add_log(f"转换成功: {line[9:].strip()}")
    elif line.startswith("[ERROR]"):
        task.add_log(f"转换失败: {line[7:].strip()}")
    elif line.startswith("[SIZE]"):
        task.add_log(f"模型大小: {line[6:].strip()} MB")
    else:
        task.add_log(line)


def run_conversion_in_venv(task: ConversionTask) -> tuple[bool, str]:
    """在虚拟环境中执行转换，日志实时写入 task.log"""
    config = task.config
    platform = config.platform

    if platform not in CHIP_PLATFORMS:
        return False, f"不支持的平台: {platform}"

    toolkitType = CHIP_PLATFORMS[platform]["toolkit"]

    # 准备虚拟环境
    task.add_log(f"目标平台: {platform}")
    task.add_log(f"需要Toolkit: {toolkitType}")

    success, msg = venvManager.prepare_env(toolkitType)

    if not success:
        return False, f"环境准备失败: {msg}"

    task.add_log(f"环境准备完成: {msg}")

    # 获取虚拟环境中的Python路径
    pythonPath = venvManager._get_python_path(toolkitType)
    convertScript = BASE_DIR / "convert_worker.py"

    # 构建转换参数
    configJson = json.dumps(config.to_dict())

    task.add_log(f"使用Python: {pythonPath}")
    task.add_log(f"转换脚本: {convertScript}")
    task.add_log(f"配置参数: {configJson}")

    # 在虚拟环境中执行转换脚本（Popen 实时读取输出）
    try:
        task.process = subprocess.Popen(
            [
                str(pythonPath),
                str(convertScript),
                "--onnx", task.onnxPath,
                "--output", task.outputPath,
                "--config", configJson
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # 实时读取 stdout
        if task.process.stdout:
            for line in task.process.stdout:
                _parse_worker_line(task, line.rstrip("\n"))

        # 等待进程结束（带超时）
        try:
            returncode = task.process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            task.process.kill()
            task.process.wait()
            task.add_log("转换超时 (>10分钟)")
            return False, "转换超时 (>10分钟)"

        # 读取 stderr
        stderr_text = task.process.stderr.read() if task.process.stderr else ""
        if stderr_text:
            task.add_log(f"stderr: {stderr_text.strip()}")

        if returncode != 0:
            return False, "转换进程返回错误"

        # 检查输出文件
        if Path(task.outputPath).exists():
            outputSize = Path(task.outputPath).stat().st_size / (1024 * 1024)
            task.add_log(f"输出文件: {task.outputPath}")
            task.add_log(f"文件大小: {outputSize:.2f} MB")
            return True, f"转换成功: {task.outputPath}"
        else:
            return False, "输出文件不存在"

    except Exception as e:
        task.add_log(f"执行异常: {str(e)}")
        return False, f"执行异常: {str(e)}"
    finally:
        task.process = None


@app.route("/")
def index():
    """主页"""
    return render_template(
        "index.html",
        platforms=CHIP_PLATFORMS,
        venv_status=venvManager.get_all_status(),
        toolkit_venv_map=TOOLKIT_VENV_MAP,
        quantized_dtypes=QUANTIZED_DTYPES,
        quantized_algorithms=QUANTIZED_ALGORITHMS,
        optimization_levels=OPTIMIZATION_LEVELS
    )


@app.route("/convert_direct", methods=["POST"])
def convert_direct():
    """直接从主页提交转换"""
    if "file" not in request.files:
        flash("未选择文件", "error")
        return redirect(url_for("index"))
        
    file = request.files["file"]
    platform = request.form.get("platform", "")
    
    if file.filename == "" or not platform:
        flash("参数不完整", "error")
        return redirect(url_for("index"))
        
    if not file.filename.lower().endswith(".onnx"):
        flash("只支持ONNX格式文件", "error")
        return redirect(url_for("index"))
        
    if platform not in CHIP_PLATFORMS:
        flash(f"不支持的平台: {platform}", "error")
        return redirect(url_for("index"))
        
    # 保存文件
    taskId = str(uuid.uuid4())[:8]
    filename = f"{taskId}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)
    
    # 构建配置
    config = ConversionConfig(platform=platform)
    
    inputHeight = request.form.get("input_height", type=int)
    inputWidth = request.form.get("input_width", type=int)
    if inputHeight and inputWidth:
        config.inputSize = (inputHeight, inputWidth)
    
    config.inputDtype = request.form.get("input_dtype", "float32") or "float32"
    
    meanStr = request.form.get("mean_values", "")
    if meanStr:
        try:
            config.meanValues = [float(x.strip()) for x in meanStr.split(",")]
        except ValueError:
            pass
            
    stdStr = request.form.get("std_values", "")
    if stdStr:
        try:
            config.stdValues = [float(x.strip()) for x in stdStr.split(",")]
        except ValueError:
            pass
    
    config.doQuantization = request.form.get("do_quantization") == "on"
    config.quantizedDtype = request.form.get("quantized_dtype", "asymmetric_quantized-u8")
    config.quantizedAlgorithm = request.form.get("quantized_algorithm", "normal")
    config.datasetPath = request.form.get("dataset_path", "") or None
    
    config.optimizationLevel = request.form.get("optimization_level", type=int, default=2)
    config.singleCoreMode = request.form.get("single_core_mode") == "on"
    
    batchSize = request.form.get("batch_size", type=int, default=1) or 1
    config.batchSize = batchSize
    
    # 创建任务
    task = ConversionTask(taskId, str(filepath), config)
    tasks[taskId] = task
    
    outputFilename = f"{taskId}_{Path(filepath).stem}.rknn"
    outputPath = OUTPUT_DIR / outputFilename
    task.outputPath = str(outputPath)
    
    # 执行转换
    task.status = "converting"
    task.add_log(f"文件: {file.filename}")
    task.add_log(f"平台: {platform}")
    task.add_log(f"配置: {config.to_dict()}")

    success, message = run_conversion_in_venv(task)

    task.endTime = datetime.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.endTime - task.startTime).total_seconds()
        task.add_log(f"耗时: {duration:.2f}秒")
    else:
        task.status = "failed"
        task.message = message

    return render_template("result.html", task=task)


@app.route("/upload", methods=["POST"])
def upload():
    """上传ONNX模型"""
    if "file" not in request.files:
        flash("未选择文件", "error")
        return redirect(url_for("index"))
        
    file = request.files["file"]
    if file.filename == "":
        flash("未选择文件", "error")
        return redirect(url_for("index"))
        
    if not file.filename.lower().endswith(".onnx"):
        flash("只支持ONNX格式文件", "error")
        return redirect(url_for("index"))
        
    taskId = str(uuid.uuid4())[:8]
    filename = f"{taskId}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)
    
    onnxInfo = get_onnx_info(str(filepath))
    
    return render_template(
        "config.html",
        taskId=taskId,
        filename=file.filename,
        filepath=str(filepath),
        platforms=CHIP_PLATFORMS,
        onnx_info=onnxInfo,
        quantized_dtypes=QUANTIZED_DTYPES,
        quantized_algorithms=QUANTIZED_ALGORITHMS,
        optimization_levels=OPTIMIZATION_LEVELS,
        venv_status=venvManager.get_all_status()
    )


@app.route("/convert", methods=["POST"])
def convert():
    """执行转换"""
    taskId = request.form.get("taskId", "")
    filepath = request.form.get("filepath", "")
    platform = request.form.get("platform", "")
    
    if not all([taskId, filepath, platform]):
        return jsonify({"success": False, "message": "参数不完整"})
        
    if platform not in CHIP_PLATFORMS:
        return jsonify({"success": False, "message": f"不支持的平台: {platform}"})
        
    # 构建配置
    config = ConversionConfig(platform=platform)
    
    inputHeight = request.form.get("input_height", type=int)
    inputWidth = request.form.get("input_width", type=int)
    if inputHeight and inputWidth:
        config.inputSize = (inputHeight, inputWidth)
    
    config.inputName = request.form.get("input_name", "") or None
    config.inputDtype = request.form.get("input_dtype", "float32") or "float32"
    
    meanStr = request.form.get("mean_values", "")
    if meanStr:
        try:
            config.meanValues = [float(x.strip()) for x in meanStr.split(",")]
        except ValueError:
            pass
            
    stdStr = request.form.get("std_values", "")
    if stdStr:
        try:
            config.stdValues = [float(x.strip()) for x in stdStr.split(",")]
        except ValueError:
            pass
    
    config.doQuantization = request.form.get("do_quantization") == "on"
    config.quantizedDtype = request.form.get("quantized_dtype", "asymmetric_quantized-u8")
    config.quantizedAlgorithm = request.form.get("quantized_algorithm", "normal")
    config.datasetPath = request.form.get("dataset_path", "") or None
    
    config.optimizationLevel = request.form.get("optimization_level", type=int, default=2)
    config.singleCoreMode = request.form.get("single_core_mode") == "on"
    
    dataSizeStr = request.form.get("model_data_size", "")
    if dataSizeStr:
        try:
            config.modelDataSize = int(dataSizeStr)
        except ValueError:
            pass
    
    config.batchSize = request.form.get("batch_size", type=int, default=1) or 1
    
    # 创建任务
    task = ConversionTask(taskId, filepath, config)
    tasks[taskId] = task
    
    outputFilename = f"{taskId}_{Path(filepath).stem}.rknn"
    outputPath = OUTPUT_DIR / outputFilename
    task.outputPath = str(outputPath)
    
    # 执行转换
    task.status = "converting"
    task.add_log(f"开始转换: {filepath}")
    task.add_log(f"配置: {config.to_dict()}")

    # 同步执行（阻塞）
    success, message = run_conversion_in_venv(task)

    task.endTime = datetime.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.endTime - task.startTime).total_seconds()
        task.add_log(f"总耗时: {duration:.2f}秒")
    else:
        task.status = "failed"
        task.message = message

    return render_template(
        "result.html",
        task=task,
        platforms=CHIP_PLATFORMS
    )


@app.route("/convert_async", methods=["POST"])
def convert_async():
    """异步执行转换（后台线程）"""
    taskId = request.form.get("taskId", "")
    filepath = request.form.get("filepath", "")
    platform = request.form.get("platform", "")
    
    if not all([taskId, filepath, platform]):
        return jsonify({"success": False, "message": "参数不完整"})
        
    if platform not in CHIP_PLATFORMS:
        return jsonify({"success": False, "message": f"不支持的平台: {platform}"})
        
    # 构建配置（同上）
    config = ConversionConfig(platform=platform)
    
    inputHeight = request.form.get("input_height", type=int)
    inputWidth = request.form.get("input_width", type=int)
    if inputHeight and inputWidth:
        config.inputSize = (inputHeight, inputWidth)
    
    config.inputName = request.form.get("input_name", "") or None
    config.inputDtype = request.form.get("input_dtype", "float32") or "float32"
    
    meanStr = request.form.get("mean_values", "")
    if meanStr:
        try:
            config.meanValues = [float(x.strip()) for x in meanStr.split(",")]
        except ValueError:
            pass
            
    stdStr = request.form.get("std_values", "")
    if stdStr:
        try:
            config.stdValues = [float(x.strip()) for x in stdStr.split(",")]
        except ValueError:
            pass
    
    config.doQuantization = request.form.get("do_quantization") == "on"
    config.quantizedDtype = request.form.get("quantized_dtype", "asymmetric_quantized-u8")
    config.quantizedAlgorithm = request.form.get("quantized_algorithm", "normal")
    config.datasetPath = request.form.get("dataset_path", "") or None
    
    config.optimizationLevel = request.form.get("optimization_level", type=int, default=2)
    config.singleCoreMode = request.form.get("single_core_mode") == "on"
    
    dataSizeStr = request.form.get("model_data_size", "")
    if dataSizeStr:
        try:
            config.modelDataSize = int(dataSizeStr)
        except ValueError:
            pass
    
    config.batchSize = request.form.get("batch_size", type=int, default=1) or 1
    
    # 创建任务
    task = ConversionTask(taskId, filepath, config)
    tasks[taskId] = task
    
    outputFilename = f"{taskId}_{Path(filepath).stem}.rknn"
    outputPath = OUTPUT_DIR / outputFilename
    task.outputPath = str(outputPath)
    
    task.status = "converting"
    task.add_log(f"开始转换: {filepath}")
    task.add_log(f"配置: {config.to_dict()}")
    
    # 启动后台线程
    thread = threading.Thread(target=run_conversion_in_venv_async, args=(task,))
    thread.start()
    
    return jsonify({
        "success": True,
        "taskId": taskId,
        "message": "转换任务已启动"
    })


@app.route("/prepare_env/<toolkit_type>")
def prepare_env(toolkit_type: str):
    """准备虚拟环境"""
    if toolkit_type not in TOOLKIT_VENV_MAP:
        return jsonify({"success": False, "message": f"未知的toolkit类型: {toolkit_type}"})
        
    success, msg = venvManager.prepare_env(toolkit_type)
    return jsonify({
        "success": success,
        "message": msg,
        "logs": [],
        "status": venvManager.get_status(toolkit_type)
    })


@app.route("/env_status")
def env_status():
    """获取虚拟环境状态"""
    return jsonify(venvManager.get_all_status())


@app.route("/download/<taskId>")
def download(taskId: str):
    """下载转换后的模型"""
    task = tasks.get(taskId)
    if not task or not task.outputPath:
        flash("任务不存在或未完成", "error")
        return redirect(url_for("index"))
        
    if not Path(task.outputPath).exists():
        flash("文件不存在", "error")
        return redirect(url_for("index"))
        
    return send_file(
        task.outputPath,
        as_attachment=True,
        download_name=Path(task.outputPath).name
    )


@app.route("/api/platforms")
def api_platforms():
    """获取支持的平台列表"""
    return jsonify(CHIP_PLATFORMS)


@app.route("/api/tasks")
def api_tasks():
    """获取任务列表API"""
    return jsonify([
        {
            "taskId": t.taskId,
            "platform": t.config.platform,
            "status": t.status,
            "message": t.message,
            "startTime": t.startTime.isoformat(),
            "endTime": t.endTime.isoformat() if t.endTime else None
        }
        for t in tasks.values()
    ])


@app.route("/api/task/<taskId>")
def api_task(taskId: str):
    """获取单个任务状态API"""
    task = tasks.get(taskId)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"})
        
    return jsonify({
        "success": True,
        "taskId": task.taskId,
        "platform": task.config.platform,
        "status": task.status,
        "message": task.message,
        "outputPath": task.outputPath,
        "startTime": task.startTime.isoformat(),
        "endTime": task.endTime.isoformat() if task.endTime else None,
        "log": task.log[-20:] if len(task.log) > 20 else task.log
    })


@app.route("/api/info/<path:filepath>")
def api_info(filepath: str):
    """获取ONNX模型信息API"""
    return jsonify(get_onnx_info(filepath))


def init_virtual_environments():
    """
    启动时自动初始化虚拟环境：
    1. 检查 uv，未安装则自动安装
    2. uv 可用时，自动创建虚拟环境并安装 rknn-toolkit / rknn-toolkit2
    3. uv 不可用时，仅打印状态检查
    """
    print("=" * 70)
    print()
    print("初始化虚拟环境")
    print(f"虚拟环境目录: {VENV_DIR}")
    print()
    print("=" * 70)

    # 步骤1: 确保 uv 已安装
    print("\n[步骤1/3] 检查 uv...")
    uvOk, uvMsg = venvManager._ensure_uv_installed()
    print(f"  结果: {uvMsg}")
    if not uvOk:
        print("\n" + "=" * 70)
        print("⚠ uv 不可用，仅执行状态检查（不自动安装环境）")
        print("=" * 70)
        return _check_env_status_only()

    # 步骤2: 确保所需 Python 版本已安装
    print("\n[步骤2/3] 确保 Python 版本...")
    requiredPyVersions = {info["pythonVersion"] for info in TOOLKIT_VENV_MAP.values()}
    for pyVersion in sorted(requiredPyVersions):
        pyOk, pyMsg = venvManager._ensure_python_installed(pyVersion)
        print(f"  {pyVersion}: {'✓' if pyOk else '✗'} {pyMsg}")
        if not pyOk:
            print(f"\n⚠ Python {pyVersion} 安装失败，跳过对应环境的自动安装")

    # 步骤3: 自动创建虚拟环境并安装 rknn
    print("\n[步骤3/3] 创建虚拟环境并安装 RKNN Toolkit...")
    results = {}
    for toolkitType, info in TOOLKIT_VENV_MAP.items():
        print(f"\n[{toolkitType}]")
        print(f"  虚拟环境: {info['venvName']}")
        print(f"  包名: {info['packageName']}")

        status = venvManager.get_status(toolkitType)
        if status.get("rknnInstalled"):
            print(f"  ✓ 已就绪，跳过")
            results[toolkitType] = {"success": True, "message": "已就绪", "skipped": True}
            continue

        print(f"  -> 开始准备环境...")
        success, msg = venvManager.prepare_env(toolkitType)
        results[toolkitType] = {"success": success, "message": msg}

    print("\n" + "=" * 70)
    print("初始化完成")

    readyCount = sum(1 for r in results.values() if r.get("success"))
    totalCount = len(TOOLKIT_VENV_MAP)
    print(f"环境状态: {readyCount}/{totalCount} 就绪")

    for toolkitType, result in results.items():
        statusIcon = "✓" if result.get("success") else "✗"
        print(f"  {statusIcon} {toolkitType}: {result.get('message')}")

    print("=" * 70)
    return results


def _check_env_status_only() -> dict:
    """仅检查环境状态，不自动安装"""
    results = {}
    for toolkitType, info in TOOLKIT_VENV_MAP.items():
        print(f"\n[{toolkitType}]")
        print(f"  虚拟环境: {info['venvName']}")
        print(f"  包名: {info['packageName']}")

        status = venvManager.get_status(toolkitType)
        venvPath = status.get("venvPath", "未知")
        pythonPath = status.get("pythonPath", "未知")
        exists = status.get("exists", False)
        pythonReady = status.get("pythonReady", False)
        rknnInstalled = status.get("rknnInstalled", False)

        print(f"  路径: {venvPath}")
        print(f"  Python: {pythonPath}")
        print(f"  存在={exists}, Python就绪={pythonReady}, rknn已安装={rknnInstalled}")

        if rknnInstalled:
            print(f"  状态: ✓ 已就绪")
            results[toolkitType] = {"success": True, "message": "已就绪", "skipped": True}
        elif exists and pythonReady:
            print(f"  状态: ○ 虚拟环境已创建，缺少 rknn 包")
            results[toolkitType] = {"success": True, "message": "待安装", "skipped": True}
        elif exists:
            print(f"  状态: ○ 虚拟环境目录存在但不完整")
            results[toolkitType] = {"success": True, "message": "待重建", "skipped": True}
        else:
            print(f"  状态: ○ 未创建")
            results[toolkitType] = {"success": True, "message": "待创建", "skipped": True}

    print("\n" + "=" * 70)
    readyCount = sum(1 for r in results.values() if r.get("message") == "已就绪")
    totalCount = len(TOOLKIT_VENV_MAP)
    print(f"环境状态: {readyCount}/{totalCount} 就绪")
    for toolkitType, result in results.items():
        statusIcon = "✓" if result.get("message") == "已就绪" else "○"
        print(f"  {statusIcon} {toolkitType}: {result.get('message')}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    # 启动时只检查状态，不自动安装
    initResults = init_virtual_environments()

    print()
    print("ONNX转RKNN Web转换工具")
    print(f"访问: http://localhost:5000")
    print("=" * 70)

    app.run(host="0.0.0.0", port=5000, debug=True)