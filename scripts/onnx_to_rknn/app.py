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

import argparse
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
from werkzeug.utils import secure_filename

# 配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "datasets"
# 虚拟环境目录使用 TMPDIR
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
VENV_DIR = TMPDIR / "onnx_to_rknn_venvs"

# 清华镜像源（加速国内下载）
TSINGHUA_PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
TSINGHUA_PYTHON_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/python/"

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
        "venv_name": "venv_toolkit",
        "package_name": "rknn-toolkit",
        "python_version": "3.8",
        "install_mode": "tar.gz",
        "tar_url": "https://github.com/rockchip-linux/rknn-toolkit/releases/download/v{version}/rknn-toolkit-v{version}-packages.tar.gz",
        "versions": ["1.7.5", "1.7.3", "1.7.1", "1.7.0", "1.6.1", "1.6.0"],
        "note": "下载tar.gz解压后按Python版本和架构选择whl安装",
    },
    "rknn-toolkit2": {
        "venv_name": "venv_toolkit2",
        "package_name": "rknn-toolkit2",
        "python_version": "3.12",
        "install_mode": "whl",
        "whl_base_url": "https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages",
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
    "RK2118": {"toolkit": "rknn-toolkit2", "target": "rk2118", "description": "RK2118 NPU"},
    "RK3562": {"toolkit": "rknn-toolkit2", "target": "rk3562", "description": "RK3562"},
    "RK3566": {"toolkit": "rknn-toolkit2", "target": "rk3566", "description": "RK3566"},
    "RK3568": {"toolkit": "rknn-toolkit2", "target": "rk3568", "description": "RK3568"},
    "RK3576": {"toolkit": "rknn-toolkit2", "target": "rk3576", "description": "RK3576"},
    "RK3588": {"toolkit": "rknn-toolkit2", "target": "rk3588", "description": "RK3588"},
    "RV1103": {"toolkit": "rknn-toolkit2", "target": "rv1103", "description": "RV1103 ISP"},
    "RV1106": {"toolkit": "rknn-toolkit2", "target": "rv1106", "description": "RV1106 ISP"},
    "RV1103B": {"toolkit": "rknn-toolkit2", "target": "rv1103b", "description": "RV1103B ISP"},
    "RV1106B": {"toolkit": "rknn-toolkit2", "target": "rv1106b", "description": "RV1106B ISP"},
    "RV1126B": {"toolkit": "rknn-toolkit2", "target": "rv1126b", "description": "RV1126B ISP"},
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
    input_size: Optional[tuple[int, int]] = None
    input_name: Optional[str] = None
    mean_values: Optional[list[float]] = None
    std_values: Optional[list[float]] = None
    input_dtype: str = "float32"
    do_quantization: bool = False
    quantized_dtype: str = "asymmetric_quantized-u8"
    quantized_algorithm: str = "normal"
    dataset_path: Optional[str] = None
    optimization_level: int = 2
    single_core_mode: bool = False
    model_data_size: Optional[int] = None
    batch_size: int = 1

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "input_size": list(self.input_size) if self.input_size else None,
            "input_name": self.input_name,
            "mean_values": self.mean_values,
            "std_values": self.std_values,
            "input_dtype": self.input_dtype,
            "do_quantization": self.do_quantization,
            "quantized_dtype": self.quantized_dtype,
            "quantized_algorithm": self.quantized_algorithm,
            "dataset_path": self.dataset_path,
            "optimization_level": self.optimization_level,
            "single_core_mode": self.single_core_mode,
            "model_data_size": self.model_data_size,
            "batch_size": self.batch_size,
        }


class ConversionTask:
    """转换任务"""

    def __init__(self, task_id: str, onnx_path: str, config: ConversionConfig):
        self.task_id = task_id
        self.onnx_path = onnx_path
        self.config = config
        self.output_path: Optional[str] = None
        self.status = "pending"
        self.message = ""
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.log: list[str] = []
        self.process: Optional[subprocess.Popen] = None
        self._log_lock = threading.Lock()

    def add_log(self, msg: str):
        """实时添加日志，支持并发安全"""
        with self._log_lock:
            self.log.append(msg)
        print(f"[{self.task_id}] {msg}")


class VirtualEnvManager:
    """虚拟环境管理器"""

    def __init__(self, venv_base_dir: Path = VENV_DIR):
        self.venv_base_dir = venv_base_dir
        self.env_status: dict[str, dict] = {}
        self._check_all_envs()

    def _get_venv_path(self, toolkit_type: str) -> Path:
        """获取虚拟环境路径"""
        venv_name = TOOLKIT_VENV_MAP[toolkit_type]["venv_name"]
        return self.venv_base_dir / venv_name

    def _get_python_path(self, toolkit_type: str) -> Path:
        """获取虚拟环境中的Python路径"""
        venv_path = self._get_venv_path(toolkit_type)
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def _get_pip_path(self, toolkit_type: str) -> Path:
        """获取虚拟环境中的pip路径"""
        venv_path = self._get_venv_path(toolkit_type)
        if sys.platform == "win32":
            return venv_path / "Scripts" / "pip.exe"
        return venv_path / "bin" / "pip"

    def _get_uv_path(self) -> Optional[str]:
        """查找 uv 可执行文件路径"""
        uv_path = shutil.which("uv")
        if uv_path:
            try:
                result = subprocess.run([uv_path, "--version"],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return uv_path
            except Exception:
                pass
        return None

    def _ensure_uv_installed(self) -> tuple[bool, str]:
        """确保 uv 已安装，未安装则自动安装"""
        uv_path = self._get_uv_path()
        if uv_path:
            return True, f"uv 已安装: {uv_path}"

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
                install_result = subprocess.run(
                    ["sh"],
                    input=result.stdout,
                    capture_output=True, text=True, timeout=120
                )
                if install_result.returncode == 0:
                    print("  -> 脚本安装完成")
                    # 刷新 PATH 缓存
                    import importlib
                    importlib.reload(shutil)
                    uv_path = shutil.which("uv")
                    if uv_path:
                        print(f"  ✓ uv 安装成功: {uv_path}")
                        return True, f"uv 安装成功: {uv_path}"
                    else:
                        # uv 可能被安装到 ~/.local/bin，需要添加到 PATH
                        local_bin = Path.home() / ".local" / "bin"
                        if local_bin.exists() and (local_bin / "uv").exists():
                            os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
                            uv_path = shutil.which("uv")
                            if uv_path:
                                print(f"  ✓ uv 安装成功 (添加到 PATH): {uv_path}")
                                return True, f"uv 安装成功: {uv_path}"
            else:
                print(f"  ✗ 下载脚本失败: {result.stderr[:200]}")
        except Exception as e:
            print(f"  ✗ 脚本安装异常: {e}")

        # 方法2: 通过 pip 安装（使用清华源）
        try:
            print("  -> 尝试通过 pip 安装 uv（清华源）...")
            pip_cmds = ["pip", "pip3", sys.executable + " -m pip"]
            for pip_cmd in pip_cmds:
                parts = pip_cmd.split() if " " in pip_cmd else [pip_cmd]
                result = subprocess.run(
                    parts + ["install", "uv", "-i", TSINGHUA_PYPI_INDEX],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    uv_path = shutil.which("uv")
                    if uv_path:
                        print(f"  ✓ uv 通过 pip 安装成功: {uv_path}")
                        return True, f"uv 安装成功: {uv_path}"
        except Exception as e:
            print(f"  ✗ pip 安装异常: {e}")

        return False, "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"

    def _ensure_python_installed(self, py_version: str) -> tuple[bool, str]:
        """使用 uv 确保指定 Python 版本已安装"""
        uv_path = self._get_uv_path()
        if not uv_path:
            return False, "uv 未安装，无法安装 Python"

        py_cmd_name = f"python{py_version}"
        print(f"\n[ensure_python] 确保 {py_cmd_name} 已安装...")

        # 检查是否已存在
        py_cmd = shutil.which(py_cmd_name)
        if py_cmd:
            try:
                result = subprocess.run([py_cmd, "--version"],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and py_version in result.stdout:
                    print(f"  ✓ {py_cmd_name} 已存在: {py_cmd}")
                    return True, f"{py_cmd_name} 已存在"
            except Exception:
                pass

        # 使用 uv 安装（实时输出进度，使用清华镜像）
        try:
            print(f"  -> 使用 uv 安装 {py_version}（清华镜像）...")
            # 设置环境变量：禁用 uv 进度条，改为普通文本输出便于实时显示
            env = os.environ.copy()
            env["UV_NO_PROGRESS"] = "1"
            env["NO_COLOR"] = "1"
            
            process = subprocess.Popen(
                [uv_path, "python", "install", py_version, "--mirror", TSINGHUA_PYTHON_MIRROR],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            if process.stdout:
                for line in process.stdout:
                    line = line.rstrip("\n")
                    if line:
                        print(f"    {line}")

            try:
                returncode = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print(f"  ✗ {py_version} 安装超时")
                return False, f"{py_version} 安装超时"

            if returncode == 0:
                print(f"  ✓ {py_version} 安装成功")
                return True, f"{py_version} 安装成功"
            else:
                print(f"  ✗ {py_version} 安装失败 (返回码: {returncode})")
                return False, f"{py_version} 安装失败 (返回码: {returncode})"
        except Exception as e:
            print(f"  ✗ {py_version} 安装异常: {e}")
            return False, f"{py_version} 安装异常: {str(e)}"

    def _find_python_executable(self, py_version: str) -> Optional[str]:
        """查找指定版本的 Python 可执行文件"""
        major, minor = py_version.split(".")[:2]
        candidates = [
            f"python{major}.{minor}",
            f"python{major}{minor}",
        ]
        if sys.platform == "win32":
            candidates.insert(0, f"py -{major}.{minor}")

        for cmd in candidates:
            python_path = shutil.which(cmd)
            if python_path:
                try:
                    result = subprocess.run(
                        [python_path, "--version"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and py_version in result.stdout:
                        return python_path
                except Exception:
                    pass

        # 回退到当前 Python（仅当版本匹配时）
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_version == py_version:
            return sys.executable
        return None

    def _check_all_envs(self):
        """检查所有虚拟环境状态"""
        for toolkit_type, info in TOOLKIT_VENV_MAP.items():
            venv_path = self._get_venv_path(toolkit_type)
            python_path = self._get_python_path(toolkit_type)

            exists = venv_path.exists()
            python_exists = python_path.exists()

            # 检查rknn是否已安装
            rknn_installed = False
            if python_exists:
                try:
                    result = subprocess.run(
                        [str(python_path), "-c", "import rknn; print('ok')"],
                        capture_output=True,
                        timeout=5
                    )
                    rknn_installed = result.returncode == 0
                except Exception:
                    rknn_installed = False

            self.env_status[toolkit_type] = {
                "venv_path": str(venv_path),
                "python_path": str(python_path),
                "exists": exists,
                "python_ready": python_exists,
                "rknn_installed": rknn_installed,
                "package_name": info["package_name"],
            }

    def get_status(self, toolkit_type: str) -> dict:
        """获取指定toolkit的虚拟环境状态"""
        return self.env_status.get(toolkit_type, {})

    def get_all_status(self) -> dict:
        """获取所有虚拟环境状态"""
        return self.env_status

    def create_venv(self, toolkit_type: str) -> tuple[bool, str]:
        """创建虚拟环境（优先使用 uv，回退到标准 venv）"""
        venv_path = self._get_venv_path(toolkit_type)
        toolkit_info = TOOLKIT_VENV_MAP[toolkit_type]
        required_py_version = toolkit_info["python_version"]
        py_cmd_name = f"python{required_py_version}"

        print(f"\n[create_venv] toolkit={toolkit_type}, requiredPy={required_py_version}")

        if venv_path.exists() and self._get_python_path(toolkit_type).exists():
            print(f"  -> 虚拟环境已存在且完整: {venv_path}")
            return True, f"虚拟环境已存在: {venv_path}"

        # 如果目录存在但环境不完整，删除重建
        if venv_path.exists():
            print(f"  -> 清理不完整的环境: {venv_path}")
            shutil.rmtree(venv_path)
            print(f"  -> 清理完成")

        # 优先尝试 uv
        uv_path = self._get_uv_path()
        if uv_path:
            print(f"  -> 使用 uv 创建虚拟环境: {uv_path}")
            try:
                cmd = [uv_path, "venv", str(venv_path), "--python", py_cmd_name]
                print(f"  -> 执行: {' '.join(cmd)}")
                env = os.environ.copy()
                env["UV_NO_PROGRESS"] = "1"
                env["NO_COLOR"] = "1"
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
                print(f"  -> uv venv 返回码: {result.returncode}")
                if result.stdout:
                    print(f"  -> stdout: {result.stdout.strip()[:200]}")
                if result.returncode != 0:
                    err = result.stderr.strip()[:300] if result.stderr else "无错误输出"
                    print(f"  ✗ uv venv 失败: {err}")
                    print(f"  -> 回退到标准 venv...")
                else:
                    print(f"  -> uv venv 创建成功")
                    python_path = self._get_python_path(toolkit_type)
                    if python_path.exists():
                        print(f"  -> Python 可执行文件确认存在: {python_path}")
                        self._check_all_envs()
                        print(f"  ✓ 虚拟环境创建完成 (uv): {venv_path}")
                        return True, f"虚拟环境创建成功 (uv): {venv_path}"
                    else:
                        print(f"  ✗ uv 创建的 Python 不存在: {python_path}")
                        print(f"  -> 回退到标准 venv...")
            except subprocess.TimeoutExpired:
                print(f"  ✗ uv venv 超时")
                print(f"  -> 回退到标准 venv...")
            except Exception as e:
                print(f"  ✗ uv venv 异常: {e}")
                print(f"  -> 回退到标准 venv...")

        # 回退到标准 venv
        print(f"  -> 使用标准 venv 创建虚拟环境...")
        python_cmd = self._find_python_executable(required_py_version)
        if not python_cmd:
            print(f"  ✗ 未找到 Python {required_py_version}")
            return False, f"未找到 Python {required_py_version}，请确保已安装"
        print(f"  -> 找到 Python: {python_cmd}")

        try:
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(venv_path)],
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

            python_path = self._get_python_path(toolkit_type)
            pip_path = self._get_pip_path(toolkit_type)
            print(f"  -> 检查虚拟环境 Python: {python_path}")

            if not python_path.exists():
                print(f"  ✗ Python 可执行文件不存在: {python_path}")
                if venv_path.exists():
                    try:
                        subdirs = [p.name for p in venv_path.iterdir()]
                        print(f"  -> venv 目录内容: {subdirs}")
                        bin_dir = venv_path / "bin"
                        if bin_dir.exists():
                            bins = [p.name for p in bin_dir.iterdir()]
                            print(f"  -> bin 目录内容: {bins[:10]}")
                    except Exception as e:
                        print(f"  -> 列出目录失败: {e}")
                return False, f"Python创建失败: {python_path}"
            print(f"  -> Python 可执行文件确认存在")

            # 确保pip存在
            print(f"  -> 检查 pip: {pip_path}")
            if not pip_path.exists():
                print(f"  -> pip 不存在，使用 ensurepip 安装...")
                ensure_result = subprocess.run(
                    [str(python_path), "-m", "ensurepip", "--upgrade"],
                    capture_output=True, text=True, timeout=60
                )
                print(f"  -> ensurepip 返回码: {ensure_result.returncode}")
                if ensure_result.returncode != 0:
                    err = ensure_result.stderr.strip()[:300] if ensure_result.stderr else "无错误输出"
                    print(f"  ✗ ensurepip 失败: {err}")
                    return False, f"ensurepip失败: {ensure_result.stderr}"
                print(f"  -> ensurepip 成功")
            else:
                print(f"  -> pip 已存在")

            self._check_all_envs()
            print(f"  ✓ 虚拟环境创建完成: {venv_path}")
            return True, f"虚拟环境创建成功: {venv_path}"

        except subprocess.TimeoutExpired:
            print(f"  ✗ 创建超时 (>60s)")
            return False, "创建超时"
        except Exception as e:
            print(f"  ✗ 创建异常: {e}")
            return False, f"创建失败: {str(e)}"

    def _build_install_cmd(self, toolkit_type: str) -> tuple[list[str], str]:
        """构建安装命令，优先使用 uv pip（清华源），回退到标准 pip"""
        uv_path = self._get_uv_path()
        python_path = self._get_python_path(toolkit_type)
        pip_path = self._get_pip_path(toolkit_type)

        if uv_path:
            return [uv_path, "pip", "install", "--python", str(python_path), "-i", TSINGHUA_PYPI_INDEX], "uv"

        use_pip_module = not pip_path.exists()
        if use_pip_module:
            return [str(python_path), "-m", "pip", "install", "-i", TSINGHUA_PYPI_INDEX], "pip"
        return [str(pip_path), "install", "-i", TSINGHUA_PYPI_INDEX], "pip"

    def install_rknn(self, toolkit_type: str) -> tuple[bool, str]:
        """在虚拟环境中安装rknn-toolkit（优先使用 uv pip）"""
        python_path = self._get_python_path(toolkit_type)
        toolkit_info = TOOLKIT_VENV_MAP[toolkit_type]
        package_name = toolkit_info["package_name"]
        python_version = toolkit_info["python_version"]
        install_mode = toolkit_info.get("install_mode", "whl")
        versions = toolkit_info.get("versions", [])
        venv_path = self._get_venv_path(toolkit_type)

        # 确保虚拟环境存在
        if not python_path.exists():
            print(f"[步骤1/2] 虚拟环境不存在，开始创建...")
            success, msg = self.create_venv(toolkit_type)
            print(msg)
            if not success:
                return False, msg
        else:
            print(f"[步骤1/2] 虚拟环境已存在: {venv_path}")

        install_cmd, tool_name = self._build_install_cmd(toolkit_type)
        print(f"  安装工具: {tool_name}")
        print(f"  安装命令: {' '.join(install_cmd)}")
        print(f"  包名: {package_name}, Python版本: {python_version}, 安装模式: {install_mode}")
        print(f"  系统: {SYSTEM_INFO['system']}, 架构: {SYSTEM_INFO['arch']}, cp标签: {get_python_cp_tag(python_version)}")

        try:
            # 跳过 pip 源安装，直接下载 whl
            print(f"[步骤2/2] 跳过 pip 源，直接下载 whl 安装...")
            if install_mode == "tar.gz":
                print(f"  模式: tar.gz（从GitHub下载大文件，可能较慢）")
                downloaded_whl = self._download_and_extract_toolkit_tar(toolkit_info, python_version)
            else:
                print(f"  模式: whl（从GitHub直接下载）")
                downloaded_whl = self._download_toolkit2_whl(toolkit_info, python_version)

            if not downloaded_whl:
                print(f"✗ 未找到可下载的 whl 文件")
                return False, "下载失败"

            print(f"安装本地 whl: {downloaded_whl.name}")
            cmd_display = " ".join(install_cmd) + f" {str(downloaded_whl)}"
            print(cmd_display)
            
            cmd = install_cmd + [str(downloaded_whl)]
            
            process = subprocess.Popen(
                cmd,
                stdout=None,  # 直接输出到终端，保留 uv 原生进度条
                stderr=None,
                text=True,
            )

            try:
                returncode = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print(f"✗ 安装超时")
                return False, "安装超时"

            if returncode == 0:
                print(f"✓ 安装成功")
                self._check_all_envs()
                return True, "安装成功"
            else:
                print(f"✗ 安装失败 (返回码: {returncode})")
                return False, f"安装失败 (返回码: {returncode})"

        except subprocess.TimeoutExpired as e:
            print(f"✗ 安装超时: {e}")
            return False, "安装超时"
        except Exception as e:
            print(f"✗ 安装异常: {str(e)}")
            return False, f"安装异常: {str(e)}"

    def _build_whl_names(self, package_name: str, python_version: str, versions: list[str]) -> list[str]:
        """
        根据系统架构和Python版本构建可能的whl文件名列表
        用于 rknn-toolkit2 直接从 GitHub raw 下载 whl
        """
        whl_names = []
        pkg_whl_name = package_name.replace("-", "_")
        cp_tag = get_python_cp_tag(python_version)
        arch = SYSTEM_INFO["arch"]
        system = SYSTEM_INFO["system"]

        for version in versions:
            if system == "linux":
                if arch == "x86_64":
                    # 实际文件名同时包含两种 manylinux 标签（用点连接）
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux_2_17_x86_64.whl")
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux2014_x86_64.whl")
                elif arch == "aarch64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl")
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux_2_17_aarch64.whl")
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux2014_aarch64.whl")
                elif arch == "armv7l":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-linux_armv7l.whl")
            elif system == "darwin":
                if arch == "x86_64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-macosx_10_9_x86_64.whl")
                elif arch == "aarch64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-macosx_11_0_arm64.whl")
            elif system == "windows":
                if arch == "x86_64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-win_amd64.whl")

        return whl_names

    def _download_file(self, url: str, dest: Path, timeout: int = 60,
                        max_retries: int = 3) -> bool:
        """带超时、速度检测、进度打印和重试的文件下载"""
        import urllib.request
        import urllib.error

        print(f"开始下载: {url}")
        connect_timeout = 120  # SSL 握手等连接阶段超时
        chunk_size = 1024 * 1024  # 1MB chunks，减少 Python 循环和系统调用次数

        for attempt in range(max_retries):
            if attempt > 0:
                delay = 2 ** (attempt - 1)  # 指数退避: 1, 2, 4 秒
                print(f"  第 {attempt + 1}/{max_retries} 次尝试，等待 {delay}s...")
                time.sleep(delay)
                if dest.exists():
                    dest.unlink()

            start_time = time.time()
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.0'
                })
                response = urllib.request.urlopen(req, timeout=connect_timeout)

                total_size = response.headers.get('Content-Length')
                total_size = int(total_size) if total_size else None
                if total_size and total_size > 500 * 1024 * 1024:
                    print(f"  警告: 文件大小 {total_size / (1024*1024):.0f}MB，下载可能需要较长时间")

                downloaded = 0
                last_report_time = start_time
                last_report_bytes = 0

                # 1MB 文件写入缓冲，减少 write 系统调用次数
                with open(dest, 'wb', buffering=1024 * 1024) as f:
                    while True:
                        chunk_start = time.time()
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)
                        chunk_time = time.time() - chunk_start

                        # 检测是否卡死: 1MB 读取超过 120 秒认为卡住（约 8.5KB/s）
                        if chunk_time > 120:
                            print(f"  下载速度过慢，单个块耗时 {chunk_time:.0f}s，中止")
                            raise TimeoutError("Download stalled")

                        # 检测总体超时
                        elapsed = time.time() - start_time
                        if elapsed > timeout:
                            print(f"  下载总体超时 ({timeout}s)，已下载 {downloaded / (1024*1024):.1f}MB")
                            raise TimeoutError(f"Download timeout after {timeout}s")

                        # 每 5 秒或每 10MB 报告一次
                        now = time.time()
                        if now - last_report_time > 5 or downloaded - last_report_bytes > 10 * 1024 * 1024:
                            mb = downloaded / (1024 * 1024)
                            speed = (downloaded - last_report_bytes) / max(now - last_report_time, 0.001)
                            if total_size:
                                pct = min(100, downloaded * 100 // total_size)
                                total_mb = total_size / (1024 * 1024)
                                print(f"  进度: {pct}% ({mb:.1f}/{total_mb:.1f} MB, {speed/1024:.1f} KB/s)")
                            else:
                                print(f"  已下载: {mb:.1f} MB ({speed/1024:.1f} KB/s)")
                            last_report_time = now
                            last_report_bytes = downloaded

                elapsed = time.time() - start_time
                file_size = dest.stat().st_size / (1024 * 1024)
                print(f"下载完成: {dest.name} ({file_size:.1f} MB, {elapsed:.1f}s)")
                return True

            except Exception as e:
                err_msg = str(e)[:120]
                print(f"  下载异常: {err_msg}")
                if dest.exists():
                    dest.unlink()
                if attempt == max_retries - 1:
                    print(f"下载失败，已重试 {max_retries} 次")
                    return False
                print(f"  即将重试...")

        return False

    def _download_and_extract_toolkit_tar(self, toolkit_info: dict, python_version: str) -> Optional[Path]:
        """下载 rknn-toolkit tar.gz 并找到匹配的 whl"""
        import tarfile

        versions = toolkit_info["versions"]
        tar_url_template = toolkit_info["tar_url"]
        cp_tag = get_python_cp_tag(python_version)
        arch = SYSTEM_INFO["arch"]

        for version in versions:
            extract_dir = WHL_DIR / f"rknn-toolkit-v{version}-packages"
            tar_name = f"rknn-toolkit-v{version}-packages.tar.gz"
            tar_path = WHL_DIR / tar_name
            tar_url = tar_url_template.format(version=version)

            print(f"尝试版本 {version}...")

            # 先检查本地是否已有解压好的 whl
            if extract_dir.exists():
                whl_files = list(extract_dir.rglob("*.whl"))
                print(f"  本地已解压，找到 {len(whl_files)} 个 whl")
                matching_whl = self._find_matching_whl(whl_files, cp_tag, arch)
                if matching_whl:
                    return matching_whl

            # 检查本地是否已有 tar.gz
            if tar_path.exists():
                print(f"  本地已有 tar.gz，尝试解压...")
            else:
                # 下载（文件可能很大，超时设长一些）
                print(f"  从 GitHub 下载 tar.gz（文件可能很大，请耐心等待或手动下载）...")
                if not self._download_file(tar_url, tar_path, timeout=7200):
                    continue

            # 解压
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    try:
                        tar.extractall(extract_dir, filter="data")
                    except TypeError:
                        # Python 旧版本不支持 filter 参数，退回手动校验防穿越
                        base_resolved = extract_dir.resolve()
                        for member in tar.getmembers():
                            member_path = (extract_dir / member.name).resolve()
                            try:
                                member_path.relative_to(base_resolved)
                            except ValueError:
                                raise RuntimeError(f"危险的 tar 路径: {member.name}")
                        tar.extractall(extract_dir)
                print(f"  解压完成: {extract_dir}")
            except Exception as e:
                print(f"  解压失败: {str(e)[:80]}")
                continue

            # 查找匹配的 whl
            whl_files = list(extract_dir.rglob("*.whl"))
            print(f"  找到 {len(whl_files)} 个 whl 文件")

            matching_whl = self._find_matching_whl(whl_files, cp_tag, arch)
            if matching_whl:
                return matching_whl

        return None

    def _find_matching_whl(self, whl_files: list[Path], cp_tag: str, arch: str) -> Optional[Path]:
        """从 whl 文件列表中找到匹配 Python 版本和架构的 whl"""
        arch_aliases = {
            "x86_64": ["x86_64", "amd64"],
            "aarch64": ["aarch64", "arm64"],
            "armv7l": ["armv7l"],
        }
        aliases = arch_aliases.get(arch, [arch])

        print(f"  查找匹配 whl (cpTag={cp_tag}, arch={arch}, aliases={aliases})")
        print(f"  候选文件数: {len(whl_files)}")
        for whl_file in whl_files:
            name = whl_file.name
            name_lower = name.lower()
            if cp_tag.lower() not in name_lower:
                continue
            for alias in aliases:
                if alias.lower() in name_lower:
                    print(f"  ✓ 匹配成功: {name} (alias={alias})")
                    return whl_file
        print(f"  ✗ 未找到匹配的 whl")
        return None

    def _download_toolkit2_whl(self, toolkit_info: dict, python_version: str) -> Optional[Path]:
        """下载 rknn-toolkit2 的 whl 文件（直接从 GitHub raw）"""
        versions = toolkit_info["versions"]
        whl_base_url = toolkit_info["whl_base_url"]
        package_name = toolkit_info["package_name"]
        arch = SYSTEM_INFO["arch"]
        # GitHub 上的子目录名：x86_64 或 arm64
        subdir = "arm64" if arch == "aarch64" else arch

        print(f"  构建候选 whl 列表 (Python={python_version}, arch={arch})...")
        for version in versions:
            whl_names = self._build_whl_names(package_name, python_version, [version])
            print(f"  版本 {version}: {len(whl_names)} 个候选文件")
            for name in whl_names:
                print(f"    - {name}")

            for idx, whl_name in enumerate(whl_names):
                whl_url = f"{whl_base_url}/{subdir}/{whl_name}"
                whl_path = WHL_DIR / whl_name
                print(f"  尝试下载 {idx+1}/{len(whl_names)}: {whl_name}")

                # 优先使用本地已下载的 whl
                if whl_path.exists() and whl_path.stat().st_size > 1000:
                    print(f"  ✓ 使用本地已下载的 whl: {whl_path.name} ({whl_path.stat().st_size / (1024*1024):.1f} MB)")
                    return whl_path

                if self._download_file(whl_url, whl_path, timeout=120):
                    return whl_path
                else:
                    print(f"  该文件不存在或下载失败，继续下一个")

        print(f"  所有候选 whl 下载均失败")
        return None

    def prepare_env(self, toolkit_type: str) -> tuple[bool, str]:
        """准备虚拟环境（创建并安装rknn）"""
        print("=" * 50)
        print(f"开始准备环境: {toolkit_type}")
        print("=" * 50)

        status = self.get_status(toolkit_type)
        print(f"当前状态: 存在={status.get('exists', False)}, "
              f"Python就绪={status.get('python_ready', False)}, "
              f"rknn已安装={status.get('rknn_installed', False)}")

        if status.get("rknn_installed"):
            print("✓ 环境已就绪，无需操作")
            return True, "环境已就绪"

        # 检查虚拟环境是否存在
        if not status.get("exists"):
            print("--- 阶段1: 创建虚拟环境 ---")
            success, msg = self.create_venv(toolkit_type)
            print(msg)
            if not success:
                print("✗ 环境准备失败: 虚拟环境创建失败")
                return False, msg
            print("✓ 虚拟环境创建完成")
        else:
            print("--- 阶段1: 虚拟环境已存在，跳过创建 ---")

        # 安装rknn
        print("--- 阶段2: 安装 RKNN Toolkit ---")
        success, msg = self.install_rknn(toolkit_type)

        if success:
            print("✓ 环境准备完成")
        else:
            print("✗ 环境准备失败")

        print("=" * 50)
        return success, msg


# 全局虚拟环境管理器
venv_manager = VirtualEnvManager()

# 存储转换任务
tasks: dict[str, ConversionTask] = {}
MAX_TASKS = 100
# 转换子进程超时（秒），默认 10 分钟，可通过 RKNN_CONVERT_TIMEOUT 环境变量调整
CONVERT_TIMEOUT = int(os.environ.get("RKNN_CONVERT_TIMEOUT", "600"))
_tasks_lock = threading.Lock()


def _safe_under_dir(base_dir: Path, untrusted_name: Optional[str]) -> Optional[Path]:
    """
    将不可信的文件名/相对路径限定在 base_dir 之内。

    返回安全的绝对路径；若输入为空、为绝对路径，或 resolve 后逃出 base_dir，则返回 None。
    """
    if not untrusted_name:
        return None
    candidate = Path(untrusted_name)
    if candidate.is_absolute():
        return None
    try:
        resolved = (base_dir / candidate).resolve()
    except (OSError, RuntimeError):
        return None
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        return None
    return resolved


def _resolve_dataset_path(raw: Optional[str]) -> Optional[str]:
    """限制用户提交的 dataset 路径必须在 DATASET_DIR 之内，越权返回 None"""
    if not raw:
        return None
    safe = _safe_under_dir(DATASET_DIR, raw)
    return str(safe) if safe else None


def _cleanup_task_files(task: ConversionTask) -> None:
    """删除任务关联的上传 / 输出文件，仅清理 UPLOAD_DIR 与 OUTPUT_DIR 之内的路径"""
    for path, base_dir in (
        (task.onnx_path, UPLOAD_DIR),
        (task.output_path, OUTPUT_DIR),
    ):
        if not path:
            continue
        try:
            abs_path = Path(path).resolve()
            abs_path.relative_to(base_dir.resolve())
        except (OSError, ValueError):
            continue
        try:
            abs_path.unlink(missing_ok=True)
        except OSError:
            pass


def _add_task(task_id: str, task: ConversionTask) -> None:
    """添加任务，超过上限时优先淘汰最早的已完成任务，避免误删运行中任务"""
    with _tasks_lock:
        if len(tasks) >= MAX_TASKS:
            finished = [
                t for t in tasks.values()
                if t.status in ("completed", "failed")
            ]
            for old_task in sorted(finished, key=lambda t: t.start_time)[:20]:
                _cleanup_task_files(old_task)
                tasks.pop(old_task.task_id, None)
        tasks[task_id] = task


app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    raise RuntimeError("SECRET_KEY 环境变量未设置，请设置后重新启动")
app.secret_key = secret_key
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


def get_onnx_info(onnx_path: str) -> dict:
    """获取ONNX模型信息"""
    try:
        import onnx
        model = onnx.load(onnx_path)

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

        file_size = Path(onnx_path).stat().st_size / (1024 * 1024)

        return {
            "success": True,
            "inputs": inputs,
            "outputs": outputs,
            "file_size": f"{file_size:.2f} MB"
        }
    except ImportError:
        return {"success": False, "error": "onnx库未安装: pip install onnx"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_conversion_in_venv_async(task: ConversionTask):
    """异步执行转换（在后台线程中）"""
    success, message = run_conversion_in_venv(task)

    task.end_time = datetime.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.end_time - task.start_time).total_seconds()
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

    toolkit_type = CHIP_PLATFORMS[platform]["toolkit"]

    # 准备虚拟环境
    task.add_log(f"目标平台: {platform}")
    task.add_log(f"需要Toolkit: {toolkit_type}")

    success, msg = venv_manager.prepare_env(toolkit_type)

    if not success:
        return False, f"环境准备失败: {msg}"

    task.add_log(f"环境准备完成: {msg}")

    # 获取虚拟环境中的Python路径
    python_path = venv_manager._get_python_path(toolkit_type)
    convert_script = BASE_DIR / "convert_worker.py"

    # 构建转换参数
    config_json = json.dumps(config.to_dict())

    task.add_log(f"使用Python: {python_path}")
    task.add_log(f"转换脚本: {convert_script}")
    task.add_log(f"配置参数: {config_json}")

    # 在虚拟环境中执行转换脚本（Popen 实时读取输出）
    try:
        task.process = subprocess.Popen(
            [
                str(python_path),
                str(convert_script),
                "--onnx", task.onnx_path,
                "--output", task.output_path,
                "--config", config_json
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 实时读取 stdout（stderr 已重定向到 stdout）
        if task.process.stdout:
            for line in task.process.stdout:
                _parse_worker_line(task, line.rstrip("\n"))

        # 等待进程结束（带超时）
        try:
            returncode = task.process.wait(timeout=CONVERT_TIMEOUT)
        except subprocess.TimeoutExpired:
            task.process.kill()
            task.process.wait()
            timeout_min = CONVERT_TIMEOUT / 60
            task.add_log(f"转换超时 (>{timeout_min:.0f}分钟)")
            return False, f"转换超时 (>{timeout_min:.0f}分钟)"

        if returncode != 0:
            return False, "转换进程返回错误"

        # 检查输出文件
        if Path(task.output_path).exists():
            output_size = Path(task.output_path).stat().st_size / (1024 * 1024)
            task.add_log(f"输出文件: {task.output_path}")
            task.add_log(f"文件大小: {output_size:.2f} MB")
            return True, f"转换成功: {task.output_path}"
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
        venv_status=venv_manager.get_all_status(),
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
    task_id = str(uuid.uuid4())[:8]
    safe_name = secure_filename(file.filename or "") or "model.onnx"
    filename = f"{task_id}_{safe_name}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    # 构建配置
    config = ConversionConfig(platform=platform)

    input_height = request.form.get("input_height", type=int)
    input_width = request.form.get("input_width", type=int)
    if input_height and input_width:
        config.input_size = (input_height, input_width)

    config.input_dtype = request.form.get("input_dtype", "float32") or "float32"

    mean_str = request.form.get("mean_values", "")
    if mean_str:
        try:
            config.mean_values = [float(x.strip()) for x in mean_str.split(",")]
        except ValueError:
            pass

    std_str = request.form.get("std_values", "")
    if std_str:
        try:
            config.std_values = [float(x.strip()) for x in std_str.split(",")]
        except ValueError:
            pass

    config.do_quantization = request.form.get("do_quantization") == "on"
    config.quantized_dtype = request.form.get("quantized_dtype", "asymmetric_quantized-u8")
    config.quantized_algorithm = request.form.get("quantized_algorithm", "normal")
    config.dataset_path = _resolve_dataset_path(request.form.get("dataset_path", ""))

    config.optimization_level = request.form.get("optimization_level", type=int, default=2)
    config.single_core_mode = request.form.get("single_core_mode") == "on"

    batch_size = request.form.get("batch_size", type=int, default=1) or 1
    config.batch_size = batch_size

    # 创建任务
    task = ConversionTask(task_id, str(filepath), config)
    _add_task(task_id, task)

    output_filename = f"{task_id}_{Path(filepath).stem}.rknn"
    output_path = OUTPUT_DIR / output_filename
    task.output_path = str(output_path)

    # 执行转换
    task.status = "converting"
    task.add_log(f"文件: {file.filename}")
    task.add_log(f"平台: {platform}")
    task.add_log(f"配置: {config.to_dict()}")

    success, message = run_conversion_in_venv(task)

    task.end_time = datetime.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.end_time - task.start_time).total_seconds()
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

    task_id = str(uuid.uuid4())[:8]
    safe_name = secure_filename(file.filename or "") or "model.onnx"
    filename = f"{task_id}_{safe_name}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    onnx_info = get_onnx_info(str(filepath))

    return render_template(
        "config.html",
        task_id=task_id,
        filename=file.filename,
        filepath=str(filepath),
        platforms=CHIP_PLATFORMS,
        onnx_info=onnx_info,
        quantized_dtypes=QUANTIZED_DTYPES,
        quantized_algorithms=QUANTIZED_ALGORITHMS,
        optimization_levels=OPTIMIZATION_LEVELS,
        venv_status=venv_manager.get_all_status()
    )


@app.route("/convert", methods=["POST"])
def convert():
    """执行转换"""
    task_id = request.form.get("task_id", "")
    filepath = request.form.get("filepath", "")
    platform = request.form.get("platform", "")

    if not all([task_id, filepath, platform]):
        return jsonify({"success": False, "message": "参数不完整"})

    if platform not in CHIP_PLATFORMS:
        return jsonify({"success": False, "message": f"不支持的平台: {platform}"})

    # 构建配置
    config = ConversionConfig(platform=platform)

    input_height = request.form.get("input_height", type=int)
    input_width = request.form.get("input_width", type=int)
    if input_height and input_width:
        config.input_size = (input_height, input_width)

    config.input_name = request.form.get("input_name", "") or None
    config.input_dtype = request.form.get("input_dtype", "float32") or "float32"

    mean_str = request.form.get("mean_values", "")
    if mean_str:
        try:
            config.mean_values = [float(x.strip()) for x in mean_str.split(",")]
        except ValueError:
            pass

    std_str = request.form.get("std_values", "")
    if std_str:
        try:
            config.std_values = [float(x.strip()) for x in std_str.split(",")]
        except ValueError:
            pass

    config.do_quantization = request.form.get("do_quantization") == "on"
    config.quantized_dtype = request.form.get("quantized_dtype", "asymmetric_quantized-u8")
    config.quantized_algorithm = request.form.get("quantized_algorithm", "normal")
    config.dataset_path = _resolve_dataset_path(request.form.get("dataset_path", ""))

    config.optimization_level = request.form.get("optimization_level", type=int, default=2)
    config.single_core_mode = request.form.get("single_core_mode") == "on"

    data_size_str = request.form.get("model_data_size", "")
    if data_size_str:
        try:
            config.model_data_size = int(data_size_str)
        except ValueError:
            pass

    config.batch_size = request.form.get("batch_size", type=int, default=1) or 1

    # 创建任务
    task = ConversionTask(task_id, filepath, config)
    _add_task(task_id, task)

    output_filename = f"{task_id}_{Path(filepath).stem}.rknn"
    output_path = OUTPUT_DIR / output_filename
    task.output_path = str(output_path)

    # 执行转换
    task.status = "converting"
    task.add_log(f"开始转换: {filepath}")
    task.add_log(f"配置: {config.to_dict()}")

    # 同步执行（阻塞）
    success, message = run_conversion_in_venv(task)

    task.end_time = datetime.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.end_time - task.start_time).total_seconds()
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
    task_id = request.form.get("task_id", "")
    filepath = request.form.get("filepath", "")
    platform = request.form.get("platform", "")

    if not all([task_id, filepath, platform]):
        return jsonify({"success": False, "message": "参数不完整"})

    if platform not in CHIP_PLATFORMS:
        return jsonify({"success": False, "message": f"不支持的平台: {platform}"})

    # 构建配置（同上）
    config = ConversionConfig(platform=platform)

    input_height = request.form.get("input_height", type=int)
    input_width = request.form.get("input_width", type=int)
    if input_height and input_width:
        config.input_size = (input_height, input_width)

    config.input_name = request.form.get("input_name", "") or None
    config.input_dtype = request.form.get("input_dtype", "float32") or "float32"

    mean_str = request.form.get("mean_values", "")
    if mean_str:
        try:
            config.mean_values = [float(x.strip()) for x in mean_str.split(",")]
        except ValueError:
            pass

    std_str = request.form.get("std_values", "")
    if std_str:
        try:
            config.std_values = [float(x.strip()) for x in std_str.split(",")]
        except ValueError:
            pass

    config.do_quantization = request.form.get("do_quantization") == "on"
    config.quantized_dtype = request.form.get("quantized_dtype", "asymmetric_quantized-u8")
    config.quantized_algorithm = request.form.get("quantized_algorithm", "normal")
    config.dataset_path = _resolve_dataset_path(request.form.get("dataset_path", ""))

    config.optimization_level = request.form.get("optimization_level", type=int, default=2)
    config.single_core_mode = request.form.get("single_core_mode") == "on"

    data_size_str = request.form.get("model_data_size", "")
    if data_size_str:
        try:
            config.model_data_size = int(data_size_str)
        except ValueError:
            pass

    config.batch_size = request.form.get("batch_size", type=int, default=1) or 1

    # 创建任务
    task = ConversionTask(task_id, filepath, config)
    _add_task(task_id, task)

    output_filename = f"{task_id}_{Path(filepath).stem}.rknn"
    output_path = OUTPUT_DIR / output_filename
    task.output_path = str(output_path)

    task.status = "converting"
    task.add_log(f"开始转换: {filepath}")
    task.add_log(f"配置: {config.to_dict()}")

    # 启动后台线程
    thread = threading.Thread(target=run_conversion_in_venv_async, args=(task,))
    thread.start()

    return jsonify({
        "success": True,
        "task_id": task_id,
        "message": "转换任务已启动"
    })


@app.route("/prepare_env/<toolkit_type>")
def prepare_env(toolkit_type: str):
    """准备虚拟环境"""
    if toolkit_type not in TOOLKIT_VENV_MAP:
        return jsonify({"success": False, "message": f"未知的toolkit类型: {toolkit_type}"})

    success, msg = venv_manager.prepare_env(toolkit_type)
    return jsonify({
        "success": success,
        "message": msg,
        "logs": [],
        "status": venv_manager.get_status(toolkit_type)
    })


@app.route("/env_status")
def env_status():
    """获取虚拟环境状态"""
    return jsonify(venv_manager.get_all_status())


@app.route("/download/<task_id>")
def download(task_id: str):
    """下载转换后的模型"""
    task = tasks.get(task_id)
    if not task or not task.output_path:
        flash("任务不存在或未完成", "error")
        return redirect(url_for("index"))

    if not Path(task.output_path).exists():
        flash("文件不存在", "error")
        return redirect(url_for("index"))

    return send_file(
        task.output_path,
        as_attachment=True,
        download_name=Path(task.output_path).name
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
            "task_id": t.task_id,
            "platform": t.config.platform,
            "status": t.status,
            "message": t.message,
            "start_time": t.start_time.isoformat(),
            "end_time": t.end_time.isoformat() if t.end_time else None
        }
        for t in tasks.values()
    ])


@app.route("/api/task/<task_id>")
def api_task(task_id: str):
    """获取单个任务状态API"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"})

    return jsonify({
        "success": True,
        "task_id": task.task_id,
        "platform": task.config.platform,
        "status": task.status,
        "message": task.message,
        "output_path": task.output_path,
        "start_time": task.start_time.isoformat(),
        "end_time": task.end_time.isoformat() if task.end_time else None,
        "log": task.log[-20:] if len(task.log) > 20 else task.log
    })


@app.route("/api/info/<path:filepath>")
def api_info(filepath: str):
    """获取ONNX模型信息API"""
    safePath = _safe_under_dir(UPLOAD_DIR, filepath)
    if safePath is None:
        return jsonify({"success": False, "error": "非法路径"}), 400
    return jsonify(get_onnx_info(str(safePath)))


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
    uv_ok, uv_msg = venv_manager._ensure_uv_installed()
    print(f"  结果: {uv_msg}")
    if not uv_ok:
        print("\n" + "=" * 70)
        print("⚠ uv 不可用，仅执行状态检查（不自动安装环境）")
        print("=" * 70)
        return _check_env_status_only()

    # 步骤2: 确保所需 Python 版本已安装
    print("\n[步骤2/3] 确保 Python 版本...")
    required_py_versions = {info["python_version"] for info in TOOLKIT_VENV_MAP.values()}
    for py_version in sorted(required_py_versions):
        py_ok, py_msg = venv_manager._ensure_python_installed(py_version)
        print(f"  {py_version}: {'✓' if py_ok else '✗'} {py_msg}")
        if not py_ok:
            print(f"\n⚠ Python {py_version} 安装失败，跳过对应环境的自动安装")

    # 步骤3: 自动创建虚拟环境并安装 rknn
    print("\n[步骤3/3] 创建虚拟环境并安装 RKNN Toolkit...")
    results = {}
    for toolkit_type, info in TOOLKIT_VENV_MAP.items():
        print(f"\n[{toolkit_type}]")
        print(f"  虚拟环境: {info['venv_name']}")
        print(f"  包名: {info['package_name']}")

        status = venv_manager.get_status(toolkit_type)
        if status.get("rknn_installed"):
            print(f"  ✓ 已就绪，跳过")
            results[toolkit_type] = {"success": True, "message": "已就绪", "skipped": True}
            continue

        print(f"  -> 开始准备环境...")
        success, msg = venv_manager.prepare_env(toolkit_type)
        results[toolkit_type] = {"success": success, "message": msg}

    print("\n" + "=" * 70)
    print("初始化完成")

    ready_count = sum(1 for r in results.values() if r.get("success"))
    total_count = len(TOOLKIT_VENV_MAP)
    print(f"环境状态: {ready_count}/{total_count} 就绪")

    for toolkit_type, result in results.items():
        status_icon = "✓" if result.get("success") else "✗"
        print(f"  {status_icon} {toolkit_type}: {result.get('message')}")

    print("=" * 70)
    return results


def _check_env_status_only() -> dict:
    """仅检查环境状态，不自动安装"""
    results = {}
    for toolkit_type, info in TOOLKIT_VENV_MAP.items():
        print(f"\n[{toolkit_type}]")
        print(f"  虚拟环境: {info['venv_name']}")
        print(f"  包名: {info['package_name']}")

        status = venv_manager.get_status(toolkit_type)
        venv_path = status.get("venv_path", "未知")
        python_path = status.get("python_path", "未知")
        exists = status.get("exists", False)
        python_ready = status.get("python_ready", False)
        rknn_installed = status.get("rknn_installed", False)

        print(f"  路径: {venv_path}")
        print(f"  Python: {python_path}")
        print(f"  存在={exists}, Python就绪={python_ready}, rknn已安装={rknn_installed}")

        if rknn_installed:
            print(f"  状态: ✓ 已就绪")
            results[toolkit_type] = {"success": True, "message": "已就绪", "skipped": True}
        elif exists and python_ready:
            print(f"  状态: ○ 虚拟环境已创建，缺少 rknn 包")
            results[toolkit_type] = {"success": True, "message": "待安装", "skipped": True}
        elif exists:
            print(f"  状态: ○ 虚拟环境目录存在但不完整")
            results[toolkit_type] = {"success": True, "message": "待重建", "skipped": True}
        else:
            print(f"  状态: ○ 未创建")
            results[toolkit_type] = {"success": True, "message": "待创建", "skipped": True}

    print("\n" + "=" * 70)
    ready_count = sum(1 for r in results.values() if r.get("message") == "已就绪")
    total_count = len(TOOLKIT_VENV_MAP)
    print(f"环境状态: {ready_count}/{total_count} 就绪")
    for toolkit_type, result in results.items():
        status_icon = "✓" if result.get("message") == "已就绪" else "○"
        print(f"  {status_icon} {toolkit_type}: {result.get('message')}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX转RKNN Web转换工具")
    parser.add_argument("--skip-init-env", action="store_true",
                        help="跳过虚拟环境初始化检查")
    args = parser.parse_args()

    if args.skip_init_env:
        print("跳过环境初始化...")
        init_results = {}
    else:
        init_results = init_virtual_environments()

    print()
    print("ONNX转RKNN Web转换工具")
    print(f"访问: http://localhost:5000")
    print("=" * 70)

    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    # 默认仅监听本机回环，避免局域网未授权访问；如需对外开放请显式设置 FLASK_HOST=0.0.0.0
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    print(f"绑定地址: {host}:{port}（如需对外暴露请设置 FLASK_HOST=0.0.0.0）")
    app.run(host=host, port=port, debug=debug_mode)
