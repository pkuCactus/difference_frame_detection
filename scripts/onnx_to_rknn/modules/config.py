#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置常量"""

import os
import platform
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "datasets"
WHL_DIR = BASE_DIR / "whl_files"

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
VENV_DIR = TMPDIR / "onnx_to_rknn_venvs"

TSINGHUA_PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
TSINGHUA_PYTHON_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/python/"

CONVERT_TIMEOUT = 10 * 60  # 10分钟

for d in [UPLOAD_DIR, OUTPUT_DIR, DATASET_DIR, VENV_DIR, WHL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_system_info() -> dict:
    machine = platform.machine().lower()
    system = platform.system().lower()
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "armv7l": "armv7l",
    }
    return {
        "system": system,
        "arch": arch_map.get(machine, machine),
        "machine": machine,
    }


SYSTEM_INFO = get_system_info()


def get_python_cp_tag(py_version: str) -> str:
    major, minor = py_version.split(".")[:2]
    return f"cp{major}{minor}"


TOOLKIT_VENV_MAP = {
    "rknn-toolkit": {
        "venv_name": "venv_toolkit",
        "package_name": "rknn-toolkit",
        "python_version": "3.8",
        "install_mode": "tar.gz",
        "tar_url": "https://github.com/rockchip-linux/rknn-toolkit/releases/download/v{version}/rknn-toolkit-v{version}-packages.tar.gz",
        "versions": ["1.7.5", "1.7.3", "1.7.1", "1.7.0", "1.6.1", "1.6.0"],
    },
    "rknn-toolkit2": {
        "venv_name": "venv_toolkit2",
        "package_name": "rknn-toolkit2",
        "python_version": "3.12",
        "install_mode": "whl",
        "whl_base_url": "https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages",
        "versions": ["2.3.2", "2.2.0", "2.1.0", "2.0.0"],
        "onnx_version": "1.18.0",
    },
}

CHIP_PLATFORMS = {
    "RK1808": {"toolkit": "rknn-toolkit", "target": "rk1808", "description": "RK1808 NPU"},
    "RV1109": {"toolkit": "rknn-toolkit", "target": "rv1109", "description": "RV1109 ISP"},
    "RV1126": {"toolkit": "rknn-toolkit", "target": "rv1126", "description": "RV1126 ISP"},
    "RK3399Pro": {"toolkit": "rknn-toolkit", "target": "rk3399pro", "description": "RK3399Pro NPU"},
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

QUANTIZED_DTYPES = {
    "w8a8": "W8A8 量化 (权重8位,激活8位,推荐)",
    "w8a16": "W8A16 量化 (权重8位,激活16位)",
    "w16a16i": "W16A16 整数量化",
    "w16a16i_dfp": "W16A16 定点浮点量化",
    "w4a16": "W4A16 量化 (权重4位,激活16位)",
}

QUANTIZED_ALGORITHMS = {
    "normal": "普通量化 (快速)",
    "mmse": "MMSE量化 (最小均方误差)",
    "kl_divergence": "KL散度量化 (精度高，速度慢)",
    "gdq": "GDQ量化 (梯度分布量化)",
}

OPTIMIZATION_LEVELS = {
    1: "基本优化",
    2: "标准优化 (推荐)",
    3: "激进优化 (可能影响精度)",
}