#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据模型"""

import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ConversionConfig:
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
        with self._log_lock:
            self.log.append(msg)
        print(f"[{self.task_id}] {msg}")

    def to_dict(self) -> dict:
        duration = None
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        return {
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
            "onnx_path": self.onnx_path,
            "output_path": self.output_path,
            "platform": self.config.platform,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": duration,
            "log_count": len(self.log),
        }