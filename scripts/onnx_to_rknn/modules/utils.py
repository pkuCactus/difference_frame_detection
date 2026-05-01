#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工具函数"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Response

from .config import BASE_DIR, CHIP_PLATFORMS, CONVERT_TIMEOUT, DATASET_DIR
from .models import ConversionConfig, ConversionTask


tasks: dict[str, ConversionTask] = {}
MAX_TASKS = 100
_tasks_lock = threading.Lock()


def _safe_under_dir(base_dir: Path, untrusted_name: Optional[str]) -> Optional[Path]:
    if not untrusted_name:
        return None
    try:
        safe_name = Path(untrusted_name).name
        if safe_name.startswith("/") or ".." in safe_name:
            return None
        full_path = (base_dir / safe_name).resolve()
        if str(full_path).startswith(str(base_dir.resolve())):
            return full_path
    except Exception:
        pass
    return None


def _resolve_dataset_path(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    safe_path = _safe_under_dir(DATASET_DIR, raw)
    if safe_path and safe_path.exists():
        return str(safe_path)
    if Path(raw).exists():
        return raw
    return None


def _cleanup_task_files(task: ConversionTask) -> None:
    try:
        if task.onnx_path and Path(task.onnx_path).exists():
            Path(task.onnx_path).unlink()
    except Exception:
        pass
    try:
        if task.output_path and Path(task.output_path).exists():
            Path(task.output_path).unlink()
    except Exception:
        pass


def _add_task(task_id: str, task: ConversionTask) -> None:
    with _tasks_lock:
        if len(tasks) >= MAX_TASKS:
            oldest_id = next(iter(tasks))
            oldest_task = tasks.pop(oldest_id)
            _cleanup_task_files(oldest_task)
        tasks[task_id] = task


def get_onnx_info(onnx_path: str) -> dict:
    """解析ONNX模型信息 - 在rknn-toolkit2虚拟环境中运行"""
    from modules import venv_manager
    import subprocess
    import json
    
    # 使用rknn-toolkit2虚拟环境的Python解析ONNX
    toolkit_type = "rknn-toolkit2"
    success, msg = venv_manager.prepare_env(toolkit_type)
    if not success:
        return {"error": f"虚拟环境准备失败: {msg}，无法解析ONNX"}
    
    python_path = venv_manager._get_python_path(toolkit_type)
    
    # 创建临时解析脚本
    parse_script = """
import sys
import json
try:
    import onnx
    model = onnx.load(sys.argv[1])
    inputs = []
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "?" for d in inp.type.tensor_type.shape.dim]
        inputs.append({"name": inp.name, "shape": shape})
    outputs = []
    for outp in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else "?" for d in outp.type.tensor_type.shape.dim]
        outputs.append({"name": outp.name, "shape": shape})
    
    is_dynamic = False
    first_input_shape = None
    first_input_name = None
    input_dtype = "float32"
    height = None
    width = None
    
    if inputs:
        first_input = inputs[0]
        first_input_name = first_input["name"]
        first_input_shape = first_input["shape"]
        is_dynamic = "?" in str(first_input_shape) or 0 in first_input_shape
        if len(first_input_shape) >= 4:
            height = first_input_shape[2] if first_input_shape[2] not in ["?", 0] else None
            width = first_input_shape[3] if first_input_shape[3] not in ["?", 0] else None
    
    result = {
        "inputs": inputs,
        "outputs": outputs,
        "opset": model.opset_import[0].version if model.opset_import else None,
        "ir_version": model.ir_version,
        "is_dynamic": is_dynamic,
        "input_name": first_input_name,
        "input_shape": str(first_input_shape) if first_input_shape else None,
        "input_dtype": input_dtype,
        "height": height,
        "width": width,
    }
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"""
    
    try:
        result = subprocess.run(
            [str(python_path), "-c", parse_script, onnx_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {"error": f"解析失败: {result.stderr}"}
        
        return json.loads(result.stdout.strip())
        
    except subprocess.TimeoutExpired:
        return {"error": "解析超时"}
    except json.JSONDecodeError as e:
        return {"error": f"解析结果JSON错误: {e}"}
    except Exception as e:
        return {"error": f"解析异常: {e}"}


def run_conversion_in_venv_async(task: ConversionTask):
    from modules import venv_manager
    success, message = run_conversion_in_venv(task)

    task.end_time = task.start_time.__class__.now()

    if success:
        task.status = "completed"
        task.message = message
        duration = (task.end_time - task.start_time).total_seconds()
        task.add_log(f"总耗时: {duration:.2f}秒")
    else:
        task.status = "failed"
        task.message = message


def generate_task_log_stream(task_id: str):
    task = tasks.get(task_id)
    if not task:
        yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'})}\n\n"
        return

    last_index = 0
    while True:
        with task._log_lock:
            current_len = len(task.log)
            new_logs = task.log[last_index:current_len]
            last_index = current_len
            status = task.status

        for log_msg in new_logs:
            yield f"data: {json.dumps({'type': 'log', 'message': log_msg})}\n\n"

        if status in ("completed", "failed"):
            yield f"data: {json.dumps({'type': 'status', 'status': status, 'message': task.message})}\n\n"
            break

        time.sleep(0.1)


def task_log_stream_response(task_id: str) -> Response:
    from flask import Response
    return Response(
        generate_task_log_stream(task_id),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_worker_line(task: ConversionTask, line: str):
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
    from modules import venv_manager
    config = task.config
    platform = config.platform

    if platform not in CHIP_PLATFORMS:
        return False, f"不支持的平台: {platform}"

    toolkit_type = CHIP_PLATFORMS[platform]["toolkit"]
    task.add_log(f"目标平台: {platform}")
    task.add_log(f"需要Toolkit: {toolkit_type}")

    success, msg = venv_manager.prepare_env(toolkit_type)
    if not success:
        return False, f"环境准备失败: {msg}"

    task.add_log(f"环境准备完成: {msg}")

    python_path = venv_manager._get_python_path(toolkit_type)
    convert_script = BASE_DIR / "convert_worker.py"
    config_json = json.dumps(config.to_dict())

    task.add_log(f"使用Python: {python_path}")
    task.add_log(f"转换脚本: {convert_script}")

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

        if task.process.stdout:
            for line in task.process.stdout:
                _parse_worker_line(task, line.rstrip("\n"))

        try:
            returncode = task.process.wait(timeout=CONVERT_TIMEOUT)
        except subprocess.TimeoutExpired:
            task.process.kill()
            task.process.wait()
            task.add_log(f"转换超时 (>={CONVERT_TIMEOUT}秒)")
            return False, "转换超时"

        if returncode != 0:
            return False, "转换进程返回错误"

        if Path(task.output_path).exists():
            output_size = Path(task.output_path).stat().st_size / (1024 * 1024)
            task.add_log(f"输出文件: {task.output_path}")
            task.add_log(f"文件大小: {output_size:.2f} MB")
            return True, f"转换成功: {task.output_path}"
        return False, "输出文件不存在"

    except Exception as e:
        task.add_log(f"执行异常: {str(e)}")
        return False, f"执行异常: {str(e)}"
    finally:
        task.process = None