#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX转RKNN Web转换工具

使用方法：
    SECRET_KEY=<key> python app.py
    访问 http://localhost:5000
"""

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from modules import (
    BASE_DIR,
    UPLOAD_DIR,
    OUTPUT_DIR,
    CHIP_PLATFORMS,
    QUANTIZED_DTYPES,
    QUANTIZED_ALGORITHMS,
    OPTIMIZATION_LEVELS,
    TOOLKIT_VENV_MAP,
    ConversionConfig,
    ConversionTask,
    venv_manager,
    tasks,
    _add_task,
    _resolve_dataset_path,
    get_onnx_info,
    run_conversion_in_venv_async,
    task_log_stream_response,
)


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY 环境变量未设置")


@app.route("/")
def index():
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

    task_id = str(uuid.uuid4())[:8]
    safe_name = secure_filename(file.filename or "") or "model.onnx"
    filename = f"{task_id}_{safe_name}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

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
    config.batch_size = request.form.get("batch_size", type=int, default=1) or 1

    task = ConversionTask(task_id, str(filepath), config)
    _add_task(task_id, task)

    output_filename = f"{task_id}_{Path(filepath).stem}.rknn"
    output_path = OUTPUT_DIR / output_filename
    task.output_path = str(output_path)

    task.status = "converting"
    task.add_log(f"文件: {file.filename}")
    task.add_log(f"平台: {platform}")

    import threading
    thread = threading.Thread(target=run_conversion_in_venv_async, args=(task,))
    thread.start()

    return render_template("converting.html", task=task)


@app.route("/task/<task_id>/log/stream")
def task_log_stream(task_id: str):
    return task_log_stream_response(task_id)


@app.route("/prepare_env/<toolkit_type>")
def prepare_env(toolkit_type: str):
    if toolkit_type not in TOOLKIT_VENV_MAP:
        return jsonify({"success": False, "message": f"未知 toolkit: {toolkit_type}"})
    success, msg = venv_manager.prepare_env(toolkit_type)
    return jsonify({"success": success, "message": msg, "status": venv_manager.get_status(toolkit_type)})


@app.route("/env_status")
def env_status():
    return jsonify(venv_manager.get_all_status())


@app.route("/download/<task_id>")
def download(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return "任务不存在", 404
    if task.status != "completed":
        return "任务未完成", 400
    if not task.output_path or not Path(task.output_path).exists():
        return "文件不存在", 404
    return send_file(task.output_path, as_attachment=True)


@app.route("/api/platforms")
def api_platforms():
    return jsonify(CHIP_PLATFORMS)


@app.route("/api/tasks")
def api_tasks():
    return jsonify({tid: t.to_dict() for tid, t in tasks.items()})


@app.route("/api/task/<task_id>")
def api_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(task.to_dict())


@app.route("/api/info/<path:filepath>")
def api_info(filepath: str):
    safe_path = UPLOAD_DIR / secure_filename(filepath)
    if not safe_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return jsonify(get_onnx_info(str(safe_path)))


def ensure_pythons_installed():
    from modules.venv_manager import VirtualEnvManager
    import subprocess
    import shutil

    print("\n[步骤2/3] 确保 Python 版本...")
    manager = VirtualEnvManager()

    for toolkit_type, info in TOOLKIT_VENV_MAP.items():
        py_version = info["python_version"]
        py_cmd = f"python{py_version}"
        if shutil.which(py_cmd):
            try:
                result = subprocess.run([py_cmd, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and py_version in result.stdout:
                    print(f"  {py_version}: ✓ {py_cmd} 已存在")
                    continue
            except Exception:
                pass

        print(f"  {py_version}: -> 安装...")
        success, msg = manager._ensure_python_installed(py_version)
        print(f"  {py_version}: {msg}")


def init_virtual_environments():
    print("\n" + "=" * 70)
    print("初始化虚拟环境")
    print(f"虚拟环境目录: {UPLOAD_DIR.parent.parent / 'onnx_to_rknn_venvs'}")
    print("=" * 70 + "\n")

    print("[步骤1/3] 检查 uv...")
    uv_path = venv_manager._get_uv_path()
    if uv_path:
        print(f"  结果: uv 已安装: {uv_path}")
    else:
        print("  结果: uv 未安装，开始安装...")
        success, msg = venv_manager._ensure_uv_installed()
        print(f"  结果: {msg}")

    ensure_pythons_installed()

    print("\n[步骤3/3] 创建虚拟环境并安装 RKNN Toolkit...")
    for toolkit_type, info in TOOLKIT_VENV_MAP.items():
        venv_name = info["venv_name"]
        package_name = info["package_name"]
        print(f"\n[{toolkit_type}]")
        print(f"  虚拟环境: {venv_name}")
        print(f"  包名: {package_name}")

        status = venv_manager.get_status(toolkit_type)
        if status.get("rknn_installed"):
            print("  ✓ 已就绪，跳过")
            continue

        print("  -> 开始准备环境...")
        success, msg = venv_manager.prepare_env(toolkit_type)
        print(f"  结果: {msg}")

    print("\n" + "=" * 70)
    print("初始化完成")
    env_status = venv_manager.get_all_status()
    ready_count = sum(1 for s in env_status.values() if s.get("rknn_installed"))
    print(f"环境状态: {ready_count}/{len(env_status)} 就绪")
    for toolkit_type, status in env_status.items():
        icon = "✓" if status.get("rknn_installed") else "○"
        state = "已就绪" if status.get("rknn_installed") else status.get("message", "未就绪")
        print(f"  {icon} {toolkit_type}: {state}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX转RKNN Web服务")
    parser.add_argument("--skip-init-env", action="store_true", help="跳过环境初始化")
    args = parser.parse_args()

    if not args.skip_init_env:
        init_virtual_environments()

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    print("\n" + "=" * 70)
    print("ONNX转RKNN Web转换工具")
    print(f"访问: http://{host}:{port}")
    print("=" * 70)
    print(f"绑定地址: {host}:{port}（如需对外暴露请设置 FLASK_HOST=0.0.0.0）")

    app.run(host=host, port=port, debug=False)