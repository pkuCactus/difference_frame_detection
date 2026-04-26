#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 app.py review 修复点的最小单元测试。

覆盖：
- _safe_under_dir：路径越权拦截（用于上传文件名 / datasetPath / api_info）
- _add_task：容量裁剪、不删除运行中任务、不再无限递归
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# 测试导入前设置 SECRET_KEY，避免 app.py 顶层 raise
os.environ.setdefault("SECRET_KEY", "test-secret")

sys.path.insert(0, str(Path(__file__).parent))

import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_tasks():
    """每个测试用例前后清空全局 tasks，避免互相污染"""
    app_module.tasks.clear()
    yield
    app_module.tasks.clear()


# ---------------------------- _safe_under_dir ---------------------------

class TestSafeUnderDir:
    """_safe_under_dir 在路径校验上的关键行为"""

    def test_normal_filename(self, tmp_path):
        result = app_module._safe_under_dir(tmp_path, "model.onnx")
        assert result is not None
        assert result == (tmp_path / "model.onnx").resolve()

    def test_path_traversal_double_dot(self, tmp_path):
        assert app_module._safe_under_dir(tmp_path, "../etc/passwd") is None

    def test_absolute_path_rejected(self, tmp_path):
        assert app_module._safe_under_dir(tmp_path, "/etc/passwd") is None

    def test_empty_string_rejected(self, tmp_path):
        assert app_module._safe_under_dir(tmp_path, "") is None

    def test_none_rejected(self, tmp_path):
        assert app_module._safe_under_dir(tmp_path, None) is None

    def test_nested_subdir_rejected_or_sanitized(self, tmp_path):
        """嵌套子路径要么被拒，要么被规约到 base 之内，不能逃逸"""
        result = app_module._safe_under_dir(tmp_path, "subdir/../../etc/passwd")
        # 规约后必须仍然在 tmp_path 之内（或返回 None）
        if result is not None:
            assert str(result).startswith(str(tmp_path.resolve()))


# ---------------------------- _add_task ---------------------------

def _make_task(taskId: str, status: str = "completed",
               start_offset_seconds: int = 0) -> "app_module.ConversionTask":
    """构造一个最小可用的 ConversionTask 用于测试"""
    cfg = app_module.ConversionConfig(platform="RK3588")
    task = app_module.ConversionTask(taskId, "/fake/path.onnx", cfg)
    task.status = status
    task.startTime = datetime.now() - timedelta(seconds=start_offset_seconds)
    return task


class TestAddTask:
    """_add_task 修复后的容量裁剪行为"""

    def test_basic_add(self):
        task = _make_task("t1")
        app_module._add_task("t1", task)
        assert "t1" in app_module.tasks
        assert app_module.tasks["t1"] is task

    def test_does_not_recurse(self):
        """回归测试：曾经因调用自身导致 RecursionError"""
        for i in range(10):
            app_module._add_task(f"t{i}", _make_task(f"t{i}"))
        assert len(app_module.tasks) == 10

    def test_capacity_eviction_keeps_running(self, monkeypatch):
        """容量满时应优先淘汰已完成任务，保留运行中任务"""
        monkeypatch.setattr(app_module, "MAX_TASKS", 5)

        # 4 个最早的运行中任务 + 1 个最近完成的任务，凑满 5 个
        for i in range(4):
            app_module._add_task(
                f"running{i}",
                _make_task(f"running{i}", status="converting",
                           start_offset_seconds=1000 + i),
            )
        app_module._add_task(
            "done0",
            _make_task("done0", status="completed", start_offset_seconds=10),
        )
        assert len(app_module.tasks) == 5

        # 触发上限：再加一个，应该淘汰已完成的，保留运行中的
        app_module._add_task("new", _make_task("new", status="converting"))
        assert "new" in app_module.tasks
        for i in range(4):
            assert f"running{i}" in app_module.tasks, "运行中任务不应被淘汰"

    def test_capacity_eviction_oldest_finished_first(self, monkeypatch):
        """已完成任务按 startTime 排序，最早的先淘汰"""
        monkeypatch.setattr(app_module, "MAX_TASKS", 3)

        app_module._add_task(
            "old",
            _make_task("old", status="completed", start_offset_seconds=10000),
        )
        app_module._add_task(
            "mid",
            _make_task("mid", status="completed", start_offset_seconds=5000),
        )
        app_module._add_task(
            "new",
            _make_task("new", status="completed", start_offset_seconds=1000),
        )
        assert len(app_module.tasks) == 3

        app_module._add_task("extra", _make_task("extra", status="completed"))

        # 最早的 "old" 应被淘汰
        assert "old" not in app_module.tasks
        assert "extra" in app_module.tasks


# --------------------- _resolve_dataset_path -------------------------

class TestResolveDatasetPath:
    """_resolve_dataset_path 必须把 dataset 路径限制在 DATASET_DIR 之内"""

    def test_empty_returns_none(self):
        assert app_module._resolve_dataset_path("") is None
        assert app_module._resolve_dataset_path(None) is None

    def test_relative_under_dataset_dir_ok(self):
        result = app_module._resolve_dataset_path("calib.txt")
        assert result is not None
        assert result.startswith(str(app_module.DATASET_DIR.resolve()))

    def test_absolute_path_rejected(self):
        assert app_module._resolve_dataset_path("/etc/passwd") is None

    def test_traversal_rejected(self):
        assert app_module._resolve_dataset_path("../../etc/passwd") is None


# --------------------- _cleanup_task_files ---------------------------

class TestCleanupTaskFiles:
    """_cleanup_task_files 仅清理 UPLOAD_DIR / OUTPUT_DIR 之内的文件"""

    def test_removes_files_under_managed_dirs(self, tmp_path, monkeypatch):
        uploadDir = tmp_path / "uploads"
        outputDir = tmp_path / "outputs"
        uploadDir.mkdir()
        outputDir.mkdir()
        monkeypatch.setattr(app_module, "UPLOAD_DIR", uploadDir)
        monkeypatch.setattr(app_module, "OUTPUT_DIR", outputDir)

        onnx = uploadDir / "abc_model.onnx"
        rknn = outputDir / "abc_model.rknn"
        onnx.write_bytes(b"x")
        rknn.write_bytes(b"y")

        task = _make_task("abc", status="completed")
        task.onnxPath = str(onnx)
        task.outputPath = str(rknn)

        app_module._cleanup_task_files(task)

        assert not onnx.exists()
        assert not rknn.exists()

    def test_refuses_files_outside_managed_dirs(self, tmp_path, monkeypatch):
        uploadDir = tmp_path / "uploads"
        outputDir = tmp_path / "outputs"
        uploadDir.mkdir()
        outputDir.mkdir()
        monkeypatch.setattr(app_module, "UPLOAD_DIR", uploadDir)
        monkeypatch.setattr(app_module, "OUTPUT_DIR", outputDir)

        # 任务文件指向被管目录之外的路径，应该被忽略
        external = tmp_path / "outside.bin"
        external.write_bytes(b"keep me")

        task = _make_task("ext", status="completed")
        task.onnxPath = str(external)
        task.outputPath = None

        app_module._cleanup_task_files(task)

        assert external.exists()


# --------------------- api_info 越权拦截 ----------------------------

class TestApiInfoTraversal:
    """/api/info/<path> 必须显式拒绝逃出 UPLOAD_DIR 的路径"""

    def test_traversal_returns_400(self):
        client = app_module.app.test_client()
        resp = client.get("/api/info/..%2Fetc%2Fpasswd")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body == {"success": False, "error": "非法路径"}
