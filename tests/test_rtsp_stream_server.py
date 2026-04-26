#!/usr/bin/env python3
"""RTSP Stream Server 单元测试"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

from rtsp_stream_server import (
    FRAME_REPORT_INTERVAL,
    FFMPEG_CODEC,
    FFMPEG_BIN,
    PIX_FMT_INPUT,
    PIX_FMT_OUTPUT,
    PRESET_ULTRAFAST,
    RTSP_DEFAULT_HOST,
    RTSP_DEFAULT_PORT,
    RTSP_DEFAULT_URL,
    STARTUP_TIMEOUT_SEC,
    TRANSPORT_TCP,
    TUNE_ZEROLATENCY,
    MediaMTXManager,
    RtspStreamServer,
    _terminate_process,
)


class TestTerminateProcess:
    """测试 _terminate_process 函数"""

    def test_none_process_returns_false(self):
        assert _terminate_process(None) is False

    def test_already_dead_process_returns_false(self):
        mock = MagicMock()
        mock.poll.return_value = 0  # 已退出
        assert _terminate_process(mock) is False

    def test_terminate_success(self):
        mock = MagicMock()
        mock.poll.return_value = None  # 运行中
        mock.wait.return_value = None

        assert _terminate_process(mock) is True
        mock.terminate.assert_called_once()
        mock.wait.assert_called_once()
        mock.kill.assert_not_called()

    def test_kill_after_timeout(self):
        mock = MagicMock()
        mock.poll.return_value = None
        mock.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]

        assert _terminate_process(mock) is True
        mock.terminate.assert_called_once()
        mock.kill.assert_called_once()
        assert mock.wait.call_count == 2


class TestMediaMTXManagerPlatformDetection:
    """测试平台检测逻辑"""

    @patch("rtsp_stream_server.platform.system", return_value="Linux")
    @patch("rtsp_stream_server.platform.machine", return_value="x86_64")
    def test_linux_amd64(self, mock_machine, mock_system):
        mgr = MediaMTXManager()
        assert "linux_amd64" in mgr.tar_file_name

    @patch("rtsp_stream_server.platform.system", return_value="Linux")
    @patch("rtsp_stream_server.platform.machine", return_value="aarch64")
    def test_linux_arm64(self, mock_machine, mock_system):
        mgr = MediaMTXManager()
        assert "linux_arm64" in mgr.tar_file_name

    @patch("rtsp_stream_server.platform.system", return_value="Linux")
    @patch("rtsp_stream_server.platform.machine", return_value="armv7l")
    def test_linux_armv7(self, mock_machine, mock_system):
        mgr = MediaMTXManager()
        assert "linux_armv7" in mgr.tar_file_name

    @patch("rtsp_stream_server.platform.system", return_value="Darwin")
    @patch("rtsp_stream_server.platform.machine", return_value="x86_64")
    def test_darwin_amd64(self, mock_machine, mock_system):
        mgr = MediaMTXManager()
        assert "darwin_amd64" in mgr.tar_file_name

    @patch("rtsp_stream_server.platform.system", return_value="Darwin")
    @patch("rtsp_stream_server.platform.machine", return_value="arm64")
    def test_darwin_arm64(self, mock_machine, mock_system):
        mgr = MediaMTXManager()
        assert "darwin_arm64" in mgr.tar_file_name

    @patch("rtsp_stream_server.platform.system", return_value="Windows")
    @patch("rtsp_stream_server.platform.machine", return_value="AMD64")
    def test_unknown_fallback(self, mock_machine, mock_system):
        mgr = MediaMTXManager()
        assert "linux_amd64" in mgr.tar_file_name


class TestMediaMTXManagerInstallation:
    """测试安装检测和下载"""

    @patch.object(Path, "exists", return_value=True)
    @patch.object(Path, "stat")
    def test_is_installed_true(self, mock_stat, mock_exists, tmp_path):
        mock_stat_result = MagicMock()
        mock_stat_result.st_mode = 0o100755
        mock_stat.return_value = mock_stat_result
        mgr = MediaMTXManager(install_dir=tmp_path)
        assert mgr.is_installed is True

    def test_is_installed_false_no_file(self, tmp_path):
        mgr = MediaMTXManager(install_dir=tmp_path)
        assert mgr.is_installed is False

    @patch.object(Path, "exists", return_value=True)
    @patch.object(Path, "stat")
    def test_is_installed_false_not_executable(self, mock_stat, mock_exists, tmp_path):
        mock_stat_result = MagicMock()
        mock_stat_result.st_mode = 0o100644
        mock_stat.return_value = mock_stat_result
        mgr = MediaMTXManager(install_dir=tmp_path)
        assert mgr.is_installed is False

    @patch("rtsp_stream_server.urllib.request.urlretrieve")
    def test_download_success(self, mock_urlretrieve, tmp_path):
        import io
        import tarfile

        mgr = MediaMTXManager(install_dir=tmp_path)
        mgr.tar_file_name = "test.tar.gz"

        # 创建模拟 tar 包
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            content = b"#!/bin/sh\necho test"
            data = io.BytesIO(content)
            info = tarfile.TarInfo(name="mediamtx")
            info.size = len(content)
            info.mode = 0o755
            tar.addfile(info, data)

        def fake_urlretrieve(url, dest):
            Path(dest).write_bytes(tar_path.read_bytes())

        mock_urlretrieve.side_effect = fake_urlretrieve

        result = mgr.download()
        assert result is True
        assert mgr.executable_path.exists()

    @patch("rtsp_stream_server.urllib.request.urlretrieve")
    def test_download_failure(self, mock_urlretrieve, tmp_path):
        import urllib.error

        mock_urlretrieve.side_effect = urllib.error.URLError("network error")
        mgr = MediaMTXManager(install_dir=tmp_path)
        result = mgr.download()
        assert result is False


class TestMediaMTXManagerStartup:
    """测试启动逻辑"""

    @patch("rtsp_stream_server.time.perf_counter")
    @patch("rtsp_stream_server.time.sleep")
    def test_wait_for_startup_success(self, mock_sleep, mock_perf):
        mgr = MediaMTXManager()
        mgr.process = MagicMock()
        mgr.process.poll.return_value = None  # 一直运行
        mock_perf.side_effect = [0, 1, 10]  # 第一次调用、循环内、超时

        result = mgr._wait_for_startup()
        assert result is True

    @patch("rtsp_stream_server.time.perf_counter")
    @patch("rtsp_stream_server.time.sleep")
    def test_wait_for_startup_failure(self, mock_sleep, mock_perf):
        mgr = MediaMTXManager()
        mgr.process = MagicMock()
        mgr.process.poll.return_value = 1  # 立即退出
        mock_perf.side_effect = [0, 0.1]

        result = mgr._wait_for_startup()
        assert result is False


class TestRtspStreamServerFFmpegCommand:
    """测试 FFmpeg 命令构建"""

    def test_build_command_structure(self):
        server = RtspStreamServer("/tmp/test.mp4", RTSP_DEFAULT_URL, fps=30)
        cmd = server._build_ffmpeg_command(640, 480)

        assert cmd[0] == FFMPEG_BIN
        assert "-f" in cmd
        assert "rawvideo" in cmd
        assert "-c:v" in cmd
        assert FFMPEG_CODEC in cmd
        assert PIX_FMT_INPUT in cmd
        assert PIX_FMT_OUTPUT in cmd
        assert PRESET_ULTRAFAST in cmd
        assert TUNE_ZEROLATENCY in cmd
        assert TRANSPORT_TCP in cmd
        assert "640x480" in cmd
        assert "30" in cmd
        assert RTSP_DEFAULT_URL in cmd

    def test_build_command_custom_url(self):
        custom_url = "rtsp://192.168.1.100:8554/live"
        server = RtspStreamServer("/tmp/test.mp4", custom_url, fps=15)
        cmd = server._build_ffmpeg_command(1920, 1080)

        assert "1920x1080" in cmd
        assert "15" in cmd
        assert custom_url in cmd


class TestRtspStreamServerLifecycle:
    """测试 RtspStreamServer 生命周期方法"""

    def test_init_defaults(self):
        server = RtspStreamServer("/tmp/test.mp4", RTSP_DEFAULT_URL)
        assert server.video_path == Path("/tmp/test.mp4")
        assert server.rtsp_url == RTSP_DEFAULT_URL
        assert server.fps == 25
        assert server.loop is True
        assert server.ffmpeg_process is None

    def test_ensure_ffmpeg_running_when_none(self):
        server = RtspStreamServer("/tmp/test.mp4", RTSP_DEFAULT_URL)
        with patch.object(server, "_start_ffmpeg") as mock_start:
            server._ensure_ffmpeg_running(640, 480)
            mock_start.assert_called_once_with(640, 480)

    def test_ensure_ffmpeg_running_when_alive(self):
        server = RtspStreamServer("/tmp/test.mp4", RTSP_DEFAULT_URL)
        server.ffmpeg_process = MagicMock()
        server.ffmpeg_process.poll.return_value = None
        with patch.object(server, "_start_ffmpeg") as mock_start:
            server._ensure_ffmpeg_running(640, 480)
            mock_start.assert_not_called()

    def test_ensure_ffmpeg_running_when_dead(self):
        server = RtspStreamServer("/tmp/test.mp4", RTSP_DEFAULT_URL)
        server.ffmpeg_process = MagicMock()
        server.ffmpeg_process.poll.return_value = 0  # 已退出
        with patch.object(server, "_start_ffmpeg") as mock_start:
            server._ensure_ffmpeg_running(640, 480)
            mock_start.assert_called_once_with(640, 480)

    def test_stop_ffmpeg(self):
        server = RtspStreamServer("/tmp/test.mp4", RTSP_DEFAULT_URL)
        mock_process = MagicMock()
        server.ffmpeg_process = mock_process

        with patch("rtsp_stream_server._terminate_process", return_value=True) as mock_term:
            server._stop_ffmpeg()
            mock_term.assert_called_once_with(mock_process)
            assert server.ffmpeg_process is None


class TestConstants:
    """测试常量定义"""

    def test_default_values(self):
        assert RTSP_DEFAULT_PORT == 8554
        assert RTSP_DEFAULT_HOST == "localhost"
        assert RTSP_DEFAULT_URL == "rtsp://localhost:8554/stream1"
        assert STARTUP_TIMEOUT_SEC == 5.0
        assert FRAME_REPORT_INTERVAL == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
