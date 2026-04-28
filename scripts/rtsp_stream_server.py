#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTSP模拟流服务器 - 循环读取视频文件并通过RTSP协议发送。"""

import argparse
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

MEDIAMTX_VERSION = "v1.9.3"
MEDIAMTX_DIR = Path(tempfile.gettempdir()) / "mediamtx"

FFMPEG_BIN = "ffmpeg"
FFMPEG_CODEC = "libx264"
PIX_FMT_INPUT = "bgr24"
PIX_FMT_OUTPUT = "yuv420p"
PRESET_ULTRAFAST = "ultrafast"
TUNE_ZEROLATENCY = "zerolatency"
TRANSPORT_TCP = "tcp"
RTSP_DEFAULT_URL = "rtsp://localhost:8554/stream1"
RTSP_DEFAULT_HOST = "localhost"
RTSP_DEFAULT_PORT = 8554

STARTUP_TIMEOUT_SEC = 5.0
WAIT_INTERVAL_SEC = 0.1
FRAME_REPORT_INTERVAL = 100


PLATFORM_MAP = {
    ("linux", "x86_64"): "linux_amd64",
    ("linux", "amd64"): "linux_amd64",
    ("linux", "aarch64"): "linux_arm64",
    ("linux", "arm64"): "linux_arm64",
}


def _terminate_process(process: Optional[subprocess.Popen], timeout: float = STARTUP_TIMEOUT_SEC) -> bool:
    if not process or process.poll() is not None:
        return False

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return True


class MediaMTXManager:
    def __init__(self, install_dir: Path = MEDIAMTX_DIR, download_dir: Optional[Path] = None):
        self.install_dir = install_dir
        self.download_dir = download_dir or Path(__file__).parent
        self.process = None
        suffix = self._detect_platform()
        self.executable_path = self.install_dir / "mediamtx"
        self.tar_file_name = f"mediamtx_{MEDIAMTX_VERSION}_{suffix}.tar.gz"
        self.tar_path = self.download_dir / self.tar_file_name
        self.download_url = (
            f"https://github.com/bluenviron/mediamtx/releases/download/"
            f"{MEDIAMTX_VERSION}/{self.tar_file_name}"
        )

    @staticmethod
    def _detect_platform() -> str:
        system = platform.system().lower()
        machine = platform.machine().lower()
        key = (system, machine)
        if key in PLATFORM_MAP:
            return PLATFORM_MAP[key]
        if system == "linux" and "arm" in machine:
            return "linux_armv7"
        if system == "darwin":
            return "darwin_amd64" if machine in ("x86_64", "amd64") else "darwin_arm64"
        return "linux_amd64"

    @property
    def is_installed(self) -> bool:
        return self.executable_path.exists() and bool(self.executable_path.stat().st_mode & 0o111)

    def download(self) -> bool:
        if self.is_installed:
            print(f"MediaMTX已安装: {self.executable_path}")
            return True

        self.install_dir.mkdir(parents=True, exist_ok=True)

        if self.tar_path.exists():
            print(f"发现本地安装包: {self.tar_path}")
        else:
            print(f"正在下载MediaMTX {MEDIAMTX_VERSION} ...")
            try:
                print(f"下载地址: {self.download_url}")
                urllib.request.urlretrieve(self.download_url, self.tar_path)
                print(f"下载完成，已保存到: {self.tar_path}")
            except (urllib.error.URLError, OSError) as e:
                print(f"下载失败: {e}")
                return False

        try:
            print("正在解压...")
            with tarfile.open(self.tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    memberPath = self.install_dir / member.name
                    try:
                        memberPath.relative_to(self.install_dir.resolve())
                    except ValueError:
                        raise tarfile.TarError(f"非法tar路径: {member.name}")
                tar.extractall(self.install_dir)

            self.executable_path.chmod(0o755)
            print(f"MediaMTX安装成功: {self.executable_path}")
            return True
        except (OSError, tarfile.TarError) as e:
            print(f"解压失败: {e}")
            return False

    def _wait_for_startup(self) -> bool:
        deadline = time.perf_counter() + STARTUP_TIMEOUT_SEC
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                print("MediaMTX启动失败")
                return False
            time.sleep(WAIT_INTERVAL_SEC)
        return True

    def start(self) -> bool:
        if not self.is_installed and not self.download():
            return False

        print("正在启动MediaMTX服务器...")

        try:
            self.process = subprocess.Popen(
                [str(self.executable_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(self.install_dir),
            )
            if not self._wait_for_startup():
                return False

            print(f"MediaMTX已启动 (PID: {self.process.pid})")
            print(f"RTSP服务地址: rtsp://{RTSP_DEFAULT_HOST}:{RTSP_DEFAULT_PORT}/")
            return True
        except OSError as e:
            print(f"启动失败: {e}")
            return False

    def stop(self) -> None:
        if _terminate_process(self.process):
            print("MediaMTX已停止")


class RtspStreamServer:
    def __init__(self, video_path: str, rtsp_url: str, fps: int = 25, loop: bool = True):
        self.video_path = Path(video_path)
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.loop = loop
        self.ffmpeg_process: Optional[subprocess.Popen] = None

    def _build_ffmpeg_command(self, width: int, height: int) -> list[str]:
        return [
            FFMPEG_BIN,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", PIX_FMT_INPUT,
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", FFMPEG_CODEC,
            "-pix_fmt", PIX_FMT_OUTPUT,
            "-preset", PRESET_ULTRAFAST,
            "-tune", TUNE_ZEROLATENCY,
            "-f", "rtsp",
            "-rtsp_transport", TRANSPORT_TCP,
            self.rtsp_url,
        ]

    def _start_ffmpeg(self, width: int, height: int) -> None:
        command = self._build_ffmpeg_command(width, height)
        self.ffmpeg_process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    def _stop_ffmpeg(self) -> None:
        _terminate_process(self.ffmpeg_process)
        self.ffmpeg_process = None

    def _ensure_ffmpeg_running(self, width: int, height: int) -> None:
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            return
        print("FFmpeg连接断开，重新启动...")
        self._stop_ffmpeg()
        self._start_ffmpeg(width, height)

    def _write_frame(self, frame: "np.ndarray", width: int, height: int) -> bool:
        try:
            self.ffmpeg_process.stdin.write(frame.tobytes())
            return True
        except BrokenPipeError:
            self._ensure_ffmpeg_running(width, height)
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                return True
            except BrokenPipeError:
                print("警告: 帧写入失败，已丢弃")
                return False

    def _handle_eof(self, frame_count: int) -> bool:
        if self.loop:
            print(f"循环播放: 已处理 {frame_count} 帧，重新开始")
            return True
        print(f"视频结束: 共处理 {frame_count} 帧")
        return False

    def _throttle_frame(self, next_frame_time: float) -> float:
        next_frame_time += 1.0 / self.fps
        sleepTime = next_frame_time - time.perf_counter()
        if sleepTime > 0:
            time.sleep(sleepTime)
        return next_frame_time

    def _print_progress(self, frame_count: int) -> None:
        if frame_count % FRAME_REPORT_INTERVAL == 0:
            print(f"已推流 {frame_count} 帧")

    def start(self) -> None:
        if cv2 is None:
            print("错误: 未安装 opencv-python，请执行: pip install opencv-python")
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {self.video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {video_fps:.2f}fps, 共{total_frames}帧")
        print(f"推流地址: {self.rtsp_url}")
        print(f"推流帧率: {self.fps}fps")
        print("开始推流... (Ctrl+C 停止)")

        self._start_ffmpeg(width, height)

        frame_count = 0
        next_frame_time = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self._handle_eof(frame_count):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        continue
                    break

                self._ensure_ffmpeg_running(width, height)
                self._write_frame(frame, width, height)

                frame_count += 1
                next_frame_time = self._throttle_frame(next_frame_time)
                self._print_progress(frame_count)

        except KeyboardInterrupt:
            print("\n用户中断，停止推流")
        finally:
            cap.release()
            self._stop_ffmpeg()


def main() -> None:
    parser = argparse.ArgumentParser(description="RTSP模拟流服务器 - 循环读取视频并推流")
    parser.add_argument("--video", "-v", required=True, help="视频文件路径")
    parser.add_argument(
        "--rtsp", "-r", default=RTSP_DEFAULT_URL, help=f"RTSP推流地址 (默认: {RTSP_DEFAULT_URL})"
    )
    parser.add_argument("--fps", "-f", type=int, default=25, help="推流帧率 (默认: 25)")
    parser.add_argument("--no-loop", action="store_true", help="禁用循环播放")
    parser.add_argument("--no-server", action="store_true", help="不自动启动MediaMTX服务器")

    args = parser.parse_args()

    if not shutil.which(FFMPEG_BIN):
        print("错误: 未找到FFmpeg，请先安装: sudo apt install ffmpeg")
        sys.exit(1)

    media_mtx = None

    try:
        if not args.no_server:
            media_mtx = MediaMTXManager()
            if not media_mtx.start():
                print("警告: MediaMTX启动失败，继续尝试推流...")

        server = RtspStreamServer(
            video_path=args.video, rtsp_url=args.rtsp, fps=args.fps, loop=not args.no_loop
        )
        server.start()

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if media_mtx:
            media_mtx.stop()


if __name__ == '__main__':
    main()
