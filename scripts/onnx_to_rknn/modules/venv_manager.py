#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""虚拟环境管理"""

import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

from .config import (
    get_python_cp_tag,
    SYSTEM_INFO,
    TOOLKIT_VENV_MAP,
    TSINGHUA_PYPI_INDEX,
    TSINGHUA_PYTHON_MIRROR,
    VENV_DIR,
    WHL_DIR,
)


class VirtualEnvManager:
    def __init__(self, venv_base_dir: Path = VENV_DIR):
        self.venv_base_dir = venv_base_dir
        self.env_status: dict[str, dict] = {}
        self._check_all_envs()

    def _get_venv_path(self, toolkit_type: str) -> Path:
        venv_name = TOOLKIT_VENV_MAP[toolkit_type]["venv_name"]
        return self.venv_base_dir / venv_name

    def _get_python_path(self, toolkit_type: str) -> Path:
        venv_path = self._get_venv_path(toolkit_type)
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def _get_pip_path(self, toolkit_type: str) -> Path:
        venv_path = self._get_venv_path(toolkit_type)
        if sys.platform == "win32":
            return venv_path / "Scripts" / "pip.exe"
        return venv_path / "bin" / "pip"

    def _get_uv_path(self) -> Optional[str]:
        uv_path = shutil.which("uv")
        if uv_path:
            try:
                result = subprocess.run([uv_path, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return uv_path
            except Exception:
                pass
        return None

    def _ensure_uv_installed(self) -> tuple[bool, str]:
        uv_path = self._get_uv_path()
        if uv_path:
            return True, f"uv 已安装: {uv_path}"

        print("\n[ensure_uv] uv 未找到，开始自动安装...")

        try:
            print("  -> 尝试通过官方脚本安装 uv...")
            result = subprocess.run(
                ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print("  -> 下载安装脚本成功，执行安装...")
                install_result = subprocess.run(
                    ["sh"], input=result.stdout, capture_output=True, text=True, timeout=120
                )
                if install_result.returncode == 0:
                    print("  -> 脚本安装完成")
                    importlib.reload(shutil)
                    uv_path = shutil.which("uv")
                    if uv_path:
                        print(f"  ✓ uv 安装成功: {uv_path}")
                        return True, f"uv 安装成功: {uv_path}"
                    else:
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

        try:
            print("  -> 尝试通过 pip 安装 uv（清华源）...")
            pip_cmds = ["pip", "pip3", sys.executable + " -m pip"]
            for pip_cmd in pip_cmds:
                parts = pip_cmd.split() if " " in pip_cmd else [pip_cmd]
                process = subprocess.Popen(
                    parts + ["install", "uv", "-i", TSINGHUA_PYPI_INDEX],
                    stdout=None, stderr=None, text=True,
                )
                try:
                    returncode = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    continue

                if returncode == 0:
                    uv_path = shutil.which("uv")
                    if uv_path:
                        print(f"  ✓ uv 通过 pip 安装成功: {uv_path}")
                        return True, f"uv 安装成功: {uv_path}"
        except Exception as e:
            print(f"  ✗ pip 安装异常: {e}")

        return False, "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"

    def _ensure_python_installed(self, py_version: str) -> tuple[bool, str]:
        uv_path = self._get_uv_path()
        if not uv_path:
            return False, "uv 未安装，无法安装 Python"

        py_cmd_name = f"python{py_version}"
        print(f"\n[ensure_python] 确保 {py_cmd_name} 已安装...")

        py_cmd = shutil.which(py_cmd_name)
        if py_cmd:
            try:
                result = subprocess.run([py_cmd, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and py_version in result.stdout:
                    print(f"  ✓ {py_cmd_name} 已存在: {py_cmd}")
                    return True, f"{py_cmd_name} 已存在"
            except Exception:
                pass

        try:
            print(f"  -> 使用 uv 安装 {py_version}（清华镜像）...")
            process = subprocess.Popen(
                [uv_path, "python", "install", py_version, "--mirror", TSINGHUA_PYTHON_MIRROR],
                stdout=None, stderr=None, text=True, bufsize=1,
            )
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
        major, minor = py_version.split(".")[:2]
        candidates = [f"python{major}.{minor}", f"python{major}{minor}"]
        if sys.platform == "win32":
            candidates.insert(0, f"py -{major}.{minor}")

        for cmd in candidates:
            python_path = shutil.which(cmd)
            if python_path:
                try:
                    result = subprocess.run([python_path, "--version"], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and py_version in result.stdout:
                        return python_path
                except Exception:
                    pass

        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_version == py_version:
            return sys.executable
        return None

    def _check_all_envs(self):
        for toolkit_type, info in TOOLKIT_VENV_MAP.items():
            venv_path = self._get_venv_path(toolkit_type)
            python_path = self._get_python_path(toolkit_type)

            exists = venv_path.exists()
            python_exists = python_path.exists()

            rknn_installed = False
            if python_exists:
                try:
                    result = subprocess.run(
                        [str(python_path), "-c", "import rknn; print('ok')"],
                        capture_output=True, timeout=5
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
        return self.env_status.get(toolkit_type, {})

    def get_all_status(self) -> dict:
        return self.env_status

    def create_venv(self, toolkit_type: str) -> tuple[bool, str]:
        venv_path = self._get_venv_path(toolkit_type)
        toolkit_info = TOOLKIT_VENV_MAP[toolkit_type]
        required_py_version = toolkit_info["python_version"]
        py_cmd_name = f"python{required_py_version}"

        print(f"\n[create_venv] toolkit={toolkit_type}, requiredPy={required_py_version}")

        if venv_path.exists() and self._get_python_path(toolkit_type).exists():
            print(f"  -> 虚拟环境已存在且完整: {venv_path}")
            return True, f"虚拟环境已存在: {venv_path}"

        if venv_path.exists():
            print(f"  -> 清理不完整的环境: {venv_path}")
            shutil.rmtree(venv_path)

        uv_path = self._get_uv_path()
        if uv_path:
            print(f"  -> 使用 uv 创建虚拟环境: {uv_path}")
            try:
                cmd = [uv_path, "venv", str(venv_path), "--python", py_cmd_name]
                env = os.environ.copy()
                env["UV_NO_PROGRESS"] = "1"
                env["NO_COLOR"] = "1"
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
                if result.returncode != 0:
                    err = result.stderr.strip()[:300] if result.stderr else "无错误输出"
                    print(f"  ✗ uv venv 失败: {err}")
                    print(f"  -> 回退到标准 venv...")
                else:
                    python_path = self._get_python_path(toolkit_type)
                    if python_path.exists():
                        self._check_all_envs()
                        print(f"  ✓ 虚拟环境创建完成 (uv): {venv_path}")
                        return True, f"虚拟环境创建成功 (uv): {venv_path}"
            except subprocess.TimeoutExpired:
                print(f"  ✗ uv venv 超时")
            except Exception as e:
                print(f"  ✗ uv venv 异常: {e}")

        print(f"  -> 使用标准 venv 创建虚拟环境...")
        python_cmd = self._find_python_executable(required_py_version)
        if not python_cmd:
            print(f"  ✗ 未找到 Python {required_py_version}")
            return False, f"未找到 Python {required_py_version}"

        try:
            result = subprocess.run([python_cmd, "-m", "venv", str(venv_path)], capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                return False, f"创建失败: {result.stderr}"

            python_path = self._get_python_path(toolkit_type)
            pip_path = self._get_pip_path(toolkit_type)

            if not python_path.exists():
                return False, f"Python创建失败: {python_path}"

            if not pip_path.exists():
                subprocess.run([str(python_path), "-m", "ensurepip", "--upgrade"], capture_output=True, timeout=60)

            self._check_all_envs()
            return True, f"虚拟环境创建成功: {venv_path}"
        except subprocess.TimeoutExpired:
            return False, "创建超时"
        except Exception as e:
            return False, f"创建失败: {str(e)}"

    def _build_install_cmd(self, toolkit_type: str) -> tuple[list[str], str]:
        uv_path = self._get_uv_path()
        python_path = self._get_python_path(toolkit_type)
        pip_path = self._get_pip_path(toolkit_type)

        if uv_path:
            return [uv_path, "pip", "install", "--python", str(python_path), "-i", TSINGHUA_PYPI_INDEX], "uv"

        if not pip_path.exists():
            return [str(python_path), "-m", "pip", "install", "-i", TSINGHUA_PYPI_INDEX], "pip"
        return [str(pip_path), "install", "-i", TSINGHUA_PYPI_INDEX], "pip"

    def install_rknn(self, toolkit_type: str) -> tuple[bool, str]:
        python_path = self._get_python_path(toolkit_type)
        toolkit_info = TOOLKIT_VENV_MAP[toolkit_type]
        python_version = toolkit_info["python_version"]
        install_mode = toolkit_info.get("install_mode", "whl")
        venv_path = self._get_venv_path(toolkit_type)

        if not python_path.exists():
            success, msg = self.create_venv(toolkit_type)
            if not success:
                return False, msg

        install_cmd, tool_name = self._build_install_cmd(toolkit_type)
        print(f"  安装工具: {tool_name}")

        try:
            if install_mode == "tar.gz":
                downloaded_whl = self._download_and_extract_toolkit_tar(toolkit_info, python_version)
            else:
                downloaded_whl = self._download_toolkit2_whl(toolkit_info, python_version)

            if not downloaded_whl:
                return False, "下载失败"

            cmd = install_cmd + [str(downloaded_whl)]
            process = subprocess.Popen(cmd, stdout=None, stderr=None, text=True)
            try:
                returncode = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return False, "安装超时"

            if returncode == 0:
                # 安装 setuptools<72
                setuptools_cmd = install_cmd + ["setuptools<72"]
                subprocess.Popen(setuptools_cmd, stdout=None, stderr=None, text=True).wait(timeout=120)

                # 安装指定版本 onnx
                onnx_version = toolkit_info.get("onnx_version")
                if onnx_version:
                    onnx_cmd = install_cmd + [f"onnx=={onnx_version}"]
                    subprocess.Popen(onnx_cmd, stdout=None, stderr=None, text=True).wait(timeout=120)

                self._check_all_envs()
                return True, "安装成功"
            return False, f"安装失败 (返回码: {returncode})"
        except Exception as e:
            return False, f"安装异常: {str(e)}"

    def _build_whl_names(self, package_name: str, python_version: str, versions: list[str]) -> list[str]:
        whl_names = []
        pkg_whl_name = package_name.replace("-", "_")
        cp_tag = get_python_cp_tag(python_version)
        arch = SYSTEM_INFO["arch"]
        system = SYSTEM_INFO["system"]

        for version in versions:
            if system == "linux":
                if arch == "x86_64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
                elif arch == "aarch64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl")
                elif arch == "armv7l":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-linux_armv7l.whl")
            elif system == "darwin":
                if arch == "x86_64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-macosx_10_9_x86_64.whl")
                elif arch == "aarch64":
                    whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-macosx_11_0_arm64.whl")
            elif system == "windows" and arch == "x86_64":
                whl_names.append(f"{pkg_whl_name}-{version}-{cp_tag}-{cp_tag}-win_amd64.whl")

        return whl_names

    def _download_file(self, url: str, dest: Path, timeout: int = 60, max_retries: int = 3) -> bool:
        import urllib.request

        print(f"开始下载: {url}")
        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(2 ** (attempt - 1))
                if dest.exists():
                    dest.unlink()

            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                response = urllib.request.urlopen(req, timeout=120)

                total_size = response.headers.get('Content-Length')
                total_size = int(total_size) if total_size else None

                downloaded = 0
                with open(dest, 'wb', buffering=1024 * 1024) as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                print(f"下载完成: {dest.name}")
                return True
            except Exception as e:
                print(f"  下载异常: {str(e)[:120]}")
                if dest.exists():
                    dest.unlink()

        return False

    def _download_and_extract_toolkit_tar(self, toolkit_info: dict, python_version: str) -> Optional[Path]:
        versions = toolkit_info["versions"]
        tar_url_template = toolkit_info["tar_url"]
        cp_tag = get_python_cp_tag(python_version)
        arch = SYSTEM_INFO["arch"]

        for version in versions:
            extract_dir = WHL_DIR / f"rknn-toolkit-v{version}-packages"
            tar_path = WHL_DIR / f"rknn-toolkit-v{version}-packages.tar.gz"
            tar_url = tar_url_template.format(version=version)

            if extract_dir.exists():
                whl_files = list(extract_dir.rglob("*.whl"))
                matching_whl = self._find_matching_whl(whl_files, cp_tag, arch)
                if matching_whl:
                    return matching_whl

            if not tar_path.exists():
                if not self._download_file(tar_url, tar_path, timeout=7200):
                    continue

            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    try:
                        tar.extractall(extract_dir, filter="data")
                    except TypeError:
                        tar.extractall(extract_dir)
            except Exception:
                continue

            whl_files = list(extract_dir.rglob("*.whl"))
            matching_whl = self._find_matching_whl(whl_files, cp_tag, arch)
            if matching_whl:
                return matching_whl

        return None

    def _find_matching_whl(self, whl_files: list[Path], cp_tag: str, arch: str) -> Optional[Path]:
        arch_aliases = {"x86_64": ["x86_64", "amd64"], "aarch64": ["aarch64", "arm64"], "armv7l": ["armv7l"]}
        aliases = arch_aliases.get(arch, [arch])

        for whl_file in whl_files:
            name_lower = whl_file.name.lower()
            if cp_tag.lower() not in name_lower:
                continue
            for alias in aliases:
                if alias.lower() in name_lower:
                    return whl_file
        return None

    def _download_toolkit2_whl(self, toolkit_info: dict, python_version: str) -> Optional[Path]:
        versions = toolkit_info["versions"]
        whl_base_url = toolkit_info["whl_base_url"]
        package_name = toolkit_info["package_name"]
        arch = SYSTEM_INFO["arch"]
        subdir = "arm64" if arch == "aarch64" else arch

        for version in versions:
            whl_names = self._build_whl_names(package_name, python_version, [version])
            for whl_name in whl_names:
                whl_url = f"{whl_base_url}/{subdir}/{whl_name}"
                whl_path = WHL_DIR / whl_name

                if whl_path.exists() and whl_path.stat().st_size > 1000:
                    return whl_path

                if self._download_file(whl_url, whl_path, timeout=120):
                    return whl_path

        return None

    def prepare_env(self, toolkit_type: str) -> tuple[bool, str]:
        status = self.get_status(toolkit_type)
        if status.get("rknn_installed"):
            return True, "环境已就绪"

        if not status.get("exists"):
            success, msg = self.create_venv(toolkit_type)
            if not success:
                return False, msg

        return self.install_rknn(toolkit_type)