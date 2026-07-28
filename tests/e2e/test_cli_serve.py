"""
E2E: paper-review serve — 验证 HTTP 服务可通过 CLI 启动并响应请求。

使用 subprocess 启动 serve 子进程，等待就绪后发送 HTTP 请求。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

# 用 requests 做 HTTP 测试
pytest.importorskip("requests")


def _paper_review_bin() -> str:
    """返回 paper-review 可执行文件路径。"""
    bindir = Path(sys.executable).parent
    candidate = bindir / "paper-review"
    if candidate.exists():
        return str(candidate)
    which = subprocess.run(["which", "paper-review"], capture_output=True, text=True, check=False)
    if which.returncode == 0:
        return which.stdout.strip()
    return f"{sys.executable} -m paper_review"


def _wait_for_serve(url: str, timeout: float = 5.0) -> bool:
    """轮询等待 serve 就绪。"""
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.3)
    return False


class TestServeE2E:
    """serve 命令 E2E 验证"""

    def test_serve_health_endpoint(self, tmp_path: Path):
        """启动 serve，验证 /health 返回 ok。"""
        import requests

        port = 18765  # 固定端口避免冲突
        url = f"http://localhost:{port}"
        data_dir = tmp_path / "serve-data"
        data_dir.mkdir(parents=True)

        proc = subprocess.Popen(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "serve",
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            assert _wait_for_serve(url, timeout=8.0), "serve 未在预期时间内就绪"
            resp = requests.get(f"{url}/health", timeout=3)
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_serve_status_endpoint(self, tmp_path: Path):
        """serve 启动后 /status 返回索引状态。"""
        import requests

        port = 18766
        url = f"http://localhost:{port}"
        data_dir = tmp_path / "serve-status-data"
        data_dir.mkdir(parents=True)

        proc = subprocess.Popen(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "serve",
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            assert _wait_for_serve(url, timeout=8.0), "serve 未就绪"
            resp = requests.get(f"{url}/status", timeout=3)
            assert resp.status_code == 200
            data = resp.json()
            assert "papers" in data
            assert data["papers"] == 0  # 空索引
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_serve_search_invalid_query(self, tmp_path: Path):
        """serve 搜索接口对无效查询返回 400。"""
        import requests

        port = 18767
        url = f"http://localhost:{port}"
        data_dir = tmp_path / "serve-search-data"
        data_dir.mkdir(parents=True)

        proc = subprocess.Popen(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "serve",
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            assert _wait_for_serve(url, timeout=8.0), "serve 未就绪"
            # POST 缺少 query 字段
            resp = requests.post(
                f"{url}/search",
                json={"limit": 5},
                timeout=3,
            )
            assert resp.status_code == 400
            assert "error" in resp.json()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
