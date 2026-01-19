from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time

from smart_interviewer.settings import settings


def _popen(cmd: list[str]) -> subprocess.Popen:
    # start_new_session=True starts a new process group (Linux/macOS),
    # so we can terminate the whole tree (reloaders, watchers, etc.).
    return subprocess.Popen(cmd, start_new_session=True, env=os.environ.copy())


def _terminate_tree(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main() -> None:
    host = str(settings.HOST)
    port = int(settings.PORT)
    reload_api = bool(settings.RELOAD)

    # Path to your streamlit file (exactly what you run today)
    ui_path = pathlib.Path("src/smart_interviewer/ui/streamlit_app.py").resolve()

    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "smart_interviewer.app:create_app",
        "--factory",
        "--host", host,
        "--port", str(port),
        "--log-level", "info",
    ]
    if reload_api:
        api_cmd.append("--reload")
        # Optional: restrict reload to avoid scanning too much (CPU saver)
        # api_cmd += ["--reload-dir", str(pathlib.Path("src/smart_interviewer").resolve())]
        # api_cmd += ["--reload-exclude", "src/smart_interviewer/ui/*"]

    ui_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        # Big CPU saver:
        "--server.fileWatcherType", "none",
        "--server.runOnSave", "false",
    ]

    print(f"[smart_interviewer] API: http://{host}:{port}")
    api_proc = _popen(api_cmd)

    print("[smart_interviewer] UI : http://localhost:8501")
    ui_proc = _popen(ui_cmd)

    try:
        while True:
            api_rc = api_proc.poll()
            ui_rc = ui_proc.poll()

            if api_rc is not None:
                raise SystemExit(f"API exited with code {api_rc}")
            if ui_rc is not None:
                raise SystemExit(f"UI exited with code {ui_rc}")

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[smart_interviewer] Ctrl+C received, shutting down...")
    finally:
        _terminate_tree(ui_proc)
        _terminate_tree(api_proc)
