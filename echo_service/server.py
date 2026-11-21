"""Lightweight REST service interface for ECHO headless operations."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
import threading

from echo_engine import (
    job_display_name,
    job_file_flags,
    job_finished,
    job_mtime,
    list_previous_jobs,
    scale_status_line,
)
from echo_service.cache import ThrottledCache
from echo_service.exporter import export_voxel_model
from echo_service.task_queue import JobManager


class EchoRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request handler for ECHO REST API endpoints."""

    def log_message(self, format, *args):
        # Suppress noisy standard logging to stdout
        pass

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        server: "EchoServer" = self.server  # type: ignore
        path = self.path.split("?")[0]

        if path in ("/api/health", "/"):
            self._send_json(200, {"ok": True, "service": "ECHO REST API", "version": "1.0.0"})
            return

        if path == "/api/status":
            state = server.job_manager.get_state()
            self._send_json(200, {"ok": True, "state": state})
            return

        if path == "/api/jobs":
            def scan():
                jobs = list_previous_jobs(server.jobs_root)
                items = []
                for j in jobs:
                    items.append({
                        "id": j.name,
                        "name": job_display_name(j),
                        "path": str(j),
                        "mtime": job_mtime(j),
                        "finished": job_finished(j),
                        "scaleStatus": scale_status_line(j),
                        "flags": job_file_flags(j),
                    })
                return items

            jobs_list = server.history_cache.get_or_scan(scan)
            self._send_json(200, {"ok": True, "jobs": jobs_list})
            return

        self._send_json(404, {"ok": False, "error": f"Endpoint {path} not found"})

    def do_POST(self):
        server: "EchoServer" = self.server  # type: ignore
        path = self.path.split("?")[0]

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = {}
        if content_length > 0:
            try:
                raw_body = self.rfile.read(content_length)
                post_data = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": f"Invalid JSON payload: {exc}"})
                return

        if path == "/api/jobs/select":
            video_path = post_data.get("path")
            if not video_path:
                self._send_json(400, {"ok": False, "error": "Missing 'path' parameter"})
                return
            ok, err = server.job_manager.select_video(Path(video_path))
            if ok:
                self._send_json(200, {"ok": True, "message": "Video selected successfully"})
            else:
                self._send_json(400, {"ok": False, "error": err})
            return

        if path == "/api/jobs/scene":
            ok, err = server.job_manager.start_scene()
            if ok:
                server.history_cache.invalidate()
                self._send_json(200, {"ok": True, "message": "Scene reconstruction started"})
            else:
                self._send_json(400, {"ok": False, "error": err})
            return

        if path == "/api/jobs/depth":
            ok, err = server.job_manager.start_depth()
            if ok:
                server.history_cache.invalidate()
                self._send_json(200, {"ok": True, "message": "Depth estimation started"})
            else:
                self._send_json(400, {"ok": False, "error": err})
            return

        if path == "/api/export":
            job_path_str = post_data.get("job")
            fmt = post_data.get("format", "glb")
            if not job_path_str:
                self._send_json(400, {"ok": False, "error": "Missing 'job' parameter"})
                return
            job_path = Path(job_path_str)
            if not job_path.exists():
                self._send_json(404, {"ok": False, "error": f"Job path {job_path} does not exist"})
                return

            res = export_voxel_model(job_path, fmt=fmt)
            status_code = 200 if res.get("ok") else 500
            self._send_json(status_code, res)
            return

        self._send_json(404, {"ok": False, "error": f"Endpoint {path} not found"})


class EchoServer(ThreadingHTTPServer):
    """Threading HTTP Server hosting the ECHO service API."""

    def __init__(
        self,
        server_address: tuple,
        job_manager: JobManager,
        jobs_root: Path,
        history_cache: Optional[ThrottledCache] = None,
    ):
        super().__init__(server_address, EchoRequestHandler)
        self.job_manager = job_manager
        self.jobs_root = Path(jobs_root)
        self.history_cache = history_cache or ThrottledCache(ttl_seconds=3.0)
        self._thread: Optional[threading.Thread] = None

    def start_background(self):
        """Start serving requests in a background daemon thread."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Gracefully shut down the server."""
        self.shutdown()
        self.server_close()
