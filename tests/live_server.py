"""Run an app under a real uvicorn server in a background thread (real HTTP, real threadpool and event loop)."""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import uvicorn


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def live_server(app, ready_timeout: float = 120.0) -> Iterator[str]:
    """Yield the base URL once `/ready` is 200 (the app's lifespan runs, so its loader starts); stop on exit."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, name="live-server", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + ready_timeout
    try:
        while True:
            try:
                if httpx.get(base + "/ready", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError(f"server on {base} not ready within {ready_timeout} s")
            time.sleep(0.05)
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=30)
