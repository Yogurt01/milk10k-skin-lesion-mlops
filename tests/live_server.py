"""Run an app under a real uvicorn server in a background thread (real HTTP, real threadpool and event loop)."""

import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx2
import uvicorn


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_ready(base: str, timeout: float = 120.0, alive: Callable[[], bool] = lambda: True) -> dict:
    """Poll `base/ready` until it is 200 and return its JSON; RuntimeError after `timeout` s or once `alive()` is
    false."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = httpx2.get(base + "/ready", timeout=5)
            if response.status_code == 200:
                return response.json()
        except httpx2.TransportError:
            pass
        if time.monotonic() > deadline or not alive():
            raise RuntimeError(f"server on {base} not ready within {timeout} s")
        time.sleep(0.05)


@contextmanager
def live_server(app, ready_timeout: float = 120.0) -> Iterator[str]:
    """Yield the base URL once `/ready` is 200 (the app's lifespan runs, so its loader starts); stop on exit."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, name="live-server", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        wait_until_ready(base, ready_timeout, alive=thread.is_alive)
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=30)
