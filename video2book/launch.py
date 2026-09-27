"""``video2book-app``: start the local web interface and open it in the browser."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser


def _open_when_ready(url: str, port: int, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.3)
    webbrowser.open(url)


def main() -> None:
    from streamlit.web import cli as stcli

    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    port = int(os.environ.get("VIDEO2BOOK_PORT", "8501"))
    url = f"http://localhost:{port}"
    if os.environ.get("VIDEO2BOOK_NO_BROWSER") != "1":
        threading.Thread(target=_open_when_ready, args=(url, port), daemon=True).start()
    print(f"Video2Book is starting at {url}  (close this window to stop it)")
    sys.argv = [
        "streamlit", "run", app,
        "--server.address", "localhost",          # only reachable from this computer
        "--server.port", str(port),
        "--server.headless", "true",              # no first-run e-mail prompt; we open the browser
        "--server.maxUploadSize", "4096",         # MB
        "--browser.gatherUsageStats", "false",    # fully offline, no telemetry
        "--client.toolbarMode", "minimal",
        "--theme.base", "light",
    ] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
