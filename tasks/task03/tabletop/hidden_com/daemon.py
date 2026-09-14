"""Bounded JSON requests over a Unix socket; never deserialize agent code."""
import argparse
import json
import os
from pathlib import Path
import socket
import struct

from .config import SOCKET, STATE
from .service import Session

MAX_REQUEST = 16384


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=SOCKET)
    parser.add_argument("--state", default=STATE)
    args = parser.parse_args()
    os.umask(0o077)
    state = Path(args.state)
    state.parent.mkdir(parents=True, exist_ok=True)
    if state.exists():
        raise RuntimeError("refusing to restart an existing episode")
    session = Session(state, video_root=state.parent / "media")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    os.chmod(args.socket, 0o666)
    server.listen(8)
    while True:
        conn, _ = server.accept()
        with conn:
            conn.settimeout(5)
            _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            try:
                with conn.makefile("rb") as stream:
                    raw = stream.readline(MAX_REQUEST+1)
                    if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                        raise ValueError("request too large or incomplete")
                    request = json.loads(raw)
                    if type(request) is not dict:
                        raise ValueError("expected JSON object")
                    result = {"ok": True, "result": session.handle(request, uid)}
            except (ValueError, TypeError, KeyError, socket.timeout, UnicodeError):
                result = {"ok": False, "error": "invalid request or operation unavailable"}
            except Exception:
                # No traceback, model paths, or hidden values reach the socket.
                session.finalize()
                result = {"ok": False, "error": "simulation stopped"}
            try:
                conn.sendall(json.dumps(result, allow_nan=False).encode()+b"\n")
            except (BrokenPipeError, socket.timeout):
                pass


if __name__ == "__main__":
    main()
