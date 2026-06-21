from __future__ import annotations

import pickle
import socket
import struct
import zlib


MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def encode_message(value) -> bytes:
    return zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), level=3)


def decode_message(payload: bytes):
    return pickle.loads(zlib.decompress(payload))


def recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError("Segmentation server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(conn: socket.socket) -> bytes:
    size = struct.unpack("!I", recv_exact(conn, 4))[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid remote message size: {size}")
    return recv_exact(conn, size)


def send_message(conn: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message exceeds {MAX_MESSAGE_BYTES} bytes")
    conn.sendall(struct.pack("!I", len(payload)) + payload)
