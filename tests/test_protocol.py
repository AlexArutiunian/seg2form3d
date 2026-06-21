import socket
import struct

from sku_rgbd.segmentation.protocol import MAX_MESSAGE_BYTES, recv_message, send_message


def test_rejects_oversized_incoming_message():
    left, right = socket.socketpair()
    try:
        right.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
        try:
            recv_message(left)
            raise AssertionError("oversized message was accepted")
        except ValueError:
            pass
    finally:
        left.close()
        right.close()


def test_rejects_oversized_outgoing_message():
    left, right = socket.socketpair()
    try:
        try:
            send_message(left, b"x" * (MAX_MESSAGE_BYTES + 1))
            raise AssertionError("oversized message was accepted")
        except ValueError:
            pass
    finally:
        left.close()
        right.close()
