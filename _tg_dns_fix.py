"""
DNS override for TigerGraph Savanna when local DNS is broken.
Import this module BEFORE any requests/pyTigerGraph calls.

Uses the IPs resolved via Google DNS (8.8.8.8):
  tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io
    -> 34.207.12.133, 100.49.123.93, 44.215.193.149, 3.228.36.105
"""

import socket as _socket

_ORIG_GETADDRINFO = _socket.getaddrinfo

_HOST_OVERRIDES = {
    "tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io": "34.207.12.133",
}


def _patched_getaddrinfo(host, port, *args, **kwargs):
    resolved = _HOST_OVERRIDES.get(host)
    if resolved:
        return _orig_call(resolved, port, *args, **kwargs)
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


def _orig_call(host, port, *args, **kwargs):
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


_socket.getaddrinfo = _patched_getaddrinfo
