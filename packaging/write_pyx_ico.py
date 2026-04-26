#!/usr/bin/env python3
"""Write a multi-resolution Windows .ico from a PNG (stdlib only)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    """Minimal RGBA8 PNG encoder (no deps)."""
    if len(rgba) != width * height * 4:
        raise ValueError("rgba size mismatch")
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + rgba[y * width * 4 : (y + 1) * width * 4] for y in range(height))
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _resize_nearest(src_w: int, src_h: int, src: bytes, dst_w: int, dst_h: int) -> bytes:
    out = bytearray(dst_w * dst_h * 4)
    for dy in range(dst_h):
        sy = min(src_h - 1, int(dy * src_h / dst_h))
        for dx in range(dst_w):
            sx = min(src_w - 1, int(dx * src_w / dst_w))
            si = (sy * src_w + sx) * 4
            oi = (dy * dst_w + dx) * 4
            out[oi : oi + 4] = src[si : si + 4]
    return bytes(out)


def _read_png_rgba(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    w = h = 0
    idat_parts: list[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4
        ctype = data[pos : pos + 4]
        pos += 4
        chunk_data = data[pos : pos + length]
        pos += length
        pos += 4  # crc
        if ctype == b"IHDR":
            w, h, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", chunk_data)
            if bit_depth != 8 or color_type != 6:
                raise ValueError("need 8-bit RGBA PNG")
        elif ctype == b"IDAT":
            idat_parts.append(chunk_data)
        elif ctype == b"IEND":
            break
    raw = zlib.decompress(b"".join(idat_parts))
    stride = w * 4
    rgba = bytearray(w * h * 4)
    i = 0
    for y in range(h):
        i += 1  # filter byte
        row = raw[i : i + stride]
        i += stride
        rgba[y * stride : (y + 1) * stride] = row
    return w, h, bytes(rgba)


def write_ico(png_path: Path, ico_path: Path, sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)) -> None:
    sw, sh, src = _read_png_rgba(png_path)
    images: list[bytes] = []
    entries: list[tuple[int, int, int, int]] = []  # w, h, png_len, offset
    offset = 6 + len(sizes) * 16
    for sz in sizes:
        rgba = _resize_nearest(sw, sh, src, sz, sz)
        png_bytes = _png_rgba(sz, sz, rgba)
        images.append(png_bytes)
        entries.append((sz, sz, len(png_bytes), offset))
        offset += len(png_bytes)

    out = bytearray()
    out += struct.pack("<HHH", 0, 1, len(sizes))  # reserved, type=icon, count
    for w, h, png_len, off in entries:
        bw = 0 if w >= 256 else w
        bh = 0 if h >= 256 else h
        out += struct.pack("<BBBBHHII", bw, bh, 0, 0, 1, 32, png_len, off)
    for img in images:
        out += img
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    ico_path.write_bytes(out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    write_ico(root / "public" / "brand" / "pyx-app-icon.png", root / "packaging" / "windows" / "pyx.ico")
    print("Wrote", root / "packaging" / "windows" / "pyx.ico")
