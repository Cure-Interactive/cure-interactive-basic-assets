from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np


WIDTH = 4096
HEIGHT = 2048
OUT = Path(__file__).with_name("T_Sky_Day_Cloudy_Partly_A_2K.png")


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def angular_delta(a: np.ndarray, b: float) -> np.ndarray:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def add_cloud(
    rgb: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    lobes: list[tuple[float, float, float, float, int]],
) -> None:
    center_lon = math.radians(center_lon_deg)
    center_lat = math.radians(center_lat_deg)
    cos_center = max(0.18, math.cos(center_lat))

    tones = np.array(
        [
            [65535, 65535, 62400],
            [59200, 61600, 62900],
            [51000, 56000, 60200],
        ],
        dtype=np.float32,
    )

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    shade = np.full((HEIGHT, WIDTH), 2, dtype=np.uint8)

    for lon_offset_deg, lat_offset_deg, rx_deg, ry_deg, tone in lobes:
        lobe_lon = center_lon + math.radians(lon_offset_deg)
        lobe_lat = center_lat + math.radians(lat_offset_deg)
        dx = angular_delta(lon, lobe_lon) * cos_center
        dy = lat - lobe_lat
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        d = (dx / rx) ** 2 + (dy / ry) ** 2
        lobe = 1.0 - smoothstep(0.82, 1.0, d)
        mask = np.maximum(mask, lobe)
        shade = np.where((lobe > 0.53) & (tone < shade), tone, shade)

    hard = mask > 0.48
    if not np.any(hard):
        return

    lower_half = lat < center_lat - math.radians(0.5)
    right_half = angular_delta(lon, center_lon) > 0.0
    shade = np.where(hard & lower_half, np.maximum(shade, 1), shade)
    shade = np.where(hard & lower_half & right_half, 2, shade)

    for tone_index in range(3):
        sel = hard & (shade == tone_index)
        rgb[sel] = tones[tone_index]


def write_png_rgb16(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.uint16)
    if arr.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Expected {(HEIGHT, WIDTH, 3)}, got {arr.shape}")

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 16, 2, 0, 0, 0)
    scanlines = bytearray()
    for row in arr:
        scanlines.append(0)
        scanlines.extend(row.byteswap().tobytes())

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    x = (np.arange(WIDTH, dtype=np.float32) + 0.5) / WIDTH
    y = (np.arange(HEIGHT, dtype=np.float32) + 0.5) / HEIGHT
    lon = x[None, :] * (2.0 * math.pi) - math.pi
    lat = (0.5 - y[:, None]) * math.pi

    horizon = 1.0 - np.abs(lat) / (0.5 * math.pi)
    zenith_bias = np.clip(lat / (0.5 * math.pi), 0.0, 1.0)
    nadir_bias = np.clip(-lat / (0.5 * math.pi), 0.0, 1.0)

    horizon_col = np.array([39200, 55200, 65535], dtype=np.float32)
    zenith_col = np.array([9600, 32200, 60400], dtype=np.float32)
    nadir_col = np.array([24500, 45500, 64200], dtype=np.float32)

    rgb = (
        horizon[..., None] * horizon_col
        + zenith_bias[..., None] * zenith_col
        + nadir_bias[..., None] * nadir_col
    )
    rgb = np.broadcast_to(rgb, (HEIGHT, WIDTH, 3)).copy()

    sun_lon = math.radians(-58.0)
    sun_lat = math.radians(22.0)
    sun_dx = angular_delta(lon, sun_lon) * math.cos(sun_lat)
    sun_dy = lat - sun_lat
    sun = np.exp(-((sun_dx / math.radians(22.0)) ** 2 + (sun_dy / math.radians(13.0)) ** 2))
    rgb += sun[..., None] * np.array([7600, 6100, 2600], dtype=np.float32)

    cloud_specs = [
        (-176, 1, [(-7, -1, 13, 6, 1), (5, 0, 16, 7, 0), (18, -2, 10, 5, 1)]),
        (-132, 15, [(-18, -3, 13, 6, 1), (-5, 2, 17, 8, 0), (11, 1, 15, 7, 1), (25, -3, 11, 5, 2)]),
        (-86, -8, [(-16, -1, 15, 7, 0), (0, 1, 20, 9, 0), (17, -2, 13, 6, 1)]),
        (-42, 25, [(-13, -2, 11, 5, 1), (-2, 1, 15, 7, 0), (11, -1, 12, 6, 1)]),
        (18, 4, [(-20, -3, 14, 6, 2), (-8, 1, 18, 8, 0), (9, 2, 17, 8, 0), (24, -2, 12, 5, 1)]),
        (66, -18, [(-18, -1, 15, 7, 1), (-4, 1, 19, 9, 0), (14, -2, 16, 7, 1), (30, -4, 9, 4, 2)]),
        (116, 11, [(-14, -2, 13, 6, 1), (0, 2, 16, 8, 0), (15, -1, 13, 6, 1)]),
        (166, -3, [(-17, -2, 13, 6, 1), (-2, 1, 18, 8, 0), (15, -1, 15, 7, 1)]),
        (-108, 48, [(-9, -1, 9, 4, 1), (1, 1, 12, 5, 0), (11, -1, 8, 4, 1)]),
        (34, 57, [(-7, 0, 8, 3, 1), (2, 1, 10, 4, 0), (10, -1, 7, 3, 1)]),
        (145, -55, [(-7, 0, 8, 3, 1), (2, 1, 10, 4, 0), (10, -1, 7, 3, 1)]),
        (-18, -49, [(-8, -1, 9, 4, 1), (2, 1, 12, 5, 0), (12, -1, 8, 4, 1)]),
    ]

    for spec in cloud_specs:
        add_cloud(rgb, lon, lat, spec[0], spec[1], spec[2])

    # Add explicit seam-spanning cloud lobes so both image edges carry the same shapes.
    add_cloud(
        rgb,
        lon,
        lat,
        180.0,
        18.0,
        [(-11, -2, 10, 5, 1), (0, 1, 16, 7, 0), (12, -1, 11, 5, 1)],
    )

    arr = np.clip(np.rint(rgb), 0, 65535).astype(np.uint16)
    arr[:, -1, :] = arr[:, 0, :]
    write_png_rgb16(OUT, arr)
    print(OUT)


if __name__ == "__main__":
    main()
