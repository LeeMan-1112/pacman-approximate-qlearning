#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pacman Arcade Renderer and Windows Media Foundation H.264 MP4 Video Writer.
Produces standard H.264 .mp4 videos without requiring external FFmpeg/OpenCV binaries.
"""

import ctypes
from ctypes import POINTER, byref, c_int32, c_int64, c_uint8, c_uint32, c_uint64, c_void_p, c_wchar_p
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", c_uint8 * 8),
    ]

    def __init__(self, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8):
        super().__init__(l, w1, w2, (c_uint8 * 8)(b1, b2, b3, b4, b5, b6, b7, b8))


MF_VERSION = 0x00020070
MFMediaType_Video = GUID(0x73646976, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)
MFVideoFormat_H264 = GUID(0x34363248, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)
MFVideoFormat_RGB32 = GUID(0x00000016, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)

MF_MT_MAJOR_TYPE = GUID(0x48EBA18E, 0xF8C9, 0x4687, 0xBF, 0x11, 0x0A, 0x74, 0xC9, 0xF9, 0x6A, 0x8F)
MF_MT_SUBTYPE = GUID(0xF7E34C9A, 0x42E8, 0x4714, 0xB7, 0x4B, 0xCB, 0x29, 0xD7, 0x2C, 0x35, 0xE5)
MF_MT_AVG_BITRATE = GUID(0x20332624, 0xFB0D, 0x4D9E, 0xBD, 0x0D, 0xCB, 0xF6, 0x78, 0x6C, 0x10, 0x2E)
MF_MT_INTERLACE_MODE = GUID(0xE2724BB8, 0xE676, 0x4806, 0xB4, 0xB2, 0xA8, 0xD6, 0xEF, 0xB4, 0x4C, 0xCD)
MF_MT_FRAME_SIZE = GUID(0x1652C33D, 0xD6B2, 0x4012, 0xB8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xA3, 0x7D)
MF_MT_FRAME_RATE = GUID(0xC459A2E8, 0x3D2C, 0x4E44, 0xB1, 0x32, 0xFE, 0xE5, 0x15, 0x6C, 0x7B, 0xB0)
MF_MT_PIXEL_ASPECT_RATIO = GUID(0xC6376A1E, 0x8D0A, 0x4027, 0xBE, 0x45, 0x6D, 0x9A, 0x0A, 0xD3, 0x9B, 0xB6)


def _vcall(obj, index, restype, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    func = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])
    return func


class MP4Writer:
    """Native Windows Media Foundation H.264 MP4 encoder."""

    def __init__(self, filepath, width, height, fps=15, bitrate=2500000):
        self.filepath = os.path.abspath(filepath)
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.frame_duration = int(10_000_000 // fps)
        self.rt_start = 0

        self.ole32 = ctypes.windll.ole32
        self.mfplat = ctypes.windll.mfplat
        self.mfreadwrite = ctypes.windll.mfreadwrite

        self.ole32.CoInitializeEx(None, 0)
        self.mfplat.MFStartup(MF_VERSION, 0)

        if os.path.exists(self.filepath):
            os.remove(self.filepath)

        self.writer = c_void_p()
        hr = self.mfreadwrite.MFCreateSinkWriterFromURL(
            c_wchar_p(self.filepath), None, None, byref(self.writer)
        )
        if hr != 0:
            raise RuntimeError(f"MFCreateSinkWriterFromURL failed: 0x{hr & 0xFFFFFFFF:08X}")

        # Output Media Type (H.264)
        mt_out = c_void_p()
        self.mfplat.MFCreateMediaType(byref(mt_out))
        _vcall(mt_out, 24, c_int32, POINTER(GUID), POINTER(GUID))(
            mt_out, byref(MF_MT_MAJOR_TYPE), byref(MFMediaType_Video)
        )
        _vcall(mt_out, 24, c_int32, POINTER(GUID), POINTER(GUID))(
            mt_out, byref(MF_MT_SUBTYPE), byref(MFVideoFormat_H264)
        )
        _vcall(mt_out, 21, c_int32, POINTER(GUID), c_uint32)(
            mt_out, byref(MF_MT_AVG_BITRATE), self.bitrate
        )
        _vcall(mt_out, 21, c_int32, POINTER(GUID), c_uint32)(
            mt_out, byref(MF_MT_INTERLACE_MODE), 2
        )
        _vcall(mt_out, 22, c_int32, POINTER(GUID), c_uint64)(
            mt_out, byref(MF_MT_FRAME_SIZE), (self.width << 32) | self.height
        )
        _vcall(mt_out, 22, c_int32, POINTER(GUID), c_uint64)(
            mt_out, byref(MF_MT_FRAME_RATE), (self.fps << 32) | 1
        )
        _vcall(mt_out, 22, c_int32, POINTER(GUID), c_uint64)(
            mt_out, byref(MF_MT_PIXEL_ASPECT_RATIO), (1 << 32) | 1
        )

        self.stream_index = c_uint32()
        _vcall(self.writer, 3, c_int32, c_void_p, POINTER(c_uint32))(
            self.writer, mt_out, byref(self.stream_index)
        )

        # Input Media Type (RGB32)
        mt_in = c_void_p()
        self.mfplat.MFCreateMediaType(byref(mt_in))
        _vcall(mt_in, 24, c_int32, POINTER(GUID), POINTER(GUID))(
            mt_in, byref(MF_MT_MAJOR_TYPE), byref(MFMediaType_Video)
        )
        _vcall(mt_in, 24, c_int32, POINTER(GUID), POINTER(GUID))(
            mt_in, byref(MF_MT_SUBTYPE), byref(MFVideoFormat_RGB32)
        )
        _vcall(mt_in, 21, c_int32, POINTER(GUID), c_uint32)(
            mt_in, byref(MF_MT_INTERLACE_MODE), 2
        )
        _vcall(mt_in, 22, c_int32, POINTER(GUID), c_uint64)(
            mt_in, byref(MF_MT_FRAME_SIZE), (self.width << 32) | self.height
        )
        _vcall(mt_in, 22, c_int32, POINTER(GUID), c_uint64)(
            mt_in, byref(MF_MT_FRAME_RATE), (self.fps << 32) | 1
        )
        _vcall(mt_in, 22, c_int32, POINTER(GUID), c_uint64)(
            mt_in, byref(MF_MT_PIXEL_ASPECT_RATIO), (1 << 32) | 1
        )

        _vcall(self.writer, 4, c_int32, c_uint32, c_void_p, c_void_p)(
            self.writer, self.stream_index.value, mt_in, None
        )
        _vcall(self.writer, 5, c_int32)(self.writer)

        _vcall(mt_out, 2, c_uint32)(mt_out)
        _vcall(mt_in, 2, c_uint32)(mt_in)

    def write_frame(self, rgb_array):
        """
        Writes an (H, W, 3) uint8 RGB frame to the MP4 video.
        Note: Media Foundation RGB32 expects bottom-up BGRA bytes.
        """
        bgra = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # Flip vertically for bottom-up DIB layout in Media Foundation RGB32
        flipped = rgb_array[::-1, :, :]
        bgra[:, :, 0] = flipped[:, :, 2]  # B
        bgra[:, :, 1] = flipped[:, :, 1]  # G
        bgra[:, :, 2] = flipped[:, :, 0]  # R
        bgra[:, :, 3] = 255               # A
        raw_bytes = bgra.tobytes()
        cb_buffer = len(raw_bytes)

        buf = c_void_p()
        self.mfplat.MFCreateMemoryBuffer(cb_buffer, byref(buf))
        p_data = c_void_p()
        _vcall(buf, 3, c_int32, POINTER(c_void_p), POINTER(c_uint32), POINTER(c_uint32))(
            buf, byref(p_data), None, None
        )
        ctypes.memmove(p_data, raw_bytes, cb_buffer)
        _vcall(buf, 4, c_int32)(buf)
        _vcall(buf, 6, c_int32, c_uint32)(buf, cb_buffer)

        sample = c_void_p()
        self.mfplat.MFCreateSample(byref(sample))
        _vcall(sample, 42, c_int32, c_void_p)(sample, buf)
        _vcall(sample, 35, c_int32, c_int64)(sample, self.rt_start)
        _vcall(sample, 37, c_int32, c_int64)(sample, self.frame_duration)

        _vcall(self.writer, 6, c_int32, c_uint32, c_void_p)(
            self.writer, self.stream_index.value, sample
        )

        _vcall(sample, 2, c_uint32)(sample)
        _vcall(buf, 2, c_uint32)(buf)
        self.rt_start += self.frame_duration

    def close(self):
        if self.writer:
            _vcall(self.writer, 11, c_int32)(self.writer)
            _vcall(self.writer, 2, c_uint32)(self.writer)
            self.writer = None
            self.mfplat.MFShutdown()


class PacmanRenderer:
    """
    Renders Pacman game state to high-clarity RGB frames (width=608, height=720)
    with HUD overlay matching Assignment 1's episode progression video style.
    """

    GHOST_COLORS = [
        (255, 42, 42),    # Blinky (Red)
        (255, 184, 255),  # Pinky (Pink)
        (0, 255, 255),    # Inky (Cyan)
        (255, 184, 81),   # Clyde (Orange)
    ]

    def __init__(self, tile_size=32, hud_height=48):
        self.tile = tile_size
        self.hud_h = hud_height
        self.maze_h = 21
        self.maze_w = 19
        self.width = self.maze_w * self.tile          # 19 * 32 = 608 (multiple of 16)
        self.height = self.maze_h * self.tile + hud_height  # 21 * 32 + 48 = 720 (multiple of 16)

        try:
            self.font_large = ImageFont.truetype("arialbd.ttf", 22)
            self.font_med = ImageFont.truetype("arialbd.ttf", 16)
            self.font_small = ImageFont.truetype("consolab.ttf", 14)
        except Exception:
            self.font_large = ImageFont.load_default()
            self.font_med = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def render_state(self, state, epoch_num, epsilon=0.0, banner_text=None):
        img = Image.new("RGB", (self.width, self.height), (8, 8, 20))
        draw = ImageDraw.Draw(img)

        # 1. Draw Top HUD Bar
        draw.rectangle([0, 0, self.width, self.hud_h - 2], fill=(16, 18, 38))
        draw.line([0, self.hud_h - 2, self.width, self.hud_h - 2], fill=(45, 90, 255), width=2)

        # Episode / Epoch indicator (styled like Assignment 1 video)
        draw.text((14, 12), f"Epoch {epoch_num}", fill=(255, 75, 75), font=self.font_large)
        score_val = int(round(state["score"]))
        pellets_left = int(np.sum(state["pellets"]) + np.sum(state["capsules"]))
        draw.text((160, 14), f"SCORE: {score_val:5d}", fill=(255, 230, 50), font=self.font_med)
        draw.text((315, 14), f"PELLETS: {pellets_left:3d}", fill=(180, 220, 255), font=self.font_med)
        draw.text((460, 14), f"STEP: {state['steps']:3d}", fill=(170, 255, 170), font=self.font_med)

        walls = state["walls"]
        pellets = state["pellets"]
        capsules = state["capsules"]
        t = self.tile
        oy = self.hud_h

        # 2. Draw Maze Walls, Pellets, and Power Capsules
        for r in range(self.maze_h):
            for c in range(self.maze_w):
                x0 = c * t
                y0 = oy + r * t
                x1 = x0 + t
                y1 = y0 + t

                if walls[r, c]:
                    draw.rectangle(
                        [x0 + 2, y0 + 2, x1 - 2, y1 - 2],
                        fill=(14, 28, 95),
                        outline=(33, 66, 225),
                        width=2,
                    )
                elif r == 8 and c == 9:
                    # Ghost house gate
                    draw.line([x0, y0 + t // 2, x1, y0 + t // 2], fill=(255, 184, 255), width=4)
                elif pellets[r, c]:
                    cx, cy = x0 + t // 2, y0 + t // 2
                    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 205, 175))
                elif capsules[r, c]:
                    cx, cy = x0 + t // 2, y0 + t // 2
                    pulse = 8 if (state["steps"] % 4 < 2) else 6
                    draw.ellipse(
                        [cx - pulse, cy - pulse, cx + pulse, cy + pulse],
                        fill=(255, 225, 140),
                        outline=(255, 255, 255),
                    )

        # 3. Draw Pacman
        pr, pc = state["pacman_pos"]
        pdir = state["pacman_dir"]
        px0 = pc * t + 3
        py0 = oy + pr * t + 3
        px1 = (pc + 1) * t - 3
        py1 = oy + (pr + 1) * t - 3

        mouth_open = (state["steps"] % 2 == 0)
        if not mouth_open:
            draw.ellipse([px0, py0, px1, py1], fill=(255, 235, 0))
        else:
            # PIL pieslice angles: 0 is right (3 o'clock), clockwise in degrees
            dir_angles = {
                3: (35, 325),   # RIGHT
                1: (125, 415),  # DOWN
                2: (215, 505),  # LEFT
                0: (305, 595),  # UP
            }
            start_ang, end_ang = dir_angles.get(pdir, (35, 325))
            draw.pieslice([px0, py0, px1, py1], start=start_ang, end=end_ang, fill=(255, 235, 0))

        # 4. Draw Ghosts
        for idx, g in enumerate(state["ghosts"]):
            gr, gc = g["pos"]
            gx0 = gc * t + 3
            gy0 = oy + gr * t + 3
            gx1 = (gc + 1) * t - 3
            gy1 = oy + (gr + 1) * t - 3
            gw = gx1 - gx0
            gh = gy1 - gy0

            scared = g["scared_timer"] > 0
            if scared:
                if g["scared_timer"] <= 8 and (state["steps"] % 2 == 0):
                    gcolor = (240, 240, 255)
                else:
                    gcolor = (35, 65, 255)
            else:
                gcolor = self.GHOST_COLORS[idx % len(self.GHOST_COLORS)]

            # Ghost dome + body
            draw.pieslice([gx0, gy0, gx1, gy0 + gh], start=180, end=360, fill=gcolor)
            draw.rectangle([gx0, gy0 + gh // 2, gx1, gy1 - 3], fill=gcolor)
            # Ghost wavy skirt
            skirt_w = gw // 3
            for s in range(3):
                sx0 = gx0 + s * skirt_w
                draw.pieslice([sx0, gy1 - 7, sx0 + skirt_w, gy1 + 1], start=0, end=180, fill=gcolor)

            # Eyes
            eye_offsets = {0: (0, -2), 1: (0, 2), 2: (-2, 0), 3: (2, 0)}
            edx, edy = eye_offsets.get(g["dir"], (0, 0))
            for ex in [gx0 + gw // 3 - 2, gx0 + (2 * gw) // 3 - 1]:
                ey = gy0 + gh // 3
                if scared:
                    draw.rectangle([ex, ey, ex + 3, ey + 3], fill=(255, 210, 180))
                else:
                    draw.ellipse([ex - 3, ey - 4, ex + 4, ey + 4], fill=(255, 255, 255))
                    draw.ellipse(
                        [ex - 1 + edx, ey - 1 + edy, ex + 2 + edx, ey + 2 + edy],
                        fill=(20, 40, 200),
                    )

        # 5. Optional end-of-episode status banner
        if banner_text:
            bx0, by0 = 74, self.height // 2 - 36
            bx1, by1 = self.width - 74, self.height // 2 + 36
            draw.rectangle([bx0, by0, bx1, by1], fill=(12, 16, 40), outline=(255, 215, 0), width=3)
            draw.text((bx0 + 22, by0 + 22), banner_text, fill=(255, 235, 80), font=self.font_large)

        return np.array(img, dtype=np.uint8)
