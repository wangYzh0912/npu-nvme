"""Direct HBM-to-NVMe S2-R0 writer and fresh-process replay helper.

This module is intentionally synchronous.  R0 first proves the complete
replacement/ACK/restart contract while keeping the training parameters stable
between capture and ACK.  It uses the native-dtype HBM DMA API for tensor
blocks; only the descriptor and small control payloads use host buffers.
"""

from __future__ import annotations

import binascii
import ctypes
import json
import math
import time
from typing import Mapping

import numpy as np

from c_bindings import lib
from chunk_helpers import build_chunks, build_chunks_host, build_ctypes_arrays
from direct_checkpoint import get_dev_ptr
from incremental_frame import (pack_r0_frame_prefix, unpack_r0_frame_prefix)
from training_state import decode_control_value, encode_control_value


BLOCK_SIZE = 4096


def _align(value, alignment=BLOCK_SIZE):
    return (int(value) + alignment - 1) // alignment * alignment


def _gf2_matrix_times(matrix, vector):
    result = 0
    index = 0
    while vector:
        if vector & 1:
            result ^= matrix[index]
        vector >>= 1
        index += 1
    return result


def _gf2_matrix_square(square, matrix):
    for index in range(32):
        square[index] = _gf2_matrix_times(matrix, matrix[index])


def crc32_combine(crc1, crc2, length2):
    """Return CRC32(A+B) from CRC32(A), CRC32(B), and len(B)."""
    if length2 < 0:
        raise ValueError("CRC suffix length must be non-negative")
    if length2 == 0:
        return (int(crc1) ^ int(crc2)) & 0xFFFFFFFF
    odd = [0] * 32
    even = [0] * 32
    odd[0] = 0xEDB88320
    row = 1
    for index in range(1, 32):
        odd[index] = row
        row <<= 1
    _gf2_matrix_square(even, odd)
    _gf2_matrix_square(odd, even)
    length = int(length2)
    first = int(crc1) & 0xFFFFFFFF
    while length:
        _gf2_matrix_square(even, odd)
        if length & 1:
            first = _gf2_matrix_times(even, first)
        length >>= 1
        if not length:
            break
        _gf2_matrix_square(odd, even)
        if length & 1:
            first = _gf2_matrix_times(odd, first)
        length >>= 1
    return (first ^ (int(crc2) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _combine_records(records, payload_bytes):
    """Combine CRCs for aligned records and their zero padding."""
    crc = 0
    cursor = 0
    zero_crc = {}
    for record in sorted(records, key=lambda item: item["payload_offset"]):
        offset = int(record["payload_offset"])
        if offset < cursor:
            raise ValueError("overlapping payload records")
        gap = offset - cursor
        if gap:
            if gap not in zero_crc:
                zero_crc[gap] = binascii.crc32(bytes(gap)) & 0xFFFFFFFF
            crc = crc32_combine(crc, zero_crc[gap], gap)
        size = int(record["payload_bytes"])
        crc = crc32_combine(crc, int(record["crc32"]), size)
        cursor = offset + size
        padded = _align(cursor)
        if padded != cursor:
            pad = padded - cursor
            if pad not in zero_crc:
                zero_crc[pad] = binascii.crc32(bytes(pad)) & 0xFFFFFFFF
            crc = crc32_combine(crc, zero_crc[pad], pad)
            cursor = padded
    if cursor != int(payload_bytes):
        raise ValueError("payload CRC records do not cover payload")
    return crc


class R0NpuWriter:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Delta raw writer retired; strict FULL only until D2 validation')


class R0NpuReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Delta in-place restore retired; use strict FULL restore')
