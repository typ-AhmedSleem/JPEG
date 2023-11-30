import struct
import numpy as np

# todo: Explore more quantization tables

QT_LUMINANCE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

QT_CHROMINANCE = np.array(
    [
        [6, 4, 4, 6, 10, 16, 20, 24],
        [5, 5, 6, 8, 10, 23, 24, 22],
        [6, 5, 6, 10, 16, 23, 28, 22],
        [6, 7, 9, 12, 20, 35, 32, 25],
        [7, 9, 15, 22, 27, 44, 41, 31],
        [10, 14, 22, 26, 32, 42, 45, 37],
        [20, 26, 31, 35, 41, 48, 48, 40],
        [29, 37, 38, 39, 45, 40, 41, 40],
    ]
)

def unpackBitsSequence(type, buffer, length):
    s = ""
    for _ in range(length):
        s += type
    return list(struct.unpack(s, buffer[:length]))


def obtainQuantizationTable(data: bytes):
    QT_SIZE = 64
    (tableId,) = struct.unpack("B", data[0:1])
    seq = unpackBitsSequence("B", data[1 : 1 + QT_SIZE], QT_SIZE)
    data = data[QT_SIZE + 1 :]
    return tableId, seq
