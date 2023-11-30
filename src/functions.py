import os
import struct
import cv2 as cv
import numpy as np
import pandas as pd
from typing import Final
import constants as const
from cv2.typing import MatLike


LOGGER_ENABLED: Final = True


def log(*args):
    if LOGGER_ENABLED:
        print(*args)


def pathExists(path: str):
    return os.path.exists(path)


def readImage(path: str):
    src = cv.imread(path)
    return cv.resize(src, const.RESIZE_SHAPE) if const.RESIZE else src


def splitToChannels(image: MatLike):
    return cv.split(image)


def dataFrame(arr):
    return pd.DataFrame(arr)


def preview(title=f"{np.random.randint(0,100)}", image: MatLike = None):
    log(f"Displayed {title} for preview.")
    cv.imshow(title, image)
    cv.waitKey(0)


def removeFF00Bytes(data):
    finalData = []
    idx = 0
    while True:
        byte, nextByte = struct.unpack("BB", data[idx : idx + 2])
        if byte == 0xFF:
            if nextByte != 0:
                break
            finalData.append(data[idx])
            idx += 2
        else:
            finalData.append(data[idx])
            idx += 1
    return finalData, idx

def dequantizeAndBuildBlock(stream, idx: int, QT, oldCoeff):
    return np.ones(shape=(8,8)), oldCoeff



def preview(title=f"{np.random.randint(0,100)}", image: MatLike = None):
    log(f"Displayed {title} for preview.")
    fname = f"{const.BASE_PATH}\\results\\{title}.jpeg"
    if title.lower() == "result":
        cv.imwrite(fname, image, [cv.IMWRITE_JPEG_QUALITY, const.DEF_QLT_FACTOR])
    else:
        cv.imwrite(fname, image)
    cv.imshow(title, image)
    cv.waitKey(0)


def convertColorspace(image: MatLike):
    sR, sB, sG = cv.split(image)
    R = sR.astype(np.float32)
    G = sB.astype(np.float32)
    B = sG.astype(np.float32)

    Y = 0.299 * R + 0.587 * G + 0.144 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    return Y, Cb, Cr


def changeQuality(table, quality: int):
    # * Ensure quality is in range from 1 to 100
    quality = max(1, min(100, quality))
    # * For a quality is below 50%, we increase quantization to remove more details from image
    if quality < 50:
        qualityFactor = 5000 / quality
    else:
        # * Lowering the quantization to increase quality and keep image details
        qualityFactor = max(1.0, 100.0 - quality)

    finalTable: MatLike = np.floor((qualityFactor * table + 50) / 100)

    # * We add this line to prevent division by zero errors
    finalTable[finalTable == 0] = 1

    log(
        f"Changed quality on quantization table. quality: {quality}, resultFactor: {qualityFactor}"
    )
    return finalTable.astype(np.int32)


# ** ENCODING FUNCTION ** #


def chrominanceDownsampling(blue: MatLike, red: MatLike, blockSize: int = 2):
    log("Downsampling Chrominance channels...")
    H, W = blue.shape
    BLOCK_SHAPE: Final = (blockSize, blockSize)
    DS_SHAPE: Final = (H // blockSize, W // blockSize)
    # * Break blue and red channels by 2x2 blocks
    Cb = []  # np.zeros(shape=DS_SHAPE, dtype=np.uint8)
    Cr = []  # np.zeros(shape=DS_SHAPE, dtype=np.uint8)
    # * Downsampling on chrominance channels
    for rowIdx in range(0, H, blockSize):
        blueRow = []
        redRow = []
        for colIdx in range(0, W, blockSize):
            blueBlock = [
                blue[rowIdx - 1, colIdx - 1],
                blue[rowIdx - 1, colIdx],
                blue[rowIdx, colIdx - 1],
                blue[rowIdx, colIdx],
            ]
            redBlock = [
                red[rowIdx - 1, colIdx - 1],
                red[rowIdx - 1, colIdx],
                red[rowIdx, colIdx - 1],
                red[rowIdx, colIdx],
            ]
            # blueBlock = blue[rowIdx : rowIdx + blockSize, colIdx : colIdx + blockSize]
            # redBlock = red[rowIdx : rowIdx + blockSize, colIdx : colIdx + blockSize]
            blueRow.append(np.average(blueBlock))
            redRow.append(np.average(redBlock))
        Cb.append(blueRow)
        Cr.append(redRow)

    log(f"Downsampled Chrominance channels. Got Cb: {blue.shape} & Cr: {red.shape}")
    # * Return the downsampled channels
    return np.array(Cb), np.array(Cr)


def shiftChannel(channel: MatLike):
    log("Shifting channel...")
    shifted: MatLike = channel - 128
    log("Shifted channel.")
    return shifted.astype(np.int8)


def dctLuminance(sY: MatLike):
    DCT_BLOCK_SIZE = 8
    H, W = sY.shape
    dctChannel = np.zeros_like(sY)
    for row in range(0, H, DCT_BLOCK_SIZE):
        for col in range(0, W, DCT_BLOCK_SIZE):
            riIdx, rfIdx = row, row + DCT_BLOCK_SIZE
            ciIdx, cfIdx = col, col + DCT_BLOCK_SIZE
            # Get the 8x8 block
            block = sY[riIdx:rfIdx, ciIdx:cfIdx]
            # Perform DCT on block
            dctBlock = cv.dct(block)
            # Put the dct block back on its location
            dctChannel[riIdx:rfIdx, ciIdx:cfIdx] = dctBlock
        # log("DCT luminance row of blocks.")
    log("Finished luminance channel DCT")
    return dctChannel


def dctChrominance(sCb: MatLike, sCr: MatLike):
    DCT_BLOCK_SIZE = 8
    H, W = sCb.shape
    dctCb = np.zeros_like(sCb)
    dctCr = np.zeros_like(sCr)
    for row in range(0, H, DCT_BLOCK_SIZE):
        for col in range(0, W, DCT_BLOCK_SIZE):
            riIdx, rfIdx = row, row + DCT_BLOCK_SIZE
            ciIdx, cfIdx = col, col + DCT_BLOCK_SIZE
            # Get the 8x8 block from each channel
            sCbBlock = dctCb[riIdx:rfIdx, ciIdx:cfIdx]
            sCrBlock = dctCr[riIdx:rfIdx, ciIdx:cfIdx]
            # Perform DCT on both block
            dctCbBlock = cv.dct(sCbBlock)
            dctCrBlock = cv.dct(sCrBlock)
            # Put the dct blocks back on their locations
            dctCb[riIdx:rfIdx, ciIdx:cfIdx] = dctCbBlock
            dctCr[riIdx:rfIdx, ciIdx:cfIdx] = dctCrBlock
        # log("DCT chrominance row of blocks from Cb & Cr.")
    log("Finished chrominance channels DCT")
    return dctCb, dctCr


def quantizeLuminance(channel: MatLike, quantizationTable: MatLike):
    log(f"Quantizing luminance channel...")
    DCT_BLOCK_SIZE = 8
    H, W = channel.shape
    qtChannel = np.zeros_like(channel)
    for row in range(0, H, DCT_BLOCK_SIZE):
        for col in range(0, W, DCT_BLOCK_SIZE):
            riIdx, rfIdx = row, min(row + DCT_BLOCK_SIZE, H)
            ciIdx, cfIdx = col, min(col + DCT_BLOCK_SIZE, W)
            # * Get the 8x8 block from channel
            qtTable = quantizationTable
            dctBlock = channel[riIdx:rfIdx, ciIdx:cfIdx]
            # * Apply quantization table on block
            if quantizationTable.shape != dctBlock.shape:
                qtTable = quantizationTable[: dctBlock.shape[0], : dctBlock.shape[1]]
            qtBlock = np.round(dctBlock / qtTable)
            # * Put the qtBlock back on its location
            qtChannel[riIdx:rfIdx, ciIdx:cfIdx] = qtBlock
        # log("Quantized luminance row of blocks.")
    log("Finished quantizing luminance channel.")
    return qtChannel


def quantizeChrominance(sCb: MatLike, sCr: MatLike, quantizationTable: MatLike):
    log(f"Quantizing chrominance channels...")
    DCT_BLOCK_SIZE = 8
    H, W = sCb.shape
    qtCb = np.zeros_like(sCb)
    qtCr = np.zeros_like(sCr)
    for row in range(0, H, DCT_BLOCK_SIZE):
        for col in range(0, W, DCT_BLOCK_SIZE):
            riIdx, rfIdx = row, min(row + DCT_BLOCK_SIZE, H)
            ciIdx, cfIdx = col, min(col + DCT_BLOCK_SIZE, W)
            # * Get the 8x8 block from each channel
            qtTable = quantizationTable
            dctCbBlock = sCb[riIdx:rfIdx, ciIdx:cfIdx]
            dctCrBlock = sCr[riIdx:rfIdx, ciIdx:cfIdx]
            # * Apply quantization table on block
            if quantizationTable.shape != dctCbBlock.shape:
                qtTable = quantizationTable[
                    : dctCbBlock.shape[0], : dctCbBlock.shape[1]
                ]
            qtCbBlock = np.round(dctCbBlock / qtTable)
            qtCrBlock = np.round(dctCrBlock / qtTable)
            # * Put the qtBlock back on its location
            qtCb[riIdx:rfIdx, ciIdx:cfIdx] = qtCbBlock
            qtCr[riIdx:rfIdx, ciIdx:cfIdx] = qtCrBlock
        # log("Quantized chrominance two row of blocks from Cb & Cr.")
    log("Finished quantizing chrominance channels.")
    return qtCb, qtCr  # sCb, sCr


def zigzag(image: MatLike):
    result = []
    goingUP = True
    H, W = image.shape
    rowIdx, colIdx = 0, 0
    while rowIdx < H and colIdx < W:
        result.append(image[rowIdx, colIdx])
        # If moving upward
        if goingUP:
            if colIdx == 0 or rowIdx == H - 1:
                goingUP = False
                if rowIdx == H - 1:
                    colIdx += 1
                else:
                    rowIdx += 1
            else:
                rowIdx += 1
                colIdx -= 1
        # If moving downward
        else:
            if rowIdx == 0 or colIdx == W - 1:
                goingUP = True
                if colIdx == W - 1:
                    rowIdx += 1
                else:
                    colIdx += 1
            else:
                rowIdx -= 1
                colIdx += 1
    return np.array(result)


def zigzagScanChannels(qY: MatLike, qCb: MatLike, qCr: MatLike):
    return (
        zigzag(qY.astype(np.int8)),
        zigzag(qCb.astype(np.int8)),
        zigzag(qCr.astype(np.int8)),
    )


def rleEncode(zgY: MatLike, zgCb: MatLike, zgCr: MatLike):
    rleY, rleCb, rleCr = [], [], []
    countY = 1
    # RLE the luminance channel
    for i in range(1, len(zgY)):
        if zgY[i] == zgY[i - 1]:
            countY += 1
        else:
            rleY.extend([countY, zgY[i - 1]])
            countY = 1
    rleY.extend([countY, zgY[-1]])

    # RLE the chrominance channels
    countCb, countCr = 1, 1
    for i in range(1, len(zgCb)):
        # Blue channel
        if zgCb[i] == zgCb[i - 1]:
            countCb += 1
        else:
            rleCb.extend([countCb, zgCb[i - 1]])
            countCb = 1
        # Red channel
        if zgCr[i] == zgCr[i - 1]:
            countCr += 1
        else:
            rleCr.extend([countCr, zgCr[i - 1]])
            countCr = 1
    rleCb.extend([countCb, zgCb[-1]])
    rleCr.extend([countCr, zgCr[-1]])

    return np.array(rleY), np.array(rleCb), np.array(rleCr)


# ** DECODING FUNCTIONS ** #
def rleDecode(rleChannel: MatLike):
    idx = 0
    zgChannel = []
    while idx < len(rleChannel):
        count = rleChannel[idx]
        value = rleChannel[idx + 1]
        zgChannel.extend([value] * count)
        idx += 2
    return zgChannel


def rleDecodeChannels(rleY: MatLike, rleCb: MatLike, rleCr: MatLike):
    # * Decode RLE for Y channel
    zgY = rleDecode(rleY)
    # * Decode RLE for Cb channel
    zgCb = rleDecode(rleCb)
    # * Decode RLE for Cr channel
    zgCr = rleDecode(rleCr)

    return np.array(zgY), np.array(zgCb), np.array(zgCr)


def buildChannelFromZigZag(zigzag: MatLike):
    # Check if the length of the zigzag array is divisible by 64 (8x8)
    if len(zigzag) % 64 != 0:
        log(
            "ERROR: Impossible to make a channel of 8x8 blocks using given zigzag array length"
        )
        exit(1)

    # *  Calculate the number of 8x8 blocks to build the entire channel
    numOfBlocks = zigzag.size // 64
    blocks = np.zeros(shape=(numOfBlocks, numOfBlocks))
    for i in range(numOfBlocks):
        # * Make a zeros 8x8 block
        # block = np.zeros((8, 8), dtype=np.int32)
        # * Fill the 8x8 block with data from zigzag
        for j in range(64):
            row = j // 8
            col = j % 8
            blocks[row][col] = zigzag[i * 64 + j]
        # blocks.append(block)
    return blocks


# ** UTILS ** #
def pad2d_if_needed(array: MatLike, target_shape):
    log(f"Trying to pad from shape {array.shape} to shape {target_shape}")
    if array.shape == target_shape:
        return array
    # Use broadcasting to make array compatible with target shape
    new_shape = (
        max(array.shape[0], target_shape[0]),
        max(array.shape[1], target_shape[1]),
    )
    new_array = np.zeros(new_shape)
    new_array[: array.shape[0], : array.shape[1]] = array
    return new_array


def getFileSize(path: str):
    return os.path.getsize(path)


def writeToFile(img: str):
    with open("output.txt", "w") as out:
        out.write(img)


def writeImageToFile(image: MatLike):
    with open("qY.csv", "w") as out:
        dataFrame(image).to_csv(out, index=False)
        log("Wrote DataFrame into qY.df")