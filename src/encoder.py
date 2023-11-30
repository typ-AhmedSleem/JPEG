from ast import Tuple
from pickle import TRUE
import cv2 as cv
import numpy as np
import pandas as pd
import quantization as qtables
import functions as func
import constants as const


class JPEGEncoder:
    def encode(self, qualityFactor: int, inputImagePath: str):
        func.log("=================== ENCODER ===================")
        # * Check if the input image path exists
        if not func.pathExists(inputImagePath):
            func.log("Input image path doesn't exist.")
            exit(1)

        func.log(
            "Image to be compressed has size of {:.1f} KBs".format(
                func.getFileSize(inputImagePath)
            )
        )
        # * Before anything, Read the input image
        imageRGB = func.readImage(inputImagePath)
        func.log(f"Read input RGB image, its shape= {imageRGB.shape}")

        # * Step 1: Transform the image color-space from RGB to YUV
        imageYUV = cv.cvtColor(imageRGB, cv.COLOR_BGR2YCrCb)
        func.log(f"Converted input to YUV image, its shape= {imageRGB.shape}")

        # * Extra step: Get W, H and split the imageYUV for its channels
        Y, Cb, Cr = func.splitToChannels(imageYUV)
        originalSize = Y.size + Cb.size | Cr.size
        func.log(f"Splitted YUV channels, shape={Y.shape}, size={Y.size}")

        # * Step 2: Perform Chrominance DownSampling on the Cb & Cr (take 2x2 as shape for one block of pixels)
        dsCb, dsCr = func.chrominanceDownsampling(blue=Cb, red=Cr, blockSize=2)
        func.log(f"Finished Chroma Downsampling. Cb={dsCb.shape} | Cr={dsCr.shape}")

        # * Before Step 3: Select quantization tables for both luminance and chroma channels and apply quality factor on them
        QTL = func.changeQuality(qtables.QT_LUMINANCE, qualityFactor)  # Luminance table.
        QTC = func.changeQuality(qtables.QT_CHROMINANCE, qualityFactor)  # Chroma table.

        # * Step 3: Apply DCT on (8x8) shape of pixel blocks on all three channels
        # Shift all channels pixels to be in range -128 to 127
        sY = func.shiftChannel(Y).astype(np.float32)
        sCb = func.shiftChannel(dsCb).astype(np.float32)
        sCr = func.shiftChannel(dsCr).astype(np.float32)
        # Apply DCT on shifted luminance channel Y, blue and red chroma channels Cb, Cr
        dctY = func.dctLuminance(sY)
        dctCb, dctCr = func.dctChrominance(sCb, sCr)
        # Apply quantization table and quality factor on all channels
        qY = func.quantizeLuminance(dctY, QTL)
        qCb, qCr = func.quantizeChrominance(dctCb, dctCr, QTC)

        # * Step 4: Perform a zigzag scan on each channel
        zgY, zgCb, zgCr = func.zigzagScanChannels(qY, qCb, qCr)
        func.log("========== ZigZag scan ==========")
        func.log(f"ZigZag Y to: {zgY.size}")
        func.log(f"ZigZag Cb to: {zgCb.size}")
        func.log(f"ZigZag Cr to: {zgCr.size}")

        # * Step 5: Encode each zigzagged channel using RunLengthEncoding (RLE)
        rleY, rleCb, rleCr = func.rleEncode(zgY, zgCb, zgCr)
        func.log("========== RLE encoding ==========")
        func.log(f"RLE Y to: {rleY.size}")
        func.log(f"RLE Cb to: {rleCb.size}")
        func.log(f"RLE Cr to: {rleCr.size}")
        func.log()

        newSize = rleY.size + rleCb.size + rleCr.size

        # ? Preview results from all steps above
        func.log("Starting preview...")
        func.preview("RGB", imageRGB)
        func.preview("YUV", imageYUV)

        func.preview("Luminance Y", Y)
        func.preview("Blue Chroma Cb", Cb)
        func.preview("Red Chroma Cr", Cr)

        func.preview("Downsampled Blue Chroma Cr", dsCb)
        func.preview("Downsampled Red Chroma Cr", dsCr)

        func.preview("Shifted Luminance sY", sY)
        func.preview("Shifted Blue Chroma sCb", sCb)
        func.preview("Shifted Luminance sCr", sCr)

        func.preview("dctY", dctY)
        func.preview("dctCb", dctCb)
        func.preview("dctCr", dctCr)

        func.preview("result", qY)
        func.preview("result", imageRGB)
        func.preview("Quantized Blue Chroma qCb", qCb)
        func.preview("Quantized Red Chroma qCr", qCr)

        func.log("Preview finished.")
        func.log(
            "Compressed image from size {:.1f} KBs to size {:.1f} KBs".format(
                originalSize / 1024, newSize / 1024
            )
        )
        func.log("JPEGEncoder encoded image successfully.")
        return qY.shape, rleY, rleCb, rleCr

if __name__ == "__main__":
    try:
        inputPath = input("Enter image path to decode: ").strip()
        quality = int(input("Choose quality [0:100]: "))
        encoder = JPEGEncoder()
        encoder.encode(quality, inputPath) # shape, rleY, rleCb, rleCr
    except FileNotFoundError:
        func.log("Can't find image at given path.")
    except PermissionError:
        func.log("Can't read image file. Permission error.")
    except KeyboardInterrupt:
        func.log("Terminated by user.")
    except ValueError:
        func.log("Quality is invalid.")