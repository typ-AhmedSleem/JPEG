import cv2 as cv
import numpy as np
import pandas as pd
from struct import unpack
from cv2.typing import MatLike

# Local imports
import huffman
from jpeg import JPEG
import functions as func
import processing as processor
import constants as const
import quantization as qtz


class JPEGDecoder:
    """
    Class that is responsible for decoding a JPEG image,
    it holds runtime for data below:
        - Huffman tables
        - Quantization tables
        - RLE image pixels for Y, Cb, Cr channels

    Outputs a 3-channel RGB image as a MatLike of shape (H, W, 3)
    """

    def __init__(self, **options):
        # * Quantization tables
        self.qTL = None  # Luminance
        self.qTC = None  # Chrominance
        # * Huffman Tables
        self.huffmanTables = {}
        # * Raw content of image
        self.image = None
        # * Decoders
        self.huffmanDecoder = huffman.HuffmanDecoder()

    def decode(self, imagePath: str):
        # * Step 1: Read the image content
        self.image = JPEG(imagePath)
        data = self.image.read()
        func.log("Image has", len(data), "bytes of data.")
        func.log("========== Step 1: Opened image and read its content")
        # * Step 2: Process data based on markers until reach end of data
        while True:
            # Process fragment of data until next marker
            marker = unpack(">H", data[0:2])[0]
            if marker in huffman.HM_MARKERS:
                func.log()
                func.log(
                    "Reached marker:'", huffman.HM_MARKERS.get(marker), "' marker."
                )
            if marker == 0xFFD8:  # * Start of Image.
                data = data[2:]
            elif marker == 0xFFD9:  # * End of Image.
                return
            else:
                # ? Sections below have data after marker, so we get its length
                chunkLen = unpack(">H", data[2:4])[0]
                chunkLen += 2
                chunk = data[4:chunkLen]
                if marker == 0xFFC4:  # * Huffman Table.
                    tableId, table = self.huffmanDecoder.decode(
                        chunk, self.huffmanTables
                    )
                    self.huffmanTables[tableId] = table
                    func.log("Decoded HuffmanTable:", tableId, "|", table)
                elif marker == 0xFFDB:  # * Quantization Table.
                    tableId, seq = qtz.obtainQuantizationTable(chunk)
                    if tableId == 0:  # Luminance QT.
                        self.qTL = seq
                    elif tableId == 1:  # Chrominance QT.
                        self.qTC = seq
                    func.log("Obtained QT: id=", tableId, "seq=", seq)
                elif marker == 0xFFE0:  # * Application Default Header
                    self.image.processImageData(chunk)
                elif marker == 0xFFC0:  # * Start of Frame
                    pass
                elif marker == 0xFFDA:  # * Start of Scan
                    chunkLen = self.image.processSOS(data, chunkLen, self.qTL, self.qTC)
                data = data[chunkLen:]
            if len(data) == 0:
                break


if __name__ == "__main__":
    try:
        decoder = JPEGDecoder()
        decoder.decode(input("Enter image path to decode: ").strip())
    except FileNotFoundError:
        func.log("Can't find image at given path.")
    except PermissionError:
        func.log("Can't read image file. Permission error.")
    except KeyboardInterrupt:
        func.log("Terminated by user.")
