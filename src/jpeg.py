import functions as func
from struct import unpack

from huffman import BitStream


class JPEG:
    """
    Class that holds data of a jpeg imagefile like:
        - Image metadata (quality, shape)
    """

    def __init__(self, path: str) -> None:
        self.width = 0
        self.height = 0
        self.path = path
        # Encoded image data
        self.data: bytes = None
        # Decoded image channels
        self.Y = None
        self.Cr = None
        self.Cb = None

    @property
    def size(self):
        return (self.height, self.width)

    def read(self):
        if self.data != None:
            return self.data
        # * Check if image exists
        if not func.pathExists(self.path):
            raise FileNotFoundError("Image can't be found on given path")
        # * Open the image file
        with open(self.path, "rb") as image:
            self.data = image.read()
        return self.data

    def processImageData(self, data: bytes):
        header, self.height, self.width, components = unpack(">BHHB", data[0:6])
        func.log("======= JPEG.obtainImageData")
        func.log("Size=", self.size, "| header=", header)
        for idx in range(components):
            offsetR, offsetC = 6 + idx * 3, 9 + idx * 3
            id, sample, qTblId = unpack("BBB", data[offsetR:offsetC])
            func.log("id=", id, "| sample=", sample, "| qTblId=", qTblId)

    def processSOS(self, data, headerLength, QTL, QTC):
        data, chunkLength = func.removeFF00Bytes(data[headerLength:])
        stream = BitStream(data)
        prevCoeffY, prevCoeffCr, prevCoeffCb = 0, 0, 0
        for y in range(self.height // 8):
            for x in range(self.width // 8):
                self.Y, oldCoeffY = func.dequantizeAndBuildBlock(
                    stream, 0, QTL, prevCoeffY
                )
                self.Cr, prevCoeffCr = func.dequantizeAndBuildBlock(
                    stream, 1, QTC, prevCoeffCr
                )
                self.Cb, prevCoeffCb = func.dequantizeAndBuildBlock(
                    stream, 1, QTC, prevCoeffCb
                )
        return chunkLength + headerLength
