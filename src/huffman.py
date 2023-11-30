from struct import unpack
import functions as func

HM_MARKERS = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}


class BitStream:
    def __init__(self, data):
        self.pos = 0
        self.data = data

    def getBit(self):
        bit = self.data[self.pos >> 3]
        sign = 7 - (self.pos & 0x7)
        self.pos += 1
        return (bit >> sign) & 1

    def getNBits(self, length):
        val = 0
        for _ in range(length):
            val = val * 2 + self.getBit()
        return val


class HuffmanTable:
    def __init__(self):
        self.content = []
        self.elements = []

    def constructBits(self, root, element, pos):
        if isinstance(root, list):
            if pos == 0:
                if len(root) < 2:
                    root.append(element)
                    return True
                return False
            for i in [0, 1]:
                if len(root) == i:
                    root.append([])
                if self.constructBits(root[i], element, pos - 1) == True:
                    return True
        return False

    def fillHuffmanTable(self, lengths, elements):
        elmIdx = 0
        self.elements = elements
        for i in range(len(lengths)):
            for _ in range(lengths[i]):
                self.constructBits(self.content, elements[elmIdx], i)
                elmIdx += 1

    def findBit(self, bitStream: BitStream):
        result = self.content
        while isinstance(result, list):
            result = result[bitStream.getBit()]
        return result

    def findCode(self, stream: BitStream):
        while True:
            code = self.findBit(stream)
            if code == 0:
                return 0
            elif code != -1:
                return code

    def __repr__(self) -> str:
        return str(self.content)


class HuffmanDecoder:
    def decode(self, data: bytes, tablesMapped):
        offset = 0
        tableId = unpack("B", data[offset : offset + 1])[0]
        offset += 1

        # Extract the 16 bytes containing length data
        lengths = unpack("BBBBBBBBBBBBBBBB", data[offset : offset + 16])
        offset += 16

        # Extract the elements after the initial 16 bytes
        elements = []
        for i in lengths:
            elements += unpack("B" * i, data[offset : offset + i])
            offset += i

        func.log("TableId: ", tableId)
        func.log("Elements: ", len(elements))
        func.log("Lengths: ", lengths)

        # * Create the Huffman table and fill it
        huffmanTable = HuffmanTable()
        huffmanTable.fillHuffmanTable(lengths, elements)
        # Shift data by offset which means that we consumed them
        data = data[offset:]

        return tableId, huffmanTable


class HuffmanEncoder:
    pass
