import numpy as np
from cv2 import idct as cvIDCT

def inverseDCT(src):
    return cvIDCT(src)