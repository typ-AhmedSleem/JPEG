from typing import Final

# Algorithm options
FILE_EXT = "jpeg"
DEF_QLT_FACTOR = 80
RESIZE_SHAPE = (600, 720)
RESIZE = False

# IO
BASE_PATH = "C:\\Users\\typ\\Development\\algorithms\\JPEG\\src\\images"
FILENAME = "result"
INPUT_PATH: Final = f"{BASE_PATH}\\output\\{FILENAME}.jpeg"
OUTPUT_PATH: Final = f"{BASE_PATH}\\output\\{FILENAME}.{FILE_EXT}"