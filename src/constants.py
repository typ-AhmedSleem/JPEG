from typing import Final

# Algorithm options
FILE_EXT = "jpeg"
DEF_QLT_FACTOR = 80
RESIZE_SHAPE = (600, 720)
RESIZE = False


# IO
LOGGER_ENABLED: Final = True
FILENAME = "original (8)"
BASE_PATH = "C:\\Users\\typ\\Development\\algorithms\\JPEG\\src\\images"
INPUT_PATH: Final = f"{BASE_PATH}\\input\\{FILENAME}.jpeg"
OUTPUT_PATH: Final = f"{BASE_PATH}\\output\\{FILENAME}.{FILE_EXT}"
