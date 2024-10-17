import os
from enum import Enum

class Constant(Enum):
    VERSION = "c_0"
    USE_CASE = "obdetection"
    IMAGE_FORMATS = [".jpg", ".png", ".jpg"]
    LOGS_FOLDER = "logs"
    LOGS_NAME = "app.log"
    CONFIG = os.path.join("config", "config_train.yaml")
    TEST_CONFIG = os.path.join("config", "config_test.yaml")
