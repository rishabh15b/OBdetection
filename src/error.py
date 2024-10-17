from enum import Enum

class Errors(Enum):
    code1 = "No images found in {}"
    code2 = "Data preprocessing script error"
    code3 = "Model training script error"
    code4 = "Inference script error"
    code5 = "Wrong Input given"
