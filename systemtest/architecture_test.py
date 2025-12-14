from utils.constants import *
from utils.architecture import *

def main():
    # Creating model and testing output shapes
    model = YOLOv3(num_classes=CONSTANTS.NUM_CLASSES)
    x = torch.randn((1, 3, CONSTANTS.IMAGE_SIZE, CONSTANTS.IMAGE_SIZE))
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # Asserting output shapes
    assert model(x)[0].shape == (1, 3, CONSTANTS.IMAGE_SIZE//32, CONSTANTS.IMAGE_SIZE//32, CONSTANTS.NUM_CLASSES + 5)
    assert model(x)[1].shape == (1, 3, CONSTANTS.IMAGE_SIZE//16, CONSTANTS.IMAGE_SIZE//16, CONSTANTS.NUM_CLASSES + 5)
    assert model(x)[2].shape == (1, 3, CONSTANTS.IMAGE_SIZE//8, CONSTANTS.IMAGE_SIZE//8, CONSTANTS.NUM_CLASSES + 5)
    print("Output shapes are correct!")