import os

from absl import app
import numpy as np
from PIL import Image
import torchvision

from jam import imagenet_util
from jam.flax import convnext

NUM_CLASSES = 1000


def preprocess_image(im, imsize):
    # TODO: Use the official preprocessing function from torchvision
    im = im.resize((imsize + 32, imsize + 32))
    im = im.crop((16, 16, 16 + imsize, 16 + imsize))
    # Convert im to tensor and normalize with channel-wise RGB
    return (np.float32(im) - imagenet_util.IMAGENET_MEAN_RGB) / imagenet_util.IMAGENET_STDDEV_RGB  # type: ignore


def main(_):
    flax_module = convnext.convnext_tiny()

    # N, H, W, C
    state_dict = torchvision.models.convnext_tiny(pretrained=True).state_dict()
    restored_variables = convnext.load_from_torch_checkpoint(state_dict)

    image = Image.open(os.path.join("tests", "testdata", "peppers.jpg"))
    label = "bell pepper"

    x = preprocess_image(image, 224)
    logits = flax_module.apply(restored_variables, x[None], is_training=False)
    which_class = imagenet_util.IMAGENET_CLASSLIST[int(logits.argmax())]  # type: ignore
    print("Predicted class:", which_class)
    print("Actual class:", label)


if __name__ == "__main__":
    app.run(main)
