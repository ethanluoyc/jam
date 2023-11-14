import os

from absl import app
import dill
import jax
import numpy as np
from PIL import Image

from jam import imagenet_util
from jam.flax import nfnet


def preprocess_image(im, imsize):
    im = im.resize((imsize + 32, imsize + 32))
    im = im.crop((16, 16, 16 + imsize, 16 + imsize))
    # Convert im to tensor and normalize with channel-wise RGB
    return (np.float32(im) - imagenet_util.IMAGENET_MEAN_RGB) / imagenet_util.IMAGENET_STDDEV_RGB  # type: ignore


def main(_):
    variant = "F0"
    with open(f"data/models/nfnet/{variant}/model.npz", "rb") as in_file:
        params = dill.load(in_file)
        params = nfnet.load_from_haiku_checkpoint(params)

    image = Image.open(os.path.join("tests", "testdata", "peppers.jpg"))
    label = "bell pepper"

    imsize = nfnet.nfnet_params[variant]["test_imsize"]

    # Prepare the forward fn
    module = nfnet.NFNet(num_classes=1000, variant=variant)

    def forward(params, inputs, is_training):
        predictions = module.apply({"params": params}, inputs, is_training=is_training)
        return predictions["logits"]  # type: ignore

    fwd = jax.jit(lambda params, inputs: forward(params, inputs, is_training=False))

    x = preprocess_image(image, imsize)
    # We split this into two cells so that we don't repeatedly jit the fwd fn.
    logits = fwd(params, x[None])  # Give X a newaxis to make it batch-size-1
    which_class = imagenet_util.IMAGENET_CLASSLIST[int(logits.argmax())]
    print("Predicted class:", which_class)
    print("Actual class:", label)


if __name__ == "__main__":
    app.run(main)
