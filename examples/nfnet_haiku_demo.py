import os

from absl import app
import dill
import haiku as hk
import jax
import numpy as np
from PIL import Image

from jam import imagenet_util
from jam.models.nfnet import nfnet_haiku


def preprocess_image(im, imsize):
    im = im.resize((imsize + 32, imsize + 32))
    im = im.crop((16, 16, 16 + imsize, 16 + imsize))
    # Convert im to tensor and normalize with channel-wise RGB
    return (np.float32(im) - imagenet_util.IMAGENET_MEAN_RGB) / imagenet_util.IMAGENET_STDDEV_RGB  # type: ignore


def main(_):
    variant = "F0"
    with open(f"data/checkpoints/nfnet/{variant}/model.npz", "rb") as in_file:
        params = dill.load(in_file)

    image = Image.open(os.path.join("tests", "testdata", "peppers.jpg"))
    label = "bell pepper"

    imsize = nfnet_haiku.nfnet_params[variant]["test_imsize"]

    # Prepare the forward fn
    def forward(inputs, is_training):
        model = nfnet_haiku.NFNet(num_classes=1000, variant=variant)
        return model(inputs, is_training=is_training)["logits"]

    net = hk.without_apply_rng(hk.transform(forward))
    fwd = jax.jit(lambda inputs: net.apply(params, inputs, is_training=False))

    x = preprocess_image(image, imsize)
    # We split this into two cells so that we don't repeatedly jit the fwd fn.
    logits = fwd(x[None])  # Give X a newaxis to make it batch-size-1
    which_class = imagenet_util.IMAGENET_CLASSLIST[int(logits.argmax())]
    print("Predicted class:", which_class)
    print("Actual class:", label)


if __name__ == "__main__":
    app.run(main)
