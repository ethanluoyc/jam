import os

from absl import app
import haiku as hk
import numpy as np
from PIL import Image
from safetensors.flax import load_file

from jam import imagenet_util
from jam.haiku import resnet

NUM_CLASSES = 1000


def preprocess_image(im, imsize):
    # TODO: Use the official preprocessing function from torchvision
    im = im.resize((imsize + 32, imsize + 32))
    im = im.crop((16, 16, 16 + imsize, 16 + imsize))
    # Convert im to tensor and normalize with channel-wise RGB
    return (np.float32(im) - imagenet_util.IMAGENET_MEAN_RGB) / imagenet_util.IMAGENET_STDDEV_RGB  # type: ignore


def main(_):
    resnet_size = 50

    haiku_module_cls = getattr(resnet, f"ResNet{resnet_size}")
    name = f"resnet{resnet_size}"

    def model_fn(x, is_training, test_local_stats=False):
        bn_config = {"decay_rate": 0.9}
        initial_conv_config = {
            "padding": [3, 3],
        }
        return haiku_module_cls(
            num_classes=1000,
            bn_config=bn_config,
            initial_conv_config=initial_conv_config,
            name=name,
        )(x, is_training=is_training, test_local_stats=test_local_stats)

    model = hk.without_apply_rng(hk.transform_with_state(model_fn))

    image = Image.open(os.path.join("tests", "testdata", "peppers.jpg"))
    label = "bell pepper"

    x = preprocess_image(image, 224)
    state_dict = load_file(
        f"data/models/torchvision/{name}-imagenet1k-v2/torch_model.safetensors"
    )
    (params, state) = resnet.load_from_torch_checkpoint(state_dict, name)
    logits, _ = model.apply(params, state, x[None], is_training=False)

    which_class = imagenet_util.IMAGENET_CLASSLIST[int(logits.argmax())]
    print("Predicted class:", which_class)
    print("Actual class:", label)


if __name__ == "__main__":
    app.run(main)
