import os

from absl import app
import haiku as hk
import numpy as np
from PIL import Image
from safetensors.flax import load_file

from jam import imagenet_util
from jam.models.r3m import r3m_haiku
from jam.models.resnet import import_resnet_haiku


def preprocess_image(im, imsize):
    im = im.resize((imsize + 32, imsize + 32))
    im = im.crop((16, 16, 16 + imsize, 16 + imsize))
    # Convert im to tensor and normalize with channel-wise RGB
    return (np.float32(im) - imagenet_util.IMAGENET_MEAN_RGB) / imagenet_util.IMAGENET_STDDEV_RGB  # type: ignore


def _load_pretrained_checkpoint(model_name):
    state_dict = load_file(f"data/checkpoints/r3m/{model_name}/torch_model.safetensors")
    return state_dict


def restore_from_torch_checkpoint_haiku(state_dict):
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("convnet."):
            filtered_state_dict[k[8:]] = v

    return import_resnet_haiku.restore_from_torch_checkpoint(
        filtered_state_dict, name="r3m/~/convnet"
    )


def main(_):
    resnet_size = 50
    model_name = "r3m-50"

    image = Image.open(os.path.join("tests", "testdata", "peppers.jpg"))
    imsize = 224

    def forward(inputs, is_training=True):
        model = r3m_haiku.R3M(resnet_size)
        return model(inputs, is_training)

    model = hk.without_apply_rng(hk.transform_with_state(forward))

    state_dict = _load_pretrained_checkpoint(model_name)
    params, state = restore_from_torch_checkpoint_haiku(state_dict)

    x = preprocess_image(image, imsize)
    embedding, _ = model.apply(
        params, state, x[None], is_training=False
    )  # Give X a newaxis to make it batch-size-1
    print(embedding[0].shape)


if __name__ == "__main__":
    app.run(main)
