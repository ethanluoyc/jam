from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from jrm.models import resnet

IMAGENET_MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class R3M(hk.Module):
    def __init__(self, resnet_size: int, name: Optional[str] = "r3m"):
        super().__init__(name)
        resnet_cls = {
            18: resnet.ResNet18,
            34: resnet.ResNet34,
            50: resnet.ResNet50,
        }[resnet_size]

        bn_config = {"decay_rate": 0.9}
        initial_conv_config = {"padding": [3, 3]}

        self.convnet = resnet_cls(
            num_classes=None,  # unused
            bn_config=bn_config,
            initial_conv_config=initial_conv_config,
            name="convnet",
        )

    def __call__(self, images, is_training, test_local_stats=False):
        network = self.convnet
        out = images
        out = network.initial_conv(out)

        if not network.resnet_v2:
            out = network.initial_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)

        out = resnet.max_pool(
            out,
            window_shape=(1, 3, 3, 1),
            strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )

        for block_group in network.block_groups:
            out = block_group(out, is_training, test_local_stats)

        if network.resnet_v2:
            out = network.final_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)

        out = jnp.mean(out, axis=(1, 2))
        return out
