from flax import linen as nn
import jax
import jax.numpy as jnp

from jam.models.resnet import resnet_flax


class R3M(nn.Module):
    resnet_size: int

    def setup(self):
        resnet_cls = {
            18: resnet_flax.ResNet18,
            34: resnet_flax.ResNet34,
            50: resnet_flax.ResNet50,
        }[self.resnet_size]

        bn_config = {"momentum": 0.9}
        initial_conv_config = {"padding": [3, 3]}

        self.convnet = resnet_cls(
            num_classes=None,  # type: ignore
            bn_config=bn_config,
            initial_conv_config=initial_conv_config,  # type: ignore
            name="convnet",
        )

    def __call__(self, images, use_running_average):
        network = self.convnet
        out = images
        out = network.initial_conv(out)

        out = network.initial_batchnorm(out, use_running_average=use_running_average)
        out = jax.nn.relu(out)

        out = resnet_flax.max_pool(
            out,
            window_shape=(1, 3, 3, 1),
            strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )

        for block_group in network.block_groups:
            out = block_group(out, use_running_average=use_running_average)

        out = jnp.mean(out, axis=(1, 2))
        return out
