import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from jam.flax.resnet import convert_torch_checkpoint
from jam.flax.resnet import resnet


class R3M(nn.Module):
    resnet_size: int

    def setup(self):
        resnet_cls = {
            18: resnet.ResNet18,
            34: resnet.ResNet34,
            50: resnet.ResNet50,
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

        out = resnet.max_pool(
            out,
            window_shape=(1, 3, 3, 1),
            strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )

        for block_group in network.block_groups:
            out = block_group(out, use_running_average=use_running_average)

        out = jnp.mean(out, axis=(1, 2))
        return out


def load_from_torch_checkpoint(state_dict):
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("convnet."):
            filtered_state_dict[k[8:]] = v

    restored = convert_torch_checkpoint.load_from_torch_checkpoint(filtered_state_dict)
    variables = flax.traverse_util.unflatten_dict(restored, sep="/")

    variables["params"] = {"convnet": variables["params"]}
    variables["batch_stats"] = {"convnet": variables["batch_stats"]}
    return variables
