# type: ignore
from typing import Any, Mapping, Optional, Sequence, Union

from flax import linen as nn
import jax
import jax.numpy as jnp

FloatStrOrBool = Union[str, float, bool]


def max_pool(x, window_shape, strides, padding):
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, window_shape, strides, padding
    )


class BlockV1(nn.Module):
    """ResNet V1 block with optional bottleneck."""

    channels: int
    stride: Union[int, Sequence[int]]
    use_projection: bool
    bn_config: Mapping[str, FloatStrOrBool]
    bottleneck: bool

    def setup(self) -> None:
        bn_config = dict(self.bn_config)
        bn_config.setdefault("use_scale", True)
        bn_config.setdefault("use_bias", True)
        bn_config.setdefault("momentum", 0.999)
        bn_config.setdefault("use_fast_variance", False)

        if self.use_projection:
            self.proj_conv = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                padding="SAME",
                name="shortcut_conv",
            )

            self.proj_batchnorm = nn.BatchNorm(name="shortcut_batchnorm", **bn_config)

        if self.bottleneck:
            conv_0 = nn.Conv(
                features=self.channels // 4,
                kernel_size=(1, 1),
                strides=1,
                use_bias=False,
                padding="SAME",
                name="conv_0",
            )
            bn_0 = nn.BatchNorm(name="batchnorm_0", **bn_config)

            # Place the stride on the second conv rather than the first
            # This is also known as the Resnet V1.5.
            conv_1 = nn.Conv(
                features=self.channels // 4,
                kernel_size=(3, 3),
                strides=self.stride,
                use_bias=False,
                padding=((1, 1), (1, 1)),
                name="conv_1",
            )

            bn_1 = nn.BatchNorm(name="batchnorm_1", **bn_config)
            layers = ((conv_0, bn_0), (conv_1, bn_1))
            conv_2 = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=1,
                use_bias=False,
                padding="SAME",
                name="conv_2",
            )

            bn_2 = nn.BatchNorm(
                name="batchnorm_2", scale_init=nn.initializers.zeros_init(), **bn_config
            )
            layers = layers + ((conv_2, bn_2),)
        else:
            conv_0 = nn.Conv(
                features=self.channels,
                kernel_size=(3, 3),
                strides=self.stride,
                use_bias=False,
                padding=((1, 1), (1, 1)),
                name="conv_0",
            )
            bn_0 = nn.BatchNorm(name="batchnorm_0", **bn_config)

            conv_1 = nn.Conv(
                features=self.channels,
                kernel_size=(3, 3),
                strides=1,
                use_bias=False,
                padding=((1, 1), (1, 1)),
                name="conv_1",
            )

            bn_1 = nn.BatchNorm(name="batchnorm_1", **bn_config)
            layers = ((conv_0, bn_0), (conv_1, bn_1))

        self.layers = layers

    def __call__(self, inputs, use_running_average: bool):
        out = shortcut = inputs

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_batchnorm(
                shortcut, use_running_average=use_running_average
            )

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out, use_running_average=use_running_average)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)


class BlockGroup(nn.Module):
    """Higher level block for ResNet implementation."""

    channels: int
    num_blocks: int
    stride: Union[int, Sequence[int]]
    bn_config: Mapping[str, FloatStrOrBool]
    bottleneck: bool
    use_projection: bool

    def setup(self):
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(
                BlockV1(
                    channels=self.channels,
                    stride=(1 if i else self.stride),
                    use_projection=(i == 0 and self.use_projection),
                    bottleneck=self.bottleneck,
                    bn_config=self.bn_config,
                    name="block_%d" % (i),
                )
            )
        self.blocks = blocks

    def __call__(self, inputs, use_running_average: bool):
        out = inputs
        for block in self.blocks:
            out = block(out, use_running_average)
        return out


def check_length(length, value, name):
    if len(value) != length:
        raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


CONFIGS = {
    18: {
        "blocks_per_group": (2, 2, 2, 2),
        "bottleneck": False,
        "channels_per_group": (64, 128, 256, 512),
        "use_projection": (False, True, True, True),
    },
    34: {
        "blocks_per_group": (3, 4, 6, 3),
        "bottleneck": False,
        "channels_per_group": (64, 128, 256, 512),
        "use_projection": (False, True, True, True),
    },
    50: {
        "blocks_per_group": (3, 4, 6, 3),
        "bottleneck": True,
        "channels_per_group": (256, 512, 1024, 2048),
        "use_projection": (True, True, True, True),
    },
    101: {
        "blocks_per_group": (3, 4, 23, 3),
        "bottleneck": True,
        "channels_per_group": (256, 512, 1024, 2048),
        "use_projection": (True, True, True, True),
    },
    152: {
        "blocks_per_group": (3, 8, 36, 3),
        "bottleneck": True,
        "channels_per_group": (256, 512, 1024, 2048),
        "use_projection": (True, True, True, True),
    },
    200: {
        "blocks_per_group": (3, 24, 36, 3),
        "bottleneck": True,
        "channels_per_group": (256, 512, 1024, 2048),
        "use_projection": (True, True, True, True),
    },
}


class ResNet(nn.Module):
    """ResNet model."""

    blocks_per_group: Sequence[int]
    num_classes: Optional[int]
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None
    bottleneck: bool = True
    channels_per_group: Sequence[int] = (256, 512, 1024, 2048)
    use_projection: Sequence[bool] = (True, True, True, True)
    logits_config: Optional[Mapping[str, Any]] = None
    name: Optional[str] = None
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None
    strides: Sequence[int] = (1, 2, 2, 2)

    def setup(self):
        bn_config = dict(self.bn_config or {})
        bn_config.setdefault("momentum", 0.9)
        bn_config.setdefault("epsilon", 1e-5)
        bn_config.setdefault("use_scale", True)
        bn_config.setdefault("use_bias", True)
        bn_config.setdefault("use_fast_variance", False)

        logits_config = dict(self.logits_config or {})
        logits_config.setdefault("kernel_init", nn.initializers.zeros_init())
        logits_config.setdefault("name", "logits")

        # Number of blocks in each group for ResNet.
        check_length(4, self.blocks_per_group, "blocks_per_group")
        check_length(4, self.channels_per_group, "channels_per_group")
        check_length(4, self.strides, "strides")

        initial_conv_config = dict(self.initial_conv_config or {})
        initial_conv_config.setdefault("features", 64)
        initial_conv_config.setdefault("kernel_size", (7, 7))
        initial_conv_config.setdefault("strides", 2)
        initial_conv_config.setdefault("use_bias", False)
        initial_conv_config.setdefault("padding", "SAME")
        initial_conv_config.setdefault("name", "initial_conv")

        self.initial_conv = nn.Conv(**initial_conv_config)

        self.initial_batchnorm = nn.BatchNorm(name="initial_batchnorm", **bn_config)

        block_groups = []
        for i, stride in enumerate(self.strides):
            block_groups.append(
                BlockGroup(
                    channels=self.channels_per_group[i],
                    num_blocks=self.blocks_per_group[i],
                    stride=stride,
                    bn_config=bn_config,
                    bottleneck=self.bottleneck,
                    use_projection=self.use_projection[i],
                    name="block_group_%d" % (i),
                )
            )

        self.block_groups = block_groups

        if self.num_classes:
            self.logits = nn.Dense(self.num_classes, **logits_config)
        else:
            self.logits = None

    def __call__(self, inputs, use_running_average=False):
        out = inputs
        out = self.initial_conv(out)
        out = self.initial_batchnorm(out, use_running_average=use_running_average)
        out = jax.nn.relu(out)

        out = max_pool(
            out,
            window_shape=(1, 3, 3, 1),
            strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )

        for block_group in self.block_groups:
            out = block_group(out, use_running_average)

        out = jnp.mean(out, axis=(1, 2))

        if self.logits is not None:
            out = self.logits(out)

        return out


def ResNet18(
    num_classes: int,
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    strides: Sequence[int] = (1, 2, 2, 2),
):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride
        of convolutions for each block in each group.
    """
    return ResNet(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        **CONFIGS[18],
    )


def ResNet34(
    num_classes: int,
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    strides: Sequence[int] = (1, 2, 2, 2),
):
    return ResNet(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        **CONFIGS[34],
    )


def ResNet50(
    num_classes: int,
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    strides: Sequence[int] = (1, 2, 2, 2),
):
    return ResNet(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        **CONFIGS[50],
    )


def ResNet101(
    num_classes: int,
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    strides: Sequence[int] = (1, 2, 2, 2),
):
    return ResNet(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        **CONFIGS[101],
    )


def ResNet152(
    num_classes: int,
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    strides: Sequence[int] = (1, 2, 2, 2),
):
    return ResNet(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        **CONFIGS[152],
    )


def ResNet200(
    num_classes: int,
    bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    logits_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
    strides: Sequence[int] = (1, 2, 2, 2),
):
    return ResNet(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        **CONFIGS[200],
    )


if __name__ == "__main__":
    resnet = ResNet18(10)
    variables = resnet.init(
        jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)), use_running_average=False
    )
