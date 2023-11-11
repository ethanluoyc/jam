import dataclasses
import functools
from typing import Any, Callable, List, Sequence, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

_default_kernel_init = nn.initializers.truncated_normal(stddev=0.02)


class StochDepth(nn.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    drop_rate: float
    scale_by_keep: bool = False
    rng_collection: str = "dropout"

    def __call__(self, x, is_training) -> jnp.ndarray:
        if not is_training:
            return x
        batch_size = x.shape[0]
        rng = self.make_rng(self.rng_collection)
        r = jax.random.uniform(rng, [batch_size, 1, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = jnp.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


@dataclasses.dataclass
class CNBlockConfig:
    channels: int
    num_blocks: int


class CNBlock(nn.Module):
    dim: int
    layer_scale: float
    stochastic_depth_prob: float
    norm_cls: Any = functools.partial(nn.LayerNorm, epsilon=1e-6)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: jax.nn.gelu(
        x, approximate=True
    )

    def setup(self) -> None:
        self.block = nn.Sequential(
            [
                nn.Conv(
                    features=self.dim,
                    kernel_size=(7, 7),
                    padding=3,
                    feature_group_count=self.dim,
                    use_bias=True,
                    kernel_init=_default_kernel_init,
                ),
                self.norm_cls(),
                nn.Dense(4 * self.dim, use_bias=True, kernel_init=_default_kernel_init),
                self.activation,
                nn.Dense(self.dim, use_bias=True, kernel_init=_default_kernel_init),
            ]
        )
        self.layer_scale_param = self.param(
            "layer_scale",
            lambda key, shape, dtype: jnp.full(shape, self.layer_scale, dtype),
            (self.dim,),
            jnp.float32,
        )
        self.stoch_depth = StochDepth(self.stochastic_depth_prob, scale_by_keep=True)

    def __call__(self, inputs: jnp.ndarray, is_training) -> jnp.ndarray:
        result = self.layer_scale_param * self.block(inputs)
        result = self.stoch_depth(result, is_training)
        result = inputs + result
        return result


class ConvNextStage(nn.Module):
    channels: int
    num_blocks: int
    layer_scale: float
    stochastic_depth_probs: Sequence[Union[float, np.ndarray]]
    stride: int = 1
    block_cls: Any = CNBlock
    norm_cls: Any = functools.partial(nn.LayerNorm, epsilon=1e-6)

    @nn.compact
    def __call__(self, inputs, is_training) -> Any:
        x = inputs
        if inputs.shape[-1] != self.channels or self.stride != 1:
            downsample = nn.Sequential(
                [
                    self.norm_cls(),
                    nn.Conv(
                        features=self.channels,
                        kernel_size=(2, 2),
                        strides=self.stride,
                        kernel_init=_default_kernel_init,
                        use_bias=True,
                    ),
                ]
            )
            x = downsample(x)

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(
                self.block_cls(
                    self.channels, self.layer_scale, self.stochastic_depth_probs[i]
                )
            )
        for block in blocks:
            x = block(x, is_training)

        return x


class ConvNeXt(nn.Module):
    block_settings: List[CNBlockConfig]
    block_cls: Any = CNBlock
    stochastic_depth_prob: float = 0.0
    layer_scale: float = 1e-6
    num_classes: int = 1000
    block_cls: Any = CNBlock
    norm_cls: Any = functools.partial(
        nn.LayerNorm, epsilon=1e-6, use_fast_variance=False
    )

    def setup(self) -> None:
        block_setting = self.block_settings

        firstconv_output_channels = block_setting[0].channels
        stem = nn.Sequential(
            [
                nn.Conv(
                    firstconv_output_channels,
                    kernel_size=(4, 4),
                    strides=4,
                    padding=0,
                    use_bias=True,
                    kernel_init=_default_kernel_init,
                ),
                self.norm_cls(),
            ]
        )
        self.stem = stem

        total_stage_blocks = sum(cnf.num_blocks for cnf in block_setting)
        sd_probs = [
            self.stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            for stage_block_id in range(total_stage_blocks)
        ]
        sd_probs = np.split(
            sd_probs, np.cumsum([cnf.num_blocks for cnf in block_setting])  # type: ignore
        )

        stages = []
        for i, cnf in enumerate(block_setting):
            stages.append(
                ConvNextStage(
                    cnf.channels,
                    cnf.num_blocks,
                    self.layer_scale,
                    stride=2 if i > 0 else 1,
                    stochastic_depth_probs=sd_probs[i],  # type: ignore
                    block_cls=self.block_cls,
                    norm_cls=self.norm_cls,
                )
            )

        self.stages = stages
        self.classifier = nn.Sequential(
            [
                self.norm_cls(),
                lambda x: jnp.reshape(x, (x.shape[0], -1)),
                nn.Dense(self.num_classes, kernel_init=_default_kernel_init),
            ],
            name="classifier",
        )

    @nn.compact
    def __call__(self, inputs, is_training: bool = True) -> jnp.ndarray:
        x = self.stem(inputs)
        for stage in self.stages:
            x = stage(x, is_training)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)
        x = self.classifier(x)
        return x


def convnext_tiny():
    block_setting = [
        CNBlockConfig(96, 3),
        CNBlockConfig(192, 3),
        CNBlockConfig(384, 9),
        CNBlockConfig(768, 3),
    ]
    stocharstic_depth_prob = 0.1
    return ConvNeXt(block_setting, stochastic_depth_prob=stocharstic_depth_prob)


def convnext_small():
    block_setting = [
        CNBlockConfig(96, 3),
        CNBlockConfig(192, 3),
        CNBlockConfig(384, 27),
        CNBlockConfig(768, 3),
    ]
    stocharstic_depth_prob = 0.4
    return ConvNeXt(block_setting, stochastic_depth_prob=stocharstic_depth_prob)


def convnext_base():
    block_setting = [
        CNBlockConfig(128, 3),
        CNBlockConfig(256, 3),
        CNBlockConfig(512, 27),
        CNBlockConfig(1024, 3),
    ]
    stocharstic_depth_prob = 0.5
    return ConvNeXt(block_setting, stochastic_depth_prob=stocharstic_depth_prob)


def convnext_large():
    block_setting = [
        CNBlockConfig(192, 3),
        CNBlockConfig(384, 3),
        CNBlockConfig(768, 27),
        CNBlockConfig(1536, 3),
    ]
    stocharstic_depth_prob = 0.5
    return ConvNeXt(block_setting, stochastic_depth_prob=stocharstic_depth_prob)
