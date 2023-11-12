import dataclasses
import functools
from typing import Any, Callable, List, Sequence, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jam.flax import common

_default_kernel_init = nn.initializers.truncated_normal(stddev=0.02)


@dataclasses.dataclass
class CNBlockConfig:
    channels: int
    num_blocks: int


class ConvNeXtBlock(nn.Module):
    dim: int
    layer_scale: float
    stochastic_depth_prob: float
    norm_cls: Any = functools.partial(nn.LayerNorm, epsilon=1e-6)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: jax.nn.gelu(
        x, approximate=False
    )
    dtype: Any = jnp.float32

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
                    name="conv",
                    dtype=self.dtype,
                ),
                self.norm_cls(name="norm"),
                nn.Dense(
                    4 * self.dim,
                    use_bias=True,
                    kernel_init=_default_kernel_init,
                    name="dense_0",
                    dtype=self.dtype,
                ),
                self.activation,
                nn.Dense(
                    self.dim,
                    use_bias=True,
                    kernel_init=_default_kernel_init,
                    name="dense_1",
                    dtype=self.dtype,
                ),
            ]
        )
        self.layer_scale_param = self.param(
            "layer_scale",
            lambda key, shape, dtype: jnp.full(shape, self.layer_scale, dtype),
            (self.dim,),
            self.dtype,
        )
        self.stochastic_depth = common.StochasticDepth(
            self.stochastic_depth_prob, scale_by_keep=True
        )

    def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        result = self.layer_scale_param * self.block(inputs)
        result = self.stochastic_depth(result, deterministic=not is_training)
        result = inputs + result
        return result


class ConvNextStage(nn.Module):
    channels: int
    num_blocks: int
    layer_scale: float
    stochastic_depth_probs: Sequence[Union[float, np.ndarray]]
    stride: int = 1
    block_cls: Any = ConvNeXtBlock
    norm_cls: Any = functools.partial(nn.LayerNorm, epsilon=1e-6)
    dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training) -> Any:
        x = inputs
        if inputs.shape[-1] != self.channels or self.stride != 1:
            downsample = nn.Sequential(
                [
                    self.norm_cls(name="downsample_norm"),
                    nn.Conv(
                        features=self.channels,
                        kernel_size=(2, 2),
                        strides=self.stride,
                        kernel_init=_default_kernel_init,
                        use_bias=True,
                        name="downsample_conv",
                        dtype=self.dtype,
                    ),
                ]
            )
            x = downsample(x)

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(
                self.block_cls(
                    self.channels,
                    self.layer_scale,
                    self.stochastic_depth_probs[i],
                    name=f"block_{i}",
                )
            )
        for block in blocks:
            x = block(x, is_training)

        return x


def _compute_per_block_stochastic_depth_probs(
    stochastic_depth_prob: float, num_blocks: List[int]
) -> List[List[float]]:
    """Computes the per-block stochastic depth probabilities."""
    total_stage_blocks = sum(num_blocks)
    probs = np.linspace(0, stochastic_depth_prob, total_stage_blocks)
    drop_rates = np.split(probs, np.cumsum([b for b in num_blocks]))
    return list(map(lambda x: x.tolist(), drop_rates))[:-1]


class ConvNeXt(nn.Module):
    block_settings: List[CNBlockConfig]
    block_cls: Any = ConvNeXtBlock
    stochastic_depth_prob: float = 0.0
    layer_scale: float = 1e-6
    num_classes: int = 1000
    block_cls: Any = ConvNeXtBlock
    norm_cls: Any = functools.partial(
        nn.LayerNorm, epsilon=1e-6, use_fast_variance=True
    )
    dtype: Any = jnp.float32

    def setup(self) -> None:
        block_setting = self.block_settings
        conv = functools.partial(nn.Conv, dtype=self.dtype)
        norm_cls = functools.partial(self.norm_cls, dtype=self.dtype)
        block_cls = functools.partial(self.block_cls, dtype=self.dtype)

        firstconv_output_channels = block_setting[0].channels
        stem = nn.Sequential(
            [
                conv(
                    firstconv_output_channels,
                    kernel_size=(4, 4),
                    strides=4,
                    padding=0,
                    use_bias=True,
                    kernel_init=_default_kernel_init,
                    name="initial_conv",
                ),
                norm_cls(name="initial_norm"),
            ]
        )
        self.stem = stem

        sd_probs = _compute_per_block_stochastic_depth_probs(
            self.stochastic_depth_prob, [b.num_blocks for b in block_setting]
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
                    block_cls=block_cls,
                    norm_cls=norm_cls,
                    name=f"stage_{i}",
                )
            )

        self.stages = stages
        self.norm = norm_cls(name="norm")
        self.head = nn.Dense(
            self.num_classes,
            kernel_init=_default_kernel_init,
            name="head",
            dtype=self.dtype,
        )

    @nn.compact
    def __call__(self, inputs, is_training: bool = True) -> jnp.ndarray:
        x = self.stem(inputs)
        for stage in self.stages:
            x = stage(x, is_training)
        x = jnp.mean(x, axis=(1, 2))
        x = self.norm(x)
        x = self.head(x)
        return x


def convnext_tiny(**kwargs):
    block_setting = [
        CNBlockConfig(96, 3),
        CNBlockConfig(192, 3),
        CNBlockConfig(384, 9),
        CNBlockConfig(768, 3),
    ]
    stocharstic_depth_prob = 0.1
    return ConvNeXt(
        block_setting, stochastic_depth_prob=stocharstic_depth_prob, **kwargs
    )


def convnext_small(**kwargs):
    block_setting = [
        CNBlockConfig(96, 3),
        CNBlockConfig(192, 3),
        CNBlockConfig(384, 27),
        CNBlockConfig(768, 3),
    ]
    stocharstic_depth_prob = 0.4
    return ConvNeXt(
        block_setting, stochastic_depth_prob=stocharstic_depth_prob, **kwargs
    )


def convnext_base(**kwargs):
    block_setting = [
        CNBlockConfig(128, 3),
        CNBlockConfig(256, 3),
        CNBlockConfig(512, 27),
        CNBlockConfig(1024, 3),
    ]
    stocharstic_depth_prob = 0.5
    return ConvNeXt(
        block_setting, stochastic_depth_prob=stocharstic_depth_prob, **kwargs
    )


def convnext_large(**kwargs):
    block_setting = [
        CNBlockConfig(192, 3),
        CNBlockConfig(384, 3),
        CNBlockConfig(768, 27),
        CNBlockConfig(1536, 3),
    ]
    stocharstic_depth_prob = 0.5
    return ConvNeXt(
        block_setting, stochastic_depth_prob=stocharstic_depth_prob, **kwargs
    )
