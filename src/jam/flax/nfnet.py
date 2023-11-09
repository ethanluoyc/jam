"""Architecture definitions for NFNets."""
from typing import Any, Optional, Sequence, Tuple, Union

import flax
from flax import linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import canonicalize_padding
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]
default_kernel_init = initializers.lecun_normal()

# F-series models
nfnet_params = {
    "F0": {
        "width": [256, 512, 1536, 1536],
        "depth": [1, 2, 6, 3],
        "train_imsize": 192,
        "test_imsize": 256,
        "RA_level": "405",
        "drop_rate": 0.2,
    },
    "F1": {
        "width": [256, 512, 1536, 1536],
        "depth": [2, 4, 12, 6],
        "train_imsize": 224,
        "test_imsize": 320,
        "RA_level": "410",
        "drop_rate": 0.3,
    },
    "F2": {
        "width": [256, 512, 1536, 1536],
        "depth": [3, 6, 18, 9],
        "train_imsize": 256,
        "test_imsize": 352,
        "RA_level": "410",
        "drop_rate": 0.4,
    },
    "F3": {
        "width": [256, 512, 1536, 1536],
        "depth": [4, 8, 24, 12],
        "train_imsize": 320,
        "test_imsize": 416,
        "RA_level": "415",
        "drop_rate": 0.4,
    },
    "F4": {
        "width": [256, 512, 1536, 1536],
        "depth": [5, 10, 30, 15],
        "train_imsize": 384,
        "test_imsize": 512,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F5": {
        "width": [256, 512, 1536, 1536],
        "depth": [6, 12, 36, 18],
        "train_imsize": 416,
        "test_imsize": 544,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F6": {
        "width": [256, 512, 1536, 1536],
        "depth": [7, 14, 42, 21],
        "train_imsize": 448,
        "test_imsize": 576,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F7": {
        "width": [256, 512, 1536, 1536],
        "depth": [8, 16, 48, 24],
        "train_imsize": 480,
        "test_imsize": 608,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
}

# Minor variants FN+, slightly wider
nfnet_params.update(
    **{
        **{
            f"{key}+": {
                **nfnet_params[key],
                "width": [384, 768, 2048, 2048],
            }
            for key in nfnet_params
        }
    }
)

# Nonlinearities with magic constants (gamma) baked in.
# Note that not all nonlinearities will be stable, especially if they are
# not perfectly monotonic. Good choices include relu, silu, and gelu.
nonlinearities = {
    "identity": lambda x: x,
    "celu": lambda x: jax.nn.celu(x) * 1.270926833152771,
    "elu": lambda x: jax.nn.elu(x) * 1.2716004848480225,
    "gelu": lambda x: jax.nn.gelu(x) * 1.7015043497085571,
    "glu": lambda x: jax.nn.glu(x) * 1.8484294414520264,
    "leaky_relu": lambda x: jax.nn.leaky_relu(x) * 1.70590341091156,
    "log_sigmoid": lambda x: jax.nn.log_sigmoid(x) * 1.9193484783172607,
    "log_softmax": lambda x: jax.nn.log_softmax(x) * 1.0002083778381348,
    "relu": lambda x: jax.nn.relu(x) * 1.7139588594436646,
    "relu6": lambda x: jax.nn.relu6(x) * 1.7131484746932983,
    "selu": lambda x: jax.nn.selu(x) * 1.0008515119552612,
    "sigmoid": lambda x: jax.nn.sigmoid(x) * 4.803835391998291,
    "silu": lambda x: jax.nn.silu(x) * 1.7881293296813965,
    "soft_sign": lambda x: jax.nn.soft_sign(x) * 2.338853120803833,
    "softplus": lambda x: jax.nn.softplus(x) * 1.9203323125839233,
    "tanh": lambda x: jnp.tanh(x) * 1.5939117670059204,
}


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class WSConv2D(nn.Module):
    """2D Convolution with Scaled Weight Standardization and affine gain+bias.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before and
        after each spatial dimension. A single int is interpreted as applying the same padding
        in all dims and assign a single int in a sequence causes the same padding
        to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
    """

    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "SAME"
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    mask: Optional[Array] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None

    def standardize_weight(self, weight, eps=1e-4):
        """Apply scaled WS with affine gain."""
        mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        # Get gain
        gain = self.param(
            "gain", nn.initializers.ones_init(), (weight.shape[-1],), weight.dtype
        )
        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
        shift = mean * scale
        return weight * scale - shift

    @nn.compact
    def __call__(self, inputs: jax.Array, eps: float = 1e-4) -> jax.Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap'ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """

        if isinstance(self.kernel_size, int):
            raise TypeError(
                "Expected Conv kernel_size to be a"
                " tuple/list of integers (eg.: [3, 3]) but got"
                f" {self.kernel_size}."
            )
        else:
            kernel_size = tuple(self.kernel_size)
            if len(kernel_size) != 2:
                raise ValueError(
                    "Convolution currently only supports 2 spatial dimensions."
                )

        def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if isinstance(padding_lax, str) and padding_lax not in ["SAME", "VALID"]:
            raise ValueError("padding must be 'SAME', 'VALID', or sequence of ints")

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        in_features = jnp.shape(inputs)[-1]

        # One shared convolutional kernel for all pixels in the output.
        assert in_features % self.feature_group_count == 0
        kernel_shape = kernel_size + (
            in_features // self.feature_group_count,
            self.features,
        )

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {kernel_shape}"
            )

        kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
        kernel = self.param("kernel", kernel_init, kernel_shape, self.param_dtype)
        kernel = self.standardize_weight(kernel, eps)

        if self.mask is not None:
            kernel *= self.mask

        bias_shape = (self.features,)
        bias_init = nn.initializers.zeros_init()
        bias = self.param("bias", bias_init, bias_shape, self.param_dtype)

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        y = lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
        y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]  # type: ignore
            y = jnp.reshape(y, output_shape)
        return y


def signal_metrics(x, i):
    """Things to measure about a NCHW tensor activation."""
    metrics = {}
    # Average channel-wise mean-squared
    metrics[f"avg_sq_mean_{i}"] = jnp.mean(jnp.mean(x, axis=[0, 1, 2]) ** 2)
    # Average channel variance
    metrics[f"avg_var_{i}"] = jnp.mean(jnp.var(x, axis=[0, 1, 2]))
    return metrics


class SqueezeExcite(nn.Module):
    """Simple Squeeze+Excite module."""

    in_ch: int
    out_ch: int
    se_ratio: float = 0.5
    hidden_ch: Optional[int] = None
    activation: Any = jax.nn.relu

    @nn.compact
    def __call__(self, x):
        in_ch, out_ch = self.in_ch, self.out_ch
        if self.se_ratio is None:
            if self.hidden_ch is None:
                raise ValueError("Must provide one of se_ratio or hidden_ch")
            hidden_ch = self.hidden_ch
        else:
            hidden_ch = max(1, int(in_ch * self.se_ratio))

        fc0 = nn.Dense(hidden_ch, use_bias=True, name="linear")  # type: ignore
        fc1 = nn.Dense(out_ch, use_bias=True, name="linear_1")
        h = jnp.mean(x, axis=[1, 2])  # Mean pool over HW extent
        h = fc1(self.activation(fc0(h)))
        h = jax.nn.sigmoid(h)[:, None, None]  # Broadcast along H, W
        return h


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


class NFNet(nn.Module):
    """Normalizer-Free Networks with an improved architecture.

    References:
      [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
      Recognition Without Normalization.
    """

    variant_dict = nfnet_params

    num_classes: int
    variant: str = "F0"
    width: float = 1.0
    se_ratio: float = 0.5
    alpha: float = 0.2
    stochdepth_rate: float = 0.1
    drop_rate: Optional[float] = None
    activation: str = "gelu"
    fc_init: Optional[Any] = None
    final_conv_mult: int = 2
    final_conv_ch: Optional[int] = None
    use_two_convs: bool = True

    which_conv: Any = WSConv2D

    def setup(self):
        # Get variant info
        # Stem
        block_params = self.variant_dict[self.variant]
        width_pattern = block_params["width"]
        depth_pattern = block_params["depth"]
        bneck_pattern = block_params.get("expansion", [0.5] * 4)
        group_pattern = block_params.get("group_width", [128] * 4)
        big_pattern = block_params.get("big_width", [True] * 4)
        activation = nonlinearities[self.activation]
        ch = width_pattern[0] // 2
        stochdepth_rate = self.stochdepth_rate
        self.stem = nn.Sequential(
            [
                self.which_conv(
                    16, kernel_size=(3, 3), strides=2, padding="SAME", name="stem_conv0"
                ),
                activation,
                self.which_conv(
                    32, kernel_size=(3, 3), strides=1, padding="SAME", name="stem_conv1"
                ),
                activation,
                self.which_conv(
                    64, kernel_size=(3, 3), strides=1, padding="SAME", name="stem_conv2"
                ),
                activation,
                self.which_conv(
                    ch, kernel_size=(3, 3), strides=2, padding="SAME", name="stem_conv3"
                ),
            ]
        )

        # Body
        blocks = []
        expected_std = 1.0
        num_blocks = sum(depth_pattern)
        index = 0  # Overall block index
        stride_pattern = [1, 2, 2, 2]
        block_args = zip(
            width_pattern,
            depth_pattern,
            bneck_pattern,
            group_pattern,
            big_pattern,
            stride_pattern,
        )
        for (
            block_width,
            stage_depth,
            expand_ratio,
            group_size,
            big_width,
            stride,
        ) in block_args:
            for block_index in range(stage_depth):
                # Scalar pre-multiplier so each block sees an N(0,1) input at init
                beta = 1.0 / expected_std
                # Block stochastic depth drop-rate
                block_stochdepth_rate = stochdepth_rate * index / num_blocks
                out_ch = int(block_width * self.width)
                blocks += [
                    NFBlock(
                        ch,
                        out_ch,
                        expansion=expand_ratio,
                        se_ratio=self.se_ratio,
                        group_size=group_size,
                        stride=stride if block_index == 0 else 1,
                        beta=beta,
                        alpha=self.alpha,
                        activation=activation,
                        which_conv=self.which_conv,
                        stochdepth_rate=block_stochdepth_rate,
                        big_width=big_width,
                        use_two_convs=self.use_two_convs,
                        name="nf_block_{}".format(index) if index > 0 else "nf_block",
                    )
                ]
                ch = out_ch
                index += 1
                # Reset expected std but still give it 1 block of growth
                if block_index == 0:
                    expected_std = 1.0
                expected_std = (expected_std**2 + self.alpha**2) ** 0.5

        self.blocks = blocks
        # Head
        if self.final_conv_mult is None:
            if self.final_conv_ch is None:
                raise ValueError("Must provide one of final_conv_mult or final_conv_ch")
            ch = self.final_conv_ch
        else:
            ch = int(self.final_conv_mult * ch)
        self.final_conv = self.which_conv(
            ch, kernel_size=(1, 1), padding="SAME", name="final_conv"
        )
        # By default, initialize with N(0, 0.01)
        if self.fc_init is None:
            fc_init = nn.initializers.normal(stddev=0.01)
        else:
            fc_init = self.fc_init
        dropout_rate = self.drop_rate
        if dropout_rate is None:
            dropout_rate = block_params["drop_rate"]
        if dropout_rate is None:
            dropout_rate = 0.0
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Dense(
            self.num_classes, kernel_init=fc_init, use_bias=True, name="linear"
        )

    def __call__(self, x, is_training=True, return_metrics=False):
        """Return the output of the final layer without any [log-]softmax."""
        # Stem
        outputs = {}
        out = self.stem(x)
        if return_metrics:
            outputs.update(signal_metrics(out, 0))
        # Blocks
        for i, block in enumerate(self.blocks):
            out, res_avg_var = block(out, is_training=is_training)
            if return_metrics:
                outputs.update(signal_metrics(out, i + 1))
                outputs[f"res_avg_var_{i}"] = res_avg_var
        # Final-conv->activation, pool, dropout, classify
        activation = nonlinearities[self.activation]

        out = activation(self.final_conv(out))
        pool = jnp.mean(out, [1, 2])
        outputs["pool"] = pool
        # Optionally apply dropout
        pool = self.dropout(pool, deterministic=not is_training)
        outputs["logits"] = self.fc(pool)
        return outputs


class NFBlock(nn.Module):
    """Normalizer-Free Net Block."""

    in_ch: int
    out_ch: int
    expansion: float = 0.5
    se_ratio: float = 0.5
    kernel_shape: int = 3
    group_size: int = 128
    stride: int = 1
    beta: float = 1.0
    alpha: float = 0.2
    which_conv: Any = WSConv2D
    activation: Any = jax.nn.gelu
    big_width: bool = True
    use_two_convs: bool = True
    stochdepth_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x, is_training):
        width = int((self.out_ch if self.big_width else self.in_ch) * self.expansion)
        # Round expanded with based on group count
        groups = width // self.group_size
        width = self.group_size * groups

        use_projection = self.stride > 1 or self.in_ch != self.out_ch
        use_stochdepth = (
            self.stochdepth_rate is not None
            and self.stochdepth_rate > 0.0
            and self.stochdepth_rate < 1.0
        )

        conv0 = self.which_conv(width, kernel_size=(1, 1), padding="SAME", name="conv0")
        # Grouped NxN conv
        conv1 = self.which_conv(
            width,
            kernel_size=(self.kernel_shape, self.kernel_shape),
            strides=self.stride,
            padding="SAME",
            feature_group_count=groups,
            name="conv1",
        )
        if self.use_two_convs:
            conv1b = self.which_conv(
                width,
                kernel_size=(self.kernel_shape, self.kernel_shape),
                strides=1,
                padding="SAME",
                feature_group_count=groups,
                name="conv1b",
            )

        # Conv 2, typically projection conv
        conv2 = self.which_conv(
            self.out_ch, kernel_size=(1, 1), padding="SAME", name="conv2"
        )
        # Use shortcut conv on channel change or downsample.
        if use_projection:
            conv_shortcut = self.which_conv(
                self.out_ch, kernel_size=(1, 1), padding="SAME", name="conv_shortcut"
            )
        # Squeeze + Excite Module
        se = SqueezeExcite(
            self.out_ch, self.out_ch, self.se_ratio, name="squeeze_excite"
        )

        # Are we using stochastic depth?
        if use_stochdepth:
            stoch_depth = StochDepth(self.stochdepth_rate)  # type: ignore

        out = self.activation(x) * self.beta
        if self.stride > 1:  # Average-pool downsample.
            shortcut = nn.avg_pool(
                out, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            )
            if use_projection:
                shortcut = conv_shortcut(shortcut)  # type: ignore
        elif use_projection:
            shortcut = conv_shortcut(out)  # type: ignore
        else:
            shortcut = x
        out = conv0(out)
        out = conv1(self.activation(out))
        if self.use_two_convs:
            out = conv1b(self.activation(out))  # type: ignore
        out = conv2(self.activation(out))
        out = (se(out) * 2) * out  # Multiply by 2 for rescaling
        # Get average residual standard deviation for reporting metrics.
        res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
        # Apply stochdepth if applicable.
        if use_stochdepth:
            out = stoch_depth(out, is_training)  # type: ignore
        # SkipInit Gain
        out = out * self.param("skip_gain", nn.initializers.zeros_init(), (), out.dtype)
        return out * self.alpha + shortcut, res_avg_var


def load_from_haiku_checkpoint(haiku_params):
    flat_params = {}
    for module_name, pdict in haiku_params.items():
        for name, value in pdict.items():
            m = module_name.lstrip("NFNet/~/").replace("~/", "")
            if name == "w":
                newname = f"{m}/kernel"
            elif name == "b":
                newname = f"{m}/bias"
            else:
                newname = f"{m}/{name}"
            flat_params[newname] = value

    return flax.traverse_util.unflatten_dict(flat_params, sep="/")
