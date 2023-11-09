# type: ignore
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Norm-Free Nets."""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

nfnet_params = {}


# F-series models
nfnet_params.update(
    **{
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
)

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


class WSConv2D(hk.Conv2D):
    """2D Convolution with Scaled Weight Standardization and affine gain+bias."""

    @hk.transparent
    def standardize_weight(self, weight, eps=1e-4):
        """Apply scaled WS with affine gain."""
        mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        # Get gain
        gain = hk.get_parameter(
            "gain", shape=(weight.shape[-1],), dtype=weight.dtype, init=jnp.ones
        )
        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
        shift = mean * scale
        return weight * scale - shift

    def __call__(self, inputs: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index] // self.feature_group_count,
            self.output_channels,
        )
        # Use fan-in scaled init, but WS is largely insensitive to this choice.
        w_init = hk.initializers.VarianceScaling(1.0, "fan_in", "normal")
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)
        weight = self.standardize_weight(w, eps)
        out = jax.lax.conv_general_dilated(
            inputs,
            weight,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.feature_group_count,
        )
        # Always add bias
        bias_shape = (self.output_channels,)
        bias = hk.get_parameter("bias", bias_shape, inputs.dtype, init=jnp.zeros)
        return out + bias


def signal_metrics(x, i):
    """Things to measure about a NCHW tensor activation."""
    metrics = {}
    # Average channel-wise mean-squared
    metrics[f"avg_sq_mean_{i}"] = jnp.mean(jnp.mean(x, axis=[0, 1, 2]) ** 2)
    # Average channel variance
    metrics[f"avg_var_{i}"] = jnp.mean(jnp.var(x, axis=[0, 1, 2]))
    return metrics


class SqueezeExcite(hk.Module):
    """Simple Squeeze+Excite module."""

    def __init__(
        self,
        in_ch,
        out_ch,
        se_ratio=0.5,
        hidden_ch=None,
        activation=jax.nn.relu,
        name=None,
    ):
        super().__init__(name=name)
        self.in_ch, self.out_ch = in_ch, out_ch
        if se_ratio is None:
            if hidden_ch is None:
                raise ValueError("Must provide one of se_ratio or hidden_ch")
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(self.in_ch * se_ratio))
        self.activation = activation
        self.fc0 = hk.Linear(self.hidden_ch, with_bias=True)
        self.fc1 = hk.Linear(self.out_ch, with_bias=True)

    def __call__(self, x):
        h = jnp.mean(x, axis=[1, 2])  # Mean pool over HW extent
        h = self.fc1(self.activation(self.fc0(h)))
        h = jax.nn.sigmoid(h)[:, None, None]  # Broadcast along H, W
        return h


class StochDepth(hk.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, name=None):
        super().__init__(name=name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def __call__(self, x, is_training) -> jnp.ndarray:
        if not is_training:
            return x
        batch_size = x.shape[0]
        r = jax.random.uniform(hk.next_rng_key(), [batch_size, 1, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = jnp.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class NFNet(hk.Module):
    """Normalizer-Free Networks with an improved architecture.

    References:
      [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
      Recognition Without Normalization.
    """

    variant_dict = nfnet_params

    def __init__(
        self,
        num_classes,
        variant="F0",
        width=1.0,
        se_ratio=0.5,
        alpha=0.2,
        stochdepth_rate=0.1,
        drop_rate=None,
        activation="gelu",
        fc_init=None,
        final_conv_mult=2,
        final_conv_ch=None,
        use_two_convs=True,
        name="NFNet",
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.variant = variant
        self.width = width
        self.se_ratio = se_ratio
        # Get variant info
        block_params = self.variant_dict[self.variant]
        self.train_imsize = block_params["train_imsize"]
        self.test_imsize = block_params["test_imsize"]
        self.width_pattern = block_params["width"]
        self.depth_pattern = block_params["depth"]
        self.bneck_pattern = block_params.get("expansion", [0.5] * 4)
        self.group_pattern = block_params.get("group_width", [128] * 4)
        self.big_pattern = block_params.get("big_width", [True] * 4)
        self.activation = nonlinearities[activation]
        if drop_rate is None:
            self.drop_rate = block_params["drop_rate"]
        else:
            self.drop_rate = drop_rate
        self.which_conv = WSConv2D
        # Stem
        ch = self.width_pattern[0] // 2
        self.stem = hk.Sequential(
            [
                self.which_conv(
                    16, kernel_shape=3, stride=2, padding="SAME", name="stem_conv0"
                ),
                self.activation,
                self.which_conv(
                    32, kernel_shape=3, stride=1, padding="SAME", name="stem_conv1"
                ),
                self.activation,
                self.which_conv(
                    64, kernel_shape=3, stride=1, padding="SAME", name="stem_conv2"
                ),
                self.activation,
                self.which_conv(
                    ch, kernel_shape=3, stride=2, padding="SAME", name="stem_conv3"
                ),
            ]
        )

        # Body
        self.blocks = []
        expected_std = 1.0
        num_blocks = sum(self.depth_pattern)
        index = 0  # Overall block index
        stride_pattern = [1, 2, 2, 2]
        block_args = zip(
            self.width_pattern,
            self.depth_pattern,
            self.bneck_pattern,
            self.group_pattern,
            self.big_pattern,
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
                self.blocks += [
                    NFBlock(
                        ch,
                        out_ch,
                        expansion=expand_ratio,
                        se_ratio=se_ratio,
                        group_size=group_size,
                        stride=stride if block_index == 0 else 1,
                        beta=beta,
                        alpha=alpha,
                        activation=self.activation,
                        which_conv=self.which_conv,
                        stochdepth_rate=block_stochdepth_rate,
                        big_width=big_width,
                        use_two_convs=use_two_convs,
                    )
                ]
                ch = out_ch
                index += 1
                # Reset expected std but still give it 1 block of growth
                if block_index == 0:
                    expected_std = 1.0
                expected_std = (expected_std**2 + alpha**2) ** 0.5

        # Head
        if final_conv_mult is None:
            if final_conv_ch is None:
                raise ValueError("Must provide one of final_conv_mult or final_conv_ch")
            ch = final_conv_ch
        else:
            ch = int(final_conv_mult * ch)
        self.final_conv = self.which_conv(
            ch, kernel_shape=1, padding="SAME", name="final_conv"
        )
        # By default, initialize with N(0, 0.01)
        if fc_init is None:
            fc_init = hk.initializers.RandomNormal(mean=0, stddev=0.01)
        self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)

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
        out = self.activation(self.final_conv(out))
        pool = jnp.mean(out, [1, 2])
        outputs["pool"] = pool
        # Optionally apply dropout
        if self.drop_rate > 0.0 and is_training:
            pool = hk.dropout(hk.next_rng_key(), self.drop_rate, pool)
        outputs["logits"] = self.fc(pool)
        return outputs


class NFBlock(hk.Module):
    """Normalizer-Free Net Block."""

    def __init__(
        self,
        in_ch,
        out_ch,
        expansion=0.5,
        se_ratio=0.5,
        kernel_shape=3,
        group_size=128,
        stride=1,
        beta=1.0,
        alpha=0.2,
        which_conv=WSConv2D,
        activation=jax.nn.gelu,
        big_width=True,
        use_two_convs=True,
        stochdepth_rate=None,
        name=None,
    ):
        super().__init__(name=name)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.kernel_shape = kernel_shape
        self.activation = activation
        self.beta, self.alpha = beta, alpha
        # Mimic resnet style bigwidth scaling?
        width = int((self.out_ch if big_width else self.in_ch) * expansion)
        # Round expanded with based on group count
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.stride = stride
        self.use_two_convs = use_two_convs
        # Conv 0 (typically expansion conv)
        self.conv0 = which_conv(
            self.width, kernel_shape=1, padding="SAME", name="conv0"
        )
        # Grouped NxN conv
        self.conv1 = which_conv(
            self.width,
            kernel_shape=kernel_shape,
            stride=stride,
            padding="SAME",
            feature_group_count=self.groups,
            name="conv1",
        )
        if self.use_two_convs:
            self.conv1b = which_conv(
                self.width,
                kernel_shape=kernel_shape,
                stride=1,
                padding="SAME",
                feature_group_count=self.groups,
                name="conv1b",
            )
        # Conv 2, typically projection conv
        self.conv2 = which_conv(
            self.out_ch, kernel_shape=1, padding="SAME", name="conv2"
        )
        # Use shortcut conv on channel change or downsample.
        self.use_projection = stride > 1 or self.in_ch != self.out_ch
        if self.use_projection:
            self.conv_shortcut = which_conv(
                self.out_ch, kernel_shape=1, padding="SAME", name="conv_shortcut"
            )
        # Squeeze + Excite Module
        self.se = SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

        # Are we using stochastic depth?
        self._has_stochdepth = (
            stochdepth_rate is not None
            and stochdepth_rate > 0.0
            and stochdepth_rate < 1.0
        )
        if self._has_stochdepth:
            self.stoch_depth = StochDepth(stochdepth_rate)

    def __call__(self, x, is_training):
        out = self.activation(x) * self.beta
        if self.stride > 1:  # Average-pool downsample.
            shortcut = hk.avg_pool(
                out, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME"
            )
            if self.use_projection:
                shortcut = self.conv_shortcut(shortcut)
        elif self.use_projection:
            shortcut = self.conv_shortcut(out)
        else:
            shortcut = x
        out = self.conv0(out)
        out = self.conv1(self.activation(out))
        if self.use_two_convs:
            out = self.conv1b(self.activation(out))
        out = self.conv2(self.activation(out))
        out = (self.se(out) * 2) * out  # Multiply by 2 for rescaling
        # Get average residual standard deviation for reporting metrics.
        res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
        # Apply stochdepth if applicable.
        if self._has_stochdepth:
            out = self.stoch_depth(out, is_training)
        # SkipInit Gain
        out = out * hk.get_parameter("skip_gain", (), out.dtype, init=jnp.zeros)
        return out * self.alpha + shortcut, res_avg_var
