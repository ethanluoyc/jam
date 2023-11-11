from typing import Optional

from flax import linen as nn
import jax
import jax.numpy as jnp


class StochasticDepth(nn.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    drop_rate: float
    scale_by_keep: bool = False
    deterministic: Optional[bool] = None

    rng_collection: str = "dropout"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.drop_rate < 0.0 or self.drop_rate > 1.0:
            raise ValueError(f"drop_rate must be in [0, 1], got {self.drop_rate}")

    def __call__(
        self,
        inputs,
        deterministic: Optional[bool] = None,
        rng: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        if (self.drop_rate == 0.0) or deterministic:
            return inputs

        if self.drop_rate == 1.0:
            return jnp.zeros_like(inputs)

        batch_size = inputs.shape[0]
        if rng is None:
            rng = self.make_rng(self.rng_collection)

        r = jax.random.uniform(rng, [batch_size, 1, 1, 1], dtype=inputs.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = jnp.floor(keep_prob + r)
        outputs = inputs
        if self.scale_by_keep:
            outputs = outputs / keep_prob
        return outputs * binary_tensor
