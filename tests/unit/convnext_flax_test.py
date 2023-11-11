from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np

from jam.flax import convnext
from jam.flax.convnext.convnext import _compute_per_block_stochastic_depth_probs


class FlaxConvNeXTTest(absltest.TestCase):
    def test_convnext_tiny(self):
        net = convnext.convnext_tiny()
        inputs = jnp.ones((1, 224, 224, 3))
        variables = net.init(jax.random.PRNGKey(0), inputs, False)
        outputs = net.apply(variables, inputs, False)
        chex.assert_shape(outputs, (1, 1000))

    def test_compute_stochdepth_probs(self):
        blocks = [3, 3]
        sd_rate = _compute_per_block_stochastic_depth_probs(0.1, blocks)
        self.assertLen(sd_rate, 2)
        self.assertEqual(sd_rate[0][0], 0)
        self.assertEqual(sd_rate[-1][-1], 0.1)
        np.testing.assert_allclose(
            np.concatenate(sd_rate), np.linspace(0, 0.1, sum(blocks))
        )
        self.assertEqual(sd_rate[-1][-1], 0.1)


if __name__ == "__main__":
    absltest.main()
