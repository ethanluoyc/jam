# type: ignore
from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from jam.flax import convnext


class FlaxConvNeXTTest(absltest.TestCase):
    def test_convnext_tiny(self):
        net = convnext.convnext_tiny()
        inputs = jnp.ones((1, 224, 224, 3))
        variables = net.init(jax.random.PRNGKey(0), inputs, False)
        outputs = net.apply(variables, inputs, False)
        chex.assert_shape(outputs, (1, 1000))


if __name__ == "__main__":
    absltest.main()
