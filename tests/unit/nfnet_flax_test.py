# type: ignore
from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from jam.flax import nfnet


class FlaxNFNetTest(absltest.TestCase):
    def test_wsconv(self):
        conv = nfnet.WSConv2D(features=6, kernel_size=(3, 3))
        inputs = jnp.ones((1, 32, 32, 3))
        variables = conv.init(jax.random.PRNGKey(0), inputs)
        outputs = conv.apply(variables, inputs)
        chex.assert_shape(outputs, (1, 32, 32, 6))

    def test_squeeze_excite(self):
        se = nfnet.SqueezeExcite(5, 3)
        inputs = jnp.ones((4, 32, 32, 5))
        variables = se.init(jax.random.PRNGKey(0), inputs)
        outputs = se.apply(variables, inputs)
        chex.assert_shape(outputs, (4, 1, 1, 3))

    def test_stochastic_depth(self):
        sd = nfnet.StochDepth(5, 3)
        inputs = jnp.ones((4, 32, 32, 5))
        outputs = sd.apply(
            {}, inputs, is_training=True, rngs={"dropout": jax.random.PRNGKey(0)}
        )
        chex.assert_equal_shape([inputs, outputs])

        # Evaluation does not require rng.
        outputs = sd.apply({}, inputs, is_training=False)
        chex.assert_equal_shape([inputs, outputs])

    def test_nf_block(self):
        block = nfnet.NFBlock(
            in_ch=3,
            out_ch=6,
            group_size=3,
            kernel_shape=3,
            stride=1,
            se_ratio=2,
        )
        inputs = jnp.ones((4, 32, 32, 3))
        variables = block.init(jax.random.PRNGKey(0), inputs, is_training=False)
        outputs, _ = block.apply(
            variables, inputs, is_training=True, rngs={"dropout": jax.random.PRNGKey(0)}
        )
        chex.assert_shape(outputs, (4, 32, 32, 6))

    def test_nf_net(self):
        net = nfnet.NFNet(10, variant="F0")
        inputs = jnp.ones((4, 224, 224, 3))
        variables = net.init(jax.random.PRNGKey(0), inputs, is_training=False)
        outputs = net.apply(
            variables, inputs, is_training=True, rngs={"dropout": jax.random.PRNGKey(0)}
        )
        chex.assert_shape(outputs["logits"], (4, 10))


if __name__ == "__main__":
    absltest.main()
