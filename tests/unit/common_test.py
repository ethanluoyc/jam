from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from jam.flax import common


class CommonTest(absltest.TestCase):
    def test_stochastic_depth(self):
        sd = common.StochasticDepth(0.1)
        inputs = jnp.ones((4, 32, 32, 5))
        outputs = sd.apply(
            {}, inputs, deterministic=False, rngs={"dropout": jax.random.PRNGKey(0)}
        )
        chex.assert_equal_shape([inputs, outputs])

        # Evaluation does not require rng.
        outputs = sd.apply({}, inputs, deterministic=True)
        chex.assert_equal_shape([inputs, outputs])

        # Evaluation does not require rng.
        sd = common.StochasticDepth(0.0)
        outputs = sd.apply({}, inputs, deterministic=False)
        chex.assert_equal_shape([inputs, outputs])

        with self.assertRaises(ValueError):
            common.StochasticDepth(-0.1)

        with self.assertRaises(ValueError):
            common.StochasticDepth(1.2)


if __name__ == "__main__":
    absltest.main()
