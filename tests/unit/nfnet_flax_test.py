# type: ignore
from absl.testing import absltest
import chex
import flax
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

    def test_nfnet_decay_params(self):
        net = nfnet.NFNet(10, variant="F0")
        inputs = jnp.ones((1, 224, 224, 3))
        params = net.init(jax.random.PRNGKey(0), inputs, is_training=False)["params"]

        def decay_mask_fn(params):

            def decay_fn(path):
                if path[-1] == "skipgain":
                    return False
                if path[-1] == "gain":
                    return False
                if "conv" in path[-2] and "bias" in path[-1]:
                    return False
                return True

            flat_params = flax.traverse_util.flatten_dict(params)
            # True for params that we want to apply weight decay to
            flat_mask = {path: decay_fn(path) for path in flat_params}
            return flax.traverse_util.unflatten_dict(flat_mask)

        def clip_mask_fn(params):
            flat_params = flax.traverse_util.flatten_dict(params)
            # True for params that we want to apply weight decay to
            flat_mask = {path: path[0] == 'linear' for path in flat_params}
            return flax.traverse_util.unflatten_dict(flat_mask)

        decay_mask = decay_mask_fn(params)
        clip_mask = clip_mask_fn(params)
        print("Decay mask")
        for k, v in flax.traverse_util.flatten_dict(decay_mask, sep="/").items():
            print(k, v)

        # print("Clip mask")
        # for k, v in flax.traverse_util.flatten_dict(clip_mask, sep="/").items():
        #     print(k, v)


if __name__ == "__main__":
    absltest.main()
