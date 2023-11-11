# type: ignore

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision

from jam.flax import convnext


class ConvNextTest(parameterized.TestCase):
    @parameterized.parameters(
        ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
    )
    def test_convnext_torch(self, model_name):
        torch_model = getattr(torchvision.models, model_name)(pretrained=True)
        torch_model.eval()
        module = getattr(convnext, model_name)()
        initial_variables = module.init(
            jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)), is_training=False
        )

        restored_params = convnext.load_from_torch_checkpoint(torch_model.state_dict())[
            "params"
        ]

        self.assertTreeSameStructure(initial_variables["params"], restored_params)
        np.random.seed(0)
        dummy_image = np.random.normal(0, 1, size=(1, 224, 224, 3)).astype(np.float32)
        flax_output = module.apply(
            {"params": restored_params},
            dummy_image,
            is_training=False,
        )
        torch_output = (
            torch_model(torch.tensor(np.transpose(dummy_image, [0, 3, 1, 2])))
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(flax_output, torch_output, atol=1e-5)

    def assertTreeSameStructure(self, tree1, tree2):
        tree1_flat = flax.traverse_util.flatten_dict(tree1, sep="/")
        tree2_flat = flax.traverse_util.flatten_dict(tree2, sep="/")
        for k in sorted(tree2_flat.keys()):
            if k not in tree1_flat:
                raise ValueError("Expect to find {} in initial_variables".format(k))

        for k in sorted(tree1_flat.keys()):
            if k not in tree2_flat:
                raise ValueError("Expect to find {} in restored variables".format(k))

        shared_keys = set(tree1_flat.keys()) & set(tree2_flat.keys())
        for k in sorted(shared_keys):
            tree1_shape = tree1_flat[k].shape
            tree2_shape = tree2_flat[k].shape
            if tree1_shape != tree2_shape:
                raise ValueError("Shape mismatch: {} vs {}".format(k, k))


if __name__ == "__main__":
    absltest.main()
