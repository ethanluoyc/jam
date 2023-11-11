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
        flax_keys = flax.traverse_util.flatten_dict(
            initial_variables["params"], sep="/"
        ).keys()
        params_flat = flax.traverse_util.flatten_dict(
            initial_variables["params"], sep="/"
        )

        restored_params = convnext.load_from_torch_checkpoint(torch_model.state_dict())[
            "params"
        ]
        restored_flat = flax.traverse_util.flatten_dict(restored_params, sep="/")

        print("In converted but not in flax")
        for k in sorted(set(restored_flat.keys()) - set(flax_keys)):
            print(k)
        print("In flax but not in converted")
        for k in sorted(set(flax_keys) - set(restored_flat.keys())):
            print(k)

        print("Different shapes")
        for k in sorted(set(flax_keys).intersection(set(restored_flat.keys()))):
            if params_flat[k].shape != restored_flat[k].shape:
                print(k, params_flat[k].shape, restored_flat[k].shape)

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
        np.testing.assert_allclose(flax_output, torch_output, atol=0.015)


if __name__ == "__main__":
    absltest.main()
