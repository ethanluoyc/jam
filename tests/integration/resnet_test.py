# type: ignore
import os

from absl.testing import absltest
from absl.testing import parameterized
import flax
import haiku as hk
import jax
import numpy as np
from safetensors.flax import load_file
import torch
import torchvision

from jam.flax import resnet as resnet_flax
from jam.haiku import resnet as resnet_haiku

RESNET_SIZES = [152, 101, 50, 34, 18]
NUM_CLASSES = 1000

CKPT_DIR = "data/checkpoints/resnet"


def _checkpoint_path(name):
    return os.path.join(CKPT_DIR, f"{name}/torch_model.safetensors")


class ResnetImporterTest(parameterized.TestCase):
    @parameterized.parameters(RESNET_SIZES)
    def test_import_resnet_weights(self, resnet_size):
        weights = torchvision.models.get_weight(f"ResNet{resnet_size}_Weights.DEFAULT")
        torch_model = torchvision.models.get_model(
            f"resnet{resnet_size}", weights=weights
        )
        torch_model.eval()

        haiku_module_cls = getattr(resnet_haiku, f"ResNet{resnet_size}")
        name = f"resnet{resnet_size}"

        def model_fn(x, is_training, test_local_stats=False):
            bn_config = {"decay_rate": 0.9}
            initial_conv_config = {
                "padding": [3, 3],
            }
            return haiku_module_cls(
                num_classes=NUM_CLASSES,
                bn_config=bn_config,
                initial_conv_config=initial_conv_config,
                name=name,
            )(x, is_training=is_training, test_local_stats=test_local_stats)

        hk_model = hk.without_apply_rng(hk.transform_with_state(model_fn))
        # N, H, W, C
        dummy_image = np.random.normal(0, 1, size=(1, 224, 224, 3)).astype(np.float32)
        state_dict = load_file(_checkpoint_path(name))
        (
            restore_params,
            restore_state,
        ) = resnet_haiku.load_from_torch_checkpoint(state_dict, name)

        hk_output, _ = hk_model.apply(
            restore_params, restore_state, dummy_image, is_training=False
        )
        torch_output = self._predict_torch(torch_model, dummy_image)
        np.testing.assert_allclose(torch_output, hk_output, atol=2e-5)

    @parameterized.parameters(RESNET_SIZES)
    def test_import_resnet_weights_flax(self, resnet_size):
        torch_model = self._get_torch_resnet(resnet_size)
        flax_module_cls = getattr(resnet_flax, f"ResNet{resnet_size}")

        bn_config = {"momentum": 0.9}
        initial_conv_config = {"padding": [3, 3]}
        flax_module = flax_module_cls(
            num_classes=NUM_CLASSES,
            bn_config=bn_config,
            initial_conv_config=initial_conv_config,
        )
        name = f"resnet{resnet_size}"
        # # N, H, W, C
        dummy_image = np.random.normal(0, 1, size=(1, 224, 224, 3)).astype(np.float32)
        initial_variables = flax_module.init(
            jax.random.PRNGKey(0), dummy_image, use_running_average=True
        )

        state_dict = load_file(_checkpoint_path(name))
        restored_variables = resnet_flax.load_from_torch_checkpoint(state_dict)
        self.assertTreeSameStructure(initial_variables, restored_variables)

        flax_output = flax_module.apply(
            restored_variables, dummy_image, use_running_average=True
        )
        torch_output = self._predict_torch(torch_model, dummy_image)

        np.testing.assert_allclose(torch_output, flax_output, atol=2e-5)

    def _get_torch_resnet(self, resnet_size):
        weights = torchvision.models.get_weight(f"ResNet{resnet_size}_Weights.DEFAULT")
        torch_model = torchvision.models.get_model(
            f"resnet{resnet_size}", weights=weights
        )
        torch_model.eval()
        return torch_model

    def _predict_torch(self, torch_model, images):
        with torch.no_grad():
            # N, C, H, W
            torch_output = torch_model(
                torch.tensor(np.transpose(images, [0, 3, 1, 2]))
            ).numpy()
        return torch_output

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
