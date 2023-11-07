from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax.numpy as jnp
import numpy as np
import torch
import torchvision

from jrm.models import resnet
from jrm.utils import import_resnet


class ResnetImporterTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "resnet152",
            "model_name": "res_net152",
            "model_cls": resnet.ResNet152,
            "torch_model_cls": torchvision.models.resnet152,
        },
        {
            "testcase_name": "resnet101",
            "model_name": "res_net101",
            "model_cls": resnet.ResNet101,
            "torch_model_cls": torchvision.models.resnet101,
        },
        {
            "testcase_name": "resnet50",
            "model_name": "res_net50",
            "model_cls": resnet.ResNet50,
            "torch_model_cls": torchvision.models.resnet50,
        },
        {
            "testcase_name": "resnet34",
            "model_name": "res_net34",
            "model_cls": resnet.ResNet34,
            "torch_model_cls": torchvision.models.resnet34,
        },
        {
            "testcase_name": "resnet18",
            "model_name": "res_net18",
            "model_cls": resnet.ResNet18,
            "torch_model_cls": torchvision.models.resnet18,
        },
    )
    def test_import_resnet_weights(self, model_name, model_cls, torch_model_cls):
        torch_model = torch_model_cls(pretrained=True)
        torch_model.eval()

        num_classes = 1000

        def forward(x, is_training, test_local_stats=False):
            bn_config = {"decay_rate": 0.9}
            initial_conv_config = {
                "padding": [3, 3],
            }
            return model_cls(
                num_classes=num_classes,
                bn_config=bn_config,
                initial_conv_config=initial_conv_config,
                name=model_name,
            )(x, is_training=is_training, test_local_stats=test_local_stats)

        resnet_hk = hk.without_apply_rng(hk.transform_with_state(forward))
        dummy_image = jnp.ones((1, 224, 224, 3))
        # init_params, init_state = resnet_hk.init(jax.random.PRNGKey(0), dummy_image, is_training=True)
        state_dict = torch_model.state_dict()
        restore_params, restore_state = import_resnet.restore_from_torch_checkpoint(
            state_dict, model_name
        )

        # N, H, W, C
        dummy_image = np.random.normal(0, 1, size=(1, 224, 224, 3)).astype(np.float32)
        hk_output, _ = resnet_hk.apply(
            restore_params, restore_state, dummy_image, is_training=False
        )
        with torch.no_grad():
            # N, C, H, W
            torch_output = torch_model(
                torch.tensor(np.transpose(dummy_image, [0, 3, 1, 2]))
            ).numpy()
        np.testing.assert_allclose(torch_output, hk_output, atol=2e-5)


if __name__ == "__main__":
    absltest.main()
