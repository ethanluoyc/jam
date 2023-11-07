from absl.testing import absltest
from absl.testing import parameterized
import jax
import mvp
import numpy as np
import torch
from transformers import FlaxViTModel
from transformers import ViTConfig

from jrm.utils import import_vit


def vit_b16():
    config = ViTConfig(
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        qkv_bias=True,
        intermediate_size=768 * 4,
        layer_norm_eps=1e-6,
        image_size=224,
        num_channels=3,
    )
    return config


def vit_s16():
    config = ViTConfig(
        patch_size=16,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=384 * 4,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        image_size=224,
        num_channels=3,
    )
    return config


def vit_l16(**kwargs):
    config = ViTConfig(
        patch_size=16,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        image_size=224,
        num_channels=3,
        **kwargs,
    )
    return config


class ImportTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "vitb-mae-egosoup",
            "model_name": "vitb-mae-egosoup",
            "config_func": vit_b16,
        },
        {
            "testcase_name": "vits-mae-hoi",
            "model_name": "vits-mae-hoi",
            "config_func": vit_s16,
        },
        {
            "testcase_name": "vits-mae-in",
            "model_name": "vits-mae-in",
            "config_func": vit_s16,
        },
        {
            "testcase_name": "vits-sup-in",
            "model_name": "vits-sup-in",
            "config_func": vit_s16,
        },
        # {
        #     "testcase_name": "vitl-256-mae-egosoup",
        #     "model_name": "vitl-256-mae-egosoup",
        #     "config_func": functools.partial(vit_l16, image_size=256),
        # },
    )
    def test_import_mvp_weights(self, model_name, config_func):
        mvp_model = mvp.load(model_name)
        mvp_model.freeze()
        mvp_model.eval()

        model = FlaxViTModel(config_func(), add_pooling_layer=False)
        restored_params = import_vit.restore_from_torch_checkpoint(
            mvp_model.state_dict()
        )
        restored_params = jax.device_put(restored_params)

        batch_size = 32
        dummy_images = np.random.uniform(0, 1, size=(batch_size, 224, 224, 3)).astype(
            np.float32
        )

        with torch.no_grad():
            mvp_model.eval()
            mvp_output = mvp_model(
                torch.from_numpy(np.transpose(dummy_images, [0, 3, 1, 2]))
            )

        converted_output = model.module.apply(
            {"params": restored_params},
            dummy_images,
            deterministic=True,
            output_hidden_states=True,
        )

        np.testing.assert_allclose(
            mvp_output.numpy(),
            converted_output.last_hidden_state[:, 0],  # type: ignore
            atol=2e-5,
        )


if __name__ == "__main__":
    absltest.main()
