from absl.testing import absltest
from absl.testing import parameterized
import jax
import mvp
import numpy as np
import torch
from transformers import FlaxViTModel
import utils  # type: ignore

from jam.flax.vit import import_vit
from jam.flax.vit import mvp_flax


def _load(name):
    config_func = mvp_flax.CONFIGS[name.split("-")[0]]
    img_size = 256 if "-256-" in name else 224
    model = FlaxViTModel(config_func(image_size=img_size), add_pooling_layer=False)
    return model


class MVPCheckpointTest(parameterized.TestCase):
    @parameterized.parameters(
        ("vitb-mae-egosoup",),
        ("vits-mae-hoi",),
        ("vitl-256-mae-egosoup",),
    )
    def test_import_mvp_weights(self, model_name):
        mvp_model = mvp.load(model_name)
        mvp_model.freeze()
        mvp_model.eval()

        model = _load(model_name)

        state_dict = utils.load_torch_pretrained_weights(f"mvp/{model_name}")
        restored_params = import_vit.restore_from_torch_checkpoint(state_dict)
        restored_params = jax.device_put(restored_params)

        batch_size = 10
        image_size = 224 if "vitl" not in model_name else 256
        dummy_images = np.random.uniform(
            0, 1, size=(batch_size, image_size, image_size, 3)
        ).astype(np.float32)

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
