from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import numpy as np
from r3m import load_r3m
import torch
import utils  # type: ignore

from jam import imagenet_util
from jam.flax import r3m as r3m_flax
from jam.haiku import r3m as r3m_haiku

R3M_RESNET_MODELS = ["r3m-18", "r3m-34", "r3m-50"]


def _normalize_image(inputs):
    return (inputs - np.array(imagenet_util.IMAGENET_MEAN_RGB)) / np.array(
        imagenet_util.IMAGENET_STDDEV_RGB
    )


class R3MCheckpointTest(parameterized.TestCase):
    @parameterized.parameters(R3M_RESNET_MODELS)
    def test_haiku_r3m(self, model_name):
        r3m = self._load_r3m_torch_module(model_name)
        resnet_size = int(model_name.split("-")[1])  # type: ignore

        def forward(inputs, is_training=True):
            model = r3m_haiku.R3M(resnet_size)
            return model(inputs, is_training)

        resnet_hk = hk.without_apply_rng(hk.transform_with_state(forward))

        state_dict = utils.load_torch_pretrained_weights("r3m", model_name)
        restore_params, restore_state = r3m_haiku.load_from_torch_checkpoint(state_dict)

        # N, H, W, C
        image = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8).astype(
            np.float32
        )

        embedding = self._predict_torch(r3m, image)

        hk_embedding, _ = resnet_hk.apply(
            restore_params,
            restore_state,
            _normalize_image(image),
            is_training=False,
        )
        np.testing.assert_allclose(embedding, hk_embedding, atol=1e-5)

    @parameterized.parameters(R3M_RESNET_MODELS)
    def test_flax_r3m(self, model_name):
        r3m = self._load_r3m_torch_module(model_name)
        resnet_size = int(model_name.split("-")[1])  # type: ignore
        flax_module = r3m_flax.R3M(resnet_size)

        image = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8).astype(
            np.float32
        )

        state_dict = utils.load_pretrained_weights("r3m", model_name)
        restored_variables = r3m_flax.load_from_torch_checkpoint(state_dict)

        embedding = self._predict_torch(r3m, image)

        flax_embedding = flax_module.apply(
            restored_variables, _normalize_image(image), use_running_average=True
        )
        np.testing.assert_allclose(embedding, flax_embedding, atol=1e-5)  # type: ignore

    def _predict_torch(self, torch_model, images):
        with torch.no_grad():
            ## R3M expects image input to be [0-255]
            return torch_model(torch.from_numpy(np.transpose(images, [0, 3, 1, 2])))

    def _load_r3m_torch_module(self, model_name):
        resnet_size = model_name.split("-")[1]
        r3m = load_r3m(f"resnet{resnet_size}").module  # resnet18, resnet34
        r3m.eval()
        return r3m


if __name__ == "__main__":
    absltest.main()
