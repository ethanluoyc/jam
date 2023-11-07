from absl.testing import absltest
from absl.testing import parameterized
import flax
import haiku as hk
import numpy as np
from r3m import load_r3m
from safetensors.flax import load_file
import torch

from jam import imagenet_util
from jam.models.r3m import r3m_flax
from jam.models.r3m import r3m_haiku
from jam.models.resnet import import_resnet_flax
from jam.models.resnet import import_resnet_haiku


def _load_pretrained_checkpoint(model_name):
    state_dict = load_file(f"data/checkpoints/r3m/{model_name}/torch_model.safetensors")
    return state_dict


def restore_from_torch_checkpoint_haiku(state_dict):
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("convnet."):
            filtered_state_dict[k[8:]] = v

    return import_resnet_haiku.restore_from_torch_checkpoint(
        filtered_state_dict, name="r3m/~/convnet"
    )


def restore_from_torch_checkpoint_flax(state_dict):
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("convnet."):
            filtered_state_dict[k[8:]] = v

    restored = import_resnet_flax.restore_from_torch_checkpoint(filtered_state_dict)
    variables = flax.traverse_util.unflatten_dict(restored, sep="/")

    variables["params"] = {"convnet": variables["params"]}
    variables["batch_stats"] = {"convnet": variables["batch_stats"]}
    return variables


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

        state_dict = _load_pretrained_checkpoint(model_name)
        restore_params, restore_state = restore_from_torch_checkpoint_haiku(state_dict)

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

        state_dict = _load_pretrained_checkpoint(model_name)
        restored_variables = restore_from_torch_checkpoint_flax(state_dict)

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
