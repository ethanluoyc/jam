from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import numpy as np
from r3m import load_r3m
import torch

from jrm.models.r3m import IMAGENET_MEAN_RGB
from jrm.models.r3m import IMAGENET_STDDEV_RGB
from jrm.models.r3m import R3M
from jrm.utils import import_resnet


def restore_from_torch_checkpoint(state_dict):
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("convnet."):
            filtered_state_dict[k[8:]] = v

    return import_resnet.restore_from_torch_checkpoint(
        filtered_state_dict, name="r3m/~/convnet"
    )


class R3MImportTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "resnet18",
            "model_name": "resnet18",
        },
        {
            "testcase_name": "resnet34",
            "model_name": "resnet34",
        },
        {
            "testcase_name": "resnet50",
            "model_name": "resnet50",
        },
    )
    def test_jax_r3m(self, model_name):
        r3m = load_r3m(model_name).module  # resnet18, resnet34
        r3m.eval()

        def _normalize_image(inputs):
            return (inputs - np.array(IMAGENET_MEAN_RGB)) / np.array(
                IMAGENET_STDDEV_RGB
            )

        def forward(inputs, is_training=True):
            model = R3M(
                int(model_name[-2:]),
            )

            out = _normalize_image(inputs)
            return model(out, is_training)

        resnet_hk = hk.without_apply_rng(hk.transform_with_state(forward))

        state_dict = r3m.state_dict()
        restore_params, restore_state = restore_from_torch_checkpoint(state_dict)

        # N, H, W, C
        image = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8).astype(
            np.float32
        )

        with torch.no_grad():
            ## R3M expects image input to be [0-255]
            embedding = r3m(torch.from_numpy(np.transpose(image, [0, 3, 1, 2])))

        hk_embedding, _ = resnet_hk.apply(
            restore_params,
            restore_state,
            image,
            is_training=False,
        )
        np.testing.assert_allclose(embedding, hk_embedding, atol=1e-5)

    # def test_restore_jax_checkpoint(self):
    #     r3m = load_r3m("resnet18").module  # resnet18, resnet34
    #     r3m.eval()

    #     def _normalize_image(inputs):
    #         return (inputs - np.array(IMAGENET_MEAN_RGB)) / np.array(
    #             IMAGENET_STDDEV_RGB
    #         )

    #     def forward(inputs, is_training=True):
    #         model = R3M(resnet_size=18)

    #         out = _normalize_image(inputs)
    #         return model(out, is_training)

    #     resnet_hk = hk.without_apply_rng(hk.transform_with_state(forward))

    #     restore_variables = checkpoint.restore_from_path("./r3m_18")
    #     restore_params = restore_variables["params"]
    #     restore_state = restore_variables["state"]

    #     # N, H, W, C
    #     image = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8).astype(
    #         np.float32
    #     )

    #     with torch.no_grad():
    #         ## R3M expects image input to be [0-255]
    #         embedding = r3m(torch.from_numpy(np.transpose(image, [0, 3, 1, 2])))

    #     hk_embedding, _ = resnet_hk.apply(
    #         restore_params,
    #         restore_state,
    #         image,
    #         is_training=False,
    #     )
    #     np.testing.assert_allclose(embedding, hk_embedding, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
