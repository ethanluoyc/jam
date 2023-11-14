# ruff: noqa: E501
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test pretrained NFNet checkpoints on ImageNet."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import dill
import haiku as hk
import numpy as np
from PIL import Image
import utils  # type: ignore

from jam import imagenet_util
from jam.flax import nfnet as nfnet_flax
from jam.haiku import nfnet as nfnet_haiku

HERE = os.path.dirname(__file__)

NFNET_TEST_VARIANTS = ["F0"]


def _load_pretrained_checkpoint(variant):
    model_dir = utils.get_model_dir(f"nfnet/{variant}")
    with open(os.path.join(model_dir, "model.npz"), "rb") as in_file:
        params = dill.load(in_file)
    return params


def preprocess_image(im, imsize):
    im = im.resize((imsize + 32, imsize + 32))
    im = im.crop((16, 16, 16 + imsize, 16 + imsize))
    # Convert im to tensor and normalize with channel-wise RGB
    return (np.float32(im) - imagenet_util.IMAGENET_MEAN_RGB) / imagenet_util.IMAGENET_STDDEV_RGB  # type: ignore


class NFNetTest(parameterized.TestCase):
    def _load_test_image(self):
        im = Image.open(os.path.join(HERE, "../testdata", "peppers.jpg"))
        return im, "bell pepper"

    @parameterized.parameters(NFNET_TEST_VARIANTS)
    def test_nfnet(self, variant):
        params = _load_pretrained_checkpoint(variant)
        im, label = self._load_test_image()
        # Resize and crop to variant test size
        imsize = nfnet_haiku.nfnet_params[variant]["test_imsize"]
        inputs = preprocess_image(im, imsize)

        # Prepare the forward fn
        def forward(inputs, is_training):
            model = nfnet_haiku.NFNet(num_classes=1000, variant=variant)
            return model(inputs, is_training=is_training)["logits"]

        net = hk.without_apply_rng(hk.transform(forward))
        fwd = lambda params, inputs: net.apply(params, inputs, is_training=False)

        # We split this into two cells so that we don't repeatedly jit the fwd fn.
        logits = fwd(params, inputs[None])  # Give X a newaxis to make it batch-size-1
        which_class = imagenet_util.IMAGENET_CLASSLIST[int(logits.argmax())]
        self.assertEqual(which_class, label)

    @parameterized.parameters(NFNET_TEST_VARIANTS)
    def test_flax_nfnet(self, variant):
        params = _load_pretrained_checkpoint(variant)
        im, label = self._load_test_image()
        imsize = nfnet_haiku.nfnet_params[variant]["test_imsize"]
        inputs = preprocess_image(im, imsize)

        # Prepare the forward fn
        def forward(inputs, is_training):
            model = nfnet_haiku.NFNet(num_classes=1000, variant=variant)
            return model(inputs, is_training=is_training)["logits"]

        net = hk.without_apply_rng(hk.transform(forward))
        fwd = lambda params, inputs: net.apply(params, inputs, is_training=False)
        hk_logits = fwd(params, inputs[None])

        # Resize and crop to variant test size
        module = nfnet_flax.NFNet(num_classes=1000, variant=variant)
        flax_params = nfnet_flax.load_from_haiku_checkpoint(params)
        output = module.apply({"params": flax_params}, inputs[None], is_training=False)
        logits = output["logits"]  # type: ignore
        which_class = imagenet_util.IMAGENET_CLASSLIST[int(logits.argmax())]
        self.assertEqual(which_class, label)
        np.testing.assert_allclose(logits, hk_logits, atol=1e-3)


if __name__ == "__main__":
    absltest.main()
