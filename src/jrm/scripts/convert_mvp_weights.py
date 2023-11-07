from absl import app
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


def main(_):
    model_name_to_config = {
        "vitb-mae-egosoup": vit_b16,
        "vits-mae-hoi": vit_s16,
        "vits-mae-in": vit_s16,
        "vits-sup-in": vit_s16,
    }
    for model_name in model_name_to_config.keys():
        config_func = model_name_to_config[model_name]
        mvp_model = mvp.load(model_name)
        mvp_model.freeze()
        mvp_model.eval()

        model = FlaxViTModel(config_func(), add_pooling_layer=False)
        restored_params = import_vit.restore_from_torch_checkpoint(
            mvp_model.state_dict()
        )
        restored_params = jax.device_put(restored_params)

        batch_size = 32
        dummy_images = np.random.uniform(0, 1, size=(batch_size, 224, 224, 3))
        dummy_images = dummy_images.astype(np.float32)

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
            converted_output.last_hidden_state[:, 0],
            atol=2e-5,
        )

        model.params = restored_params
        model.save_pretrained(f"data/jax_checkpoints/{model_name}")


if __name__ == "__main__":
    app.run(main)
