from transformers import ViTConfig


def vit_b16(*, image_size):
    config = ViTConfig(
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        qkv_bias=True,
        intermediate_size=768 * 4,
        layer_norm_eps=1e-6,
        image_size=image_size,
        num_channels=3,
    )
    return config


def vit_s16(*, image_size):
    config = ViTConfig(
        patch_size=16,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=384 * 4,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        num_channels=3,
        image_size=image_size,
    )
    return config


def vit_l16(*, image_size):
    config = ViTConfig(
        patch_size=16,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        image_size=image_size,
        num_channels=3,
    )
    return config


CONFIGS = {
    "vits": vit_s16,
    "vitb": vit_b16,
    "vitl": vit_l16,
}
