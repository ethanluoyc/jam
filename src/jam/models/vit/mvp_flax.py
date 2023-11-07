from transformers import FlaxViTModel
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


CHECKPOINTS = {
    "vits-mae-hoi": {
        "ckpt_url": "https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth",
        "ckpt_md5": "fe6e30eb4256fae298ea0a6c6b4c1ae7",
    },
    "vits-mae-in": {
        "ckpt_url": "https://berkeley.box.com/shared/static/qlsjkv03nngu37eyvtjikfe7rz14k66d.pth",
        "ckpt_md5": "29a004bd4332f97cd22f55c1da26bc15",
    },
    "vits-sup-in": {
        "ckpt_url": "https://berkeley.box.com/shared/static/95a4ncqrh1o7llne2b1gpsfip4dt65m4.pth",
        "ckpt_md5": "f8f23ba960af1017783c9b975875d36d",
    },
    "vitb-mae-egosoup": {
        "ckpt_url": "https://berkeley.box.com/shared/static/0ckepd2ja3pi570z89ogd899cn387yut.pth",
        "ckpt_md5": "526093597ac1dc55df618bcbdfe8da4a",
    },
    "vitl-256-mae-egosoup": {
        "ckpt_url": "https://berkeley.box.com/shared/static/6p0pc47mlpp4hhwlin2hf035lxlgddxr.pth",
        "ckpt_md5": "5352b0b6c04f044f67eba41b667fcde6",
    },
}

_CONFIG_FUNCS = {
    "vits": vit_s16,
    "vitb": vit_b16,
    "vitl": vit_l16,
}


def load(name):
    assert name in CHECKPOINTS.keys(), "Model {} not available".format(name)
    # pretrained = cache_url(name, _MODELS[name])
    config_func = _CONFIG_FUNCS[name.split("-")[0]]
    img_size = 256 if "-256-" in name else 224
    # model, _ = model_func(img_size=img_size)
    model = FlaxViTModel(config_func(image_size=img_size), add_pooling_layer=False)
    return model
