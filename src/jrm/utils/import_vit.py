import numpy as np

from jrm.utils import checkpoint_importer

timm_vit_importer = checkpoint_importer.CheckpointTranslator()


@timm_vit_importer.add(r"cls_token")
def cls_token(key, val):
    return "embeddings/cls_token", val


@timm_vit_importer.add(r"blocks.(\d+).attn.qkv.(weight|bias)")
def attention_qkv(key, val, block, slot):
    newname = {
        "weight": "kernel",
        "bias": "bias",
    }[slot]
    return f"encoder/layer/{block}/attention/attention/qkv/{newname}", val


@timm_vit_importer.add(r"blocks.(\d+).attn.proj.(weight|bias)")
def attention_proj(key, val, block, slot):
    newname = {
        "weight": "kernel",
        "bias": "bias",
    }[slot]
    return f"encoder/layer/{block}/attention/output/dense/{newname}", val


@timm_vit_importer.add(r"blocks.(\d+).mlp.fc([12]).(weight|bias)")
def mlp_block(key, val, block, fc, slot):
    newname = {
        "weight": "kernel",
        "bias": "bias",
    }[slot]
    if int(fc) == 1:
        dd = "intermediate"
    else:
        dd = "output"
    return f"encoder/layer/{block}/{dd}/dense/{newname}", val


@timm_vit_importer.add(r"blocks.(\d+).norm(\d).(weight|bias)")
def layernorm(key, val, block, ln, slot):
    newname = {
        "weight": "scale",
        "bias": "bias",
    }[slot]
    if int(ln) == 1:
        dd = "layernorm_before"
    else:
        dd = "layernorm_after"
    return f"encoder/layer/{block}/{dd}/{newname}", val


@timm_vit_importer.add(r"patch_embed.proj.(weight|bias)")
def patch_embed(key, val, slot):
    newname = {
        "weight": "kernel",
        "bias": "bias",
    }[slot]
    if newname == "kernel":
        val = np.transpose(val, [2, 3, 1, 0])
    return f"embeddings/patch_embeddings/projection/{newname}", val


@timm_vit_importer.add(r"pos_embed")
def position_embedding(key, val):
    return "embeddings/position_embeddings", val


@timm_vit_importer.add(r"norm.(weight|bias)")
def encoder_norm(key, val, slot):
    newname = {"weight": "scale", "bias": "bias"}[slot]
    return f"layernorm/{newname}", val


def restore_from_torch_checkpoint(state_dict):
    import flax

    state_dict_np = {k: v.detach().numpy() for k, v in state_dict.items()}
    converted = timm_vit_importer.apply(state_dict_np)
    fixup = {}
    for k, v in converted.items():
        if v.ndim == 2:
            v = np.transpose(v, [1, 0])
        if "qkv" in k:
            query, key, value = (
                np.split(v, 3, axis=1) if len(v.shape) == 2 else np.split(v, 3, axis=0)
            )
            for kk, vv in zip(["query", "key", "value"], [query, key, value]):
                fixup[k.replace("qkv/", kk + "/")] = vv
        else:
            fixup[k] = v

    fixup = flax.traverse_util.unflatten_dict(fixup, sep="/")
    return fixup
