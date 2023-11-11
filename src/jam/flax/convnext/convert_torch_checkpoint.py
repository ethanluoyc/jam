import flax
import numpy as np

from jam.utils import checkpoint_utils

convnext_importer = checkpoint_utils.CheckpointTranslator()


def transpose_conv_weights(w):
    return np.transpose(w, [2, 3, 1, 0])


@convnext_importer.add(r"features.0.0.(weight|bias)")
def initial_conv(key, val, weight_or_bias):
    newname = {"weight": "kernel", "bias": "bias"}[weight_or_bias]
    newkey = f"stem/layers_0/{newname}"
    if newname == "kernel":
        val = transpose_conv_weights(val)
    return newkey, val


@convnext_importer.add(r"features.0.1.(weight|bias)")
def initial_norm(key, val, weight_or_bias):
    newname = {"weight": "scale", "bias": "bias"}[weight_or_bias]
    newkey = f"stem/layers_1/{newname}"
    return newkey, val


@convnext_importer.add(r"features.(1|3|5|7).(\d+).layer_scale")
def cn_block_layer_scale(key, val, stage, block_id):
    new_stage = (int(stage) - 1) // 2
    newkey = f"stages_{new_stage}/CNBlock_{block_id}/layer_scale"
    return newkey, val.reshape(-1)


@convnext_importer.add(r"features.(1|3|5|7).(\d+).block.0.(weight|bias)")
def cn_block_block_conv(key, val, stage, block, weight_or_bias):
    newname = {"weight": "kernel", "bias": "bias"}[weight_or_bias]
    new_stage = (int(stage) - 1) // 2
    newkey = f"stages_{new_stage}/CNBlock_{block}/block/layers_0/{newname}"
    if weight_or_bias == "weight":
        val = transpose_conv_weights(val)
    return newkey, val


@convnext_importer.add(r"features.(1|3|5|7).(\d+).block.(3|5).(weight|bias)")
def cn_block_block_dense(key, val, stage, block, dense_idx, weight_or_bias):
    new_idx = {3: 2, 5: 4}[int(dense_idx)]
    newname = {"weight": "kernel", "bias": "bias"}[weight_or_bias]
    new_stage = (int(stage) - 1) // 2
    newkey = f"stages_{new_stage}/CNBlock_{block}/block/layers_{new_idx}/{newname}"
    if weight_or_bias == "weight":
        val = np.transpose(val, [1, 0])
    return newkey, val


@convnext_importer.add(r"features.(1|3|5|7).(\d+).block.2.(weight|bias)")
def cn_block_block_norm(key, val, stage, block, weight_or_bias):
    newname = {"weight": "scale", "bias": "bias"}[weight_or_bias]
    new_stage = (int(stage) - 1) // 2
    newkey = f"stages_{new_stage}/CNBlock_{block}/block/layers_1/{newname}"
    return newkey, val


@convnext_importer.add(r"features.(2|4|6).0.(weight|bias)")
def block_projection_norm(key, val, layer, weight_or_bias):
    newname = {"weight": "scale", "bias": "bias"}[weight_or_bias]
    newkey = f"stages_{int(layer) // 2}/LayerNorm_0/{newname}"
    return newkey, val


@convnext_importer.add(r"features.(2|4|6).1.(weight|bias)")
def block_projection_conv(key, val, layer, weight_or_bias):
    newname = {"weight": "kernel", "bias": "bias"}[weight_or_bias]
    newkey = f"stages_{int(layer) // 2}/Conv_0/{newname}"
    if weight_or_bias == "weight":
        val = transpose_conv_weights(val)
    return newkey, val


@convnext_importer.add(r"classifier.0.(weight|bias)")
def classifier_norm(key, val, weight_or_bias):
    newname = {"weight": "scale", "bias": "bias"}[weight_or_bias]
    newkey = f"classifier/layers_0/{newname}"
    return newkey, val


@convnext_importer.add(r"classifier.2.(weight|bias)")
def classifier_dense(key, val, weight_or_bias):
    newname = {"weight": "kernel", "bias": "bias"}[weight_or_bias]
    newkey = f"classifier/layers_2/{newname}"
    if weight_or_bias == "weight":
        val = np.transpose(val, [1, 0])
    return newkey, val


def load_from_torch_checkpoint(state_dict):
    converted_dict = convnext_importer.apply(
        state_dict=checkpoint_utils.as_numpy(state_dict)
    )
    converted_dict = {k: v for k, v in converted_dict.items()}
    return {"params": flax.traverse_util.unflatten_dict(converted_dict, "/")}
