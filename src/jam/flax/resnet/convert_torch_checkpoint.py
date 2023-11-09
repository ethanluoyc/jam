import flax
import numpy as np

from jam.utils import checkpoint_utils

resnet_importer = checkpoint_utils.CheckpointTranslator()


def transpose_conv_weights(w):
    return np.transpose(w, [2, 3, 1, 0])


@resnet_importer.add(r"layer(\d)\.(\d+)\.conv(\d)\.(weight|bias)")
def block(key, val, layer, block, conv, weight_or_bias):
    newname = {"weight": "kernel", "bias": "bias"}[weight_or_bias]
    newkey = (
        f"block_group_{int(layer) - 1}/block_{block}/conv_{int(conv) - 1}/{newname}"
    )
    if newname == "kernel":
        val = transpose_conv_weights(val)
    return newkey, val


@resnet_importer.add(
    r"layer(\d)\.(\d+)\.bn(\d)\.(weight|bias|running_mean|running_var|num_batches_tracked)"
)
def bn(key, val, layer, block, bn, slot):
    newname = {
        "weight": "scale",
        "bias": "bias",
        "num_batches_tracked": "counter",
        "running_mean": "mean",
        "running_var": "var",
    }[slot]
    newkey = (
        f"block_group_{int(layer) - 1}/block_{block}/batchnorm_{int(bn) - 1}/{newname}"
    )
    # if slot != "num_batches_tracked":
    #     val = np.reshape(val, [1, 1, 1, -1])
    return newkey, val


@resnet_importer.add(
    r"layer(\d)\.(\d+)\.downsample\.(\d)\.(weight|bias|running_mean|running_var|num_batches_tracked)"
)
def downsample(key, val, layer, block, conv, slot):
    if int(conv) == 0:
        newname = {
            "weight": "kernel",
            "bias": "bias",
        }[slot]
        if newname == "kernel":
            val = transpose_conv_weights(val)
        newkey = (
            f"block_group_{int(layer) - 1}/block_{int(block)}/shortcut_conv/{newname}"
        )
    elif int(conv) == 1:
        newname = {
            "weight": "scale",
            "bias": "bias",
            "num_batches_tracked": "counter",
            "running_mean": "mean",
            "running_var": "var",
        }[slot]
        newkey = f"block_group_{int(layer) - 1}/block_{int(block)}/shortcut_batchnorm/{newname}"
    else:
        raise ValueError(f"Invalid conv number {conv}")
    return newkey, val


@resnet_importer.add(r"conv1\.weight")
def initial_conv(key, val):
    return "initial_conv/kernel", transpose_conv_weights(val)


@resnet_importer.add(r"bn1\.(weight|bias|running_mean|running_var|num_batches_tracked)")
def initial_bn(key, val, slot):
    newname = {
        "weight": "scale",
        "bias": "bias",
        "num_batches_tracked": "counter",
        "running_mean": "mean",
        "running_var": "var",
    }[slot]
    newkey = "initial_batchnorm/" + newname
    return newkey, val


@resnet_importer.add(r"fc\.(weight|bias)")
def final_logits(key, val, slot):
    newkey = {"weight": "logits/kernel", "bias": "logits/bias"}[slot]
    if slot == "weight":
        val = np.transpose(val, [1, 0])
    return newkey, val


def load_from_torch_checkpoint(state_dict):
    converted_dict = resnet_importer.apply(
        state_dict=checkpoint_utils.as_numpy(state_dict)
    )
    converted_dict = {k: v for k, v in converted_dict.items()}
    converted_variables = {}
    for k, v in converted_dict.items():
        if "counter" in k:
            pass
        elif "batchnorm" in k:
            if "scale" in k or "bias" in k:
                converted_variables[f"params/{k}"] = v
            else:
                converted_variables[f"batch_stats/{k}"] = v
        else:
            converted_variables[f"params/{k}"] = v

    return flax.traverse_util.unflatten_dict(converted_variables, sep="/")
