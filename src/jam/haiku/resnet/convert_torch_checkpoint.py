# type: ignore
import numpy as np

from jam.utils import checkpoint_utils

resnet_importer = checkpoint_utils.CheckpointTranslator()


def transpose_conv_weights(w):
    return np.transpose(w, [2, 3, 1, 0])


@resnet_importer.add(r"layer(\d)\.(\d+)\.conv(\d)\.(weight|bias)")
def block(key, val, layer, block, conv, weight_or_bias):
    newname = {"weight": "w", "bias": "b"}[weight_or_bias]
    newkey = (
        f"block_group_{int(layer) - 1}/~/block_{block}/~/conv_{int(conv) - 1}/{newname}"
    )
    if newname == "w":
        val = transpose_conv_weights(val)
    return newkey, val


@resnet_importer.add(
    r"layer(\d)\.(\d+)\.bn(\d)\.(weight|bias|running_mean|running_var|num_batches_tracked)"
)
def bn(key, val, layer, block, bn, slot):
    newname = {
        "weight": "scale",
        "bias": "offset",
        "num_batches_tracked": "~/mean_ema/counter",
        "running_mean": "~/mean_ema/average",
        "running_var": "~/var_ema/average",
    }[slot]
    newkey = f"block_group_{int(layer) - 1}/~/block_{block}/~/batchnorm_{int(bn) - 1}/{newname}"
    if slot != "num_batches_tracked":
        val = np.reshape(val, [1, 1, 1, -1])
    return newkey, val


@resnet_importer.add(
    r"layer(\d)\.(\d+)\.downsample\.(\d)\.(weight|bias|running_mean|running_var|num_batches_tracked)"
)
def downsample(key, val, layer, block, conv, slot):
    if int(conv) == 0:
        newname = {
            "weight": "w",
            "bias": "b",
        }[slot]
        if newname == "w":
            val = transpose_conv_weights(val)
        newkey = f"block_group_{int(layer) - 1}/~/block_{int(block)}/~/shortcut_conv/{newname}"
    elif int(conv) == 1:
        newname = {
            "weight": "scale",
            "bias": "offset",
            "num_batches_tracked": "~/mean_ema/counter",
            "running_mean": "~/mean_ema/average",
            "running_var": "~/var_ema/average",
        }[slot]
        if slot != "num_batches_tracked":
            val = np.reshape(val, [1, 1, 1, -1])
        newkey = f"block_group_{int(layer) - 1}/~/block_{int(block)}/~/shortcut_batchnorm/{newname}"
    return newkey, val


@resnet_importer.add(r"conv1\.weight")
def initial_conv(key, val):
    return "initial_conv/w", transpose_conv_weights(val)


@resnet_importer.add(r"bn1\.(weight|bias|running_mean|running_var|num_batches_tracked)")
def initial_bn(key, val, slot):
    newname = {
        "weight": "scale",
        "bias": "offset",
        "num_batches_tracked": "~/mean_ema/counter",
        "running_mean": "~/mean_ema/average",
        "running_var": "~/var_ema/average",
    }[slot]
    newkey = "initial_batchnorm/" + newname
    if slot != "num_batches_tracked":
        val = np.reshape(val, [1, 1, 1, -1])
    return newkey, val


@resnet_importer.add(r"fc\.(weight|bias)")
def final_logits(key, val, slot):
    newkey = {"weight": "logits/w", "bias": "logits/b"}[slot]
    if slot == "weight":
        val = np.transpose(val, [1, 0])
    return newkey, val


def load_from_torch_checkpoint(state_dict, name):
    converted_dict = resnet_importer.apply(checkpoint_utils.as_numpy(state_dict))
    converted_dict = {f"{name}/~/{k}": v for k, v in converted_dict.items()}
    for k in list(converted_dict.keys()):
        if k.endswith("mean_ema/counter"):
            converted_dict[k.replace("mean_ema", "var_ema")] = converted_dict[k]
        if k.endswith("average"):
            converted_dict[k.replace("average", "hidden")] = converted_dict[k]

    params, state = {}, {}

    for k in converted_dict.keys():
        parts = k.split("/")
        module = "/".join(parts[:-1])
        name = parts[-1]
        if module.endswith("_ema"):
            restore_container = state
        else:
            restore_container = params

        if module not in restore_container:
            restore_container[module] = {}

        restore_container[module][name] = converted_dict[k]

    return params, state
