# type: ignore
"""Download and prepare checkpoints for usage in JAX."""
import os

from absl import app
from absl import flags
import gdown
from safetensors.torch import save_file
import torch
import torchvision

from jam.models.nfnet.checkpoints import CHECKPOINTS as NFNET_CHECKPOINTS
from jam.models.r3m.checkpoints import CHECKPOINTS as R3M_CHECKPOINTS
from jam.models.vit.mvp_flax import CHECKPOINTS as MVP_CHECKPOINTS

TORCH_SAFETENSORS_FILENAME = "torch_model.safetensors"
TORCH_CKPT_FILENAME = "torch_model.pt"

RESNET_CKPTS = {
    "resnet152": {
        "model_fn": torchvision.models.resnet152,
        "weights": torchvision.models.ResNet152_Weights.IMAGENET1K_V2,
    },
    "resnet101": {
        "model_fn": torchvision.models.resnet101,
        "weights": torchvision.models.ResNet101_Weights.IMAGENET1K_V2,
    },
    "resnet50": {
        "model_fn": torchvision.models.resnet50,
        "weights": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    },
    "resnet34": {
        "model_fn": torchvision.models.resnet34,
        "weights": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
    },
    "resnet18": {
        "model_fn": torchvision.models.resnet18,
        "weights": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    },
}


def prepare_mvp_checkpoint(model_dir, model_name):
    model_dir = os.path.join(model_dir, model_name)
    model_config = MVP_CHECKPOINTS[model_name]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, TORCH_CKPT_FILENAME)
    st_path = os.path.join(model_dir, TORCH_SAFETENSORS_FILENAME)

    gdown.cached_download(
        url=model_config["ckpt_url"],
        path=model_path,
        md5=model_config["ckpt_md5"],
        quiet=False,
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = state_dict["model"]
    save_file(state_dict, st_path)


def prepare_nfnet_checkpoint(model_dir, model_name):
    model_config = NFNET_CHECKPOINTS[model_name]
    model_dir = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.npz")
    gdown.cached_download(
        url=model_config["ckpt_url"],
        path=model_path,
        md5=model_config["ckpt_md5"],
        quiet=False,
    )


def prepare_r3m_checkpoint(model_dir, model_name):
    model_config = R3M_CHECKPOINTS[model_name]
    model_dir = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, TORCH_CKPT_FILENAME)
    st_path = os.path.join(model_dir, TORCH_SAFETENSORS_FILENAME)
    config_path = os.path.join(model_dir, "config.yaml")

    gdown.cached_download(
        url=model_config["ckpt_url"],
        path=model_path,
        md5=model_config["ckpt_md5"],
        quiet=False,
    )
    gdown.cached_download(
        url=model_config["ckpt_url"],
        path=config_path,
        md5=model_config["ckpt_md5"],
        quiet=False,
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = state_dict["r3m"]
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "module."
    )
    save_file(state_dict, st_path)


def prepare_resnet_checkpoint(model_dir, model_name):
    model_config = RESNET_CKPTS[model_name]
    model_dir = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    st_path = os.path.join(model_dir, TORCH_SAFETENSORS_FILENAME)
    torch_model = model_config["model_fn"](weights=model_config["weights"])
    state_dict = torch_model.state_dict()
    save_file(state_dict, st_path)


MODEL_GROUPS = {
    "r3m": (R3M_CHECKPOINTS, prepare_r3m_checkpoint),
    "nfnet": (NFNET_CHECKPOINTS, prepare_nfnet_checkpoint),
    "resnet": (RESNET_CKPTS, prepare_resnet_checkpoint),
    "mvp": (MVP_CHECKPOINTS, prepare_mvp_checkpoint),
}

_MODEL = flags.DEFINE_multi_string(
    "model", list(MODEL_GROUPS.keys()), "Path to checkpoint to convert"
)
_MODEL_DIR = flags.DEFINE_string(
    "model_dir", "data/checkpoints", "Path to checkpoint to convert"
)


def main(_):
    for group in _MODEL.value:
        configs, handler = MODEL_GROUPS[group]
        for model_name in configs:
            handler(os.path.join(_MODEL_DIR.value, group), model_name)


if __name__ == "__main__":
    app.run(main)
