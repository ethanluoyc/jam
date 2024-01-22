import functools
import os

MODEL_URLS = {
    # # NFNets
    # "nfnet/F0": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F0_haiku.npz",
    #     "ckpt_md5": "0a41039da1cb43adf47f1b7d86ddfe0f",
    # },
    # "nfnet/F1": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F1_haiku.npz",
    #     "ckpt_md5": "c9126bbfaf855f9ddb7ff3b902558781",
    # },
    # "nfnet/F2": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F2_haiku.npz",
    #     "ckpt_md5": "7a018d8d1498020e2f5a8719a1e7f53f",
    # },
    # "nfnet/F3": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F3_haiku.npz",
    #     "ckpt_md5": "d8e1e78d26db659c486150dbaccbe78a",
    # },
    # "nfnet/F4": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F4_haiku.npz",
    #     "ckpt_md5": "f5974a7e3388aa81b91a9f1403e759a4",
    # },
    # "nfnet/F5": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F5_haiku.npz",
    #     "ckpt_md5": "8639e925de951030651ac775b9f915c3",
    # },
    # "nfnet/F6": {
    #     "ckpt_url": "https://storage.googleapis.com/dm-nfnets/F6_haiku.npz",
    #     "ckpt_md5": "b16cca40ac1ecec2bd5e3273866da0c1",
    # },
    # R3M
    "r3m/r3m-50": {
        "ckpt_url": "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA",
        "ckpt_md5": "aa8e41a52670aaacbafe8e4532052d15",
        # "config_url": "https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8",
        # "config_md5": "f1929b19202d85c0829e41f302007bcc",
    },
    "r3m/r3m-18": {
        "ckpt_url": "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-",
        "ckpt_md5": "777854e2548f91480b0bd538a06ba017",
        # "config_url": "https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6",
        # "config_md5": "e226d5115543dee07fd51a17c7866bc9",
    },
    "r3m/r3m-34": {
        "ckpt_url": "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE",
        "ckpt_md5": "2f3bf7cdb165a86245d2c309a16d254f",
        # "config_url": "https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW",
        # "config_md5": "ebbb6f4259fe866d4d31571aedfbb018",
    },
    # # MVP
    # "mvp/vits-mae-hoi": {
    #     "ckpt_url": "https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth",
    #     "ckpt_md5": "fe6e30eb4256fae298ea0a6c6b4c1ae7",
    # },
    # # "vits-mae-in": {
    # #     "ckpt_url": "https://berkeley.box.com/shared/static/qlsjkv03nngu37eyvtjikfe7rz14k66d.pth",
    # #     "ckpt_md5": "29a004bd4332f97cd22f55c1da26bc15",
    # # },
    # # "vits-sup-in": {
    # #     "ckpt_url": "https://berkeley.box.com/shared/static/95a4ncqrh1o7llne2b1gpsfip4dt65m4.pth",
    # #     "ckpt_md5": "f8f23ba960af1017783c9b975875d36d",
    # # },
    # "mvp/vitb-mae-egosoup": {
    #     "ckpt_url": "https://berkeley.box.com/shared/static/0ckepd2ja3pi570z89ogd899cn387yut.pth",
    #     "ckpt_md5": "526093597ac1dc55df618bcbdfe8da4a",
    # },
    # "mvp/vitl-256-mae-egosoup": {
    #     "ckpt_url": "https://berkeley.box.com/shared/static/6p0pc47mlpp4hhwlin2hf035lxlgddxr.pth",
    #     "ckpt_md5": "5352b0b6c04f044f67eba41b667fcde6",
    # },
    # torchvision
    # resnet
    "torchvision/resnet18-imagenet1k-v1": {
        "ckpt_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "ckpt_md5": "e0b1c919e74f9a193d36871d9964bf7d",
    },
    "torchvision/resnet34-imagenet1k-v1": {
        "ckpt_url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "ckpt_md5": "78fe1097b28dbda1373a700020afeed9",
    },
    "torchvision/resnet50-imagenet1k-v2": {
        "ckpt_url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "ckpt_md5": "012571d812f34f8442473d8b827077b5",
    },
    "torchvision/resnet101-imagenet1k-v2": {
        "ckpt_url": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        "ckpt_md5": "52bf5a2e5f5fb162fa8c4387b506db43",
    },
    "torchvision/resnet152-imagenet1k-v2": {
        "ckpt_url": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
        "ckpt_md5": "0a49c0d5b76e59649d8de405d5034e84",
    },
    # convnext
    "torchvision/convnext-base-imagenet1k-v1": {
        "ckpt_url": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        "ckpt_md5": "47c2dcfa71ee41c5a240fa6e7c6bea01",
    },
    "torchvision/convnext-small-imagenet1k-v1": {
        "ckpt_url": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
        "ckpt_md5": "389fc81e4c8aae264facf624f7671d0b",
    },
    "torchvision/convnext-large-imagenet1k-v1": {
        "ckpt_url": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        "ckpt_md5": "6d78bac6b1c99928d6d346f17a9b6904",
    },
    "torchvision/convnext-tiny-imagenet1k-v1": {
        "ckpt_url": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "ckpt_md5": "fcc1480a11e933010f4d7e0ccb7e33b1",
    },
}

TORCH_SAFETENSORS_FILENAME = "torch_model.safetensors"
TORCH_CKPT_FILENAME = "torch_model.pt"
HAIKU_CKPT_FILENAME = "model.npz"


@functools.lru_cache()
def list_models():
    return sorted(list(MODEL_URLS.keys()))


def download_and_convert(model_name, model_dir, skip_download: bool = False):
    import gdown

    url = MODEL_URLS[model_name]["ckpt_url"]
    md5 = MODEL_URLS[model_name].get("ckpt_md5", None)
    model_dir = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    if model_name.startswith("nfnet"):
        if skip_download:
            gdown.cached_download(
                url=url, path=os.path.join(model_dir, HAIKU_CKPT_FILENAME), md5=md5
            )
    else:
        from safetensors.torch import save_file
        import torch

        if skip_download:
            gdown.cached_download(
                url=url, path=os.path.join(model_dir, TORCH_CKPT_FILENAME), md5=md5
            )
        ckpt_path = os.path.join(model_dir, TORCH_CKPT_FILENAME)
        st_path = os.path.join(model_dir, TORCH_SAFETENSORS_FILENAME)

        state_dict = torch.load(ckpt_path, map_location="cpu")
        if model_name.startswith("r3m"):
            state_dict = state_dict["r3m"]
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                state_dict, "module."
            )
        if model_name.startswith("mvp"):
            state_dict = state_dict["model"]
        print("Converting to SafeTensors")
        save_file(state_dict, st_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", action='store_true', help = "List available models")
    args = parser.parse_args()

    if args.list_models:
        print("Available models: ")
        for model in list_models():
            print("  - ", model)