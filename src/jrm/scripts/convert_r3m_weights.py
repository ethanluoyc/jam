import os

from absl import app
from absl import flags
from r3m import load_r3m

from jrm import checkpoint
from jrm.utils import import_resnet

_OUTPUT_DIR = flags.DEFINE_string("output_dir", None, "Output directory.")


def restore_from_torch_checkpoint(state_dict):
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("convnet."):
            filtered_state_dict[k[8:]] = v

    return import_resnet.restore_from_torch_checkpoint(
        filtered_state_dict, name="r3m/~/convnet"
    )


def main(_):
    for model_name in ["resnet18", "resnet34", "resnet50"]:
        export_model_name = model_name.replace("resnet", "r3m-")
        r3m = load_r3m(model_name).module  # resnet18, resnet34
        r3m.eval()

        state_dict = r3m.state_dict()
        restore_params, restore_state = restore_from_torch_checkpoint(state_dict)

        variables = {"params": restore_params, "state": restore_state}
        output_dir = _OUTPUT_DIR.value

        checkpoint.save_to_path(os.path.join(output_dir, export_model_name), variables)


if __name__ == "__main__":
    app.run(main)
