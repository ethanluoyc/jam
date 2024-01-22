import fnmatch

from absl import app
from absl import flags

from jam import model_zoo

_MODEL = flags.DEFINE_string("model", "*", "Name of the model to convert")
_MODEL_DIR = flags.DEFINE_string(
    "model_dir", "data/models", "Path to checkpoint to convert"
)
_SKIP_DOWNLOAD = flags.DEFINE_boolean(
    "skip_download", False, "Skip downloading the checkpoint"
)

def main(_):
    selected_models = []
    for pat in _MODEL.value.split(","):
        selected_models += fnmatch.filter(model_zoo.list_models(), pat)

    selected_models = sorted(list(set(selected_models)))

    for model_name in selected_models:
        try: 
            model_zoo.download_and_convert(model_name, _MODEL_DIR.value, _SKIP_DOWNLOAD.value)
        except Exception as e:
            print(f"Failed to convert {model_name}: {e}")


if __name__ == "__main__":
    app.run(main)
