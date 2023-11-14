from safetensors.flax import load_file


def get_model_dir(model_name):
    return f"data/models/{model_name}"


def load_torch_pretrained_weights(model_name):
    state_dict = load_file(f"data/models/{model_name}/torch_model.safetensors")
    return state_dict
