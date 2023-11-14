from absl import app
import jax
import numpy as np
from safetensors.flax import load_file

from jam.flax.vit import import_vit
from jam.flax.vit import mvp_flax


def main(_):
    model_name = "vits-mae-hoi"
    model = mvp_flax.load(model_name)

    state_dict = load_file(f"data/models/mvp/{model_name}/torch_model.safetensors")
    restored_params = import_vit.restore_from_torch_checkpoint(state_dict)
    restored_params = jax.device_put(restored_params)

    batch_size = 3
    image_size = 224 if "vitl" not in model_name else 256
    dummy_images = np.random.uniform(
        0, 1, size=(batch_size, image_size, image_size, 3)
    ).astype(np.float32)

    output = model.module.apply(
        {"params": restored_params},
        dummy_images,
        deterministic=True,
        output_hidden_states=True,
    )

    print(output.last_hidden_state.shape)  # type: ignore


if __name__ == "__main__":
    app.run(main)
