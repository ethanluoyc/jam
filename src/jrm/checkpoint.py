import os
import pickle
from typing import Any

from absl import logging
import jax
import numpy as np
import tree

CheckpointState = Any

_ARRAY_NAME = "array_nest"
_EXEMPLAR_NAME = "nest_exemplar"


def restore_from_path(ckpt_dir: str) -> CheckpointState:
    """Restore the state stored in ckpt_dir."""
    array_path = os.path.join(ckpt_dir, _ARRAY_NAME)
    exemplar_path = os.path.join(ckpt_dir, _EXEMPLAR_NAME)

    with open(exemplar_path, "rb") as f:
        exemplar = pickle.load(f)

    with open(array_path, "rb") as f:
        files = np.load(f, allow_pickle=True)
        flat_state = [files[key] for key in files.files]
    unflattened_tree = tree.unflatten_as(exemplar, flat_state)

    def maybe_convert_to_python(value, numpy):
        return value if numpy else value.item()

    return tree.map_structure(maybe_convert_to_python, unflattened_tree, exemplar)


def save_to_path(ckpt_dir: str, state: CheckpointState):
    """Save the state in ckpt_dir."""

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    is_numpy = lambda x: isinstance(x, (np.ndarray, jax.Array))
    flat_state = tree.flatten(state)
    nest_exemplar = tree.map_structure(is_numpy, state)

    array_path = os.path.join(ckpt_dir, _ARRAY_NAME)
    logging.info("Saving flattened array nest to %s", array_path)

    def _disabled_seek(*_):
        raise AttributeError("seek() is disabled on this object.")

    with open(array_path, "wb") as f:
        setattr(f, "seek", _disabled_seek)
        np.savez(f, *flat_state)

    exemplar_path = os.path.join(ckpt_dir, _EXEMPLAR_NAME)
    logging.info("Saving nest exemplar to %s", exemplar_path)
    with open(exemplar_path, "wb") as f:
        pickle.dump(nest_exemplar, f)
