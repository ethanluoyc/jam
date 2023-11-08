import re

import numpy as np
import tree


class CheckpointTranslator:
    def __init__(self):
        self.rules = []

    def add(self, pattern: str):
        def register_translation_fn_decorator(fn):
            self.rules.append((re.compile(pattern + "$"), fn))
            return fn

        return register_translation_fn_decorator

    def apply(self, state_dict):
        unmatched = {}
        new_dict = {}
        for key, value in state_dict.items():
            matched = False
            for pattern, rule_fn in self.rules:
                if pattern.match(key):
                    groups = pattern.match(key).groups()
                    new_k, new_v = rule_fn(key, value, *groups)
                    if new_k is not None:
                        new_dict[new_k] = new_v
                    matched = True
                    break
            if not matched:
                unmatched[key] = value
        if unmatched:
            raise ValueError(f"Unmatched keys: f{unmatched.keys()}")
        return new_dict


def as_numpy(nest):
    return tree.map_structure(lambda x: np.asarray(x), nest)


def load_torch_state_dict(path: str, numpy: bool = True):
    import torch

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    if numpy:
        state_dict = as_numpy(state_dict)

    return state_dict
