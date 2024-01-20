# Jam - JAX models

Jam is a collection of ML models (mostly vision models for now) implemented in
Flax/Haiku. It includes model implementation, as well as pretrained weights converted
from the other sources.

Jam is currently written to allow easy access to some pretrained models that provide
PyTorch checkpoints. These pretrained models may be used for a variety of purposes,
such as transfer learning, or as feature extractor in some vision-based RL tasks.
There are preliminary examples for training some of these models from scratch but
they are not yet fully tested/benchmarked.

## Supported pretrained models
1. ConvNeXt (via torchvision), flax
2. ResNet (via torchvision), haiku and flax
3. MVP (via https://github.com/ir413/mvp/), flax
4. NFNet (via https://github.com/google-deepmind/deepmind-research/blob/master/nfnets), haiku and flax
5. R3M (via https://github.com/facebookresearch/r3m/tree/main), haiku and flax

See all available models as follows: 
```bash
python -m jam.model_zoo --list-models
```

## Download models
To download all models:
```bash
python src/jam/scripts/download_and_convert.py # downloads all models
```

To download specific models:
```
python src/jam/scripts/download_and_convert.py r3m/r3m-18
```

## Examples
See [examples](./examples/).
