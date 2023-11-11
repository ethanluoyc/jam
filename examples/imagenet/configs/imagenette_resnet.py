import ml_collections


def get_config():
    """Get the hyperparameter configuration for Fake data benchmark."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = "ResNet18"
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "imagenette"

    config.learning_rate = 0.1
    config.warmup_epochs = 0.5
    config.momentum = 0.9
    config.batch_size = 128
    config.shuffle_buffer_size = 8 * 128
    config.prefetch = 1

    config.num_epochs = 100.0
    config.log_every_steps = 100

    config.cache = False
    config.half_precision = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config
