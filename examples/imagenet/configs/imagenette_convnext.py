import ml_collections


def get_config():
    """Get the hyperparameter configuration for Fake data benchmark."""
    # Override default configuration to avoid duplication of field definition.
    config = ml_collections.ConfigDict()

    config.dataset = "imagenette"
    config.model = "convnext_tiny"

    config.learning_rate = 4e-3
    config.warmup_epochs = 5.0
    config.batch_size = 128
    config.num_grad_accumulation_steps = 1
    config.shuffle_buffer_size = 16 * 128
    config.prefetch = 10
    config.half_precision = False

    config.num_epochs = 100.0
    config.log_every_steps = 100

    config.cache = False
    config.weight_decay = 0.05
    config.use_autoaugment = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config
