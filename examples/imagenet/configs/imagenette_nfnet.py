import ml_collections


def get_config():
    """Get the hyperparameter configuration for Fake data benchmark."""
    # Override default configuration to avoid duplication of field definition.
    config = ml_collections.ConfigDict()

    config.dataset = "imagenette"
    config.model = "convnext_tiny"

    config.warmup_epochs = 5.0
    config.batch_size = 64
    config.num_grad_accumulation_steps = 1
    # 64 * 64 == 4096 which is the original batch size in the NFNet paper.
    config.learning_rate = (
        config.num_grad_accumulation_steps * config.batch_size * 0.1 / 256.0
    )
    config.momentum = 0.9
    # config.learning_rate = 4e-3
    config.shuffle_buffer_size = 16 * 64
    config.prefetch = 10

    config.num_epochs = 100.0
    config.log_every_steps = 100

    config.cache = False
    config.weight_decay = 2e-5
    config.use_autoaugment = True

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config
