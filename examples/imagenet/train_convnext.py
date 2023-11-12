# type: ignore
# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""
# TODO: check batch norm synchronization

import functools
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import input_pipeline
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from jam.flax import convnext

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args, is_training=False)

    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
    return variables["params"]


def cross_entropy_loss(logits, labels):
    num_classes = logits.shape[-1]
    onehot_labels = common_utils.onehot(
        labels, num_classes=num_classes
    )  # TODO: fix this
    smoothed_labels = optax.smooth_labels(onehot_labels, 0.1)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=smoothed_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def train_step(state, batch, learning_rate_fn):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        logits = state.apply_fn(
            {"params": params},
            batch["image"],
            is_training=True,
            rngs={"dropout": jax.random.fold_in(jax.random.PRNGKey(0), state.step)},
        )
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, (logits,)

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name="batch")
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name="batch")

    (logits,) = aux[1]
    metrics = compute_metrics(logits, batch["label"])
    metrics["learning_rate"] = lr

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state,
            ),
            params=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin), new_state.params, state.params
            ),
            dynamic_scale=dynamic_scale,
        )
        metrics["scale"] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, batch):
    variables = {"params": state.params}
    logits = state.apply_fn(
        variables,
        batch["image"],
        is_training=False,
        mutable=False,
    )
    return compute_metrics(logits, batch["label"])


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(
    dataset_builder,
    batch_size,
    image_size,
    dtype,
    train,
    cache,
    shuffle_buffer_size,
    prefetch,
    use_autoaugment,
):
    ds = input_pipeline.create_split(
        dataset_builder,
        batch_size,
        image_size=image_size,
        dtype=dtype,
        train=train,
        cache=cache,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch=prefetch,
        use_autoaugment=use_autoaugment,
    )
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


class TrainState(train_state.TrainState):
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    logging.info("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
    """Create initial training state."""
    params = initialized(rng, image_size, model)

    def decay_mask_fn(params):
        flat_params = flax.traverse_util.flatten_dict(params)
        # True for params that we want to apply weight decay to
        flat_mask = {
            path: (path[-1] != "bias" and "norm" not in path[-2])
            for path in flat_params
        }
        return flax.traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        weight_decay=config.weight_decay,
        mask=decay_mask_fn,
    )
    if config.half_precision:
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, dynamic_scale=dynamic_scale
    )
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      Final TrainState.
    """

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    rng = random.key(0)

    image_size = 224

    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    local_batch_size = config.batch_size // jax.process_count()
    if config.half_precision:
        input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    dataset_builder = tfds.builder(config.dataset)
    train_iter = create_input_iter(
        dataset_builder,
        local_batch_size,
        image_size,
        input_dtype,
        train=True,
        cache=config.cache,
        shuffle_buffer_size=config.shuffle_buffer_size,
        prefetch=config.prefetch,
        use_autoaugment=config.use_autoaugment,
    )
    eval_iter = create_input_iter(
        dataset_builder,
        local_batch_size,
        image_size,
        input_dtype,
        train=False,
        cache=config.cache,
        shuffle_buffer_size=None,
        prefetch=config.prefetch,
        use_autoaugment=config.use_autoaugment,
    )

    steps_per_epoch = (
        dataset_builder.info.splits["train"].num_examples // config.batch_size
    )

    if config.num_train_steps <= 0:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits["validation"].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    base_learning_rate = config.learning_rate

    model = getattr(convnext, config.model)(
        num_classes=dataset_builder.info.features["label"].num_classes,
        dtype=jnp.float16 if config.half_precision else jnp.float32,
    )

    learning_rate_fn = create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch
    )

    state = create_train_state(rng, config, model, image_size, learning_rate_fn)
    # state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn),
        axis_name="batch",
    )
    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics = p_train_step(state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info("Initial compilation completed.")

        if config.get("log_every_steps"):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train_{k}": v
                    for k, v in jax.tree_util.tree_map(
                        lambda x: x.mean(), train_metrics
                    ).items()
                }
                summary["steps_per_second"] = config.log_every_steps / (
                    time.time() - train_metrics_last_t
                )
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []
            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info(
                "eval epoch: %d, loss: %.4f, accuracy: %.2f",
                epoch,
                summary["loss"],
                summary["accuracy"] * 100,
            )
            writer.write_scalars(
                step + 1, {f"eval_{key}": val for key, val in summary.items()}
            )
            writer.flush()
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            # save_checkpoint(state, workdir)
            pass

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    return state


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
