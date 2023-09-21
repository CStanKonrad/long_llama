import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule
)
import FoT.data_pipeline as data_pipeline
import EasyLM.logging_utils as logging_utils
from EasyLM.training_utils import get_gradient_step

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    train_log_freq=50,
    eval_freq=100,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    log_all_worker=False,
    logger_dir="./",
    jax_distributed=JaxDistributedConfig.get_default_config(),
    train_cross_batch_range=1,
    train_cross_batch_stepping=True,
    eval_cross_batch_range=0,
    eval_cross_batch_stepping=True,
    flip_sharding_in_cross_batch=False,
    scan_cross_batch=False,
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = logging_utils.create_logger(
        log_dir=FLAGS.logger_dir, enable=FLAGS.log_all_worker or (jax.process_index() == 0)
    )
    set_random_seed(FLAGS.seed)

    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        raise ValueError("Insufficient vocab size")

    llama_config.update(dict(
        flip_sharding_in_cross_batch=FLAGS.flip_sharding_in_cross_batch,
        scan_cross_batch=FLAGS.scan_cross_batch,
    ))
    

    base_llama_config_dict = llama_config.to_dict()

    train_llama_config =  LLaMAConfig.from_dict(base_llama_config_dict)
    train_llama_config.update(dict( cross_batch_range=FLAGS.train_cross_batch_range, cross_batch_stepping=FLAGS.train_cross_batch_stepping, dataset_packing=data_pipeline.get_dataset_packing(data_pipeline=dataset.config.data_pipeline)))


    if FLAGS.eval_steps > 0:
        eval_llama_config =  LLaMAConfig.from_dict(base_llama_config_dict)
        eval_llama_config.update(dict(cross_batch_range=FLAGS.eval_cross_batch_range, cross_batch_stepping=FLAGS.eval_cross_batch_stepping, dataset_packing=data_pipeline.get_dataset_packing(data_pipeline=eval_dataset.config.data_pipeline)))
        
    model = FlaxLLaMAForCausalLMModule(
        train_llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    if FLAGS.eval_steps > 0:
        eval_model = FlaxLLaMAForCausalLMModule(
            eval_llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
        )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng, model, llama_config):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((llama_config.dataset_packing, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((llama_config.dataset_packing, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((llama_config.dataset_packing, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
    
    def init_train_fn(rng):
        return init_fn(rng, model, train_llama_config)
    
    def init_eval_fn(rng):
        return init_fn(rng, eval_model, eval_llama_config)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        def loss_and_accuracy(params):
            logits = model.apply(
                params, batch['input_tokens'], deterministic=False,
                rngs=rng_generator(train_llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](get_gradient_step(train_state)),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = eval_model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(eval_llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_train_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )


    if FLAGS.eval_steps > 0:
        eval_state_shapes = jax.eval_shape(init_eval_fn, next_rng())
        eval_state_partition =  match_partition_rules(
            LLaMAConfig.get_partition_rules(), eval_state_shapes
        )

        assert eval_state_shapes == train_state_shapes
        assert eval_state_partition == train_state_partition

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, FLAGS.logger_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_train_fn = pjit(
        init_train_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(get_gradient_step(train_state)))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_train_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(get_gradient_step(train_state)))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        # For gradient accumulation
        dataset_iterator = iter(dataset)
        train_substeps = FLAGS.optimizer.accumulate_gradient_steps
        def get_microbatch(full_batch, train_substep):
            bs = full_batch["input_tokens"].shape[0]
            assert train_substeps <= bs and bs % train_substeps == 0
            microbatch_size = bs // train_substeps

            return jax.tree_map(
                lambda x: x[
                    train_substep
                    * microbatch_size : (train_substep + 1)
                    * microbatch_size
                ],
                full_batch,
            )


        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        train_log_aggregator = logging_utils.LogAggregator()

        eval_log_aggregator = logging_utils.LogAggregator(
            provide_latest=False,
        )

        def defragment():
            try:
                jax.lib.xla_bridge.get_backend().defragment()
            except:
                pass

        defragment()

        first_step = True

        for step in step_counter:


            (full_batch, dataset_metrics) = next(dataset_iterator)
            for substep in range(train_substeps):
                batch = get_microbatch(full_batch, substep)

                should_run_eval = FLAGS.eval_freq > 0 and FLAGS.eval_steps > 0 and substep == 0 and (step % FLAGS.eval_freq == 0 or first_step) 

                should_log_train = step % FLAGS.train_log_freq == 0 and substep == 0 # TODO change to train_substeps

                if should_run_eval:
                    defragment()
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, eval_dataset_metrics = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_log_aggregator.add(eval_metrics)

                    eval_logs = {}
                    eval_logs.update(eval_log_aggregator.get_logs())
                    eval_logs.update(logging_utils.metrics_assign_group(eval_dataset_metrics, "dataset"))
                    eval_logs = logging_utils.metrics_assign_group(eval_logs, "eval")
                    logger.write_scalars(step, eval_logs)
                    logger.flush()
                    tqdm.write("\n" + pprint.pformat(eval_logs) + "\n")
                    defragment()

                train_state, sharded_rng, train_metrics = sharded_train_step(
                    train_state, sharded_rng, batch
                )

                train_log_aggregator.add(train_metrics)

                if should_log_train:
                    train_logs = {}
                    train_logs.update(train_log_aggregator.get_logs())
                    train_logs.update(logging_utils.metrics_assign_group(dataset_metrics, "dataset"))
                    train_logs = logging_utils.metrics_assign_group(train_logs, "train")
                    logger.write_scalars(step, train_logs)
                    logger.flush()
                    tqdm.write("\n" + pprint.pformat(train_logs) + "\n")



                

                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    save_checkpoint(train_state)

                first_step = False

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
