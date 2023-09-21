from typing import Dict, Any
import logging
import sys
import os
from absl import flags
from clu import platform
from datetime import datetime

flags.DEFINE_string("workdir", "./", "workdir for logs and checkpoints of the experiment")
flags.DEFINE_boolean("pod_job", False, "For running on tpu pods with slurm")
FLAGS = flags.FLAGS

LOGGER = logging.Logger("Experiment", level=logging.INFO)
LOGGER_HANDLER = logging.StreamHandler(sys.stderr)
LOGGER_HANDLER.setFormatter(logging.Formatter("[%(asctime)s] FoT Tunning [%(levelname)s] : %(message)s"))
LOGGER.addHandler(LOGGER_HANDLER)


def override_flags(overrides: Dict[str, Any]):
    for k, v in overrides.items():
        field_names = k.split(".")
        logging.info(f"Flags: Overriding {k} to {v}")
        f = FLAGS
        for fn in field_names[:-1]:
            f = f.__getattr__(fn)

        f.__setattr__(field_names[-1], v)


def prepare_for_run(config_dict: Dict[str, Any]):
    override_flags(overrides=config_dict)
    if FLAGS.pod_job:
        import jax

        for k in os.environ.keys():
            if k.startswith("SLURM"):
                os.environ.pop(k, None)
        jax.distributed.initialize()


def create_workdir():
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir")


def add_time_to_workdir():
    cur_time = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    FLAGS.workdir = os.path.join(FLAGS.workdir, str(cur_time))

def run_from_dict(main_fn, config_dict: Dict[str, Any], post_override_callback):
    prepare_for_run(config_dict=config_dict)
    post_override_callback()
    create_workdir()
    main_fn(None)

