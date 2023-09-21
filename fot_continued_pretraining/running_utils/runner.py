from absl import flags
import mlxu
import json
from EasyLM.models.llama.llama_train import main as fot_main
from .runner_utils import *
flags.DEFINE_string("config", None, "Path to config file")

FLAGS = flags.FLAGS


def config_from_json():
    with open(FLAGS.config, "r") as f:
        return json.load(f)
    

    
def run_fot(_):
    def post_override_callback():
        add_time_to_workdir()
        FLAGS.logger_dir = FLAGS.workdir

    config_dict = config_from_json()
    return run_from_dict(main_fn=fot_main, config_dict=config_dict, post_override_callback=post_override_callback)


if __name__ == "__main__":
    mlxu.run(run_fot)
    